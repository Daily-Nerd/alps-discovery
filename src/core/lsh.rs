// ALPS-SLN Protocol — LSH-based Query Signatures and Chemistry Matching
//
// Implements MinHash LSH for set-similarity with configurable dimensions
// and shingle modes. Generates semantic signatures for capability matching
// and query signatures for Tendril routing.
//
// Key properties:
// - Similar keys produce query signatures with high Jaccard similarity
// - Unrelated keys produce query signatures with low similarity
// - Hybrid shingling (word + character n-grams) improves matching on
//   short natural language capability strings
//
// Chemistry matching:
// - Chemistry deposit uses element-wise minimum to preserve MinHash union property
// - Similarity = matching positions / total positions (Jaccard estimate)

use crate::core::chemistry::QuerySignature;
use crate::core::config::{LshConfig, LshFamily, ShingleMode};

/// Number of bytes in a MinHash signature (matches Chemistry/QuerySignature size).
pub const SIGNATURE_SIZE: usize = 64;

/// A MinHash hash family that generates LSH signatures from byte sequences.
///
/// Uses xxhash with different seeds to produce independent hash functions.
/// Each hash function maps an input to a minimum hash value, producing a
/// signature where similar inputs yield similar signatures (locality-sensitive).
pub struct MinHasher {
    /// Seeds for independent hash functions, one per signature dimension.
    seeds: Vec<u64>,
}

impl MinHasher {
    /// Creates a new MinHasher with the specified number of dimensions.
    ///
    /// Each dimension uses a unique seed derived from a base seed to ensure
    /// independent hash functions. The signature is truncated or padded
    /// to fit `SIGNATURE_SIZE` bytes.
    pub fn new(dimensions: usize) -> Self {
        // Generate deterministic seeds: one per dimension.
        // Use a simple linear congruential approach seeded from dimension index.
        let seeds: Vec<u64> = (0..dimensions)
            .map(|i| {
                // Golden ratio hash to spread seeds uniformly
                let base = 0x9E3779B97F4A7C15u64;
                base.wrapping_mul(i as u64 + 1)
            })
            .collect();
        Self { seeds }
    }

    /// Generates a MinHash signature from a byte key using the specified shingle mode.
    ///
    /// For each hash function, the minimum hash value across all shingles
    /// is retained, producing a signature where similar sets of shingles
    /// yield similar minimum hash values.
    ///
    /// Returns a `SIGNATURE_SIZE`-byte array suitable for use as a
    /// `semantic_signature` in a Digest or as a `QuerySignature`.
    pub fn hash_key(&self, key: &[u8], mode: &ShingleMode) -> [u8; SIGNATURE_SIZE] {
        // Lowercase the input for case-insensitive matching.
        let lowered: Vec<u8> = key.iter().map(|b| b.to_ascii_lowercase()).collect();
        let key = &lowered;

        let mut signature = [0u8; SIGNATURE_SIZE];
        let num_hashes = self.seeds.len().min(SIGNATURE_SIZE);

        for (i, &seed) in self.seeds.iter().take(num_hashes).enumerate() {
            let min_hash = match mode {
                ShingleMode::Bytes(size) => self.min_hash_bytes(key, *size, seed),
                ShingleMode::Words => self.min_hash_words(key, seed),
                ShingleMode::Hybrid { byte_width } => {
                    let byte_min = self.min_hash_bytes(key, *byte_width, seed);
                    let word_min = self.min_hash_words(key, seed);
                    byte_min.min(word_min)
                }
            };

            // Map the 64-bit min hash to a single byte via XOR-folding.
            // XOR-folding all 8 bytes together produces much better distribution
            // than taking only the lowest byte, because it mixes information from
            // all bits of the hash value.
            signature[i] = fold_hash_to_byte(min_hash);
        }

        // Fill remaining positions if dimensions < SIGNATURE_SIZE
        // with deterministic values derived from the key hash.
        if num_hashes < SIGNATURE_SIZE {
            let fill_hash = xxhash_with_seed(key, 0xDEADBEEF);
            for (i, slot) in signature.iter_mut().enumerate().skip(num_hashes) {
                *slot = fold_hash_to_byte(fill_hash.wrapping_add(i as u64));
            }
        }

        signature
    }

    /// Compute min hash across byte n-gram shingles.
    fn min_hash_bytes(&self, key: &[u8], shingle_size: usize, seed: u64) -> u64 {
        let mut min_hash = u64::MAX;
        if key.len() < shingle_size {
            // Short key: hash the entire key as one shingle
            min_hash = xxhash_with_seed(key, seed);
        } else {
            for window in key.windows(shingle_size) {
                let h = xxhash_with_seed(window, seed);
                min_hash = min_hash.min(h);
            }
        }
        min_hash
    }

    /// Compute min hash across word-level shingles (unigrams).
    /// Splits on whitespace, underscore, hyphen, period, comma.
    fn min_hash_words(&self, key: &[u8], seed: u64) -> u64 {
        let mut min_hash = u64::MAX;
        let mut found_word = false;
        for word in key
            .split(|&b| b == b' ' || b == b'_' || b == b'-' || b == b'.' || b == b',' || b == b'\t')
        {
            if !word.is_empty() {
                let h = xxhash_with_seed(word, seed);
                min_hash = min_hash.min(h);
                found_word = true;
            }
        }
        if !found_word {
            return xxhash_with_seed(key, seed);
        }
        min_hash
    }

    /// Computes the estimated Jaccard similarity between two signatures.
    ///
    /// Returns the fraction of positions where the byte values match,
    /// which approximates the Jaccard similarity of the underlying sets.
    pub fn similarity(a: &[u8; SIGNATURE_SIZE], b: &[u8; SIGNATURE_SIZE]) -> f64 {
        let matching = a.iter().zip(b.iter()).filter(|(x, y)| x == y).count();
        matching as f64 / SIGNATURE_SIZE as f64
    }
}

/// Generates a QuerySignature from a byte key using the configured LSH family.
///
/// This is the primary entry point for creating query signatures for Tendril
/// routing. The returned signature can be compared against hypha Chemistry
/// using `Chemistry::similarity()`.
///
/// Both MinHash and RandomProjection families produce SIGNATURE_SIZE-byte
/// signatures that are compatible with Chemistry accumulation.
pub fn compute_query_signature(key: &[u8], config: &LshConfig) -> QuerySignature {
    let minhash = compute_signature(key, config);
    QuerySignature::new(minhash)
}

/// Generates the semantic signature for a Nutrient digest.
///
/// This signature is deposited on hyphae as chemistry when a Nutrient
/// traverses them, enabling future Tendrils with similar query signatures
/// to be attracted to those hyphae.
pub fn compute_semantic_signature(key: &[u8], config: &LshConfig) -> [u8; SIGNATURE_SIZE] {
    compute_signature(key, config)
}

/// Internal: generates a signature using the configured LSH family.
fn compute_signature(key: &[u8], config: &LshConfig) -> [u8; SIGNATURE_SIZE] {
    match config.family {
        LshFamily::MinHash => {
            let hasher = MinHasher::new(config.effective_dimensions());
            hasher.hash_key(key, &config.shingle_mode)
        }
        LshFamily::RandomProjection => {
            // RandomProjection treats the key as a numeric vector and projects
            // it onto random hyperplanes. Each bit of the signature indicates
            // which side of a hyperplane the input falls on.
            //
            // For byte keys, we interpret the key as a vector of u8 values
            // and project using hash-derived random vectors.
            random_projection_signature(key, config.effective_dimensions())
        }
    }
}

/// Generates a random projection signature for numeric key spaces.
///
/// Projects the byte key (interpreted as a numeric vector) onto
/// random hyperplanes defined by hash-derived random vectors.
/// The result approximates cosine similarity between key vectors.
fn random_projection_signature(key: &[u8], dimensions: usize) -> [u8; SIGNATURE_SIZE] {
    let mut signature = [0u8; SIGNATURE_SIZE];
    let num_projections = dimensions.min(SIGNATURE_SIZE * 8); // 1 bit per projection

    for proj_idx in 0..num_projections {
        // Each projection uses a random vector derived from the projection index.
        let seed = 0xA5A5A5A5u64.wrapping_mul(proj_idx as u64 + 1);
        let mut dot_product: i64 = 0;

        for (byte_idx, &byte_val) in key.iter().enumerate() {
            // Generate a random coefficient for this dimension.
            let coeff_hash = xxhash_with_seed(&(byte_idx as u64).to_le_bytes(), seed);
            // Map to {-1, +1} based on high bit
            let coeff: i64 = if coeff_hash & 1 == 0 { 1 } else { -1 };
            dot_product += coeff * byte_val as i64;
        }

        // Set the bit based on the sign of the dot product
        let byte_pos = proj_idx / 8;
        let bit_pos = proj_idx % 8;
        if byte_pos < SIGNATURE_SIZE && dot_product >= 0 {
            signature[byte_pos] |= 1 << bit_pos;
        }
    }

    signature
}

/// Creates a wildcard query signature for germination Tendrils.
///
/// Returns all-0xFF signature that partially matches any hypha's chemistry,
/// ensuring germination Tendrils are routed via diameter-weighted exploration
/// rather than being blocked by the minimum chemistry threshold.
pub fn wildcard_signature() -> QuerySignature {
    QuerySignature::default() // All 0xFF
}

/// XOR-folds a 64-bit hash value down to 8 bits.
///
/// Mixes all 8 bytes of the hash into a single byte, providing much better
/// distribution than simply taking the lowest byte. This is critical for
/// MinHash signatures where collision rate directly impacts similarity accuracy.
#[inline]
fn fold_hash_to_byte(hash: u64) -> u8 {
    let bytes = hash.to_le_bytes();
    bytes[0] ^ bytes[1] ^ bytes[2] ^ bytes[3] ^ bytes[4] ^ bytes[5] ^ bytes[6] ^ bytes[7]
}

/// Computes xxhash of data with a given seed.
///
/// Uses xxhash-rust's XXH3 64-bit variant for fast, high-quality hashing.
fn xxhash_with_seed(data: &[u8], seed: u64) -> u64 {
    xxhash_rust::xxh3::xxh3_64_with_seed(data, seed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::config::LshConfig;

    fn default_mode() -> ShingleMode {
        ShingleMode::Hybrid { byte_width: 3 }
    }

    #[test]
    fn signature_is_deterministic() {
        let hasher = MinHasher::new(SIGNATURE_SIZE);
        let s1 = hasher.hash_key(b"legal translation services", &default_mode());
        let s2 = hasher.hash_key(b"legal translation services", &default_mode());
        assert_eq!(s1, s2);
    }

    #[test]
    fn different_inputs_produce_different_signatures() {
        let hasher = MinHasher::new(SIGNATURE_SIZE);
        let s1 = hasher.hash_key(b"legal translation", &default_mode());
        let s2 = hasher.hash_key(b"code review", &default_mode());
        assert_ne!(s1, s2);
    }

    #[test]
    fn identical_inputs_have_perfect_similarity() {
        let hasher = MinHasher::new(SIGNATURE_SIZE);
        let sig = hasher.hash_key(b"legal translation services", &default_mode());
        assert_eq!(MinHasher::similarity(&sig, &sig), 1.0);
    }

    #[test]
    fn disjoint_inputs_have_low_similarity() {
        let hasher = MinHasher::new(SIGNATURE_SIZE);
        let s1 = hasher.hash_key(b"legal translation services", &default_mode());
        let s2 = hasher.hash_key(b"quantum physics experiments", &default_mode());
        let sim = MinHasher::similarity(&s1, &s2);
        assert!(
            sim < 0.25,
            "disjoint inputs should have low similarity, got {:.3}",
            sim
        );
    }

    #[test]
    fn similar_inputs_rank_higher_than_dissimilar() {
        let hasher = MinHasher::new(SIGNATURE_SIZE);
        let query = hasher.hash_key(b"translate legal document", &default_mode());
        let similar = hasher.hash_key(b"legal translation service", &default_mode());
        let dissimilar = hasher.hash_key(b"image processing pipeline", &default_mode());

        let sim_score = MinHasher::similarity(&query, &similar);
        let dis_score = MinHasher::similarity(&query, &dissimilar);
        assert!(
            sim_score > dis_score,
            "similar ({:.3}) should exceed dissimilar ({:.3})",
            sim_score,
            dis_score
        );
    }

    #[test]
    fn short_key_handling() {
        let hasher = MinHasher::new(SIGNATURE_SIZE);
        // Keys shorter than shingle size should not panic
        let s1 = hasher.hash_key(b"ab", &default_mode());
        let s2 = hasher.hash_key(b"a", &default_mode());
        let s3 = hasher.hash_key(b"", &default_mode());
        // They should produce valid signatures
        assert_eq!(s1.len(), SIGNATURE_SIZE);
        assert_eq!(s2.len(), SIGNATURE_SIZE);
        assert_eq!(s3.len(), SIGNATURE_SIZE);
    }

    #[test]
    fn case_insensitive_matching() {
        let hasher = MinHasher::new(SIGNATURE_SIZE);
        let s1 = hasher.hash_key(b"Translate Legal Contract", &default_mode());
        let s2 = hasher.hash_key(b"translate legal contract", &default_mode());
        assert_eq!(
            MinHasher::similarity(&s1, &s2),
            1.0,
            "case should not affect matching"
        );
    }

    #[test]
    fn word_mode_matches_shared_words() {
        let hasher = MinHasher::new(SIGNATURE_SIZE);
        let mode = ShingleMode::Words;
        // These share "translate" and "legal"
        let s1 = hasher.hash_key(b"translate legal contract", &mode);
        let s2 = hasher.hash_key(b"translate legal documents with expertise", &mode);
        let sim = MinHasher::similarity(&s1, &s2);
        assert!(
            sim > 0.1,
            "shared words should produce meaningful similarity, got {:.3}",
            sim
        );
    }

    #[test]
    fn hybrid_mode_beats_bytes_only_for_natural_language() {
        let hasher = MinHasher::new(SIGNATURE_SIZE);
        let query = hasher.hash_key(b"translate legal contract", &default_mode());
        let cap = hasher.hash_key(
            b"translate text between languages with legal expertise",
            &default_mode(),
        );
        let hybrid_sim = MinHasher::similarity(&query, &cap);

        let bytes_mode = ShingleMode::Bytes(3);
        let query_b = hasher.hash_key(b"translate legal contract", &bytes_mode);
        let cap_b = hasher.hash_key(
            b"translate text between languages with legal expertise",
            &bytes_mode,
        );
        let bytes_sim = MinHasher::similarity(&query_b, &cap_b);

        assert!(
            hybrid_sim >= bytes_sim,
            "hybrid ({:.3}) should be >= bytes-only ({:.3})",
            hybrid_sim,
            bytes_sim
        );
    }

    #[test]
    fn compute_signature_uses_config() {
        let config = LshConfig::default();
        let s1 = compute_signature(b"test key", &config);
        let s2 = compute_signature(b"test key", &config);
        assert_eq!(s1, s2, "same config should produce same signature");
    }

    #[test]
    fn query_and_semantic_signatures_match() {
        let config = LshConfig::default();
        let qs = compute_query_signature(b"test key", &config);
        let ss = compute_semantic_signature(b"test key", &config);
        assert_eq!(
            qs.minhash, ss,
            "query and semantic signatures should match for same key"
        );
    }

    #[test]
    fn wildcard_signature_is_all_0xff() {
        let ws = wildcard_signature();
        assert!(ws.minhash.iter().all(|&b| b == 0xFF));
    }

    #[test]
    fn similarity_range_is_valid() {
        let hasher = MinHasher::new(SIGNATURE_SIZE);
        let inputs = [
            b"legal translation".as_slice(),
            b"code review".as_slice(),
            b"data processing".as_slice(),
            b"summarize document".as_slice(),
        ];
        for a in &inputs {
            for b in &inputs {
                let sa = hasher.hash_key(a, &default_mode());
                let sb = hasher.hash_key(b, &default_mode());
                let sim = MinHasher::similarity(&sa, &sb);
                assert!(
                    (0.0..=1.0).contains(&sim),
                    "similarity should be in [0,1], got {}",
                    sim
                );
            }
        }
    }

    #[test]
    fn statistical_unrelated_keys_low_average() {
        let hasher = MinHasher::new(SIGNATURE_SIZE);
        let keys: Vec<&[u8]> = vec![
            b"legal translation services",
            b"quantum physics experiments",
            b"underwater basket weaving",
            b"cryptocurrency mining hardware",
            b"organic chemistry formulas",
            b"satellite orbit calculations",
            b"medieval history research",
            b"deep sea exploration vessels",
            b"volcanic eruption prediction",
            b"arctic wildlife conservation",
        ];

        let mut total_sim = 0.0;
        let mut count = 0;
        for i in 0..keys.len() {
            for j in (i + 1)..keys.len() {
                let sa = hasher.hash_key(keys[i], &default_mode());
                let sb = hasher.hash_key(keys[j], &default_mode());
                total_sim += MinHasher::similarity(&sa, &sb);
                count += 1;
            }
        }
        let avg = total_sim / count as f64;
        assert!(
            avg < 0.2,
            "average similarity of unrelated keys should be low, got {:.3}",
            avg
        );
    }
}
