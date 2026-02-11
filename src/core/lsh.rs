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

    /// Computes b-bit corrected Jaccard similarity between two signatures.
    ///
    /// Accounts for random collision probability introduced by XOR-folding
    /// 64-bit hashes to 8-bit values. For b=8, A_b = 1/256 ≈ 0.00390625.
    ///
    /// Formula: J_corrected = (J_observed - A_b) / (1 - A_b)
    ///
    /// This correction removes the upward bias caused by hash collisions,
    /// making disjoint sets converge toward similarity ≈ 0.0 instead of A_b.
    pub fn similarity_corrected(a: &[u8; SIGNATURE_SIZE], b: &[u8; SIGNATURE_SIZE]) -> f64 {
        const B: u32 = 8; // 8-bit signatures (u8)
        let a_b = 1.0 / (1u64 << B) as f64; // A_b = 1/2^b = 1/256

        let j_observed = Self::similarity(a, b);

        // Apply b-bit correction formula
        let j_corrected = (j_observed - a_b) / (1.0 - a_b);

        // Clamp to [0.0, 1.0] to handle numerical edge cases
        j_corrected.clamp(0.0, 1.0)
    }

    /// Computes similarity with a 95% confidence interval.
    ///
    /// MinHash similarity is a binomial estimator: each byte position is an
    /// independent Bernoulli trial with probability = true Jaccard similarity.
    /// The standard error is `sqrt(j * (1 - j) / n)` where `n = SIGNATURE_SIZE`.
    ///
    /// When the top-2 agents' confidence intervals overlap, their similarity
    /// estimates are statistically indistinguishable — a principled replacement
    /// for the hardcoded 5% tie-breaking threshold.
    pub fn similarity_with_confidence(
        a: &[u8; SIGNATURE_SIZE],
        b: &[u8; SIGNATURE_SIZE],
    ) -> ConfidenceInterval {
        let matching = a.iter().zip(b.iter()).filter(|(x, y)| x == y).count();
        let j = matching as f64 / SIGNATURE_SIZE as f64;
        let se = (j * (1.0 - j) / SIGNATURE_SIZE as f64).sqrt();
        ConfidenceInterval {
            point_estimate: j,
            lower_bound: (j - 1.96 * se).max(0.0),
            upper_bound: (j + 1.96 * se).min(1.0),
        }
    }

    /// Computes b-bit corrected similarity with a 95% confidence interval.
    ///
    /// Uses the corrected Jaccard estimate and corrected standard error
    /// formula that accounts for random collisions from XOR-folding.
    ///
    /// Corrected SE formula: SE = sqrt(C1 * (1 - C1) / n)
    /// where C1 = J_corrected + (1 - J_corrected) * A_b
    ///
    /// This provides statistically valid confidence intervals for the
    /// corrected Jaccard similarity estimate.
    pub fn similarity_with_confidence_corrected(
        a: &[u8; SIGNATURE_SIZE],
        b: &[u8; SIGNATURE_SIZE],
    ) -> ConfidenceInterval {
        const B: u32 = 8; // 8-bit signatures (u8)
        let a_b = 1.0 / (1u64 << B) as f64; // A_b = 1/2^b = 1/256

        // Use corrected Jaccard as point estimate
        let j_corrected = Self::similarity_corrected(a, b);

        // Compute corrected standard error
        // C1 = J + (1 - J) * A_b (expected matches including random collisions)
        let c1 = j_corrected + (1.0 - j_corrected) * a_b;
        let se = (c1 * (1.0 - c1) / SIGNATURE_SIZE as f64).sqrt();

        ConfidenceInterval {
            point_estimate: j_corrected,
            lower_bound: (j_corrected - 1.96 * se).max(0.0),
            upper_bound: (j_corrected + 1.96 * se).min(1.0),
        }
    }
}

/// 95% confidence interval for a Jaccard similarity estimate.
///
/// Based on the binomial standard error of MinHash: each of the 64 byte
/// positions is an independent Bernoulli trial.
#[derive(Debug, Clone, PartialEq)]
pub struct ConfidenceInterval {
    /// Point estimate of Jaccard similarity [0.0, 1.0].
    pub point_estimate: f64,
    /// Lower bound of 95% CI.
    pub lower_bound: f64,
    /// Upper bound of 95% CI.
    pub upper_bound: f64,
}

impl ConfidenceInterval {
    /// Returns true if this interval overlaps with another.
    ///
    /// Overlapping CIs indicate the two similarity estimates are statistically
    /// indistinguishable at the 95% confidence level.
    pub fn overlaps(&self, other: &ConfidenceInterval) -> bool {
        self.lower_bound <= other.upper_bound && other.lower_bound <= self.upper_bound
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
    fn confidence_interval_identical_inputs() {
        let hasher = MinHasher::new(SIGNATURE_SIZE);
        let sig = hasher.hash_key(b"legal translation services", &default_mode());
        let ci = MinHasher::similarity_with_confidence(&sig, &sig);
        assert_eq!(ci.point_estimate, 1.0);
        // SE = sqrt(1.0 * 0.0 / 64) = 0, so bounds collapse to 1.0
        assert_eq!(ci.lower_bound, 1.0);
        assert_eq!(ci.upper_bound, 1.0);
    }

    #[test]
    fn confidence_interval_bounds_contain_point_estimate() {
        let hasher = MinHasher::new(SIGNATURE_SIZE);
        let s1 = hasher.hash_key(b"translate legal document", &default_mode());
        let s2 = hasher.hash_key(b"legal translation service", &default_mode());
        let ci = MinHasher::similarity_with_confidence(&s1, &s2);
        assert!(ci.lower_bound <= ci.point_estimate);
        assert!(ci.point_estimate <= ci.upper_bound);
        assert!(ci.lower_bound >= 0.0);
        assert!(ci.upper_bound <= 1.0);
    }

    #[test]
    fn confidence_interval_width_matches_formula() {
        let hasher = MinHasher::new(SIGNATURE_SIZE);
        let s1 = hasher.hash_key(b"translate legal document", &default_mode());
        let s2 = hasher.hash_key(b"code review and testing", &default_mode());
        let ci = MinHasher::similarity_with_confidence(&s1, &s2);
        let j = ci.point_estimate;
        let expected_se = (j * (1.0 - j) / SIGNATURE_SIZE as f64).sqrt();
        let expected_lower = (j - 1.96 * expected_se).max(0.0);
        let expected_upper = (j + 1.96 * expected_se).min(1.0);
        assert!(
            (ci.lower_bound - expected_lower).abs() < 1e-10,
            "lower bound mismatch: {} vs {}",
            ci.lower_bound,
            expected_lower
        );
        assert!(
            (ci.upper_bound - expected_upper).abs() < 1e-10,
            "upper bound mismatch: {} vs {}",
            ci.upper_bound,
            expected_upper
        );
    }

    #[test]
    fn confidence_interval_overlaps_identical() {
        let ci = ConfidenceInterval {
            point_estimate: 0.5,
            lower_bound: 0.38,
            upper_bound: 0.62,
        };
        assert!(ci.overlaps(&ci), "interval should overlap with itself");
    }

    #[test]
    fn confidence_interval_overlaps_adjacent() {
        let ci1 = ConfidenceInterval {
            point_estimate: 0.4,
            lower_bound: 0.28,
            upper_bound: 0.52,
        };
        let ci2 = ConfidenceInterval {
            point_estimate: 0.5,
            lower_bound: 0.38,
            upper_bound: 0.62,
        };
        assert!(ci1.overlaps(&ci2), "overlapping CIs should report overlap");
        assert!(ci2.overlaps(&ci1), "overlaps should be symmetric");
    }

    #[test]
    fn confidence_interval_no_overlap_disjoint() {
        let ci1 = ConfidenceInterval {
            point_estimate: 0.2,
            lower_bound: 0.10,
            upper_bound: 0.30,
        };
        let ci2 = ConfidenceInterval {
            point_estimate: 0.8,
            lower_bound: 0.70,
            upper_bound: 0.90,
        };
        assert!(!ci1.overlaps(&ci2), "disjoint CIs should not overlap");
        assert!(!ci2.overlaps(&ci1), "no-overlap should be symmetric");
    }

    #[test]
    fn confidence_interval_known_jaccard_coverage() {
        // Construct two signatures with exactly k matching positions out of 64.
        // The true Jaccard is k/64. The CI should contain k/64.
        for k in [0, 10, 32, 50, 64] {
            let mut a = [0u8; SIGNATURE_SIZE];
            let mut b = [0u8; SIGNATURE_SIZE];
            // First k positions match, rest differ
            for i in 0..SIGNATURE_SIZE {
                a[i] = i as u8;
                b[i] = if i < k {
                    i as u8
                } else {
                    (i as u8).wrapping_add(128)
                };
            }
            let ci = MinHasher::similarity_with_confidence(&a, &b);
            let true_jaccard = k as f64 / SIGNATURE_SIZE as f64;
            assert!(
                (ci.point_estimate - true_jaccard).abs() < 1e-10,
                "k={}: point estimate {} != true Jaccard {}",
                k,
                ci.point_estimate,
                true_jaccard
            );
            assert!(
                ci.lower_bound <= true_jaccard && true_jaccard <= ci.upper_bound,
                "k={}: CI [{}, {}] does not contain true Jaccard {}",
                k,
                ci.lower_bound,
                ci.upper_bound,
                true_jaccard
            );
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

    // --- Task 6.1: B-bit collision correction tests ---

    #[test]
    fn b_bit_corrected_similarity_for_disjoint_sets() {
        // Completely disjoint sets should have corrected similarity → 0.0
        // (uncorrected similarity is biased upward by collision probability A_b = 1/256)
        let hasher = MinHasher::new(SIGNATURE_SIZE);
        let s1 = hasher.hash_key(b"quantum physics experiments", &default_mode());
        let s2 = hasher.hash_key(b"underwater basket weaving", &default_mode());

        let corrected = MinHasher::similarity_corrected(&s1, &s2);
        let uncorrected = MinHasher::similarity(&s1, &s2);

        // Corrected similarity should be close to 0.0 for disjoint sets
        assert!(
            corrected < 0.05,
            "disjoint sets should have corrected similarity near 0.0, got {:.3}",
            corrected
        );

        // Verify correction reduces or maintains similarity (≤) compared to uncorrected
        // (when uncorrected is very low, correction may clamp both to 0.0)
        assert!(
            corrected <= uncorrected,
            "corrected ({:.3}) should be ≤ uncorrected ({:.3})",
            corrected,
            uncorrected
        );
    }

    #[test]
    fn b_bit_correction_formula_applied_correctly() {
        // Construct signatures with known matching count
        // to verify the correction formula: J_corrected = (J_observed - A_b) / (1 - A_b)
        let mut a = [0u8; SIGNATURE_SIZE];
        let mut b = [0u8; SIGNATURE_SIZE];

        // 10 matching positions out of 64
        for i in 0..SIGNATURE_SIZE {
            a[i] = i as u8;
            b[i] = if i < 10 {
                i as u8 // Match
            } else {
                (i as u8).wrapping_add(128) // Mismatch
            };
        }

        let j_observed = 10.0 / 64.0;
        let a_b = 1.0 / 256.0; // For b=8
        let expected_corrected = (j_observed - a_b) / (1.0 - a_b);

        let actual_corrected = MinHasher::similarity_corrected(&a, &b);

        assert!(
            (actual_corrected - expected_corrected).abs() < 1e-10,
            "correction formula mismatch: got {:.6}, expected {:.6}",
            actual_corrected,
            expected_corrected
        );
    }

    #[test]
    fn b_bit_corrected_similarity_identical_inputs_still_one() {
        let hasher = MinHasher::new(SIGNATURE_SIZE);
        let sig = hasher.hash_key(b"legal translation services", &default_mode());

        let corrected = MinHasher::similarity_corrected(&sig, &sig);

        assert_eq!(
            corrected, 1.0,
            "identical inputs should still have corrected similarity = 1.0"
        );
    }

    // --- Task 6.2: Corrected confidence interval tests ---

    #[test]
    fn b_bit_corrected_confidence_interval_formula() {
        // Test that corrected CI uses formula: SE = sqrt(C1 * (1 - C1) / n)
        // where C1 = J + (1 - J) * A_b
        let mut a = [0u8; SIGNATURE_SIZE];
        let mut b = [0u8; SIGNATURE_SIZE];

        // 20 matching positions out of 64
        for i in 0..SIGNATURE_SIZE {
            a[i] = i as u8;
            b[i] = if i < 20 {
                i as u8 // Match
            } else {
                (i as u8).wrapping_add(128) // Mismatch
            };
        }

        let ci = MinHasher::similarity_with_confidence_corrected(&a, &b);

        // Verify corrected Jaccard is used
        let j_corrected = MinHasher::similarity_corrected(&a, &b);
        assert!(
            (ci.point_estimate - j_corrected).abs() < 1e-10,
            "CI point estimate should use corrected Jaccard"
        );

        // Verify corrected SE formula
        let a_b = 1.0 / 256.0;
        let c1 = j_corrected + (1.0 - j_corrected) * a_b;
        let expected_se = (c1 * (1.0 - c1) / SIGNATURE_SIZE as f64).sqrt();
        let expected_lower = (j_corrected - 1.96 * expected_se).max(0.0);
        let expected_upper = (j_corrected + 1.96 * expected_se).min(1.0);

        assert!(
            (ci.lower_bound - expected_lower).abs() < 1e-10,
            "lower bound should use corrected SE: got {:.6}, expected {:.6}",
            ci.lower_bound,
            expected_lower
        );
        assert!(
            (ci.upper_bound - expected_upper).abs() < 1e-10,
            "upper bound should use corrected SE: got {:.6}, expected {:.6}",
            ci.upper_bound,
            expected_upper
        );
    }

    #[test]
    fn b_bit_corrected_ci_bounds_contain_point_estimate() {
        let hasher = MinHasher::new(SIGNATURE_SIZE);
        let s1 = hasher.hash_key(b"translate legal document", &default_mode());
        let s2 = hasher.hash_key(b"legal translation service", &default_mode());

        let ci = MinHasher::similarity_with_confidence_corrected(&s1, &s2);

        assert!(ci.lower_bound <= ci.point_estimate);
        assert!(ci.point_estimate <= ci.upper_bound);
        assert!(ci.lower_bound >= 0.0);
        assert!(ci.upper_bound <= 1.0);
    }

    #[test]
    fn b_bit_corrected_ci_identical_inputs_tight_bounds() {
        let hasher = MinHasher::new(SIGNATURE_SIZE);
        let sig = hasher.hash_key(b"legal translation services", &default_mode());

        let ci = MinHasher::similarity_with_confidence_corrected(&sig, &sig);

        assert_eq!(ci.point_estimate, 1.0);
        // For J=1.0, C1 = 1.0 + 0 * A_b = 1.0, SE = sqrt(1.0 * 0.0 / 64) = 0
        // So bounds should collapse to 1.0
        assert!(
            (ci.lower_bound - 1.0).abs() < 1e-10,
            "lower bound should be ~1.0 for identical inputs"
        );
        assert!(
            (ci.upper_bound - 1.0).abs() < 1e-10,
            "upper bound should be ~1.0 for identical inputs"
        );
    }

    // -----------------------------------------------------------------------
    // Property-Based Tests (Requirement 1, 17)
    // -----------------------------------------------------------------------

    use proptest::prelude::*;

    proptest! {
        /// Property test: CI coverage verification with ground truth (Requirement 1).
        ///
        /// Verifies that corrected confidence intervals contain the true Jaccard
        /// similarity at ≥93% coverage rate over many trials.
        #[test]
        fn proptest_ci_coverage_with_ground_truth(target_jaccard in 0.05f64..0.95f64) {
            const SET_SIZE: usize = 100;

            use xxhash_rust::xxh3::xxh3_64_with_seed;

            // Generate set A (100 distinct random-ish tokens using hash-based generation)
            // This ensures tokens don't share byte n-grams accidentally
            let set_a: Vec<String> = (0..SET_SIZE)
                .map(|i| {
                    let hash = xxh3_64_with_seed(b"set_a", i as u64);
                    format!("tkn{:016x}", hash) // Hex string ensures distinct byte patterns
                })
                .collect();

            // Compute overlap size to achieve target Jaccard
            // J = |A ∩ B| / |A ∪ B|
            // For |A| = |B| = 100: J = overlap / (200 - overlap)
            // Solving for overlap: overlap = J * 200 / (2 - J)
            let overlap_size = ((target_jaccard * (2.0 * SET_SIZE as f64)) / (2.0 - target_jaccard)).floor() as usize;
            let overlap_size = overlap_size.min(SET_SIZE); // Cap at SET_SIZE

            // Create set B: overlap_size tokens from A + (SET_SIZE - overlap_size) fresh tokens
            let mut set_b: Vec<String> = set_a.iter().take(overlap_size).cloned().collect();
            for i in overlap_size..SET_SIZE {
                let hash = xxh3_64_with_seed(b"set_b", i as u64);
                set_b.push(format!("tkn{:016x}", hash));
            }

            // Compute true Jaccard from actual sets
            let set_a_set: std::collections::HashSet<_> = set_a.iter().collect();
            let set_b_set: std::collections::HashSet<_> = set_b.iter().collect();
            let intersection_size = set_a_set.intersection(&set_b_set).count();
            let union_size = set_a_set.union(&set_b_set).count();
            let true_jaccard = if union_size > 0 {
                intersection_size as f64 / union_size as f64
            } else {
                0.0
            };

            // Generate MinHash signatures using Words mode (token-level, not byte n-grams)
            let hasher = MinHasher::new(SIGNATURE_SIZE);
            let mode = ShingleMode::Words; // Use word-level tokens to match set semantics

            // Join tokens with spaces to create capability strings
            let text_a = set_a.join(" ");
            let text_b = set_b.join(" ");

            let sig_a = hasher.hash_key(text_a.as_bytes(), &mode);
            let sig_b = hasher.hash_key(text_b.as_bytes(), &mode);

            // Compute corrected CI
            let ci = MinHasher::similarity_with_confidence_corrected(&sig_a, &sig_b);

            // Verify CI contains true Jaccard (this is a statistical property,
            // individual trials may fail, but ≥93% should pass)
            prop_assert!(
                ci.lower_bound <= true_jaccard && true_jaccard <= ci.upper_bound,
                "CI [{:.4}, {:.4}] should contain true Jaccard {:.4} (target was {:.4}, overlap={}/{})",
                ci.lower_bound, ci.upper_bound, true_jaccard, target_jaccard, overlap_size, SET_SIZE
            );
        }

        /// Property test: CI bounds ordering (Requirement 17).
        ///
        /// Verifies that for all confidence interval computations,
        /// lower_bound ≤ point_estimate ≤ upper_bound.
        #[test]
        fn proptest_ci_bounds_ordering(
            jaccard in 0.0f64..=1.0f64,
        ) {
            // Create synthetic signatures with specific Jaccard similarity
            let matches = (jaccard * SIGNATURE_SIZE as f64).round() as usize;
            let mut sig_a = [0u8; SIGNATURE_SIZE];
            let mut sig_b = [0u8; SIGNATURE_SIZE];

            for i in 0..SIGNATURE_SIZE {
                sig_a[i] = i as u8;
                sig_b[i] = if i < matches { i as u8 } else { (i + 100) as u8 };
            }

            let ci = MinHasher::similarity_with_confidence_corrected(&sig_a, &sig_b);

            prop_assert!(ci.lower_bound <= ci.point_estimate,
                "lower_bound {} should be ≤ point_estimate {}", ci.lower_bound, ci.point_estimate);
            prop_assert!(ci.point_estimate <= ci.upper_bound,
                "point_estimate {} should be ≤ upper_bound {}", ci.point_estimate, ci.upper_bound);
            prop_assert!(ci.lower_bound >= 0.0,
                "lower_bound {} should be ≥ 0.0", ci.lower_bound);
            prop_assert!(ci.upper_bound <= 1.0,
                "upper_bound {} should be ≤ 1.0", ci.upper_bound);
        }
    }
}
