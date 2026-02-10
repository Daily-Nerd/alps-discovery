// ALPS-SLN Protocol — LSH-based Query Signatures and Chemistry Matching
//
// Task 3.4: Implements MinHash LSH for set-similarity with configurable
// dimensions. Generates semantic signatures for Nutrient digests and
// query signatures for Tendril routing.
//
// Key properties (R3.3):
// - Similar keys produce query signatures with cosine similarity > 0.7
// - Unrelated keys produce query signatures with cosine similarity < 0.3 (in expectation)
//
// Chemistry matching (R18.5):
// - Chemistry deposit uses element-wise minimum to preserve MinHash union property
// - Similarity = matching positions / total positions (Jaccard estimate)

use crate::core::chemistry::QuerySignature;
use crate::core::config::{LshConfig, LshFamily};

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

    /// Generates a MinHash signature from a byte key.
    ///
    /// The key is treated as a set of overlapping shingles (n-grams).
    /// For each hash function, the minimum hash value across all shingles
    /// is retained, producing a signature where similar sets of shingles
    /// yield similar minimum hash values.
    ///
    /// Returns a `SIGNATURE_SIZE`-byte array suitable for use as a
    /// `semantic_signature` in a Digest or as a `QuerySignature`.
    pub fn hash_key(&self, key: &[u8]) -> [u8; SIGNATURE_SIZE] {
        let mut signature = [0u8; SIGNATURE_SIZE];
        let num_hashes = self.seeds.len().min(SIGNATURE_SIZE);

        // Generate shingles from the key (overlapping 3-byte windows).
        // For keys shorter than 3 bytes, use the entire key as a single shingle.
        let shingle_size = 3;

        for (i, &seed) in self.seeds.iter().take(num_hashes).enumerate() {
            let mut min_hash = u64::MAX;

            if key.len() < shingle_size {
                // Short key: hash the entire key as one shingle
                min_hash = xxhash_with_seed(key, seed);
            } else {
                // Hash each shingle and keep the minimum
                for window in key.windows(shingle_size) {
                    let h = xxhash_with_seed(window, seed);
                    min_hash = min_hash.min(h);
                }
            }

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
            let hasher = MinHasher::new(config.dimensions);
            hasher.hash_key(key)
        }
        LshFamily::RandomProjection => {
            // RandomProjection treats the key as a numeric vector and projects
            // it onto random hyperplanes. Each bit of the signature indicates
            // which side of a hyperplane the input falls on.
            //
            // For byte keys, we interpret the key as a vector of u8 values
            // and project using hash-derived random vectors.
            random_projection_signature(key, config.dimensions)
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
