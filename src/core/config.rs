// ALPS Discovery SDK — Configuration Types
//
// Minimal configuration types needed for local agent discovery.

use serde::{Deserialize, Serialize};

/// Controls how input text is decomposed into shingles for MinHash.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShingleMode {
    /// Character n-grams of the specified byte width.
    Bytes(usize),
    /// Word-level tokens (split on whitespace, underscore, hyphen).
    /// Better for natural language queries.
    Words,
    /// Both byte shingles AND word shingles combined.
    /// A word match OR a character n-gram match can contribute.
    /// Best recall for short natural language capability strings.
    Hybrid {
        /// Byte n-gram width (typically 3).
        byte_width: usize,
    },
}

/// Configuration for the LSH (Locality-Sensitive Hashing) subsystem.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LshConfig {
    /// Minimum similarity for a discovery result to be returned (default: 0.1).
    /// Results below this threshold are filtered as noise.
    pub similarity_threshold: f64,
    /// Dissimilarity threshold for considering a non-match.
    pub dissimilarity_threshold: f64,
    /// Number of hash dimensions (default: 128 for MinHash).
    pub dimensions: usize,
    /// LSH family selection (default: MinHash for set-similarity).
    pub family: LshFamily,
    /// How to decompose input text into shingles (default: Hybrid with 3-byte n-grams).
    pub shingle_mode: ShingleMode,
}

impl Default for LshConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.1,
            dissimilarity_threshold: 0.3,
            dimensions: 64,
            family: LshFamily::MinHash,
            shingle_mode: ShingleMode::Hybrid { byte_width: 3 },
        }
    }
}

impl LshConfig {
    /// Maximum effective dimensions (clamped to signature size).
    pub const MAX_DIMENSIONS: usize = 64;

    /// Returns the effective dimension count, clamped to signature size.
    ///
    /// Dimensions beyond `SIGNATURE_SIZE` (64) use a deterministic fill hash
    /// rather than independent hash functions, which breaks MinHash independence.
    pub fn effective_dimensions(&self) -> usize {
        self.dimensions.min(Self::MAX_DIMENSIONS)
    }
}

/// LSH hash family selection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LshFamily {
    /// MinHash for set-similarity (Jaccard). Default, 128 dimensions.
    MinHash,
    /// Random projection for numeric key spaces (cosine similarity).
    RandomProjection,
}

/// Query configuration for discovery signals.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QueryConfig {
    /// Alpha parameter for truth bias weighting.
    pub truth_bias_alpha: f64,
    /// Similarity threshold for query satisfaction.
    pub satisfaction_threshold: f64,
}

impl Default for QueryConfig {
    fn default() -> Self {
        Self {
            truth_bias_alpha: 0.7,
            satisfaction_threshold: 0.5,
        }
    }
}

/// Configuration for the spore (placeholder for enzyme trait signature).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SporeConfig {
    /// Number of storage rings.
    pub ring_count: usize,
}

impl Default for SporeConfig {
    fn default() -> Self {
        Self { ring_count: 5 }
    }
}
