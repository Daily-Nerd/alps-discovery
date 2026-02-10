// ALPS Discovery SDK — Configuration Types
//
// Minimal configuration types needed for local agent discovery.

use serde::{Deserialize, Serialize};

/// Configuration for the LSH (Locality-Sensitive Hashing) subsystem.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LshConfig {
    /// Similarity threshold for considering a match.
    pub similarity_threshold: f64,
    /// Dissimilarity threshold for considering a non-match.
    pub dissimilarity_threshold: f64,
    /// Number of hash dimensions (default: 128 for MinHash).
    pub dimensions: usize,
    /// LSH family selection (default: MinHash for set-similarity).
    pub family: LshFamily,
}

impl Default for LshConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.7,
            dissimilarity_threshold: 0.3,
            dimensions: 128,
            family: LshFamily::MinHash,
        }
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
