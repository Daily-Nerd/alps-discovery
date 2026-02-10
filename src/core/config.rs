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

/// Configuration for the adaptive exploration budget (epsilon-greedy).
///
/// Controls the explore/exploit trade-off in tie-breaking. When agents
/// have overlapping confidence intervals, exploration (random shuffle)
/// happens with probability `epsilon`. Epsilon decays as feedback
/// accumulates: `epsilon = max(floor, initial × decay_rate^feedback_count)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExplorationConfig {
    /// Starting exploration probability (default: 0.8 — heavy exploration early).
    pub epsilon_initial: f64,
    /// Minimum exploration probability (default: 0.05 — always some exploration).
    pub epsilon_floor: f64,
    /// Exponential decay rate per feedback event (default: 0.99).
    pub epsilon_decay_rate: f64,
}

impl Default for ExplorationConfig {
    fn default() -> Self {
        Self {
            epsilon_initial: 0.8,
            epsilon_floor: 0.05,
            epsilon_decay_rate: 0.99,
        }
    }
}

impl ExplorationConfig {
    /// Compute current epsilon given total feedback count.
    ///
    /// Uses floating-point exponentiation to handle feedback counts up to u64::MAX
    /// without overflow. The result is clamped to [epsilon_floor, epsilon_initial].
    pub fn current_epsilon(&self, feedback_count: u64) -> f64 {
        let epsilon = self.epsilon_initial * self.epsilon_decay_rate.powf(feedback_count as f64);
        epsilon.clamp(self.epsilon_floor, self.epsilon_initial)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exploration_config_epsilon_handles_u64_max() {
        let config = ExplorationConfig::default();
        let epsilon = config.current_epsilon(u64::MAX);

        // Should clamp to floor, not panic or return NaN/infinity
        assert!(epsilon >= config.epsilon_floor);
        assert!(epsilon <= config.epsilon_initial);
        assert!(!epsilon.is_nan());
        assert!(!epsilon.is_infinite());
    }

    #[test]
    fn exploration_config_epsilon_decay() {
        let config = ExplorationConfig {
            epsilon_initial: 0.8,
            epsilon_floor: 0.05,
            epsilon_decay_rate: 0.99,
        };

        // After 0 feedback, should be initial
        let epsilon0 = config.current_epsilon(0);
        assert!((epsilon0 - 0.8).abs() < f64::EPSILON);

        // After 100 feedback, should have decayed
        let epsilon100 = config.current_epsilon(100);
        assert!(epsilon100 < epsilon0);
        assert!(epsilon100 >= config.epsilon_floor);

        // After many feedbacks, should approach floor
        let epsilon_large = config.current_epsilon(10_000);
        assert!((epsilon_large - config.epsilon_floor).abs() < 0.01);
    }
}
