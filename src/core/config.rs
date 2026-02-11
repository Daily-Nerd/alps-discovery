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
///
/// Retained for Tendril struct compatibility but currently unused in
/// local discovery path. May be removed in a future version.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QueryConfig;

impl Default for QueryConfig {
    fn default() -> Self {
        Self
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

    /// Validate configuration parameters (Requirement 11).
    ///
    /// # Errors
    ///
    /// Returns `DiscoveryError::Config` if:
    /// - epsilon_initial < epsilon_floor
    /// - epsilon_floor <= 0
    /// - epsilon_decay_rate <= 0 or > 1.0
    pub fn validate(&self) -> Result<(), crate::error::DiscoveryError> {
        use crate::error::DiscoveryError;

        if self.epsilon_floor <= 0.0 {
            return Err(DiscoveryError::Config(
                "epsilon_floor must be > 0".to_string(),
            ));
        }
        if self.epsilon_initial < self.epsilon_floor {
            return Err(DiscoveryError::Config(format!(
                "epsilon_initial ({}) must be >= epsilon_floor ({})",
                self.epsilon_initial, self.epsilon_floor
            )));
        }
        if self.epsilon_decay_rate <= 0.0 || self.epsilon_decay_rate > 1.0 {
            return Err(DiscoveryError::Config(format!(
                "epsilon_decay_rate must be in (0, 1], got {}",
                self.epsilon_decay_rate
            )));
        }
        Ok(())
    }
}

/// Unified configuration for ALPS Discovery.
///
/// Consolidates all tuning parameters: LSH, enzyme, exploration, feedback, and diameter settings.
/// This provides a single point of configuration for the LocalNetwork.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiscoveryConfig {
    /// LSH configuration (similarity thresholds, dimensions, shingle mode).
    pub lsh: LshConfig,
    /// Enzyme configuration (max disagreement split, quorum mode).
    pub enzyme: SLNEnzymeConfig,
    /// Exploration configuration (epsilon decay).
    pub exploration: ExplorationConfig,
    /// Feedback relevance threshold for per-query feedback matching (default: 0.3).
    pub feedback_relevance_threshold: f64,
    /// Tie-breaking epsilon for strict score equality detection (default: 1e-4).
    pub tie_epsilon: f64,
    /// Minimum tau floor to prevent zero-trap absorbing state (default: 0.001).
    pub tau_floor: f64,
    /// Maximum feedback records to retain per agent (default: 100).
    pub max_feedback_records: usize,
    /// Initial diameter value for new agents (default: 0.5).
    pub diameter_initial: f64,
    /// Minimum diameter value (default: 0.01).
    pub diameter_min: f64,
    /// Maximum diameter value (default: 2.0).
    pub diameter_max: f64,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            lsh: LshConfig::default(),
            enzyme: SLNEnzymeConfig::default(),
            exploration: ExplorationConfig::default(),
            feedback_relevance_threshold: 0.3,
            tie_epsilon: 1e-4,
            tau_floor: 0.001,
            max_feedback_records: 100,
            diameter_initial: 0.5,
            diameter_min: 0.01,
            diameter_max: 2.0,
        }
    }
}

impl DiscoveryConfig {
    /// Validate all configuration parameters.
    ///
    /// # Errors
    ///
    /// Returns `DiscoveryError::Config` if any parameter is out of valid range.
    pub fn validate(&self) -> Result<(), crate::error::DiscoveryError> {
        use crate::error::DiscoveryError;

        // Validate exploration config
        self.exploration.validate()?;

        // Validate feedback relevance threshold
        if self.feedback_relevance_threshold < 0.0 || self.feedback_relevance_threshold > 1.0 {
            return Err(DiscoveryError::Config(format!(
                "feedback_relevance_threshold must be in [0, 1], got {}",
                self.feedback_relevance_threshold
            )));
        }

        // Validate tie epsilon
        if self.tie_epsilon <= 0.0 {
            return Err(DiscoveryError::Config(format!(
                "tie_epsilon must be > 0, got {}",
                self.tie_epsilon
            )));
        }

        // Validate tau floor
        if self.tau_floor <= 0.0 {
            return Err(DiscoveryError::Config(format!(
                "tau_floor must be > 0, got {}",
                self.tau_floor
            )));
        }

        // Validate diameter range
        if self.diameter_min <= 0.0 {
            return Err(DiscoveryError::Config(format!(
                "diameter_min must be > 0, got {}",
                self.diameter_min
            )));
        }
        if self.diameter_max <= self.diameter_min {
            return Err(DiscoveryError::Config(format!(
                "diameter_max ({}) must be > diameter_min ({})",
                self.diameter_max, self.diameter_min
            )));
        }
        if self.diameter_initial < self.diameter_min || self.diameter_initial > self.diameter_max {
            return Err(DiscoveryError::Config(format!(
                "diameter_initial ({}) must be in [{}, {}]",
                self.diameter_initial, self.diameter_min, self.diameter_max
            )));
        }

        Ok(())
    }
}

use crate::core::enzyme::SLNEnzymeConfig;

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // DiscoveryConfig Tests (Requirement 14)
    // -----------------------------------------------------------------------

    #[test]
    fn discovery_config_default_is_valid() {
        let config = DiscoveryConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn discovery_config_validates_feedback_threshold() {
        let config = DiscoveryConfig {
            feedback_relevance_threshold: 1.5, // Invalid: > 1.0
            ..Default::default()
        };
        assert!(config.validate().is_err());

        let config = DiscoveryConfig {
            feedback_relevance_threshold: -0.1, // Invalid: < 0.0
            ..Default::default()
        };
        assert!(config.validate().is_err());

        let config = DiscoveryConfig {
            feedback_relevance_threshold: 0.5, // Valid
            ..Default::default()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn discovery_config_validates_tie_epsilon() {
        let config = DiscoveryConfig {
            tie_epsilon: 0.0, // Invalid: must be > 0
            ..Default::default()
        };
        assert!(config.validate().is_err());

        let config = DiscoveryConfig {
            tie_epsilon: -0.001, // Invalid: must be > 0
            ..Default::default()
        };
        assert!(config.validate().is_err());

        let config = DiscoveryConfig {
            tie_epsilon: 1e-6, // Valid
            ..Default::default()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn discovery_config_validates_tau_floor() {
        let config = DiscoveryConfig {
            tau_floor: 0.0, // Invalid: must be > 0
            ..Default::default()
        };
        assert!(config.validate().is_err());

        let config = DiscoveryConfig {
            tau_floor: -0.001, // Invalid: must be > 0
            ..Default::default()
        };
        assert!(config.validate().is_err());

        let config = DiscoveryConfig {
            tau_floor: 0.0001, // Valid
            ..Default::default()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn discovery_config_validates_diameter_range() {
        // Invalid: diameter_min <= 0
        let config = DiscoveryConfig {
            diameter_min: 0.0,
            ..Default::default()
        };
        assert!(config.validate().is_err());

        // Invalid: diameter_max <= diameter_min
        let config = DiscoveryConfig {
            diameter_min: 0.5,
            diameter_max: 0.3,
            ..Default::default()
        };
        assert!(config.validate().is_err());

        // Invalid: diameter_initial < diameter_min
        let config = DiscoveryConfig {
            diameter_initial: 0.005,
            diameter_min: 0.01,
            ..Default::default()
        };
        assert!(config.validate().is_err());

        // Invalid: diameter_initial > diameter_max
        let config = DiscoveryConfig {
            diameter_initial: 3.0,
            diameter_max: 2.0,
            ..Default::default()
        };
        assert!(config.validate().is_err());

        // Valid: diameter_initial in range
        let config = DiscoveryConfig {
            diameter_initial: 0.5,
            ..Default::default()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn discovery_config_exposes_all_parameters() {
        let config = DiscoveryConfig {
            lsh: LshConfig::default(),
            enzyme: SLNEnzymeConfig::default(),
            exploration: ExplorationConfig::default(),
            feedback_relevance_threshold: 0.4,
            tie_epsilon: 2e-4,
            tau_floor: 0.002,
            max_feedback_records: 200,
            diameter_initial: 0.6,
            diameter_min: 0.02,
            diameter_max: 1.5,
        };

        // Verify all fields are accessible
        assert_eq!(config.feedback_relevance_threshold, 0.4);
        assert_eq!(config.tie_epsilon, 2e-4);
        assert_eq!(config.tau_floor, 0.002);
        assert_eq!(config.max_feedback_records, 200);
        assert_eq!(config.diameter_initial, 0.6);
        assert_eq!(config.diameter_min, 0.02);
        assert_eq!(config.diameter_max, 1.5);
    }

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

    // -----------------------------------------------------------------------
    // Configuration Validation (Requirement 11)
    // -----------------------------------------------------------------------

    #[test]
    fn exploration_config_rejects_initial_below_floor() {
        use crate::error::DiscoveryError;
        let config = ExplorationConfig {
            epsilon_initial: 0.01,
            epsilon_floor: 0.05,
            epsilon_decay_rate: 0.99,
        };
        let result = config.validate();
        assert!(matches!(result, Err(DiscoveryError::Config(_))));
        assert!(result.unwrap_err().to_string().contains("floor"));
    }

    #[test]
    fn exploration_config_rejects_zero_floor() {
        use crate::error::DiscoveryError;
        let config = ExplorationConfig {
            epsilon_initial: 0.8,
            epsilon_floor: 0.0,
            epsilon_decay_rate: 0.99,
        };
        let result = config.validate();
        assert!(matches!(result, Err(DiscoveryError::Config(_))));
        assert!(result.unwrap_err().to_string().contains("floor"));
    }

    #[test]
    fn exploration_config_rejects_invalid_decay_rate() {
        use crate::error::DiscoveryError;
        let config = ExplorationConfig {
            epsilon_initial: 0.8,
            epsilon_floor: 0.05,
            epsilon_decay_rate: 1.5, // > 1.0 invalid
        };
        let result = config.validate();
        assert!(matches!(result, Err(DiscoveryError::Config(_))));
        assert!(result.unwrap_err().to_string().contains("decay"));
    }

    #[test]
    fn exploration_config_rejects_zero_decay_rate() {
        use crate::error::DiscoveryError;
        let config = ExplorationConfig {
            epsilon_initial: 0.8,
            epsilon_floor: 0.05,
            epsilon_decay_rate: 0.0,
        };
        let result = config.validate();
        assert!(matches!(result, Err(DiscoveryError::Config(_))));
    }

    #[test]
    fn exploration_config_accepts_valid_parameters() {
        let config = ExplorationConfig::default();
        let result = config.validate();
        assert!(result.is_ok());
    }
}
