// ALPS Discovery SDK — Error Types
//
// Structured error types for all discovery operations.

use thiserror::Error;

/// Main error type for ALPS Discovery operations.
///
/// All public API methods return `Result<T, DiscoveryError>` to enable
/// proper error handling and avoid panics in production.
#[derive(Debug, Error)]
pub enum DiscoveryError {
    /// Error from the pluggable scorer.
    #[error("Scorer error: {0}")]
    Scorer(#[from] ScorerError),

    /// Agent not found in registry.
    #[error("Unknown agent: {agent_id}")]
    UnknownAgent { agent_id: String },

    /// Persistence operation failed (save/load).
    #[error("Persistence error: {0}")]
    Persistence(String),

    /// I/O operation failed.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Configuration validation failed.
    #[error("Configuration error: {0}")]
    Config(String),

    /// Circuit breaker is open for an agent (too many failures).
    #[error("Circuit breaker open for agent: {agent_id}")]
    CircuitBreakerOpen { agent_id: String },

    /// Mutex was poisoned but recovered.
    #[error("Mutex poisoned (recovered): {0}")]
    MutexPoisoned(String),
}

/// Error type for scorer implementations.
#[derive(Debug, Error)]
pub enum ScorerError {
    /// Query validation failed.
    #[error("Invalid query: {0}")]
    InvalidQuery(String),

    /// Scorer index is corrupted or inconsistent.
    #[error("Index corrupted: {0}")]
    IndexCorrupted(String),

    /// Generic scorer failure.
    #[error("Scorer failed: {0}")]
    Other(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    #[test]
    fn discovery_error_from_io_error() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let disc_err = DiscoveryError::from(io_err);
        assert!(matches!(disc_err, DiscoveryError::Io(_)));
        assert!(disc_err.to_string().contains("file not found"));
    }

    #[test]
    fn discovery_error_from_scorer_error() {
        let scorer_err = ScorerError::InvalidQuery("empty query".to_string());
        let disc_err = DiscoveryError::from(scorer_err);
        assert!(matches!(disc_err, DiscoveryError::Scorer(_)));
        assert!(disc_err.to_string().contains("empty query"));
    }

    #[test]
    fn unknown_agent_error_includes_agent_id() {
        let err = DiscoveryError::UnknownAgent {
            agent_id: "test-agent".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("test-agent"));
        assert!(msg.contains("Unknown agent"));
    }

    #[test]
    fn config_error_includes_message() {
        let err = DiscoveryError::Config("epsilon_floor must be > 0".to_string());
        assert!(err.to_string().contains("epsilon_floor must be > 0"));
    }

    #[test]
    fn circuit_breaker_error_includes_agent_id() {
        let err = DiscoveryError::CircuitBreakerOpen {
            agent_id: "failing-agent".to_string(),
        };
        assert!(err.to_string().contains("failing-agent"));
        assert!(err.to_string().contains("Circuit breaker open"));
    }

    #[test]
    fn mutex_poisoned_error_includes_context() {
        let err = DiscoveryError::MutexPoisoned("enzyme lock".to_string());
        assert!(err.to_string().contains("enzyme lock"));
        assert!(err.to_string().contains("Mutex poisoned"));
    }

    #[test]
    fn scorer_error_invalid_query() {
        let err = ScorerError::InvalidQuery("query too long".to_string());
        assert!(err.to_string().contains("Invalid query"));
        assert!(err.to_string().contains("query too long"));
    }

    #[test]
    fn scorer_error_index_corrupted() {
        let err = ScorerError::IndexCorrupted("missing entry".to_string());
        assert!(err.to_string().contains("Index corrupted"));
        assert!(err.to_string().contains("missing entry"));
    }
}
