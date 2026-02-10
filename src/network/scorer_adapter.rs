// ALPS Discovery — Scorer Adapter
//
// Wraps the pluggable Scorer trait with LSH configuration and signature
// computation convenience methods.

use crate::core::chemistry::QuerySignature;
use crate::core::config::LshConfig;
use crate::core::lsh::compute_query_signature;
use crate::scorer::{MinHashScorer, Scorer};

/// Wraps a Scorer with LSH configuration for unified scoring access.
pub struct ScorerAdapter {
    scorer: Box<dyn Scorer>,
    pub(crate) lsh_config: LshConfig,
}

impl ScorerAdapter {
    /// Create a new ScorerAdapter with the default MinHash scorer.
    pub fn new(lsh_config: LshConfig) -> Self {
        let scorer = MinHashScorer::new(lsh_config.clone());
        Self {
            scorer: Box::new(scorer),
            lsh_config,
        }
    }

    /// Create with a custom scorer.
    pub fn with_scorer(scorer: Box<dyn Scorer>, lsh_config: LshConfig) -> Self {
        Self { scorer, lsh_config }
    }

    /// Score all agents against a query string.
    pub fn score(&self, query: &str) -> Result<Vec<(String, f64)>, String> {
        self.scorer.score(query)
    }

    /// Access the underlying scorer for Query algebra evaluation.
    pub fn scorer(&self) -> &dyn Scorer {
        &*self.scorer
    }

    /// Mutable access to the underlying scorer (e.g. for registration).
    pub fn scorer_mut(&mut self) -> &mut dyn Scorer {
        &mut *self.scorer
    }

    /// Compute a query signature from raw bytes.
    pub fn compute_query_signature(&self, key: &[u8]) -> QuerySignature {
        compute_query_signature(key, &self.lsh_config)
    }

    /// Access the LSH config.
    pub fn lsh_config(&self) -> &LshConfig {
        &self.lsh_config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::config::LshConfig;

    #[test]
    fn index_and_score() {
        let mut adapter = ScorerAdapter::new(LshConfig::default());
        adapter
            .scorer_mut()
            .index_capabilities("agent-a", &["legal translation"]);
        let results = adapter.score("legal translation").unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn remove_agent_clears() {
        let mut adapter = ScorerAdapter::new(LshConfig::default());
        adapter
            .scorer_mut()
            .index_capabilities("agent-a", &["legal translation"]);
        adapter.scorer_mut().remove_agent("agent-a");
        let results = adapter.score("legal translation").unwrap();
        assert!(results.is_empty());
    }
}
