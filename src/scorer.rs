// ALPS Discovery SDK — Pluggable Scorer Interface
//
// The Scorer trait defines how agents are matched to queries.
// MinHashScorer is the built-in default using locality-sensitive hashing.

use std::collections::HashMap;

use crate::core::config::LshConfig;
use crate::core::lsh::{compute_query_signature, compute_semantic_signature, MinHasher};

/// Pluggable scoring interface for agent-to-query matching.
///
/// Implementations index agent capabilities and score them against
/// natural-language queries. The default `MinHashScorer` uses MinHash
/// LSH for set-similarity; users can supply custom scorers (e.g.
/// embedding-based) via `LocalNetwork::with_scorer()`.
pub trait Scorer: Send + Sync {
    /// Index an agent's capabilities for future scoring.
    fn index_capabilities(&mut self, agent_id: &str, capabilities: &[&str]);

    /// Remove an agent from the index.
    fn remove_agent(&mut self, agent_id: &str);

    /// Score all indexed agents against a query.
    ///
    /// Returns `(agent_id, similarity)` pairs for agents with similarity > 0.
    /// The caller applies threshold filtering, diameter adjustment, and feedback.
    fn score(&self, query: &str) -> Result<Vec<(String, f64)>, String>;
}

/// MinHash-based scorer using locality-sensitive hashing.
///
/// Stores per-capability MinHash signatures for each agent. Scoring
/// computes the max per-capability Jaccard similarity estimate.
pub struct MinHashScorer {
    /// Per-agent capability signatures.
    agents: HashMap<String, Vec<[u8; 64]>>,
    /// LSH configuration (shingle mode, dimensions, threshold).
    config: LshConfig,
}

impl MinHashScorer {
    /// Create a new MinHashScorer with the given configuration.
    pub fn new(config: LshConfig) -> Self {
        Self {
            agents: HashMap::new(),
            config,
        }
    }

    /// Returns a reference to the LSH config.
    pub fn config(&self) -> &LshConfig {
        &self.config
    }
}

impl Default for MinHashScorer {
    fn default() -> Self {
        Self::new(LshConfig::default())
    }
}

impl Scorer for MinHashScorer {
    fn index_capabilities(&mut self, agent_id: &str, capabilities: &[&str]) {
        let sigs: Vec<[u8; 64]> = capabilities
            .iter()
            .map(|cap| compute_semantic_signature(cap.as_bytes(), &self.config))
            .collect();
        self.agents.insert(agent_id.to_string(), sigs);
    }

    fn remove_agent(&mut self, agent_id: &str) {
        self.agents.remove(agent_id);
    }

    fn score(&self, query: &str) -> Result<Vec<(String, f64)>, String> {
        let query_sig = compute_query_signature(query.as_bytes(), &self.config);
        Ok(self
            .agents
            .iter()
            .map(|(agent_id, cap_sigs)| {
                let sim = cap_sigs
                    .iter()
                    .map(|sig| MinHasher::similarity(sig, &query_sig.minhash))
                    .fold(0.0f64, f64::max);
                (agent_id.clone(), sim)
            })
            .filter(|(_, sim)| *sim > 0.0)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minhash_scorer_index_and_score() {
        let mut scorer = MinHashScorer::default();
        scorer.index_capabilities("agent-a", &["legal translation", "EN-DE"]);
        let results = scorer.score("legal translation").unwrap();
        assert!(!results.is_empty());
        let (name, sim) = &results[0];
        assert_eq!(name, "agent-a");
        assert!(*sim > 0.0);
    }

    #[test]
    fn minhash_scorer_remove_agent() {
        let mut scorer = MinHashScorer::default();
        scorer.index_capabilities("agent-a", &["legal translation"]);
        scorer.remove_agent("agent-a");
        let results = scorer.score("legal translation").unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn minhash_scorer_multiple_agents_ranked() {
        let mut scorer = MinHashScorer::default();
        scorer.index_capabilities("translate", &["legal translation services"]);
        scorer.index_capabilities("summarize", &["document summarization"]);
        let results = scorer.score("legal translation").unwrap();
        assert!(!results.is_empty());
        // translate should have higher similarity for "legal translation"
        let translate_sim = results
            .iter()
            .find(|(n, _)| n == "translate")
            .map(|(_, s)| *s)
            .unwrap_or(0.0);
        let summarize_sim = results
            .iter()
            .find(|(n, _)| n == "summarize")
            .map(|(_, s)| *s)
            .unwrap_or(0.0);
        assert!(
            translate_sim > summarize_sim,
            "translate ({:.3}) should outscore summarize ({:.3})",
            translate_sim,
            summarize_sim
        );
    }

    #[test]
    fn minhash_scorer_empty_returns_empty() {
        let scorer = MinHashScorer::default();
        let results = scorer.score("anything").unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn minhash_scorer_reindex_replaces() {
        let mut scorer = MinHashScorer::default();
        scorer.index_capabilities("agent-a", &["legal translation"]);
        scorer.index_capabilities("agent-a", &["data processing"]);
        // After reindex, should match "data processing" not "legal translation" as well
        let results = scorer.score("data processing").unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].0, "agent-a");
    }
}
