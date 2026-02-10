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

/// TF-IDF scorer using term frequency–inverse document frequency weighting
/// with cosine similarity.
///
/// Unlike MinHashScorer (set overlap), TfIdfScorer weights rare domain
/// terms (e.g. "patent", "regulatory") higher than common terms (e.g.
/// "process", "service"), providing better semantic matching for natural
/// language capability descriptions.
///
/// Zero external dependencies — uses only Rust std.
pub struct TfIdfScorer {
    /// Per-agent term-frequency vectors: agent_id → (term → tf).
    agents: HashMap<String, HashMap<String, f64>>,
    /// Document frequency: term → number of agents containing it.
    df: HashMap<String, usize>,
    /// Total number of agents (documents) for IDF computation.
    num_agents: usize,
}

impl TfIdfScorer {
    /// Create a new empty TfIdfScorer.
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
            df: HashMap::new(),
            num_agents: 0,
        }
    }

    /// Tokenize text into lowercase terms.
    fn tokenize(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() >= 2)
            .map(|w| w.to_string())
            .collect()
    }

    /// Compute TF vector from a list of capability strings.
    fn compute_tf(capabilities: &[&str]) -> HashMap<String, f64> {
        let mut counts: HashMap<String, f64> = HashMap::new();
        let mut total = 0.0;
        for cap in capabilities {
            for token in Self::tokenize(cap) {
                *counts.entry(token).or_insert(0.0) += 1.0;
                total += 1.0;
            }
        }
        // Normalize by total term count.
        if total > 0.0 {
            for v in counts.values_mut() {
                *v /= total;
            }
        }
        counts
    }

    /// Compute IDF for a term: ln(1 + N / (1 + df)).
    ///
    /// Uses the smoothed variant that is always positive (even with 1 agent).
    fn idf(&self, term: &str) -> f64 {
        let df = self.df.get(term).copied().unwrap_or(0) as f64;
        (1.0 + self.num_agents as f64 / (1.0 + df)).ln()
    }

    /// Rebuild document frequency counts from current agents.
    fn rebuild_df(&mut self) {
        self.df.clear();
        self.num_agents = self.agents.len();
        for tf in self.agents.values() {
            for term in tf.keys() {
                *self.df.entry(term.clone()).or_insert(0) += 1;
            }
        }
    }

    /// Compute cosine similarity between a query TF vector and an agent TF-IDF vector.
    fn cosine_similarity(
        &self,
        query_tf: &HashMap<String, f64>,
        agent_tf: &HashMap<String, f64>,
    ) -> f64 {
        let mut dot = 0.0;
        let mut norm_q = 0.0;
        let mut norm_a = 0.0;

        // Collect all terms from both.
        let mut all_terms: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for t in query_tf.keys() {
            all_terms.insert(t.as_str());
        }
        for t in agent_tf.keys() {
            all_terms.insert(t.as_str());
        }

        for term in all_terms {
            let idf = self.idf(term);
            let q_tfidf = query_tf.get(term).copied().unwrap_or(0.0) * idf;
            let a_tfidf = agent_tf.get(term).copied().unwrap_or(0.0) * idf;
            dot += q_tfidf * a_tfidf;
            norm_q += q_tfidf * q_tfidf;
            norm_a += a_tfidf * a_tfidf;
        }

        let denom = norm_q.sqrt() * norm_a.sqrt();
        if denom > 0.0 {
            dot / denom
        } else {
            0.0
        }
    }
}

impl Default for TfIdfScorer {
    fn default() -> Self {
        Self::new()
    }
}

impl Scorer for TfIdfScorer {
    fn index_capabilities(&mut self, agent_id: &str, capabilities: &[&str]) {
        let tf = Self::compute_tf(capabilities);
        self.agents.insert(agent_id.to_string(), tf);
        self.rebuild_df();
    }

    fn remove_agent(&mut self, agent_id: &str) {
        self.agents.remove(agent_id);
        self.rebuild_df();
    }

    fn score(&self, query: &str) -> Result<Vec<(String, f64)>, String> {
        let query_tf = Self::compute_tf(&[query]);
        Ok(self
            .agents
            .iter()
            .map(|(agent_id, agent_tf)| {
                let sim = self.cosine_similarity(&query_tf, agent_tf);
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

    // -----------------------------------------------------------------------
    // TfIdfScorer tests
    // -----------------------------------------------------------------------

    #[test]
    fn tfidf_scorer_index_and_score() {
        let mut scorer = TfIdfScorer::new();
        scorer.index_capabilities("agent-a", &["legal translation services"]);
        let results = scorer.score("legal translation").unwrap();
        assert!(!results.is_empty());
        let (name, sim) = &results[0];
        assert_eq!(name, "agent-a");
        assert!(*sim > 0.0);
    }

    #[test]
    fn tfidf_scorer_remove_agent() {
        let mut scorer = TfIdfScorer::new();
        scorer.index_capabilities("agent-a", &["legal translation"]);
        scorer.remove_agent("agent-a");
        let results = scorer.score("legal translation").unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn tfidf_scorer_ranks_specific_over_generic() {
        let mut scorer = TfIdfScorer::new();
        // Both agents share "translation" but patent-expert has rare domain terms.
        scorer.index_capabilities(
            "patent-expert",
            &["patent filing translation with regulatory compliance expertise"],
        );
        scorer.index_capabilities(
            "general-translate",
            &["translation between multiple languages general purpose"],
        );
        // Query shares "translation" with both, plus domain terms with patent-expert.
        let results = scorer
            .score("patent filing translation regulatory")
            .unwrap();
        assert!(
            results.len() >= 2,
            "expected >= 2 results, got {}: {:?}",
            results.len(),
            results
        );
        let patent_sim = results
            .iter()
            .find(|(n, _)| n == "patent-expert")
            .map(|(_, s)| *s)
            .unwrap_or(0.0);
        let general_sim = results
            .iter()
            .find(|(n, _)| n == "general-translate")
            .map(|(_, s)| *s)
            .unwrap_or(0.0);
        assert!(
            patent_sim > general_sim,
            "patent-expert ({:.3}) should outscore general-translate ({:.3}) on domain-specific query",
            patent_sim, general_sim
        );
    }

    #[test]
    fn tfidf_scorer_empty_returns_empty() {
        let scorer = TfIdfScorer::new();
        let results = scorer.score("anything").unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn tfidf_scorer_weights_rare_terms_higher() {
        let mut scorer = TfIdfScorer::new();
        // "translation" appears in both agents, "patent" only in one
        scorer.index_capabilities("patent-agent", &["patent translation services"]);
        scorer.index_capabilities("general-agent", &["translation services"]);
        scorer.index_capabilities("other-agent", &["image processing"]);

        // Query for "patent" should strongly favor patent-agent
        let results = scorer.score("patent translation").unwrap();
        let patent_sim = results
            .iter()
            .find(|(n, _)| n == "patent-agent")
            .map(|(_, s)| *s)
            .unwrap_or(0.0);
        let general_sim = results
            .iter()
            .find(|(n, _)| n == "general-agent")
            .map(|(_, s)| *s)
            .unwrap_or(0.0);
        assert!(
            patent_sim > general_sim,
            "patent-agent ({:.3}) should outscore general-agent ({:.3}) because 'patent' is a rare distinguishing term",
            patent_sim, general_sim
        );
    }
}
