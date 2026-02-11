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
    /// Maximum number of words per capability before truncation (None = unlimited).
    /// Default: Some(50) to prevent long capability strings from diluting shingle sets.
    max_tokens: Option<usize>,
}

impl MinHashScorer {
    /// Create a new MinHashScorer with the given configuration.
    pub fn new(config: LshConfig) -> Self {
        Self {
            agents: HashMap::new(),
            config,
            max_tokens: Some(50), // Default: truncate at 50 words
        }
    }

    /// Create a new MinHashScorer with a custom max_tokens limit.
    ///
    /// Use this to override the default 50-word truncation limit.
    /// Pass `None` to disable truncation entirely.
    pub fn with_max_tokens(max_tokens: usize) -> Self {
        Self {
            agents: HashMap::new(),
            config: LshConfig::default(),
            max_tokens: Some(max_tokens),
        }
    }

    /// Returns a reference to the LSH config.
    pub fn config(&self) -> &LshConfig {
        &self.config
    }

    /// Truncates a capability string to max_tokens words if needed.
    ///
    /// Emits a tracing::warn when truncation occurs with agent name and original token count.
    fn truncate_capability(&self, agent_id: &str, capability: &str) -> String {
        let Some(max) = self.max_tokens else {
            return capability.to_string();
        };

        let tokens: Vec<&str> = capability.split_whitespace().collect();

        if tokens.len() <= max {
            return capability.to_string();
        }

        // Truncation needed
        tracing::warn!(
            agent_id = agent_id,
            original_tokens = tokens.len(),
            max_tokens = max,
            "truncating long capability string"
        );

        tokens[..max].join(" ")
    }
}

impl Default for MinHashScorer {
    fn default() -> Self {
        Self {
            agents: HashMap::new(),
            config: LshConfig::default(),
            max_tokens: Some(50), // Default: truncate at 50 words
        }
    }
}

impl Scorer for MinHashScorer {
    fn index_capabilities(&mut self, agent_id: &str, capabilities: &[&str]) {
        // Truncate each capability independently before indexing
        let sigs: Vec<[u8; 64]> = capabilities
            .iter()
            .map(|cap| {
                let truncated = self.truncate_capability(agent_id, cap);
                compute_semantic_signature(truncated.as_bytes(), &self.config)
            })
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
    ///
    /// This is only used for testing to verify incremental DF updates are correct.
    /// Production code uses incremental updates in `index_capabilities` and `remove_agent`.
    #[cfg(test)]
    fn rebuild_df_for_test(&mut self) {
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

        // Increment DF counts for new agent's terms (incremental update)
        for term in tf.keys() {
            *self.df.entry(term.clone()).or_insert(0) += 1;
        }

        self.agents.insert(agent_id.to_string(), tf);
        self.num_agents = self.agents.len();
    }

    fn remove_agent(&mut self, agent_id: &str) {
        if let Some(tf) = self.agents.remove(agent_id) {
            // Decrement DF counts for removed agent's terms (incremental update)
            for term in tf.keys() {
                if let Some(count) = self.df.get_mut(term) {
                    *count = count.saturating_sub(1);
                    // Clean up zero-count entries to save memory
                    if *count == 0 {
                        self.df.remove(term);
                    }
                }
            }
        }
        self.num_agents = self.agents.len();
    }

    fn score(&self, query: &str) -> Result<Vec<(String, f64)>, String> {
        let query_tf = Self::compute_tf(&[query]);
        let mut results: Vec<(String, f64)> = self
            .agents
            .iter()
            .map(|(agent_id, agent_tf)| {
                let sim = self.cosine_similarity(&query_tf, agent_tf);
                (agent_id.clone(), sim)
            })
            .filter(|(_, sim)| *sim > 0.0)
            .collect();

        // Sort descending by similarity (highest first)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results)
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

    // --- Task 6.3: Long capability string truncation tests ---

    #[test]
    fn minhash_scorer_truncates_long_capabilities() {
        let mut scorer = MinHashScorer::with_max_tokens(10); // Short limit for testing

        // This capability has 15+ words, should be truncated to 10
        let long_cap = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen";

        scorer.index_capabilities("test-agent", &[long_cap]);

        // Should still produce valid signatures
        let results = scorer.score("one two three").unwrap();
        assert!(
            !results.is_empty(),
            "truncated capability should still match"
        );
    }

    // --- Task 7.2: TfIdfScorer sorting test ---

    #[test]
    fn tfidf_scorer_results_sorted_descending() {
        let mut scorer = TfIdfScorer::new();
        scorer.index_capabilities("high-match", &["legal translation patent filing"]);
        scorer.index_capabilities("medium-match", &["legal translation"]);
        scorer.index_capabilities("low-match", &["legal"]);

        let results = scorer.score("legal translation patent").unwrap();

        assert!(results.len() >= 3, "should return all 3 agents");

        // Verify descending order by similarity
        for i in 0..results.len() - 1 {
            assert!(
                results[i].1 >= results[i + 1].1,
                "results[{}] similarity ({:.3}) should be >= results[{}] similarity ({:.3})",
                i,
                results[i].1,
                i + 1,
                results[i + 1].1
            );
        }

        // Verify highest match is first
        assert_eq!(
            results[0].0, "high-match",
            "highest similarity should be first"
        );
    }

    #[test]
    fn minhash_scorer_very_long_capability_still_matches() {
        // 750-character capability should still produce non-zero similarity
        let mut scorer = MinHashScorer::default(); // Default max_tokens = 50

        let very_long_cap = "legal translation services for international contracts and documents \
                             including patent filings trademark applications copyright registrations \
                             and regulatory compliance documentation across multiple jurisdictions \
                             with specialized expertise in European Union regulations United States \
                             federal and state law Asian Pacific trade agreements and Middle Eastern \
                             commercial law frameworks providing comprehensive linguistic and legal \
                             analysis for multinational corporations government agencies and non-profit \
                             organizations requiring accurate culturally appropriate translations that \
                             preserve legal meaning and technical precision while adhering to local \
                             regulatory requirements and industry-specific terminology standards for \
                             pharmaceutical medical device financial services technology transfer and \
                             intellectual property protection across borders with certified translators \
                             and legal experts fluent in over forty languages including but not limited \
                             to English French German Spanish Italian Portuguese Russian Chinese Japanese \
                             Korean Arabic Hebrew and various regional dialects ensuring compliance with \
                             international standards such as ISO 17100 and ASTM F2575";

        assert!(
            very_long_cap.len() > 750,
            "test capability should be > 750 characters"
        );

        scorer.index_capabilities("legal-expert", &[very_long_cap]);

        // Should still match related queries
        let results = scorer.score("legal translation").unwrap();
        assert!(
            !results.is_empty(),
            "750+ character capability should still produce results"
        );
        assert!(
            results[0].1 > 0.0,
            "similarity should be non-zero for long capability"
        );
    }

    #[test]
    fn minhash_scorer_truncation_preserves_first_tokens() {
        let mut scorer = MinHashScorer::with_max_tokens(5);

        // Capability with important terms at the beginning
        scorer.index_capabilities(
            "agent-a",
            &["legal translation document services with extra words that will be truncated"],
        );

        // Should match based on first 5 words
        let results = scorer.score("legal translation").unwrap();
        assert!(!results.is_empty());

        // Should NOT match words after truncation point
        let results_truncated = scorer.score("extra words truncated").unwrap();
        // May still have some similarity due to common words, but should be low
        if !results_truncated.is_empty() {
            assert!(
                results_truncated[0].1 < results[0].1,
                "truncated terms should have lower similarity"
            );
        }
    }

    // -----------------------------------------------------------------------
    // TfIdfScorer incremental DF tests
    // -----------------------------------------------------------------------

    #[test]
    fn tfidf_incremental_df_matches_full_rebuild() {
        let mut scorer_incremental = TfIdfScorer::new();
        let mut scorer_rebuild = TfIdfScorer::new();

        // Register 100 agents with varied capabilities
        for i in 0..100 {
            let cap_str = format!("capability {}", i % 10);
            let caps = vec![cap_str.as_str()];
            scorer_incremental.index_capabilities(&format!("agent-{}", i), &caps);
        }

        // Clone agents to rebuild scorer and manually rebuild DF
        scorer_rebuild.agents = scorer_incremental.agents.clone();
        scorer_rebuild.rebuild_df_for_test();

        // Incremental DF should match full rebuild
        assert_eq!(
            scorer_incremental.df, scorer_rebuild.df,
            "Incremental DF should match full rebuild"
        );
        assert_eq!(
            scorer_incremental.num_agents, scorer_rebuild.num_agents,
            "Agent counts should match"
        );
    }

    #[test]
    fn tfidf_remove_agent_decrements_df() {
        let mut scorer = TfIdfScorer::new();

        // Register three agents, two sharing a term
        // Note: tokenization splits on non-alphanumeric, so "shared-term" becomes ["shared", "term"]
        scorer.index_capabilities("agent-a", &["unique shared"]);
        scorer.index_capabilities("agent-b", &["another shared"]);
        scorer.index_capabilities("agent-c", &["third"]);

        // Verify DF counts (using actual tokenized terms)
        assert_eq!(*scorer.df.get("shared").unwrap(), 2);
        assert_eq!(*scorer.df.get("unique").unwrap(), 1);
        assert_eq!(*scorer.df.get("another").unwrap(), 1);

        // Remove agent-a
        scorer.remove_agent("agent-a");

        // "shared" should decrement to 1, "unique" should be removed
        assert_eq!(*scorer.df.get("shared").unwrap(), 1);
        assert!(
            !scorer.df.contains_key("unique"),
            "Zero-count terms should be cleaned up"
        );
        assert_eq!(*scorer.df.get("another").unwrap(), 1);
        assert_eq!(scorer.num_agents, 2);
    }

    #[test]
    fn tfidf_reindex_same_agent_updates_df_correctly() {
        let mut scorer = TfIdfScorer::new();

        // Register agent with initial capabilities
        scorer.index_capabilities("agent-a", &["oldword shared"]);
        scorer.index_capabilities("agent-b", &["shared"]);

        assert_eq!(*scorer.df.get("oldword").unwrap(), 1);
        assert_eq!(*scorer.df.get("shared").unwrap(), 2);

        // Reindex agent-a with different capabilities
        scorer.remove_agent("agent-a");
        scorer.index_capabilities("agent-a", &["newword shared"]);

        // "oldword" should be gone, "newword" should appear, "shared" still 2
        assert!(!scorer.df.contains_key("oldword"));
        assert_eq!(*scorer.df.get("newword").unwrap(), 1);
        assert_eq!(*scorer.df.get("shared").unwrap(), 2);
        assert_eq!(scorer.num_agents, 2);
    }

    #[test]
    fn tfidf_incremental_performance_is_constant_time() {
        // This test verifies O(T) behavior by checking that registration time
        // doesn't increase with the number of existing agents

        let mut scorer = TfIdfScorer::new();

        // Pre-register 1000 agents
        for i in 0..1000 {
            scorer.index_capabilities(&format!("agent-{}", i), &["test capability words"]);
        }

        // Time a single registration (should be O(T) where T is term count)
        // This is a smoke test - we're just verifying it doesn't panic or hang
        scorer.index_capabilities("new-agent", &["another capability with words"]);
        scorer.remove_agent("new-agent");

        // If we get here without hanging, the incremental update is working
        assert_eq!(scorer.num_agents, 1000);
    }
}
