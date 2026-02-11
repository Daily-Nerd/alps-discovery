// ALPS Discovery — Co-Occurrence Query Expansion
//
// Self-improving matching via learned term co-occurrence from feedback.
// Tracks (query_token, agent_cap_token) pairs to expand queries with
// related terms, improving matching over time without external models.

use std::collections::HashMap;

/// Self-improving query expander using co-occurrence learning from feedback.
///
/// Tracks which agent capability tokens co-occur with query tokens in successful
/// feedback events. On discovery, expands query tokens with top-k co-occurring
/// agent tokens learned from feedback history. This enables the network to learn
/// synonym relationships and improve matching over time without external NLP models.
///
/// Memory is bounded via count-based eviction: when the matrix reaches max_entries,
/// prune entries below a minimum count threshold (incrementing threshold if needed).
#[derive(Debug, Clone)]
pub struct CoOccurrenceExpander {
    /// Sparse co-occurrence matrix: (query_token, agent_cap_token) -> count
    matrix: HashMap<(String, String), u32>,
    /// Minimum feedback events before enabling expansion (default: 10)
    min_feedback_threshold: usize,
    /// Number of top co-occurring tokens to add per query token (default: 3)
    top_k: usize,
    /// Total number of feedback events recorded (for threshold gating)
    total_feedback_count: usize,
    /// Maximum matrix entries before triggering eviction (default: 50,000)
    max_entries: usize,
    /// Minimum count threshold for pruning during eviction (default: 2)
    prune_below_count: u32,
}

impl CoOccurrenceExpander {
    /// Create a new co-occurrence expander with default configuration.
    pub fn new() -> Self {
        Self {
            matrix: HashMap::new(),
            min_feedback_threshold: 10,
            top_k: 3,
            total_feedback_count: 0,
            max_entries: 50_000,
            prune_below_count: 2,
        }
    }

    /// Create a co-occurrence expander with custom configuration.
    pub fn with_config(
        min_feedback_threshold: usize,
        top_k: usize,
        max_entries: usize,
        prune_below_count: u32,
    ) -> Self {
        Self {
            matrix: HashMap::new(),
            min_feedback_threshold,
            top_k,
            total_feedback_count: 0,
            max_entries,
            prune_below_count,
        }
    }

    /// Expand query tokens with top-k co-occurring terms (if threshold met).
    ///
    /// Returns the original query tokens plus top-k co-occurring agent capability
    /// tokens for each query token, learned from feedback history. Expansion is
    /// disabled until min_feedback_threshold feedback events have been recorded.
    pub fn expand_query(&self, query_tokens: &[String]) -> Vec<String> {
        if self.total_feedback_count < self.min_feedback_threshold {
            // Not enough feedback yet, return original query
            return query_tokens.to_vec();
        }

        let mut expanded = query_tokens.to_vec();
        for token in query_tokens {
            let cooccurring = self.top_k_cooccurring(token, self.top_k);
            expanded.extend(cooccurring);
        }
        expanded
    }

    /// Record co-occurrence from feedback (with bounded memory).
    ///
    /// Increments the count for all (query_token, agent_cap_token) pairs.
    /// When the matrix reaches max_entries, prunes low-count entries to
    /// maintain bounded memory usage.
    pub fn record_feedback(&mut self, query_tokens: &[String], agent_cap_tokens: &[String]) {
        for qt in query_tokens {
            for act in agent_cap_tokens {
                *self.matrix.entry((qt.clone(), act.clone())).or_insert(0) += 1;
            }
        }
        self.total_feedback_count += 1;

        // Prune when at capacity
        if self.matrix.len() >= self.max_entries {
            self.prune_low_counts();
        }
    }

    /// Prune entries below count threshold (count-based eviction).
    ///
    /// Removes all entries with count < prune_below_count. If the matrix
    /// is still over capacity after pruning, increments the threshold and
    /// prunes again until below max_entries.
    fn prune_low_counts(&mut self) {
        let mut threshold = self.prune_below_count;
        loop {
            self.matrix.retain(|_, count| *count >= threshold);
            if self.matrix.len() < self.max_entries {
                break;
            }
            threshold += 1; // Increase threshold if still over capacity
        }
    }

    /// Get the top-k agent capability tokens that co-occur with the given query token.
    fn top_k_cooccurring(&self, query_token: &str, k: usize) -> Vec<String> {
        let mut pairs: Vec<_> = self
            .matrix
            .iter()
            .filter(|((qt, _), _)| qt == query_token)
            .map(|((_, act), count)| (act.clone(), *count))
            .collect();
        pairs.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
        pairs.into_iter().take(k).map(|(act, _)| act).collect()
    }

    /// Get the total number of feedback events recorded.
    pub fn feedback_count(&self) -> usize {
        self.total_feedback_count
    }

    /// Get the current number of entries in the co-occurrence matrix.
    pub fn matrix_size(&self) -> usize {
        self.matrix.len()
    }

    /// Get a reference to the internal co-occurrence matrix (for persistence).
    pub(crate) fn matrix(&self) -> &HashMap<(String, String), u32> {
        &self.matrix
    }

    /// Restore the co-occurrence matrix from a saved state (for persistence).
    pub(crate) fn restore_matrix(
        &mut self,
        matrix: HashMap<(String, String), u32>,
        total_feedback_count: usize,
    ) {
        self.matrix = matrix;
        self.total_feedback_count = total_feedback_count;
    }
}

impl Default for CoOccurrenceExpander {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_query_below_threshold() {
        // GIVEN: A fresh expander with no feedback
        let expander = CoOccurrenceExpander::new();
        let query_tokens = vec!["translate".to_string(), "text".to_string()];

        // WHEN: Expanding a query below the threshold
        let expanded = expander.expand_query(&query_tokens);

        // THEN: Returns original query tokens (no expansion yet)
        assert_eq!(expanded, query_tokens);
        assert_eq!(expanded.len(), 2);
    }

    #[test]
    fn test_record_feedback_basic() {
        // GIVEN: A fresh expander
        let mut expander = CoOccurrenceExpander::new();
        let query_tokens = vec!["translate".to_string()];
        let agent_caps = vec!["convert".to_string(), "transform".to_string()];

        // WHEN: Recording feedback
        expander.record_feedback(&query_tokens, &agent_caps);

        // THEN: Co-occurrence counts are incremented
        assert_eq!(expander.feedback_count(), 1);
        assert_eq!(expander.matrix_size(), 2); // (translate,convert) and (translate,transform)
        assert_eq!(
            *expander
                .matrix
                .get(&("translate".to_string(), "convert".to_string()))
                .unwrap(),
            1
        );
        assert_eq!(
            *expander
                .matrix
                .get(&("translate".to_string(), "transform".to_string()))
                .unwrap(),
            1
        );
    }

    #[test]
    fn test_record_feedback_multiple() {
        // GIVEN: An expander with initial feedback
        let mut expander = CoOccurrenceExpander::new();
        let query_tokens = vec!["translate".to_string()];
        let agent_caps = vec!["convert".to_string()];

        // WHEN: Recording multiple feedback events for the same pair
        expander.record_feedback(&query_tokens, &agent_caps);
        expander.record_feedback(&query_tokens, &agent_caps);
        expander.record_feedback(&query_tokens, &agent_caps);

        // THEN: Count is accumulated
        assert_eq!(expander.feedback_count(), 3);
        assert_eq!(
            *expander
                .matrix
                .get(&("translate".to_string(), "convert".to_string()))
                .unwrap(),
            3
        );
    }

    #[test]
    fn test_expand_query_after_threshold() {
        // GIVEN: An expander with enough feedback to enable expansion
        let mut expander = CoOccurrenceExpander::with_config(
            3, // min_feedback_threshold
            2, // top_k
            50_000, 2,
        );

        // Record feedback: "translate" co-occurs with "convert" (3x) and "transform" (2x)
        for _ in 0..3 {
            expander.record_feedback(&["translate".to_string()], &["convert".to_string()]);
        }
        for _ in 0..2 {
            expander.record_feedback(&["translate".to_string()], &["transform".to_string()]);
        }

        // WHEN: Expanding a query after threshold
        let query_tokens = vec!["translate".to_string()];
        let expanded = expander.expand_query(&query_tokens);

        // THEN: Returns original token + top-2 co-occurring tokens
        assert!(expanded.contains(&"translate".to_string()));
        assert!(expanded.contains(&"convert".to_string()));
        assert!(expanded.contains(&"transform".to_string()));
        assert_eq!(expanded.len(), 3); // translate + convert + transform
    }

    #[test]
    fn test_top_k_ordering() {
        // GIVEN: An expander with varying co-occurrence counts
        let mut expander = CoOccurrenceExpander::with_config(1, 2, 50_000, 2);

        // Record feedback with different frequencies
        for _ in 0..5 {
            expander.record_feedback(&["translate".to_string()], &["convert".to_string()]);
        }
        for _ in 0..3 {
            expander.record_feedback(&["translate".to_string()], &["transform".to_string()]);
        }
        for _ in 0..1 {
            expander.record_feedback(&["translate".to_string()], &["interpret".to_string()]);
        }

        // WHEN: Getting top-2 co-occurring tokens
        let top_tokens = expander.top_k_cooccurring("translate", 2);

        // THEN: Returns tokens ordered by count (convert=5, transform=3, interpret=1)
        assert_eq!(top_tokens[0], "convert");
        assert_eq!(top_tokens[1], "transform");
        assert_eq!(top_tokens.len(), 2);
    }

    #[test]
    fn test_prune_low_counts_at_capacity() {
        // GIVEN: An expander with very small max_entries
        let mut expander = CoOccurrenceExpander::with_config(
            1, // min_feedback_threshold
            3, // top_k
            5, // max_entries (very small for testing)
            2, // prune_below_count
        );

        // WHEN: Recording feedback that exceeds capacity
        // Add 3 entries with count=3 (should be kept)
        for _ in 0..3 {
            expander.record_feedback(&["a".to_string()], &["x".to_string()]);
            expander.record_feedback(&["b".to_string()], &["y".to_string()]);
            expander.record_feedback(&["c".to_string()], &["z".to_string()]);
        }

        // Add 3 entries with count=1 (should be pruned)
        expander.record_feedback(&["d".to_string()], &["w".to_string()]);
        expander.record_feedback(&["e".to_string()], &["v".to_string()]);
        expander.record_feedback(&["f".to_string()], &["u".to_string()]);

        // THEN: Matrix is pruned to stay below max_entries
        assert!(expander.matrix_size() <= 5);
        // High-count entries should be retained
        assert!(expander
            .matrix
            .contains_key(&("a".to_string(), "x".to_string())));
        // Low-count entries should be pruned
        assert!(!expander
            .matrix
            .contains_key(&("d".to_string(), "w".to_string())));
    }

    #[test]
    fn test_prune_increments_threshold_if_needed() {
        // GIVEN: An expander where pruning at initial threshold is insufficient
        let mut expander = CoOccurrenceExpander::with_config(
            1, // min_feedback_threshold
            3, // top_k
            3, // max_entries (very small)
            2, // prune_below_count
        );

        // WHEN: Recording feedback where all entries have count >= 2
        for _ in 0..3 {
            expander.record_feedback(&["a".to_string()], &["x".to_string()]);
            expander.record_feedback(&["b".to_string()], &["y".to_string()]);
            expander.record_feedback(&["c".to_string()], &["z".to_string()]);
            expander.record_feedback(&["d".to_string()], &["w".to_string()]);
        }

        // THEN: Pruning increments threshold until matrix fits
        assert!(expander.matrix_size() <= 3);
    }
}
