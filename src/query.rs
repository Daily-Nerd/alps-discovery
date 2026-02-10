// ALPS Discovery SDK — Capability Algebra
//
// Composable query expressions for agent discovery. Supports set-theoretic
// composition: AND (all), OR (any), NOT (exclude), and weighted combinations.
//
// Each operator composes per-agent similarity scores from sub-queries:
// - All: min across sub-queries (agent must match every term)
// - Any: max across sub-queries (agent can match any term)
// - Exclude: base_score × (1 - exclusion_score) (penalise unwanted matches)
// - Weighted: weighted average of sub-query similarities

use std::collections::HashMap;

use crate::scorer::Scorer;

/// A composable query expression for agent discovery.
///
/// Queries can be simple text strings or composed using set-theoretic
/// operators. The discovery pipeline evaluates each sub-query through the
/// configured `Scorer` and combines per-agent similarities according to
/// the operator semantics.
///
/// # Examples (Rust)
///
/// ```ignore
/// use alps_discovery::query::Query;
///
/// // AND: agent must match both
/// let q = Query::all(["legal translation", "German language"]);
///
/// // OR: agent matches either
/// let q = Query::any(["translate", "interpret"]);
///
/// // Exclude: match "translate" but penalise "medical" matches
/// let q = Query::from("translate").exclude("medical");
///
/// // Weighted: boost "translate" importance
/// let q = Query::weighted([("translate", 2.0), ("legal", 1.0)]);
/// ```
#[derive(Debug, Clone)]
pub enum Query {
    /// Simple text query (existing behaviour).
    Text(String),
    /// All sub-queries must match. Score = min across sub-queries.
    All(Vec<Query>),
    /// Any sub-query can match. Score = max across sub-queries.
    Any(Vec<Query>),
    /// Match base query, penalise agents matching exclusion.
    /// Score = base_score × (1.0 − exclusion_score).
    Exclude {
        base: Box<Query>,
        exclusion: Box<Query>,
    },
    /// Weighted combination of sub-queries.
    /// Score = Σ(weight_i × score_i) / Σ(weight_i).
    Weighted(Vec<(Query, f64)>),
}

impl Query {
    /// Create an `All` query (AND / intersection semantics).
    pub fn all(queries: impl IntoIterator<Item = impl Into<Query>>) -> Self {
        Query::All(queries.into_iter().map(Into::into).collect())
    }

    /// Create an `Any` query (OR / union semantics).
    pub fn any(queries: impl IntoIterator<Item = impl Into<Query>>) -> Self {
        Query::Any(queries.into_iter().map(Into::into).collect())
    }

    /// Create a `Weighted` query (boosted combination).
    pub fn weighted(entries: impl IntoIterator<Item = (impl Into<Query>, f64)>) -> Self {
        Query::Weighted(entries.into_iter().map(|(q, w)| (q.into(), w)).collect())
    }

    /// Chain an exclusion onto this query.
    ///
    /// Returns `Exclude { base: self, exclusion: query }`.
    /// Agents matching the exclusion have their score reduced proportionally.
    pub fn exclude(self, query: impl Into<Query>) -> Self {
        Query::Exclude {
            base: Box::new(self),
            exclusion: Box::new(query.into()),
        }
    }

    /// Evaluate this query against a scorer, returning per-agent similarity scores.
    ///
    /// Recursively evaluates sub-queries and combines results according to
    /// the operator semantics. Only agents with score > 0 are returned.
    pub fn evaluate(&self, scorer: &dyn Scorer) -> Result<HashMap<String, f64>, String> {
        match self {
            Query::Text(text) => {
                let scores = scorer.score(text)?;
                Ok(scores.into_iter().collect())
            }
            Query::All(queries) => {
                if queries.is_empty() {
                    return Ok(HashMap::new());
                }
                let mut maps: Vec<HashMap<String, f64>> = Vec::with_capacity(queries.len());
                for q in queries {
                    maps.push(q.evaluate(scorer)?);
                }
                // Agent must appear in ALL sub-results; score = min.
                let first = &maps[0];
                let mut result = HashMap::new();
                for (agent, &score) in first {
                    let min_score = maps[1..].iter().fold(score, |acc, m| {
                        m.get(agent).copied().unwrap_or(0.0).min(acc)
                    });
                    if min_score > 0.0 {
                        result.insert(agent.clone(), min_score);
                    }
                }
                Ok(result)
            }
            Query::Any(queries) => {
                if queries.is_empty() {
                    return Ok(HashMap::new());
                }
                let mut result: HashMap<String, f64> = HashMap::new();
                for q in queries {
                    let map = q.evaluate(scorer)?;
                    for (agent, score) in map {
                        let entry = result.entry(agent).or_insert(0.0);
                        *entry = entry.max(score);
                    }
                }
                Ok(result)
            }
            Query::Exclude { base, exclusion } => {
                let base_scores = base.evaluate(scorer)?;
                let exclusion_scores = exclusion.evaluate(scorer)?;
                let mut result = HashMap::new();
                for (agent, base_score) in base_scores {
                    let penalty = exclusion_scores.get(&agent).copied().unwrap_or(0.0);
                    let adjusted = base_score * (1.0 - penalty);
                    if adjusted > 0.0 {
                        result.insert(agent, adjusted);
                    }
                }
                Ok(result)
            }
            Query::Weighted(entries) => {
                if entries.is_empty() {
                    return Ok(HashMap::new());
                }
                let total_weight: f64 = entries.iter().map(|(_, w)| w).sum();
                if total_weight <= 0.0 {
                    return Ok(HashMap::new());
                }
                let mut accumulated: HashMap<String, f64> = HashMap::new();
                for (q, weight) in entries {
                    let map = q.evaluate(scorer)?;
                    for (agent, score) in map {
                        *accumulated.entry(agent).or_insert(0.0) += score * weight;
                    }
                }
                for score in accumulated.values_mut() {
                    *score /= total_weight;
                }
                accumulated.retain(|_, s| *s > 0.0);
                Ok(accumulated)
            }
        }
    }

    /// Returns the first text leaf in the query tree.
    ///
    /// Used as the "primary" query for feedback matching and signal construction
    /// when evaluating composite queries.
    pub fn primary_text(&self) -> Option<&str> {
        match self {
            Query::Text(s) => Some(s.as_str()),
            Query::All(qs) | Query::Any(qs) => qs.iter().find_map(|q| q.primary_text()),
            Query::Exclude { base, .. } => base.primary_text(),
            Query::Weighted(entries) => entries.iter().find_map(|(q, _)| q.primary_text()),
        }
    }
}

impl From<&str> for Query {
    fn from(s: &str) -> Self {
        Query::Text(s.to_string())
    }
}

impl From<String> for Query {
    fn from(s: String) -> Self {
        Query::Text(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scorer::MinHashScorer;

    fn setup_scorer() -> MinHashScorer {
        let mut scorer = MinHashScorer::default();
        scorer.index_capabilities(
            "translate",
            &["legal translation", "EN-DE", "German language"],
        );
        scorer.index_capabilities(
            "medical",
            &["medical records", "health diagnosis", "medical translation"],
        );
        scorer.index_capabilities("summarize", &["document summarization", "legal briefs"]);
        scorer
    }

    #[test]
    fn text_query_matches_scorer() {
        let scorer = setup_scorer();
        let query = Query::Text("legal translation".to_string());
        let scores = query.evaluate(&scorer).unwrap();
        assert!(scores.contains_key("translate"));
        assert!(scores["translate"] > 0.0);
    }

    #[test]
    fn all_requires_both_sub_queries() {
        let scorer = setup_scorer();
        let query = Query::all(["legal translation", "German language"]);
        let scores = query.evaluate(&scorer).unwrap();
        // translate has both "legal translation" and "German language" capabilities
        assert!(
            scores.contains_key("translate"),
            "translate should match both, got: {:?}",
            scores
        );
        // summarize only matches "legal" loosely, not "German language"
        let translate_score = scores.get("translate").copied().unwrap_or(0.0);
        let summarize_score = scores.get("summarize").copied().unwrap_or(0.0);
        assert!(
            translate_score > summarize_score,
            "translate ({:.3}) should outscore summarize ({:.3})",
            translate_score,
            summarize_score
        );
    }

    #[test]
    fn any_matches_either_sub_query() {
        let scorer = setup_scorer();
        let query = Query::any(["legal translation", "document summarization"]);
        let scores = query.evaluate(&scorer).unwrap();
        assert!(
            scores.contains_key("translate"),
            "translate should match via 'legal translation'"
        );
        assert!(
            scores.contains_key("summarize"),
            "summarize should match via 'document summarization'"
        );
    }

    #[test]
    fn exclude_penalises_matching_agents() {
        let scorer = setup_scorer();
        let base = Query::Text("medical translation".to_string());
        let base_scores = base.evaluate(&scorer).unwrap();
        let base_medical = base_scores.get("medical").copied().unwrap_or(0.0);

        let excluded = Query::Text("medical translation".to_string()).exclude("medical records");
        let exc_scores = excluded.evaluate(&scorer).unwrap();
        let exc_medical = exc_scores.get("medical").copied().unwrap_or(0.0);

        assert!(
            exc_medical < base_medical,
            "medical with exclusion ({:.3}) should be less than without ({:.3})",
            exc_medical,
            base_medical
        );
    }

    #[test]
    fn weighted_combines_sub_queries() {
        let scorer = setup_scorer();
        let query = Query::weighted([("legal translation", 2.0), ("document summarization", 1.0)]);
        let scores = query.evaluate(&scorer).unwrap();
        assert!(
            scores.contains_key("translate"),
            "translate should appear in weighted results"
        );
        assert!(
            scores.contains_key("summarize"),
            "summarize should appear in weighted results"
        );
    }

    #[test]
    fn empty_queries_return_empty() {
        let scorer = setup_scorer();
        assert!(Query::All(vec![]).evaluate(&scorer).unwrap().is_empty());
        assert!(Query::Any(vec![]).evaluate(&scorer).unwrap().is_empty());
        assert!(Query::Weighted(vec![])
            .evaluate(&scorer)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn primary_text_traverses_tree() {
        let q = Query::all(["translate legal", "German"]);
        assert_eq!(q.primary_text(), Some("translate legal"));

        let q = Query::from("hello").exclude("world");
        assert_eq!(q.primary_text(), Some("hello"));

        let q = Query::weighted([("alpha", 1.0), ("beta", 2.0)]);
        assert_eq!(q.primary_text(), Some("alpha"));
    }

    #[test]
    fn from_str_creates_text_query() {
        let q: Query = "hello world".into();
        assert!(matches!(&q, Query::Text(s) if s == "hello world"));
    }

    #[test]
    fn exclude_chaining_is_composable() {
        let scorer = setup_scorer();
        let query = Query::all(["legal translation"]).exclude("medical records");
        let scores = query.evaluate(&scorer).unwrap();
        // translate should still appear (it doesn't match "medical records" strongly)
        assert!(
            scores.contains_key("translate"),
            "translate should survive exclude, got: {:?}",
            scores
        );
    }

    #[test]
    fn all_score_is_minimum() {
        let scorer = setup_scorer();
        // Get individual scores for translate agent
        let q1_scores = Query::from("legal translation").evaluate(&scorer).unwrap();
        let q2_scores = Query::from("German language").evaluate(&scorer).unwrap();
        let q1_translate = q1_scores.get("translate").copied().unwrap_or(0.0);
        let q2_translate = q2_scores.get("translate").copied().unwrap_or(0.0);
        let expected_min = q1_translate.min(q2_translate);

        let all_scores = Query::all(["legal translation", "German language"])
            .evaluate(&scorer)
            .unwrap();
        let all_translate = all_scores.get("translate").copied().unwrap_or(0.0);
        assert!(
            (all_translate - expected_min).abs() < 1e-10,
            "All score ({:.4}) should equal min of sub-scores ({:.4}, {:.4})",
            all_translate,
            q1_translate,
            q2_translate
        );
    }

    #[test]
    fn any_score_is_maximum() {
        let scorer = setup_scorer();
        let q1_scores = Query::from("legal translation").evaluate(&scorer).unwrap();
        let q2_scores = Query::from("document summarization")
            .evaluate(&scorer)
            .unwrap();
        let q1_translate = q1_scores.get("translate").copied().unwrap_or(0.0);
        let q2_translate = q2_scores.get("translate").copied().unwrap_or(0.0);
        let expected_max = q1_translate.max(q2_translate);

        let any_scores = Query::any(["legal translation", "document summarization"])
            .evaluate(&scorer)
            .unwrap();
        let any_translate = any_scores.get("translate").copied().unwrap_or(0.0);
        assert!(
            (any_translate - expected_max).abs() < 1e-10,
            "Any score ({:.4}) should equal max of sub-scores ({:.4}, {:.4})",
            any_translate,
            q1_translate,
            q2_translate
        );
    }
}
