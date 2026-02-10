// ALPS Discovery — Discovery Pipeline
//
// Pure functions for the scoring/ranking/filtering pipeline.
// Takes pre-computed scorer output and enzyme evaluation,
// combines with feedback and filters, produces ranked results.

use std::collections::{BTreeMap, HashMap};

use crate::core::action::EnzymeAction;
use crate::core::config::QueryConfig;
use crate::core::enzyme::{KernelEvaluation, SLNEnzymeConfig, ScorerContext};
use crate::core::hyphae::Hypha;
use crate::core::lsh::{ConfidenceInterval, MinHasher};
use crate::core::signal::{Signal, Tendril};
use crate::core::types::{HyphaId, KernelType, TrailId};

use super::enzyme_adapter::EnzymeAdapter;
use super::filter::Filters;
use super::registry::{AgentRecord, FeedbackIndex};
use super::scorer_adapter::ScorerAdapter;

/// Strength of per-query feedback adjustment to diameter (max +/-20%).
/// Research: "Low-exploitation + high-fan-out outperforms high-exploitation
/// + low-fan-out in sparse networks."
pub const FEEDBACK_STRENGTH: f64 = 0.2;

/// Number of independent samples (signature bytes) for CI computation.
const CI_SAMPLE_SIZE: f64 = 64.0;

/// A single discovery result with scoring breakdown.
#[derive(Debug, Clone)]
pub struct DiscoveryResult {
    /// Agent name.
    pub agent_name: String,
    /// Raw Chemistry similarity to the query [0.0, 1.0].
    pub similarity: f64,
    /// Combined routing score (similarity x enzyme x feedback).
    pub score: f64,
    /// Agent endpoint (URI, URL, module path, etc.) if provided at registration.
    pub endpoint: Option<String>,
    /// Arbitrary metadata (protocol, version, framework, etc.) if provided.
    pub metadata: HashMap<String, String>,
}

/// Extended discovery result with full scoring breakdown for debugging.
#[derive(Debug, Clone)]
pub struct ExplainedResult {
    /// Agent name.
    pub agent_name: String,
    /// Raw similarity from the scorer [0.0, 1.0].
    pub raw_similarity: f64,
    /// 95% confidence interval on the similarity estimate.
    pub similarity_ci: ConfidenceInterval,
    /// Agent diameter (routing weight from feedback history).
    pub diameter: f64,
    /// Normalized composite enzyme score [0.0, 1.0].
    pub enzyme_score: f64,
    /// Per-query feedback factor [-1.0, 1.0].
    pub feedback_factor: f64,
    /// Final combined score.
    pub final_score: f64,
    /// Agent endpoint if provided at registration.
    pub endpoint: Option<String>,
    /// Agent metadata if provided at registration.
    pub metadata: HashMap<String, String>,
}

/// Confidence level of a discovery decision based on kernel agreement.
#[derive(Debug, Clone, PartialEq)]
pub enum DiscoveryConfidence {
    /// All kernels agree on the top agent.
    Unanimous,
    /// Majority of kernels agree. One dissents.
    Majority { dissenting_kernel: KernelType },
    /// No majority. Caller should consider parallel execution.
    Split { alternative_agents: Vec<String> },
}

/// Discovery response with confidence signal.
#[derive(Debug, Clone)]
pub struct DiscoveryResponse {
    /// Ranked discovery results.
    pub results: Vec<DiscoveryResult>,
    /// Kernel agreement confidence level.
    pub confidence: DiscoveryConfidence,
    /// Recommended number of agents to invoke in parallel.
    pub recommended_parallelism: usize,
}

/// Internal scored candidate used by the shared discovery pipeline.
/// Public for benchmark access when `bench` feature is enabled.
pub struct ScoredCandidate {
    pub agent_name: String,
    pub raw_similarity: f64,
    pub similarity_ci: ConfidenceInterval,
    pub diameter: f64,
    pub enzyme_score: f64,
    pub feedback_factor: f64,
    pub final_score: f64,
    pub endpoint: Option<String>,
    pub metadata: HashMap<String, String>,
}

/// Lightweight candidate with borrowed fields for the pre-filter scoring phase.
pub(crate) struct CandidateRef<'a> {
    pub agent_name: &'a str,
    pub raw_similarity: f64,
    pub similarity_ci: ConfidenceInterval,
    pub diameter: f64,
    pub enzyme_score: f64,
    pub feedback_factor: f64,
    pub final_score: f64,
    pub endpoint: &'a Option<String>,
    pub metadata: &'a HashMap<String, String>,
}

impl ScoredCandidate {
    /// Convert to a public DiscoveryResult (drops internal scoring fields).
    pub(crate) fn into_result(self) -> DiscoveryResult {
        DiscoveryResult {
            agent_name: self.agent_name,
            similarity: self.raw_similarity,
            score: self.final_score,
            endpoint: self.endpoint,
            metadata: self.metadata,
        }
    }

    /// Convert to a public ExplainedResult (preserves all scoring fields).
    pub(crate) fn into_explained(self) -> ExplainedResult {
        ExplainedResult {
            agent_name: self.agent_name,
            raw_similarity: self.raw_similarity,
            similarity_ci: self.similarity_ci,
            diameter: self.diameter,
            enzyme_score: self.enzyme_score,
            feedback_factor: self.feedback_factor,
            final_score: self.final_score,
            endpoint: self.endpoint,
            metadata: self.metadata,
        }
    }
}

/// Compute a 95% confidence interval from a raw similarity point estimate.
/// Public for benchmark access when `bench` feature is enabled.
pub fn similarity_to_ci(sim: f64) -> ConfidenceInterval {
    let se = (sim * (1.0 - sim) / CI_SAMPLE_SIZE).sqrt();
    ConfidenceInterval {
        point_estimate: sim,
        lower_bound: (sim - 1.96 * se).max(0.0),
        upper_bound: (sim + 1.96 * se).min(1.0),
    }
}

/// Compute per-query feedback factor for an agent using banded LSH index.
///
/// Uses the FeedbackIndex to find near-neighbor feedback records in O(k)
/// instead of scanning all records in O(n). Returns a value in [-1.0, 1.0]
/// representing how well this agent has performed on similar queries.
/// Public for benchmark access when `bench` feature is enabled.
pub fn compute_feedback_factor(
    feedback: &FeedbackIndex,
    query_minhash: &[u8; 64],
    relevance_threshold: f64,
) -> f64 {
    if feedback.is_empty() {
        return 0.0;
    }

    let mut weighted_sum = 0.0;
    let mut weight_sum = 0.0;

    // Use banded LSH to find candidate near-neighbors.
    let candidates = feedback.find_candidates(query_minhash);

    for fb in candidates {
        let relevance = MinHasher::similarity(&fb.query_minhash, query_minhash);
        if relevance >= relevance_threshold {
            weighted_sum += relevance * fb.outcome;
            weight_sum += relevance;
        }
    }

    if weight_sum > 0.0 {
        (weighted_sum / weight_sum).clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

/// Derive confidence from kernel evaluation top picks.
pub(crate) fn derive_confidence(
    kernel_eval: &KernelEvaluation,
    agents: &BTreeMap<String, AgentRecord>,
    enzyme_config: &SLNEnzymeConfig,
) -> (DiscoveryConfidence, usize) {
    let top_picks = &kernel_eval.top_picks;
    if top_picks.is_empty() {
        return (DiscoveryConfidence::Unanimous, 1);
    }

    // Count votes per hypha.
    let mut votes: BTreeMap<&HyphaId, usize> = BTreeMap::new();
    for (_, hid) in top_picks {
        *votes.entry(hid).or_insert(0) += 1;
    }

    let total = top_picks.len();
    let (best_hid, best_count) = votes
        .iter()
        .max_by_key(|(_, c)| *c)
        .map(|(h, c)| (*h, *c))
        .unwrap_or((&HyphaId([0u8; 32]), 0));

    if best_count == total {
        // All kernels agree.
        (DiscoveryConfidence::Unanimous, 1)
    } else if best_count * 2 >= total {
        // Majority agrees — find the dissenter.
        let dissenting_kernel = top_picks
            .iter()
            .find(|(_, hid)| *hid != *best_hid)
            .map(|(kt, _)| *kt)
            .unwrap_or(KernelType::NoveltySeeking);
        (DiscoveryConfidence::Majority { dissenting_kernel }, 1)
    } else {
        // No majority — split.
        let hypha_to_name: HashMap<&HyphaId, &str> = agents
            .iter()
            .map(|(name, record)| (&record.hypha.id, name.as_str()))
            .collect();
        let alternative_agents: Vec<String> = votes
            .keys()
            .filter(|hid| **hid != best_hid)
            .filter_map(|hid| hypha_to_name.get(hid).map(|s| s.to_string()))
            .collect();

        let distinct_picks = votes.len();
        let max_split = enzyme_config.max_disagreement_split;
        let parallelism = distinct_picks.min(max_split);
        (
            DiscoveryConfidence::Split { alternative_agents },
            parallelism,
        )
    }
}

// ---------------------------------------------------------------------------
// Core discovery pipeline
// ---------------------------------------------------------------------------

/// Run the discovery pipeline for a string query.
///
/// Scores all agents via the scorer, then delegates to `run_pipeline_with_scores`.
/// `exploration_epsilon` controls the probability of shuffling tied agents (0.0 = pure exploit, 1.0 = always explore).
/// Public for benchmark access when `bench` feature is enabled.
pub fn run_pipeline(
    agents: &BTreeMap<String, AgentRecord>,
    scorer: &ScorerAdapter,
    enzyme: &mut EnzymeAdapter,
    query: &str,
    filters: Option<&Filters>,
    exploration_epsilon: f64,
) -> (Vec<ScoredCandidate>, KernelEvaluation) {
    let raw_scores = match scorer.score(query) {
        Ok(scores) => scores,
        Err(e) => {
            eprintln!("alps-discovery: scorer.score() error: {}", e);
            return (
                Vec::new(),
                KernelEvaluation {
                    agent_scores: HashMap::new(),
                    top_picks: Vec::new(),
                },
            );
        }
    };
    let score_map: HashMap<String, f64> = raw_scores.into_iter().collect();
    run_pipeline_with_scores(
        agents,
        scorer,
        enzyme,
        query,
        score_map,
        filters,
        exploration_epsilon,
    )
}

/// Inner discovery pipeline operating on pre-computed scores.
///
/// `query_text` is used for the Signal/Tendril construction and feedback matching.
/// `score_map` provides per-agent similarity scores (from Scorer or Query algebra).
/// Public for benchmark access when `bench` feature is enabled.
pub fn run_pipeline_with_scores(
    agents: &BTreeMap<String, AgentRecord>,
    scorer: &ScorerAdapter,
    enzyme: &mut EnzymeAdapter,
    query_text: &str,
    score_map: HashMap<String, f64>,
    filters: Option<&Filters>,
    exploration_epsilon: f64,
) -> (Vec<ScoredCandidate>, KernelEvaluation) {
    if agents.is_empty() {
        return (
            Vec::new(),
            KernelEvaluation {
                agent_scores: HashMap::new(),
                top_picks: Vec::new(),
            },
        );
    }

    let query_sig = scorer.compute_query_signature(query_text.as_bytes());

    // Build signal and collect hyphae for enzyme evaluation.
    let signal = Signal::Tendril(Tendril {
        trail_id: TrailId([0u8; 32]),
        query_signature: query_sig.clone(),
        query_config: QueryConfig::default(),
    });
    let hyphae: Vec<&Hypha> = agents.values().map(|r| &r.hypha).collect();

    // Build a map from agent name to hypha_id for enzyme score lookup.
    let name_to_hypha: HashMap<&str, &HyphaId> = agents
        .iter()
        .map(|(name, record)| (name.as_str(), &record.hypha.id))
        .collect();

    // Build scorer context (HyphaId → raw_similarity) for unified enzyme evaluation.
    let scorer_context: ScorerContext = score_map
        .iter()
        .filter_map(|(name, &sim)| {
            name_to_hypha
                .get(name.as_str())
                .map(|hid| ((*hid).clone(), sim))
        })
        .collect();

    // Run enzyme evaluation with scorer context.
    let kernel_eval = enzyme.evaluate_with_scores(&signal, &hyphae, &scorer_context);

    // Build lightweight references — no cloning yet.
    let mut refs: Vec<CandidateRef<'_>> = agents
        .iter()
        .filter_map(|(name, record)| {
            let sim = score_map.get(name.as_str()).copied().unwrap_or(0.0);
            if sim < scorer.lsh_config.similarity_threshold {
                return None;
            }

            let feedback_factor = compute_feedback_factor(
                &record.feedback,
                &query_sig.minhash,
                scorer.lsh_config.similarity_threshold,
            );

            // Look up enzyme score for this agent.
            let enzyme_score = name_to_hypha
                .get(name.as_str())
                .and_then(|hid| kernel_eval.agent_scores.get(*hid))
                .copied()
                .unwrap_or(0.0);

            // Formula: raw_similarity × (0.5 + 0.5 × enzyme_score) × (1.0 + feedback_factor × FEEDBACK_STRENGTH)
            let score =
                sim * (0.5 + 0.5 * enzyme_score) * (1.0 + feedback_factor * FEEDBACK_STRENGTH);

            Some(CandidateRef {
                agent_name: name.as_str(),
                raw_similarity: sim,
                similarity_ci: similarity_to_ci(sim),
                diameter: record.hypha.state.diameter,
                enzyme_score,
                feedback_factor,
                final_score: score,
                endpoint: &record.endpoint,
                metadata: &record.metadata,
            })
        })
        .collect();

    // Apply metadata filters before sorting.
    if let Some(filters) = filters {
        refs.retain(|r| {
            filters
                .iter()
                .all(|(key, filter)| match r.metadata.get(key) {
                    Some(value) => filter.matches(value),
                    None => false,
                })
        });
    }

    refs.sort_by(|a, b| {
        b.final_score
            .partial_cmp(&a.final_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Adaptive exploration: two-tier tie-breaking.
    //
    // Tier 1 — TRUE TIES: agents with identical scores (within float epsilon) are
    // ALWAYS shuffled for fair distribution. This ensures equally-qualified agents
    // get equal opportunity regardless of registration order.
    //
    // Tier 2 — EXPLORATION: with probability `exploration_epsilon`, extend the shuffle
    // to the wider CI-overlap group. This allows discovering agents whose confidence
    // intervals overlap the top agent but whose point estimates differ. As epsilon
    // decays with feedback, exploration narrows toward pure exploitation.
    if refs.len() > 1 {
        use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
        use xxhash_rust::xxh3::xxh3_64_with_seed;

        // Monotonic counter for unique seeds across calls. Avoids SystemTime
        // clock resolution issues that caused repeated seeds on fast calls.
        static TIE_BREAK_COUNTER: AtomicU64 = AtomicU64::new(0);

        let top_score = refs[0].final_score;

        // Threshold for "effectively identical" scores. Must be large enough to
        // absorb floating-point noise from enzyme kernel evaluation (temporal
        // recency kernel introduces ~1e-6 differences between agents registered
        // microseconds apart), but small enough not to conflate genuinely
        // different capability matches.
        const STRICT_TIE_EPSILON: f64 = 1e-4;

        let strict_tie_count = refs
            .iter()
            .take_while(|r| (r.final_score - top_score).abs() < STRICT_TIE_EPSILON)
            .count();

        let top_ci = &refs[0].similarity_ci;
        let ci_tie_count = refs
            .iter()
            .take_while(|r| r.similarity_ci.overlaps(top_ci))
            .count();

        // Start with strict ties (always shuffled for fairness).
        let mut shuffle_count = strict_tie_count;

        // Epsilon extends to CI-overlap group for exploration.
        if ci_tie_count > shuffle_count && exploration_epsilon > 0.0 {
            let counter = TIE_BREAK_COUNTER.fetch_add(1, AtomicOrdering::Relaxed);
            let explore_seed = xxh3_64_with_seed(query_text.as_bytes(), counter);
            let explore_roll = (explore_seed % 1000) as f64 / 1000.0;
            if explore_roll < exploration_epsilon {
                shuffle_count = ci_tie_count;
            }
        }

        if shuffle_count > 1 {
            let counter = TIE_BREAK_COUNTER.fetch_add(1, AtomicOrdering::Relaxed);
            let base_seed = xxh3_64_with_seed(query_text.as_bytes(), counter);

            let shuffle_slice = &mut refs[..shuffle_count];
            // Fisher-Yates shuffle using splitmix64 PRNG.
            let mut rng_state = base_seed;
            for i in (1..shuffle_slice.len()).rev() {
                rng_state = rng_state.wrapping_add(0x9e3779b97f4a7c15);
                let mut z = rng_state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
                z ^= z >> 31;
                let j = (z as usize) % (i + 1);
                shuffle_slice.swap(i, j);
            }
        }
    }

    // Promote survivors to owned ScoredCandidate.
    let results: Vec<ScoredCandidate> = refs
        .into_iter()
        .map(|r| ScoredCandidate {
            agent_name: r.agent_name.to_string(),
            raw_similarity: r.raw_similarity,
            similarity_ci: r.similarity_ci,
            diameter: r.diameter,
            enzyme_score: r.enzyme_score,
            feedback_factor: r.feedback_factor,
            final_score: r.final_score,
            endpoint: r.endpoint.clone(),
            metadata: r.metadata.clone(),
        })
        .collect();

    // Run the enzyme process to update internal state (feeds LoadBalancingKernel).
    let decision = enzyme.process(&signal, &hyphae, &scorer_context);

    // Update forwards_count for the enzyme's pick.
    let picked = match &decision.action {
        EnzymeAction::Forward { target } => vec![target.clone()],
        EnzymeAction::Split { targets } => targets.clone(),
        _ => vec![],
    };
    for hypha_id in &picked {
        for record in agents.values() {
            if record.hypha.id == *hypha_id {
                record.hypha.state.forwards_count.increment();
            }
        }
    }

    (results, kernel_eval)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn similarity_to_ci_perfect() {
        let ci = similarity_to_ci(1.0);
        assert_eq!(ci.point_estimate, 1.0);
        assert_eq!(ci.lower_bound, 1.0);
        assert_eq!(ci.upper_bound, 1.0);
    }

    #[test]
    fn similarity_to_ci_zero() {
        let ci = similarity_to_ci(0.0);
        assert_eq!(ci.point_estimate, 0.0);
        assert_eq!(ci.lower_bound, 0.0);
        assert_eq!(ci.upper_bound, 0.0);
    }

    #[test]
    fn similarity_to_ci_midpoint() {
        let ci = similarity_to_ci(0.5);
        assert!(ci.lower_bound < 0.5);
        assert!(ci.upper_bound > 0.5);
        assert!(ci.lower_bound >= 0.0);
        assert!(ci.upper_bound <= 1.0);
    }

    #[test]
    fn feedback_factor_empty_is_zero() {
        let feedback = FeedbackIndex::new();
        let minhash = [0u8; 64];
        assert_eq!(compute_feedback_factor(&feedback, &minhash, 0.1), 0.0);
    }

    #[test]
    fn feedback_factor_positive() {
        use crate::network::registry::FeedbackRecord;
        let mut feedback = FeedbackIndex::new();
        let minhash = [42u8; 64];
        feedback.insert(FeedbackRecord {
            query_minhash: minhash,
            outcome: 1.0,
        });
        let factor = compute_feedback_factor(&feedback, &minhash, 0.1);
        assert!(factor > 0.0);
    }
}
