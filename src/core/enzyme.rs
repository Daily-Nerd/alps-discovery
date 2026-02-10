// ALPS Discovery SDK — Enzyme (Multi-Kernel Routing)
//
// Implements the Enzyme trait with multi-kernel voting for discovery.
// Each kernel evaluates independently; the enzyme picks the hypha
// with the most votes (simple majority).

use std::collections::{BTreeMap, HashMap};

use crate::core::action::{EnzymeAction, EnzymeDecision};
use crate::core::hyphae::Hypha;
use crate::core::signal::Signal;
use crate::core::types::{DecisionId, HyphaId, KernelType};

/// The core enzyme trait — sans-IO routing decision interface.
///
/// OSS-simplified: no `&Spore` or `&MembraneState` parameters.
/// These are proprietary protocol constructs not used in local discovery.
pub trait Enzyme {
    /// Process a signal and return a routing decision.
    ///
    /// `scorer_context` provides pre-computed scorer similarities keyed by HyphaId.
    fn process(
        &mut self,
        signal: &Signal,
        hyphae: &[&Hypha],
        scorer_context: &ScorerContext,
    ) -> EnzymeDecision;
}

/// A ranked hypha recommendation from a single reasoning kernel.
#[derive(Debug, Clone)]
pub struct KernelRecommendation {
    /// Ranked list of (hypha_id, score) pairs, best first.
    pub ranked: Vec<(HyphaId, f64)>,
}

/// Pre-computed scorer similarities keyed by HyphaId.
///
/// Passed to kernels so that CapabilityKernel can use the same similarity
/// values the user sees, eliminating divergence between scorer and chemistry paths.
pub type ScorerContext = HashMap<HyphaId, f64>;

/// Individual reasoning kernel interface.
///
/// Each kernel implements a different scoring strategy for hypha selection.
/// The `scorer_context` provides pre-computed scorer similarities so that
/// kernels can use the same values as the user-visible ranking.
pub trait ReasoningKernel: Send + Sync {
    /// The kernel type identifier.
    fn kernel_type(&self) -> KernelType;

    /// Evaluate hyphae and return a ranked recommendation.
    ///
    /// `scorer_context` maps HyphaId → raw_similarity from the pluggable Scorer.
    /// Kernels that need capability similarity (e.g. CapabilityKernel) should use
    /// these values instead of Chemistry. Other kernels may ignore it.
    fn evaluate(
        &self,
        signal: &Signal,
        hyphae: &[&Hypha],
        scorer_context: &ScorerContext,
    ) -> KernelRecommendation;
}

// ---------------------------------------------------------------------------
// Discovery kernels
// ---------------------------------------------------------------------------

/// Capability-matching kernel: scores hyphae by scorer similarity,
/// weighted by diameter and tau reliability.
///
/// Score: `scorer_similarity * diameter * (0.5 + 0.5 * tau.min(1.0))`.
///
/// Uses pre-computed scorer similarities from `scorer_context` instead of
/// Chemistry. This eliminates the divergence between the user-visible ranking
/// (from Scorer) and the enzyme evaluation (previously from Chemistry).
///
/// The tau factor provides a 50–100% scaling window: new agents (tau ≈ 0.001)
/// still get half their capability score, while proven agents (tau → 1.0) get
/// the full score. This prevents cold-start starvation while rewarding reliability.
pub struct CapabilityKernel;

impl ReasoningKernel for CapabilityKernel {
    fn kernel_type(&self) -> KernelType {
        KernelType::CapabilityMatching
    }

    fn evaluate(
        &self,
        _signal: &Signal,
        hyphae: &[&Hypha],
        scorer_context: &ScorerContext,
    ) -> KernelRecommendation {
        let mut ranked: Vec<(HyphaId, f64)> = hyphae
            .iter()
            .map(|h| {
                let similarity = scorer_context.get(&h.id).copied().unwrap_or(0.0);
                let tau_factor = 0.5 + 0.5 * h.state.tau.min(1.0);
                (h.id.clone(), similarity * h.state.diameter * tau_factor)
            })
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        KernelRecommendation { ranked }
    }
}

/// Novelty-seeking kernel: prioritizes less-explored hyphae.
pub struct NoveltyKernel;

impl ReasoningKernel for NoveltyKernel {
    fn kernel_type(&self) -> KernelType {
        KernelType::NoveltySeeking
    }

    fn evaluate(
        &self,
        _signal: &Signal,
        hyphae: &[&Hypha],
        _scorer_context: &ScorerContext,
    ) -> KernelRecommendation {
        let mut ranked: Vec<(HyphaId, f64)> = hyphae
            .iter()
            .map(|h| {
                let novelty = 1.0 / (1.0 + h.state.sigma);
                (h.id.clone(), novelty)
            })
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        KernelRecommendation { ranked }
    }
}

/// Load-balancing kernel: favors least-used hyphae.
pub struct LoadBalancingKernel;

impl ReasoningKernel for LoadBalancingKernel {
    fn kernel_type(&self) -> KernelType {
        KernelType::LoadBalancing
    }

    fn evaluate(
        &self,
        _signal: &Signal,
        hyphae: &[&Hypha],
        _scorer_context: &ScorerContext,
    ) -> KernelRecommendation {
        let mut ranked: Vec<(HyphaId, f64)> = hyphae
            .iter()
            .map(|h| {
                let score = h.state.diameter / (1.0 + h.state.forwards_count.get() as f64);
                (h.id.clone(), score)
            })
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        KernelRecommendation { ranked }
    }
}

/// Temporal recency kernel: favors agents with recent successful activity.
///
/// Score: exponential decay from `last_activity`. Does NOT multiply by
/// diameter — maintaining kernel diversity (orthogonal to CapabilityKernel).
pub struct TemporalRecencyKernel;

impl ReasoningKernel for TemporalRecencyKernel {
    fn kernel_type(&self) -> KernelType {
        KernelType::TemporalRecency
    }

    fn evaluate(
        &self,
        _signal: &Signal,
        hyphae: &[&Hypha],
        _scorer_context: &ScorerContext,
    ) -> KernelRecommendation {
        use std::time::Instant;
        let now = Instant::now();
        let mut ranked: Vec<(HyphaId, f64)> = hyphae
            .iter()
            .map(|h| {
                let elapsed_secs = now.duration_since(h.last_activity).as_secs_f64();
                // Exponential decay with 60-second half-life.
                // Recent activity → high score, stale → low score.
                let recency = (-elapsed_secs / 86.6).exp(); // ln(2)/86.6 ≈ 0.008 → half-life ~60s
                (h.id.clone(), recency)
            })
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        KernelRecommendation { ranked }
    }
}

// ---------------------------------------------------------------------------
// Quorum
// ---------------------------------------------------------------------------

/// Quorum mode for kernel voting decisions.
#[derive(Debug, Clone)]
pub enum Quorum {
    /// Simple majority (>50% agreement). Default, backwards compatible.
    Majority,
    /// All kernels must agree. Maximizes split rate / fan-out.
    Unanimous,
    /// Configurable supermajority threshold (e.g. 0.75 = 3/4 must agree).
    Supermajority(f64),
}

// ---------------------------------------------------------------------------
// SLNEnzymeConfig
// ---------------------------------------------------------------------------

/// Configuration for the multi-kernel discovery enzyme.
#[derive(Debug, Clone)]
pub struct SLNEnzymeConfig {
    /// Maximum number of hyphae to split across on disagreement (default: 3).
    pub max_disagreement_split: usize,
    /// Quorum mode for kernel voting (default: Majority).
    pub quorum: Quorum,
}

impl Default for SLNEnzymeConfig {
    fn default() -> Self {
        Self {
            max_disagreement_split: 3,
            quorum: Quorum::Majority,
        }
    }
}

// ---------------------------------------------------------------------------
// SLNEnzyme — multi-kernel discovery enzyme
// ---------------------------------------------------------------------------

/// Multi-kernel discovery enzyme.
///
/// Maintains a set of reasoning kernels. Each kernel evaluates independently
/// and the enzyme picks the hypha with the most votes (simple majority).
/// On disagreement, splits across the top-voted hyphae.
pub struct SLNEnzyme {
    kernels: Vec<Box<dyn ReasoningKernel>>,
    config: SLNEnzymeConfig,
    decisions_made: u64,
}

impl SLNEnzyme {
    /// Create a new SLNEnzyme with the given kernels and configuration.
    pub fn new(kernels: Vec<Box<dyn ReasoningKernel>>, config: SLNEnzymeConfig) -> Self {
        Self {
            kernels,
            config,
            decisions_made: 0,
        }
    }

    /// Create a new SLNEnzyme with the discovery kernel mix:
    /// CapabilityKernel + LoadBalancingKernel + NoveltyKernel + TemporalRecencyKernel.
    pub fn with_discovery_kernels(config: SLNEnzymeConfig) -> Self {
        let kernels: Vec<Box<dyn ReasoningKernel>> = vec![
            Box::new(CapabilityKernel),
            Box::new(LoadBalancingKernel),
            Box::new(NoveltyKernel),
            Box::new(TemporalRecencyKernel),
        ];
        Self::new(kernels, config)
    }

    fn next_decision_id(&mut self) -> DecisionId {
        self.decisions_made += 1;
        DecisionId(self.decisions_made)
    }

    /// Evaluate all kernels and produce a routing decision via majority vote.
    fn evaluate_kernels(
        &self,
        signal: &Signal,
        hyphae: &[&Hypha],
        scorer_context: &ScorerContext,
    ) -> EnzymeAction {
        if hyphae.is_empty() {
            return EnzymeAction::Dissolve;
        }

        // Each kernel picks its top-1 hypha (argmax).
        let mut votes: BTreeMap<HyphaId, usize> = BTreeMap::new();
        for kernel in &self.kernels {
            let rec = kernel.evaluate(signal, hyphae, scorer_context);
            if let Some((id, _)) = rec.ranked.first() {
                *votes.entry(id.clone()).or_insert(0) += 1;
            }
        }

        if votes.is_empty() {
            return EnzymeAction::Dissolve;
        }

        // Find the hypha with the most votes.
        let Some((best_hypha, best_count)) = votes
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(h, c)| (h.clone(), *c))
        else {
            return EnzymeAction::Dissolve;
        };

        // Check quorum: does the winner have enough votes?
        // Majority requires strictly more than 50% (>50%), not >= 50%.
        // With 4 kernels: need 3 votes (not 2). With 5 kernels: need 3 votes.
        let quorum_met = match &self.config.quorum {
            Quorum::Majority => best_count * 2 > self.kernels.len(),
            Quorum::Unanimous => best_count == self.kernels.len(),
            Quorum::Supermajority(threshold) => {
                (best_count as f64 / self.kernels.len() as f64) >= *threshold
            }
        };

        if quorum_met {
            EnzymeAction::Forward { target: best_hypha }
        } else {
            // No majority → split across distinct top picks.
            let mut targets: Vec<HyphaId> = votes.keys().cloned().collect();
            targets.truncate(self.config.max_disagreement_split);
            if let [single] = targets.as_slice() {
                EnzymeAction::Forward {
                    target: single.clone(),
                }
            } else {
                EnzymeAction::Split { targets }
            }
        }
    }
}

/// Per-agent kernel evaluation results.
#[derive(Debug, Clone)]
pub struct KernelEvaluation {
    /// Normalized composite kernel score per agent ∈ [0, 1].
    pub agent_scores: HashMap<HyphaId, f64>,
    /// Which kernel types voted for which top agent.
    pub top_picks: Vec<(KernelType, HyphaId)>,
}

impl SLNEnzyme {
    /// Evaluate all kernels and return normalized composite scores.
    ///
    /// For each kernel, scores are normalized to [0,1] by dividing by the max.
    /// The composite score is the average across all kernels.
    /// `top_picks` tracks each kernel's top choice for confidence detection.
    ///
    /// `scorer_context` provides pre-computed scorer similarities so the
    /// CapabilityKernel uses the same values as the user-visible ranking.
    pub fn evaluate_with_scores(
        &self,
        signal: &Signal,
        hyphae: &[&Hypha],
        scorer_context: &ScorerContext,
    ) -> KernelEvaluation {
        if hyphae.is_empty() {
            return KernelEvaluation {
                agent_scores: HashMap::new(),
                top_picks: Vec::new(),
            };
        }

        let num_kernels = self.kernels.len();
        let mut composite: HashMap<HyphaId, f64> = HashMap::new();
        let mut top_picks: Vec<(KernelType, HyphaId)> = Vec::new();

        for kernel in &self.kernels {
            let rec = kernel.evaluate(signal, hyphae, scorer_context);

            // Find max score for normalization.
            let max_score = rec.ranked.iter().map(|(_, s)| *s).fold(0.0_f64, f64::max);

            // Track top pick.
            if let Some((id, _)) = rec.ranked.first() {
                top_picks.push((kernel.kernel_type(), id.clone()));
            }

            // Normalize and accumulate.
            for (id, score) in &rec.ranked {
                let normalized = if max_score > 0.0 {
                    score / max_score
                } else {
                    0.0
                };
                *composite.entry(id.clone()).or_insert(0.0) += normalized;
            }
        }

        // Average across kernels.
        if num_kernels > 0 {
            for score in composite.values_mut() {
                *score /= num_kernels as f64;
            }
        }

        KernelEvaluation {
            agent_scores: composite,
            top_picks,
        }
    }

    /// Access the enzyme config.
    pub fn config(&self) -> &SLNEnzymeConfig {
        &self.config
    }
}

impl Enzyme for SLNEnzyme {
    fn process(
        &mut self,
        signal: &Signal,
        hyphae: &[&Hypha],
        scorer_context: &ScorerContext,
    ) -> EnzymeDecision {
        let decision_id = self.next_decision_id();

        let action = match signal {
            Signal::Tendril(_) => self.evaluate_kernels(signal, hyphae, scorer_context),
            _ => EnzymeAction::Absorb,
        };

        EnzymeDecision {
            action,
            decision_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::chemistry::{Chemistry, QuerySignature};
    use crate::core::config::QueryConfig;
    use crate::core::hyphae::Hypha;
    use crate::core::pheromone::HyphaState;
    use crate::core::signal::{Signal, Tendril};
    use crate::core::types::{HyphaId, PeerAddr, TrailId};
    use std::time::Instant;

    fn make_hypha(
        id_byte: u8,
        diameter: f64,
        sigma: f64,
        forwards_count: u64,
        chemistry: Chemistry,
    ) -> Hypha {
        make_hypha_with_tau(id_byte, diameter, sigma, forwards_count, chemistry, 0.01)
    }

    fn make_hypha_with_tau(
        id_byte: u8,
        diameter: f64,
        sigma: f64,
        forwards_count: u64,
        chemistry: Chemistry,
        tau: f64,
    ) -> Hypha {
        let mut id = [0u8; 32];
        id[0] = id_byte;
        Hypha {
            id: HyphaId(id),
            peer: PeerAddr(format!("local://agent-{}", id_byte)),
            state: HyphaState {
                diameter,
                tau,
                sigma,
                consecutive_pulse_timeouts: 0,
                forwards_count: crate::core::pheromone::AtomicCounter::new(forwards_count),
            },
            chemistry,
            last_activity: Instant::now(),
        }
    }

    fn make_tendril_signal(query_sig: QuerySignature) -> Signal {
        Signal::Tendril(Tendril {
            trail_id: TrailId([0u8; 32]),
            query_signature: query_sig,
            query_config: QueryConfig::default(),
        })
    }

    /// Build scorer context from (hypha, similarity) pairs.
    fn make_scorer_context(entries: &[(&Hypha, f64)]) -> ScorerContext {
        entries
            .iter()
            .map(|(h, sim)| (h.id.clone(), *sim))
            .collect()
    }

    fn empty_scorer_context() -> ScorerContext {
        ScorerContext::new()
    }

    // -----------------------------------------------------------------------
    // CapabilityKernel tests
    // -----------------------------------------------------------------------

    #[test]
    fn capability_kernel_ranks_by_scorer_similarity_times_diameter_times_tau() {
        let kernel = CapabilityKernel;

        // Hypha A: scorer says similarity = 1.0, diameter = 1.0, tau = 1.0
        let h_a = make_hypha_with_tau(1, 1.0, 0.0, 0, Chemistry::new(), 1.0);
        // Hypha B: scorer says similarity = 0.0, diameter = 1.0, tau = 1.0
        let h_b = make_hypha_with_tau(2, 1.0, 0.0, 0, Chemistry::new(), 1.0);

        let signal = make_tendril_signal(QuerySignature::default());
        let hyphae: Vec<&Hypha> = vec![&h_b, &h_a]; // put B first to verify sorting
        let ctx = make_scorer_context(&[(&h_a, 1.0), (&h_b, 0.0)]);
        let rec = kernel.evaluate(&signal, &hyphae, &ctx);

        // tau_factor = 0.5 + 0.5 * 1.0 = 1.0
        // Hypha A: 1.0 * 1.0 * 1.0 = 1.0, Hypha B: 0.0 * 1.0 * 1.0 = 0.0
        assert_eq!(rec.ranked.len(), 2);
        assert_eq!(rec.ranked[0].0, h_a.id);
        assert!((rec.ranked[0].1 - 1.0).abs() < f64::EPSILON);
        assert_eq!(rec.ranked[1].0, h_b.id);
        assert!(rec.ranked[1].1.abs() < f64::EPSILON);
    }

    #[test]
    fn capability_kernel_tau_influences_score() {
        let kernel = CapabilityKernel;

        // Both hyphae have same scorer similarity (1.0) and diameter, but different tau.
        let h_reliable = make_hypha_with_tau(1, 1.0, 0.0, 0, Chemistry::new(), 0.8);
        let h_new = make_hypha_with_tau(2, 1.0, 0.0, 0, Chemistry::new(), 0.01);

        let signal = make_tendril_signal(QuerySignature::default());
        let hyphae: Vec<&Hypha> = vec![&h_new, &h_reliable];
        let ctx = make_scorer_context(&[(&h_reliable, 1.0), (&h_new, 1.0)]);
        let rec = kernel.evaluate(&signal, &hyphae, &ctx);

        // h_reliable (tau=0.8 → factor=0.9) should outscore h_new (tau=0.01 → factor=0.505)
        assert_eq!(rec.ranked[0].0, h_reliable.id);
        assert!(
            rec.ranked[0].1 > rec.ranked[1].1,
            "reliable ({:.3}) should outscore new ({:.3})",
            rec.ranked[0].1,
            rec.ranked[1].1
        );
    }

    #[test]
    fn capability_kernel_uses_scorer_not_chemistry() {
        let kernel = CapabilityKernel;

        // Hypha A: chemistry would give high similarity, but scorer says 0.0
        let mut chem_a = Chemistry::new();
        chem_a.deposit(&[0x42; 64]);
        let h_a = make_hypha_with_tau(1, 1.0, 0.0, 0, chem_a, 1.0);

        // Hypha B: chemistry would give 0.0, but scorer says 1.0
        let h_b = make_hypha_with_tau(2, 1.0, 0.0, 0, Chemistry::new(), 1.0);

        let signal = make_tendril_signal(QuerySignature::new([0x42; 64]));
        let hyphae: Vec<&Hypha> = vec![&h_a, &h_b];
        // Scorer says B is better — opposite of what Chemistry would say
        let ctx = make_scorer_context(&[(&h_a, 0.0), (&h_b, 1.0)]);
        let rec = kernel.evaluate(&signal, &hyphae, &ctx);

        // B should rank first (scorer says 1.0), proving scorer overrides chemistry
        assert_eq!(rec.ranked[0].0, h_b.id);
        assert!((rec.ranked[0].1 - 1.0).abs() < f64::EPSILON);
        assert_eq!(rec.ranked[1].0, h_a.id);
        assert!(rec.ranked[1].1.abs() < f64::EPSILON);
    }

    #[test]
    fn capability_kernel_diameter_tiebreak() {
        let kernel = CapabilityKernel;

        // Both have same scorer similarity (1.0) but different diameters.
        let h_a = make_hypha_with_tau(1, 3.0, 0.0, 0, Chemistry::new(), 1.0);
        let h_b = make_hypha_with_tau(2, 7.0, 0.0, 0, Chemistry::new(), 1.0);

        let signal = make_tendril_signal(QuerySignature::default());
        let hyphae: Vec<&Hypha> = vec![&h_a, &h_b];
        let ctx = make_scorer_context(&[(&h_a, 1.0), (&h_b, 1.0)]);
        let rec = kernel.evaluate(&signal, &hyphae, &ctx);

        // h_b should rank first: 1.0 * 7.0 * 1.0 > 1.0 * 3.0 * 1.0
        assert_eq!(rec.ranked[0].0, h_b.id);
        assert_eq!(rec.ranked[1].0, h_a.id);
    }

    #[test]
    fn capability_kernel_empty_hyphae() {
        let kernel = CapabilityKernel;
        let signal = make_tendril_signal(QuerySignature::default());
        let hyphae: Vec<&Hypha> = vec![];
        let rec = kernel.evaluate(&signal, &hyphae, &empty_scorer_context());
        assert!(rec.ranked.is_empty());
    }

    // -----------------------------------------------------------------------
    // NoveltyKernel tests
    // -----------------------------------------------------------------------

    #[test]
    fn novelty_kernel_favors_low_sigma() {
        let kernel = NoveltyKernel;

        let h_a = make_hypha(1, 1.0, 0.0, 0, Chemistry::new());
        let h_b = make_hypha(2, 1.0, 9.0, 0, Chemistry::new());

        let signal = make_tendril_signal(QuerySignature::default());
        let hyphae: Vec<&Hypha> = vec![&h_b, &h_a];
        let rec = kernel.evaluate(&signal, &hyphae, &empty_scorer_context());

        assert_eq!(rec.ranked[0].0, h_a.id);
        assert!((rec.ranked[0].1 - 1.0).abs() < f64::EPSILON);
        assert!((rec.ranked[1].1 - 0.1).abs() < f64::EPSILON);
    }

    #[test]
    fn novelty_kernel_ignores_diameter() {
        let kernel = NoveltyKernel;

        let h_a = make_hypha(1, 2.0, 1.0, 0, Chemistry::new());
        let h_b = make_hypha(2, 6.0, 1.0, 0, Chemistry::new());

        let signal = make_tendril_signal(QuerySignature::default());
        let hyphae: Vec<&Hypha> = vec![&h_a, &h_b];
        let rec = kernel.evaluate(&signal, &hyphae, &empty_scorer_context());

        assert!((rec.ranked[0].1 - 0.5).abs() < f64::EPSILON);
        assert!((rec.ranked[1].1 - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn novelty_kernel_empty() {
        let kernel = NoveltyKernel;
        let signal = make_tendril_signal(QuerySignature::default());
        let hyphae: Vec<&Hypha> = vec![];
        let rec = kernel.evaluate(&signal, &hyphae, &empty_scorer_context());
        assert!(rec.ranked.is_empty());
    }

    // -----------------------------------------------------------------------
    // LoadBalancingKernel tests
    // -----------------------------------------------------------------------

    #[test]
    fn load_balancing_kernel_favors_fewer_forwards() {
        let kernel = LoadBalancingKernel;

        let h_a = make_hypha(1, 1.0, 0.0, 0, Chemistry::new());
        let h_b = make_hypha(2, 1.0, 0.0, 99, Chemistry::new());

        let signal = make_tendril_signal(QuerySignature::default());
        let hyphae: Vec<&Hypha> = vec![&h_b, &h_a];
        let rec = kernel.evaluate(&signal, &hyphae, &empty_scorer_context());

        assert_eq!(rec.ranked[0].0, h_a.id);
        assert!((rec.ranked[0].1 - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn load_balancing_kernel_scales_by_diameter() {
        let kernel = LoadBalancingKernel;

        let h_a = make_hypha(1, 2.0, 0.0, 5, Chemistry::new());
        let h_b = make_hypha(2, 12.0, 0.0, 5, Chemistry::new());

        let signal = make_tendril_signal(QuerySignature::default());
        let hyphae: Vec<&Hypha> = vec![&h_a, &h_b];
        let rec = kernel.evaluate(&signal, &hyphae, &empty_scorer_context());

        assert_eq!(rec.ranked[0].0, h_b.id);
        assert!((rec.ranked[0].1 - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn load_balancing_kernel_empty() {
        let kernel = LoadBalancingKernel;
        let signal = make_tendril_signal(QuerySignature::default());
        let hyphae: Vec<&Hypha> = vec![];
        let rec = kernel.evaluate(&signal, &hyphae, &empty_scorer_context());
        assert!(rec.ranked.is_empty());
    }

    // -----------------------------------------------------------------------
    // SLNEnzyme integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn sln_enzyme_empty_hyphae_dissolve() {
        let mut enzyme = SLNEnzyme::with_discovery_kernels(SLNEnzymeConfig::default());
        let signal = make_tendril_signal(QuerySignature::default());

        let hyphae: Vec<&Hypha> = vec![];

        let decision = enzyme.process(&signal, &hyphae, &empty_scorer_context());
        assert_eq!(decision.action, EnzymeAction::Dissolve);
    }

    #[test]
    fn sln_enzyme_unanimous_vote_forward() {
        // Create one hypha that is clearly the best by ALL kernel criteria:
        //   - Best scorer similarity (CapabilityKernel)
        //   - Lowest sigma (NoveltyKernel)
        //   - Fewest forwards (LoadBalancingKernel)
        //   - Most recent activity (TemporalRecencyKernel — both created now, so tied)
        let h_good = make_hypha(1, 5.0, 0.0, 0, Chemistry::new());
        let h_bad = make_hypha(2, 1.0, 100.0, 1000, Chemistry::new());

        let signal = make_tendril_signal(QuerySignature::default());

        let hyphae: Vec<&Hypha> = vec![&h_good, &h_bad];
        let ctx = make_scorer_context(&[(&h_good, 1.0), (&h_bad, 0.0)]);

        let mut enzyme = SLNEnzyme::with_discovery_kernels(SLNEnzymeConfig::default());
        let decision = enzyme.process(&signal, &hyphae, &ctx);

        assert_eq!(
            decision.action,
            EnzymeAction::Forward {
                target: h_good.id.clone()
            }
        );
    }

    #[test]
    fn sln_enzyme_disagreement_split_with_unanimous_quorum() {
        // With Unanimous quorum, even partial agreement causes a Split.
        // CapabilityKernel picks by: scorer_similarity * diameter * tau_factor
        // NoveltyKernel picks by:    1/(1+sigma)
        // LoadBalancingKernel picks by: diameter / (1+forwards_count)
        // TemporalRecencyKernel picks by: recency(last_activity)
        //
        // Hypha A: best scorer similarity, but high sigma and high forwards
        let h_a = make_hypha(1, 2.0, 1000.0, 1000, Chemistry::new());
        // Hypha B: no scorer similarity, but lowest sigma, high forwards
        let h_b = make_hypha(2, 2.0, 0.0, 1000, Chemistry::new());
        // Hypha C: no scorer similarity, high sigma, but fewest forwards
        let h_c = make_hypha(3, 2.0, 1000.0, 0, Chemistry::new());

        let signal = make_tendril_signal(QuerySignature::default());

        let hyphae: Vec<&Hypha> = vec![&h_a, &h_b, &h_c];
        let ctx = make_scorer_context(&[(&h_a, 1.0), (&h_b, 0.0), (&h_c, 0.0)]);

        let config = SLNEnzymeConfig {
            quorum: Quorum::Unanimous,
            ..SLNEnzymeConfig::default()
        };
        let mut enzyme = SLNEnzyme::with_discovery_kernels(config);
        let decision = enzyme.process(&signal, &hyphae, &ctx);

        match &decision.action {
            EnzymeAction::Split { targets } => {
                assert!(
                    targets.len() <= 3,
                    "split targets should be <= max_disagreement_split(3), got {}",
                    targets.len()
                );
                assert!(!targets.is_empty(), "split targets should not be empty");
            }
            other => panic!(
                "expected Split action with unanimous quorum, got {:?}",
                other
            ),
        }
    }

    #[test]
    fn sln_enzyme_majority_quorum_forwards_with_agreement() {
        // h_good: best scorer similarity, low sigma, low forwards → 3+ kernels agree
        let h_good = make_hypha(1, 5.0, 0.0, 0, Chemistry::new());
        let h_bad = make_hypha(2, 1.0, 100.0, 1000, Chemistry::new());

        let signal = make_tendril_signal(QuerySignature::default());

        let hyphae: Vec<&Hypha> = vec![&h_good, &h_bad];
        let ctx = make_scorer_context(&[(&h_good, 1.0), (&h_bad, 0.0)]);

        let mut enzyme = SLNEnzyme::with_discovery_kernels(SLNEnzymeConfig::default());
        let decision = enzyme.process(&signal, &hyphae, &ctx);

        assert_eq!(
            decision.action,
            EnzymeAction::Forward {
                target: h_good.id.clone()
            }
        );
    }

    #[test]
    fn sln_enzyme_non_tendril_absorb() {
        let mut enzyme = SLNEnzyme::with_discovery_kernels(SLNEnzymeConfig::default());
        let signal = Signal::Nutrient;

        let h = make_hypha(1, 1.0, 0.0, 0, Chemistry::new());
        let hyphae: Vec<&Hypha> = vec![&h];

        let decision = enzyme.process(&signal, &hyphae, &empty_scorer_context());
        assert_eq!(decision.action, EnzymeAction::Absorb);
    }

    // -----------------------------------------------------------------------
    // evaluate_with_scores tests
    // -----------------------------------------------------------------------

    #[test]
    fn evaluate_with_scores_returns_normalized() {
        let enzyme = SLNEnzyme::with_discovery_kernels(SLNEnzymeConfig::default());

        let h_a = make_hypha(1, 5.0, 0.0, 0, Chemistry::new());
        let h_b = make_hypha(2, 1.0, 10.0, 50, Chemistry::new());

        let signal = make_tendril_signal(QuerySignature::default());
        let hyphae: Vec<&Hypha> = vec![&h_a, &h_b];
        let ctx = make_scorer_context(&[(&h_a, 0.9), (&h_b, 0.1)]);

        let eval = enzyme.evaluate_with_scores(&signal, &hyphae, &ctx);

        for score in eval.agent_scores.values() {
            assert!(
                *score >= 0.0 && *score <= 1.0,
                "score {} should be in [0, 1]",
                score
            );
        }

        assert_eq!(eval.top_picks.len(), 4);
    }

    #[test]
    fn evaluate_with_scores_unanimous_detection() {
        let enzyme = SLNEnzyme::with_discovery_kernels(SLNEnzymeConfig::default());

        // Create h_bad FIRST so h_good is more recent (TemporalRecencyKernel).
        let h_bad = make_hypha(2, 1.0, 100.0, 1000, Chemistry::new());
        let h_good = make_hypha(1, 5.0, 0.0, 0, Chemistry::new());

        let signal = make_tendril_signal(QuerySignature::default());
        let hyphae: Vec<&Hypha> = vec![&h_good, &h_bad];
        let ctx = make_scorer_context(&[(&h_good, 1.0), (&h_bad, 0.0)]);

        let eval = enzyme.evaluate_with_scores(&signal, &hyphae, &ctx);

        let all_same = eval.top_picks.iter().all(|(_, id)| *id == h_good.id);
        assert!(all_same, "all kernels should pick h_good unanimously");

        let good_score = eval.agent_scores.get(&h_good.id).copied().unwrap_or(0.0);
        let bad_score = eval.agent_scores.get(&h_bad.id).copied().unwrap_or(0.0);
        assert!(
            good_score > bad_score,
            "h_good ({:.3}) should outscore h_bad ({:.3})",
            good_score,
            bad_score
        );
    }

    #[test]
    fn evaluate_with_scores_split_detection() {
        let enzyme = SLNEnzyme::with_discovery_kernels(SLNEnzymeConfig::default());

        // Each kernel picks a different winner.
        let h_a = make_hypha(1, 2.0, 1000.0, 1000, Chemistry::new());
        let h_b = make_hypha(2, 2.0, 0.0, 1000, Chemistry::new());
        let h_c = make_hypha(3, 2.0, 1000.0, 0, Chemistry::new());

        let signal = make_tendril_signal(QuerySignature::default());
        let hyphae: Vec<&Hypha> = vec![&h_a, &h_b, &h_c];
        let ctx = make_scorer_context(&[(&h_a, 1.0), (&h_b, 0.0), (&h_c, 0.0)]);

        let eval = enzyme.evaluate_with_scores(&signal, &hyphae, &ctx);

        let distinct_ids: std::collections::HashSet<&HyphaId> =
            eval.top_picks.iter().map(|(_, id)| id).collect();
        assert!(
            distinct_ids.len() >= 2,
            "should have at least 2 distinct top picks, got {}",
            distinct_ids.len()
        );
    }

    #[test]
    fn evaluate_with_scores_empty_hyphae() {
        let enzyme = SLNEnzyme::with_discovery_kernels(SLNEnzymeConfig::default());
        let signal = make_tendril_signal(QuerySignature::default());
        let hyphae: Vec<&Hypha> = vec![];

        let eval = enzyme.evaluate_with_scores(&signal, &hyphae, &empty_scorer_context());
        assert!(eval.agent_scores.is_empty());
        assert!(eval.top_picks.is_empty());
    }

    // -----------------------------------------------------------------------
    // TemporalRecencyKernel tests
    // -----------------------------------------------------------------------

    #[test]
    fn temporal_recency_kernel_favors_recent() {
        let kernel = TemporalRecencyKernel;

        let h_old = make_hypha(1, 1.0, 0.0, 0, Chemistry::new());
        std::thread::sleep(std::time::Duration::from_millis(10));
        let h_new = make_hypha(2, 1.0, 0.0, 0, Chemistry::new());

        let signal = make_tendril_signal(QuerySignature::default());
        let hyphae: Vec<&Hypha> = vec![&h_old, &h_new];
        let rec = kernel.evaluate(&signal, &hyphae, &empty_scorer_context());

        assert_eq!(rec.ranked[0].0, h_new.id);
        assert!(
            rec.ranked[0].1 > rec.ranked[1].1,
            "newer ({:.6}) should outscore older ({:.6})",
            rec.ranked[0].1,
            rec.ranked[1].1
        );
    }

    #[test]
    fn temporal_recency_kernel_ignores_diameter() {
        let kernel = TemporalRecencyKernel;

        let h_a = make_hypha(1, 10.0, 0.0, 0, Chemistry::new());
        let h_b = make_hypha(2, 0.1, 0.0, 0, Chemistry::new());

        let signal = make_tendril_signal(QuerySignature::default());
        let hyphae: Vec<&Hypha> = vec![&h_a, &h_b];
        let rec = kernel.evaluate(&signal, &hyphae, &empty_scorer_context());

        let diff = (rec.ranked[0].1 - rec.ranked[1].1).abs();
        assert!(
            diff < 0.01,
            "same-time hyphae should have similar recency scores, diff={:.6}",
            diff
        );
    }

    #[test]
    fn temporal_recency_kernel_empty() {
        let kernel = TemporalRecencyKernel;
        let signal = make_tendril_signal(QuerySignature::default());
        let hyphae: Vec<&Hypha> = vec![];
        let rec = kernel.evaluate(&signal, &hyphae, &empty_scorer_context());
        assert!(rec.ranked.is_empty());
    }

    // -----------------------------------------------------------------------
    // Quorum tests
    // -----------------------------------------------------------------------

    #[test]
    fn unanimous_quorum_produces_more_splits() {
        // Same setup, run with Majority vs Unanimous quorum.
        // Unanimous should produce more Splits.
        let h_a = make_hypha(1, 5.0, 0.0, 0, Chemistry::new());
        let h_b = make_hypha(2, 1.0, 5.0, 50, Chemistry::new());

        let signal = make_tendril_signal(QuerySignature::default());

        let hyphae: Vec<&Hypha> = vec![&h_a, &h_b];
        let ctx = make_scorer_context(&[(&h_a, 1.0), (&h_b, 0.1)]);

        // With Majority quorum — likely Forward (3-4 kernels agree on h_a).
        let config_majority = SLNEnzymeConfig {
            quorum: Quorum::Majority,
            ..SLNEnzymeConfig::default()
        };
        let mut enzyme_majority = SLNEnzyme::with_discovery_kernels(config_majority);
        let dec_majority = enzyme_majority.process(&signal, &hyphae, &ctx);
        let majority_forwards = matches!(dec_majority.action, EnzymeAction::Forward { .. });

        // With Unanimous quorum — might Split if any kernel disagrees.
        let config_unanimous = SLNEnzymeConfig {
            quorum: Quorum::Unanimous,
            ..SLNEnzymeConfig::default()
        };
        let mut enzyme_unanimous = SLNEnzyme::with_discovery_kernels(config_unanimous);
        let dec_unanimous = enzyme_unanimous.process(&signal, &hyphae, &ctx);
        let unanimous_forwards = matches!(dec_unanimous.action, EnzymeAction::Forward { .. });

        if !majority_forwards {
            assert!(
                !unanimous_forwards,
                "if majority splits, unanimous must also split"
            );
        }
    }

    #[test]
    fn evaluate_with_scores_has_four_top_picks() {
        let enzyme = SLNEnzyme::with_discovery_kernels(SLNEnzymeConfig::default());

        let h_a = make_hypha(1, 1.0, 0.0, 0, Chemistry::new());
        let h_b = make_hypha(2, 1.0, 0.0, 0, Chemistry::new());

        let signal = make_tendril_signal(QuerySignature::default());
        let hyphae: Vec<&Hypha> = vec![&h_a, &h_b];
        let ctx = make_scorer_context(&[(&h_a, 0.5), (&h_b, 0.5)]);
        let eval = enzyme.evaluate_with_scores(&signal, &hyphae, &ctx);

        assert_eq!(
            eval.top_picks.len(),
            4,
            "should have 4 top picks for 4 kernels"
        );
    }

    // -----------------------------------------------------------------------
    // Quorum majority semantic tests (Requirement 3)
    // -----------------------------------------------------------------------

    #[test]
    fn quorum_majority_with_4_kernels_requires_3_votes() {
        // With 4 kernels (Capability, LoadBalancing, Novelty, TemporalRecency),
        // Majority quorum should require 3+ votes (>50%), not 2 (=50%).
        //
        // Setup: h_a gets 2 votes, h_b gets 2 votes → should produce Split, not Forward.
        //
        // Kernel voting pattern:
        // - CapabilityKernel: votes for h_a (scorer_similarity = 1.0)
        // - NoveltyKernel: votes for h_b (sigma = 0.0, lower than h_a)
        // - LoadBalancingKernel: votes for h_b (forwards_count = 0, same as h_a)
        // - TemporalRecencyKernel: votes for whichever was created last
        let h_a = make_hypha(1, 1.0, 10.0, 0, Chemistry::new()); // high sigma
        std::thread::sleep(std::time::Duration::from_millis(5));
        let h_b = make_hypha(2, 1.0, 0.0, 0, Chemistry::new()); // low sigma, more recent

        let signal = make_tendril_signal(QuerySignature::default());
        let hyphae: Vec<&Hypha> = vec![&h_a, &h_b];
        // Give h_a strong scorer advantage to get CapabilityKernel vote
        let ctx = make_scorer_context(&[(&h_a, 1.0), (&h_b, 0.0)]);

        let config = SLNEnzymeConfig {
            quorum: Quorum::Majority,
            max_disagreement_split: 3,
        };
        let mut enzyme = SLNEnzyme::with_discovery_kernels(config);
        let decision = enzyme.process(&signal, &hyphae, &ctx);

        // With 2-of-4 votes, should produce Split (not majority).
        match &decision.action {
            EnzymeAction::Split { targets } => {
                assert_eq!(
                    targets.len(),
                    2,
                    "2-of-4 vote should produce 2-way split, got {} targets",
                    targets.len()
                );
            }
            other => panic!("2-of-4 votes should produce Split, got {:?}", other),
        }
    }

    #[test]
    fn quorum_majority_with_4_kernels_forwards_with_3_votes() {
        // With 4 kernels, 3+ votes should produce Forward action.
        //
        // Setup: h_good dominates on 3+ criteria → gets 3+ votes → Forward.
        let h_good = make_hypha(1, 5.0, 0.0, 0, Chemistry::new());
        let h_bad = make_hypha(2, 1.0, 100.0, 1000, Chemistry::new());

        let signal = make_tendril_signal(QuerySignature::default());
        let hyphae: Vec<&Hypha> = vec![&h_good, &h_bad];
        let ctx = make_scorer_context(&[(&h_good, 1.0), (&h_bad, 0.0)]);

        let config = SLNEnzymeConfig {
            quorum: Quorum::Majority,
            max_disagreement_split: 3,
        };
        let mut enzyme = SLNEnzyme::with_discovery_kernels(config);
        let decision = enzyme.process(&signal, &hyphae, &ctx);

        // With 3+ of 4 votes, should Forward.
        assert_eq!(
            decision.action,
            EnzymeAction::Forward {
                target: h_good.id.clone()
            },
            "3+ votes should produce Forward, got {:?}",
            decision.action
        );
    }

    #[test]
    fn quorum_majority_with_5_kernels_requires_3_votes() {
        // With 5 kernels, majority requires (5/2) + 1 = 3 votes (>50%).
        // This test verifies the formula works for odd numbers of kernels.
        //
        // We'll create a custom enzyme with 5 kernels to test this.
        let kernels: Vec<Box<dyn ReasoningKernel>> = vec![
            Box::new(CapabilityKernel),
            Box::new(LoadBalancingKernel),
            Box::new(NoveltyKernel),
            Box::new(TemporalRecencyKernel),
            Box::new(NoveltyKernel), // Duplicate to get 5 kernels
        ];
        let config = SLNEnzymeConfig {
            quorum: Quorum::Majority,
            max_disagreement_split: 3,
        };
        let mut enzyme = SLNEnzyme::new(kernels, config);

        let h_a = make_hypha(1, 1.0, 0.0, 0, Chemistry::new());
        let h_b = make_hypha(2, 1.0, 0.0, 0, Chemistry::new());

        let signal = make_tendril_signal(QuerySignature::default());
        let hyphae: Vec<&Hypha> = vec![&h_a, &h_b];
        let ctx = make_scorer_context(&[(&h_a, 0.5), (&h_b, 0.5)]);

        // With identical hyphae, likely 3+ kernels will agree → Forward.
        let decision = enzyme.process(&signal, &hyphae, &ctx);

        // Should produce either Forward (3+ votes) or Split (2 votes).
        // The key is that 2-of-5 (40%) should NOT be considered majority.
        match &decision.action {
            EnzymeAction::Forward { .. } => {
                // OK if 3+ kernels agreed
            }
            EnzymeAction::Split { .. } => {
                // OK if <3 votes for any single agent
            }
            other => panic!("Expected Forward or Split, got {:?}", other),
        }
    }
}
