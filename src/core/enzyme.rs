// ALPS Discovery SDK — Enzyme (Multi-Kernel Routing)
//
// Implements the Enzyme trait with multi-kernel voting for discovery.
// Each kernel evaluates independently; the enzyme picks the hypha
// with the most votes (simple majority).

use std::collections::{BTreeMap, HashMap};

use crate::core::action::{EnzymeAction, EnzymeDecision};
use crate::core::hyphae::Hypha;
use crate::core::membrane::MembraneState;
use crate::core::signal::Signal;
use crate::core::spore::tree::Spore;
use crate::core::types::{DecisionId, HyphaId, KernelType};

/// The core enzyme trait — sans-IO routing decision interface.
pub trait Enzyme {
    /// Process a signal and return a routing decision.
    fn process(
        &mut self,
        signal: &Signal,
        spore: &Spore,
        hyphae: &[&Hypha],
        membrane: &MembraneState,
    ) -> EnzymeDecision;
}

/// A ranked hypha recommendation from a single reasoning kernel.
#[derive(Debug, Clone)]
pub struct KernelRecommendation {
    /// Ranked list of (hypha_id, score) pairs, best first.
    pub ranked: Vec<(HyphaId, f64)>,
}

/// Individual reasoning kernel interface.
///
/// Each kernel implements a different scoring strategy for hypha selection.
pub trait ReasoningKernel: Send + Sync {
    /// The kernel type identifier.
    fn kernel_type(&self) -> KernelType;

    /// Evaluate hyphae and return a ranked recommendation.
    fn evaluate(&self, signal: &Signal, hyphae: &[&Hypha]) -> KernelRecommendation;
}

// ---------------------------------------------------------------------------
// Discovery kernels
// ---------------------------------------------------------------------------

/// Capability-matching kernel: scores hyphae by Chemistry similarity to the
/// query signature. Primary kernel for discovery.
///
/// Score: `similarity(hypha.chemistry, query_signature) * diameter`.
pub struct CapabilityKernel;

impl ReasoningKernel for CapabilityKernel {
    fn kernel_type(&self) -> KernelType {
        KernelType::CapabilityMatching
    }

    fn evaluate(&self, signal: &Signal, hyphae: &[&Hypha]) -> KernelRecommendation {
        let query_sig = match signal {
            Signal::Tendril(t) => &t.query_signature,
            _ => {
                let mut ranked: Vec<(HyphaId, f64)> = hyphae
                    .iter()
                    .map(|h| (h.id.clone(), h.state.diameter))
                    .collect();
                ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                return KernelRecommendation { ranked };
            }
        };

        let mut ranked: Vec<(HyphaId, f64)> = hyphae
            .iter()
            .map(|h| {
                let similarity = h.chemistry.similarity(query_sig);
                (h.id.clone(), similarity * h.state.diameter)
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

    fn evaluate(&self, _signal: &Signal, hyphae: &[&Hypha]) -> KernelRecommendation {
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

    fn evaluate(&self, _signal: &Signal, hyphae: &[&Hypha]) -> KernelRecommendation {
        let mut ranked: Vec<(HyphaId, f64)> = hyphae
            .iter()
            .map(|h| {
                let score = h.state.diameter / (1.0 + h.state.forwards_count as f64);
                (h.id.clone(), score)
            })
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        KernelRecommendation { ranked }
    }
}

// ---------------------------------------------------------------------------
// SLNEnzymeConfig
// ---------------------------------------------------------------------------

/// Configuration for the multi-kernel discovery enzyme.
#[derive(Debug, Clone)]
pub struct SLNEnzymeConfig {
    /// Maximum number of hyphae to split across on disagreement (default: 2).
    pub max_disagreement_split: usize,
}

impl Default for SLNEnzymeConfig {
    fn default() -> Self {
        Self {
            max_disagreement_split: 2,
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
    /// CapabilityKernel + LoadBalancingKernel + NoveltyKernel.
    pub fn with_discovery_kernels(config: SLNEnzymeConfig) -> Self {
        let kernels: Vec<Box<dyn ReasoningKernel>> = vec![
            Box::new(CapabilityKernel),
            Box::new(LoadBalancingKernel),
            Box::new(NoveltyKernel),
        ];
        Self::new(kernels, config)
    }

    fn next_decision_id(&mut self) -> DecisionId {
        self.decisions_made += 1;
        DecisionId(self.decisions_made)
    }

    /// Evaluate all kernels and produce a routing decision via majority vote.
    fn evaluate_kernels(&self, signal: &Signal, hyphae: &[&Hypha]) -> EnzymeAction {
        if hyphae.is_empty() {
            return EnzymeAction::Dissolve;
        }

        // Each kernel picks its top-1 hypha (argmax).
        let mut votes: BTreeMap<HyphaId, usize> = BTreeMap::new();
        for kernel in &self.kernels {
            let rec = kernel.evaluate(signal, hyphae);
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

        // Majority or unanimity → forward to the winner.
        if best_count * 2 >= self.kernels.len() {
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
    pub fn evaluate_with_scores(&self, signal: &Signal, hyphae: &[&Hypha]) -> KernelEvaluation {
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
            let rec = kernel.evaluate(signal, hyphae);

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
        _spore: &Spore,
        hyphae: &[&Hypha],
        _membrane: &MembraneState,
    ) -> EnzymeDecision {
        let decision_id = self.next_decision_id();

        let action = match signal {
            Signal::Tendril(_) => self.evaluate_kernels(signal, hyphae),
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
    use crate::core::config::{QueryConfig, SporeConfig};
    use crate::core::hyphae::Hypha;
    use crate::core::membrane::MembraneState;
    use crate::core::pheromone::HyphaState;
    use crate::core::signal::{Signal, Tendril};
    use crate::core::spore::tree::Spore;
    use crate::core::types::{HyphaId, PeerAddr, TrailId};
    use std::time::{Duration, Instant};

    fn make_hypha(
        id_byte: u8,
        diameter: f64,
        sigma: f64,
        forwards_count: u64,
        chemistry: Chemistry,
    ) -> Hypha {
        let mut id = [0u8; 32];
        id[0] = id_byte;
        Hypha {
            id: HyphaId(id),
            peer: PeerAddr(format!("local://agent-{}", id_byte)),
            state: HyphaState {
                diameter,
                tau: 0.01,
                sigma,
                omega: 0.0,
                consecutive_pulse_timeouts: 0,
                forwards_count,
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

    fn make_membrane() -> MembraneState {
        MembraneState {
            permeability: 1.0,
            deep_processing_active: false,
            buffered_count: 0,
            floor_duration: Duration::ZERO,
            below_sporulation_duration: Duration::ZERO,
            total_admitted: 0,
            total_dissolved: 0,
            total_processed: 0,
            admitted_rate: 0.0,
            dissolved_rate: 0.0,
        }
    }

    // -----------------------------------------------------------------------
    // CapabilityKernel tests
    // -----------------------------------------------------------------------

    #[test]
    fn capability_kernel_ranks_by_similarity_times_diameter() {
        let kernel = CapabilityKernel;

        // Hypha A: chemistry matches query (all 0x42), diameter = 1.0
        let mut chem_a = Chemistry::new();
        chem_a.deposit(&[0x42; 64]);
        let h_a = make_hypha(1, 1.0, 0.0, 0, chem_a);

        // Hypha B: chemistry does NOT match query (all 0x00), diameter = 1.0
        let mut chem_b = Chemistry::new();
        chem_b.deposit(&[0x00; 64]);
        let h_b = make_hypha(2, 1.0, 0.0, 0, chem_b);

        let query = QuerySignature::new([0x42; 64]);
        let signal = make_tendril_signal(query);

        let hyphae: Vec<&Hypha> = vec![&h_b, &h_a]; // put B first to verify sorting
        let rec = kernel.evaluate(&signal, &hyphae);

        // Hypha A should rank first (similarity=1.0*1.0=1.0 vs similarity=0.0*1.0=0.0)
        assert_eq!(rec.ranked.len(), 2);
        assert_eq!(rec.ranked[0].0, h_a.id);
        assert!((rec.ranked[0].1 - 1.0).abs() < f64::EPSILON);
        assert_eq!(rec.ranked[1].0, h_b.id);
        assert!(rec.ranked[1].1.abs() < f64::EPSILON);
    }

    #[test]
    fn capability_kernel_non_tendril_fallback() {
        let kernel = CapabilityKernel;

        let h_a = make_hypha(1, 2.0, 0.0, 0, Chemistry::new());
        let h_b = make_hypha(2, 5.0, 0.0, 0, Chemistry::new());

        let signal = Signal::Nutrient;
        let hyphae: Vec<&Hypha> = vec![&h_a, &h_b];
        let rec = kernel.evaluate(&signal, &hyphae);

        // Should rank by diameter only — h_b (5.0) before h_a (2.0)
        assert_eq!(rec.ranked.len(), 2);
        assert_eq!(rec.ranked[0].0, h_b.id);
        assert!((rec.ranked[0].1 - 5.0).abs() < f64::EPSILON);
        assert_eq!(rec.ranked[1].0, h_a.id);
        assert!((rec.ranked[1].1 - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn capability_kernel_equal_similarity_tiebreak() {
        let kernel = CapabilityKernel;

        // Both have the same chemistry (default 0xFF) so same similarity.
        let h_a = make_hypha(1, 3.0, 0.0, 0, Chemistry::new());
        let h_b = make_hypha(2, 7.0, 0.0, 0, Chemistry::new());

        // Query = default 0xFF matches both perfectly (similarity=1.0).
        let query = QuerySignature::default();
        let signal = make_tendril_signal(query);
        let hyphae: Vec<&Hypha> = vec![&h_a, &h_b];
        let rec = kernel.evaluate(&signal, &hyphae);

        // h_b should rank first: score = 1.0 * 7.0 vs 1.0 * 3.0
        assert_eq!(rec.ranked[0].0, h_b.id);
        assert_eq!(rec.ranked[1].0, h_a.id);
    }

    #[test]
    fn capability_kernel_empty_hyphae() {
        let kernel = CapabilityKernel;
        let signal = make_tendril_signal(QuerySignature::default());
        let hyphae: Vec<&Hypha> = vec![];
        let rec = kernel.evaluate(&signal, &hyphae);
        assert!(rec.ranked.is_empty());
    }

    // -----------------------------------------------------------------------
    // NoveltyKernel tests
    // -----------------------------------------------------------------------

    #[test]
    fn novelty_kernel_favors_low_sigma() {
        let kernel = NoveltyKernel;

        // Hypha A: sigma = 0.0 → novelty = 1/(1+0) = 1.0
        let h_a = make_hypha(1, 1.0, 0.0, 0, Chemistry::new());
        // Hypha B: sigma = 9.0 → novelty = 1/(1+9) = 0.1
        let h_b = make_hypha(2, 1.0, 9.0, 0, Chemistry::new());

        let signal = make_tendril_signal(QuerySignature::default());
        let hyphae: Vec<&Hypha> = vec![&h_b, &h_a];
        let rec = kernel.evaluate(&signal, &hyphae);

        assert_eq!(rec.ranked[0].0, h_a.id);
        assert!((rec.ranked[0].1 - 1.0).abs() < f64::EPSILON);
        assert!((rec.ranked[1].1 - 0.1).abs() < f64::EPSILON);
    }

    #[test]
    fn novelty_kernel_ignores_diameter() {
        let kernel = NoveltyKernel;

        // Same sigma, different diameters — NoveltyKernel should score identically.
        let h_a = make_hypha(1, 2.0, 1.0, 0, Chemistry::new()); // novelty = 1/(1+1) = 0.5
        let h_b = make_hypha(2, 6.0, 1.0, 0, Chemistry::new()); // novelty = 1/(1+1) = 0.5

        let signal = make_tendril_signal(QuerySignature::default());
        let hyphae: Vec<&Hypha> = vec![&h_a, &h_b];
        let rec = kernel.evaluate(&signal, &hyphae);

        // Both should have the same score (0.5) since diameter is not used.
        assert!((rec.ranked[0].1 - 0.5).abs() < f64::EPSILON);
        assert!((rec.ranked[1].1 - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn novelty_kernel_empty() {
        let kernel = NoveltyKernel;
        let signal = make_tendril_signal(QuerySignature::default());
        let hyphae: Vec<&Hypha> = vec![];
        let rec = kernel.evaluate(&signal, &hyphae);
        assert!(rec.ranked.is_empty());
    }

    // -----------------------------------------------------------------------
    // LoadBalancingKernel tests
    // -----------------------------------------------------------------------

    #[test]
    fn load_balancing_kernel_favors_fewer_forwards() {
        let kernel = LoadBalancingKernel;

        // Hypha A: forwards=0 → score = 1.0/(1+0)=1.0
        let h_a = make_hypha(1, 1.0, 0.0, 0, Chemistry::new());
        // Hypha B: forwards=99 → score = 1.0/(1+99)=0.01
        let h_b = make_hypha(2, 1.0, 0.0, 99, Chemistry::new());

        let signal = make_tendril_signal(QuerySignature::default());
        let hyphae: Vec<&Hypha> = vec![&h_b, &h_a];
        let rec = kernel.evaluate(&signal, &hyphae);

        assert_eq!(rec.ranked[0].0, h_a.id);
        assert!((rec.ranked[0].1 - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn load_balancing_kernel_scales_by_diameter() {
        let kernel = LoadBalancingKernel;

        // Same forwards_count, different diameters.
        let h_a = make_hypha(1, 2.0, 0.0, 5, Chemistry::new()); // score = 2.0/(1+5) = 0.333..
        let h_b = make_hypha(2, 12.0, 0.0, 5, Chemistry::new()); // score = 12.0/(1+5) = 2.0

        let signal = make_tendril_signal(QuerySignature::default());
        let hyphae: Vec<&Hypha> = vec![&h_a, &h_b];
        let rec = kernel.evaluate(&signal, &hyphae);

        assert_eq!(rec.ranked[0].0, h_b.id);
        assert!((rec.ranked[0].1 - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn load_balancing_kernel_empty() {
        let kernel = LoadBalancingKernel;
        let signal = make_tendril_signal(QuerySignature::default());
        let hyphae: Vec<&Hypha> = vec![];
        let rec = kernel.evaluate(&signal, &hyphae);
        assert!(rec.ranked.is_empty());
    }

    // -----------------------------------------------------------------------
    // SLNEnzyme integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn sln_enzyme_empty_hyphae_dissolve() {
        let mut enzyme = SLNEnzyme::with_discovery_kernels(SLNEnzymeConfig::default());
        let signal = make_tendril_signal(QuerySignature::default());
        let spore = Spore::new(SporeConfig::default());
        let membrane = make_membrane();
        let hyphae: Vec<&Hypha> = vec![];

        let decision = enzyme.process(&signal, &spore, &hyphae, &membrane);
        assert_eq!(decision.action, EnzymeAction::Dissolve);
    }

    #[test]
    fn sln_enzyme_unanimous_vote_forward() {
        // Create one hypha that is clearly the best by ALL kernel criteria:
        //   - Best chemistry match (CapabilityKernel)
        //   - Lowest sigma (NoveltyKernel)
        //   - Fewest forwards (LoadBalancingKernel)
        let mut chem_good = Chemistry::new();
        chem_good.deposit(&[0x42; 64]);
        let h_good = make_hypha(1, 5.0, 0.0, 0, chem_good);

        // A clearly worse hypha on all axes.
        let mut chem_bad = Chemistry::new();
        chem_bad.deposit(&[0x00; 64]);
        let h_bad = make_hypha(2, 1.0, 100.0, 1000, chem_bad);

        let query = QuerySignature::new([0x42; 64]);
        let signal = make_tendril_signal(query);
        let spore = Spore::new(SporeConfig::default());
        let membrane = make_membrane();
        let hyphae: Vec<&Hypha> = vec![&h_good, &h_bad];

        let mut enzyme = SLNEnzyme::with_discovery_kernels(SLNEnzymeConfig::default());
        let decision = enzyme.process(&signal, &spore, &hyphae, &membrane);

        assert_eq!(
            decision.action,
            EnzymeAction::Forward {
                target: h_good.id.clone()
            }
        );
    }

    #[test]
    fn sln_enzyme_disagreement_split() {
        // We need 3 hyphae where each kernel picks a different winner.
        //
        // CapabilityKernel picks by: similarity * diameter
        // NoveltyKernel picks by:    1/(1+sigma)  (diameter-independent)
        // LoadBalancingKernel picks by: diameter / (1+forwards_count)
        //
        // Hypha A: best chemistry match, but high sigma and high forwards
        //   - chemistry matches query perfectly, diameter=2.0, sigma=1000, forwards=1000
        //   - CapabilityKernel: 1.0 * 2.0 = 2.0
        //   - NoveltyKernel:    1/(1+1000) ≈ 0.001
        //   - LoadBalancing:    2.0/(1+1000) ≈ 0.002
        let mut chem_a = Chemistry::new();
        chem_a.deposit(&[0x42; 64]);
        let h_a = make_hypha(1, 2.0, 1000.0, 1000, chem_a);

        // Hypha B: poor chemistry match, but lowest sigma, high forwards
        //   - chemistry = all 0x00, diameter=2.0, sigma=0.0, forwards=1000
        //   - CapabilityKernel: 0.0 * 2.0 = 0.0
        //   - NoveltyKernel:    1/(1+0) = 1.0
        //   - LoadBalancing:    2.0/(1+1000) ≈ 0.002
        let mut chem_b = Chemistry::new();
        chem_b.deposit(&[0x00; 64]);
        let h_b = make_hypha(2, 2.0, 0.0, 1000, chem_b);

        // Hypha C: poor chemistry match, high sigma, but fewest forwards
        //   - chemistry = all 0x00, diameter=2.0, sigma=1000, forwards=0
        //   - CapabilityKernel: 0.0 * 2.0 = 0.0
        //   - NoveltyKernel:    1/(1+1000) ≈ 0.001
        //   - LoadBalancing:    2.0/(1+0) = 2.0
        let mut chem_c = Chemistry::new();
        chem_c.deposit(&[0x00; 64]);
        let h_c = make_hypha(3, 2.0, 1000.0, 0, chem_c);

        let query = QuerySignature::new([0x42; 64]);
        let signal = make_tendril_signal(query);
        let spore = Spore::new(SporeConfig::default());
        let membrane = make_membrane();
        let hyphae: Vec<&Hypha> = vec![&h_a, &h_b, &h_c];

        let mut enzyme = SLNEnzyme::with_discovery_kernels(SLNEnzymeConfig::default());
        let decision = enzyme.process(&signal, &spore, &hyphae, &membrane);

        // 3 kernels, 3 different top picks → no majority → Split
        // max_disagreement_split = 2, so targets.len() <= 2
        match &decision.action {
            EnzymeAction::Split { targets } => {
                assert!(
                    targets.len() <= 2,
                    "split targets should be <= max_disagreement_split(2), got {}",
                    targets.len()
                );
                assert!(!targets.is_empty(), "split targets should not be empty");
            }
            other => panic!("expected Split action on disagreement, got {:?}", other),
        }
    }

    #[test]
    fn sln_enzyme_non_tendril_absorb() {
        let mut enzyme = SLNEnzyme::with_discovery_kernels(SLNEnzymeConfig::default());
        let signal = Signal::Nutrient;
        let spore = Spore::new(SporeConfig::default());
        let membrane = make_membrane();

        let h = make_hypha(1, 1.0, 0.0, 0, Chemistry::new());
        let hyphae: Vec<&Hypha> = vec![&h];

        let decision = enzyme.process(&signal, &spore, &hyphae, &membrane);
        assert_eq!(decision.action, EnzymeAction::Absorb);
    }

    // -----------------------------------------------------------------------
    // evaluate_with_scores tests
    // -----------------------------------------------------------------------

    #[test]
    fn evaluate_with_scores_returns_normalized() {
        let enzyme = SLNEnzyme::with_discovery_kernels(SLNEnzymeConfig::default());

        let mut chem_a = Chemistry::new();
        chem_a.deposit(&[0x42; 64]);
        let h_a = make_hypha(1, 5.0, 0.0, 0, chem_a);

        let mut chem_b = Chemistry::new();
        chem_b.deposit(&[0x00; 64]);
        let h_b = make_hypha(2, 1.0, 10.0, 50, chem_b);

        let query = QuerySignature::new([0x42; 64]);
        let signal = make_tendril_signal(query);
        let hyphae: Vec<&Hypha> = vec![&h_a, &h_b];

        let eval = enzyme.evaluate_with_scores(&signal, &hyphae);

        // All agent scores should be in [0, 1].
        for score in eval.agent_scores.values() {
            assert!(
                *score >= 0.0 && *score <= 1.0,
                "score {} should be in [0, 1]",
                score
            );
        }

        // Should have top picks for each kernel (3 kernels).
        assert_eq!(eval.top_picks.len(), 3);
    }

    #[test]
    fn evaluate_with_scores_unanimous_detection() {
        let enzyme = SLNEnzyme::with_discovery_kernels(SLNEnzymeConfig::default());

        // One clearly dominant hypha: best similarity, lowest sigma, fewest forwards.
        let mut chem_good = Chemistry::new();
        chem_good.deposit(&[0x42; 64]);
        let h_good = make_hypha(1, 5.0, 0.0, 0, chem_good);

        let mut chem_bad = Chemistry::new();
        chem_bad.deposit(&[0x00; 64]);
        let h_bad = make_hypha(2, 1.0, 100.0, 1000, chem_bad);

        let query = QuerySignature::new([0x42; 64]);
        let signal = make_tendril_signal(query);
        let hyphae: Vec<&Hypha> = vec![&h_good, &h_bad];

        let eval = enzyme.evaluate_with_scores(&signal, &hyphae);

        // All 3 kernels should pick the same agent.
        let all_same = eval.top_picks.iter().all(|(_, id)| *id == h_good.id);
        assert!(all_same, "all kernels should pick h_good unanimously");

        // h_good should have a higher composite score.
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

        // Same setup as sln_enzyme_disagreement_split — each kernel picks different winner.
        let mut chem_a = Chemistry::new();
        chem_a.deposit(&[0x42; 64]);
        let h_a = make_hypha(1, 2.0, 1000.0, 1000, chem_a);

        let mut chem_b = Chemistry::new();
        chem_b.deposit(&[0x00; 64]);
        let h_b = make_hypha(2, 2.0, 0.0, 1000, chem_b);

        let mut chem_c = Chemistry::new();
        chem_c.deposit(&[0x00; 64]);
        let h_c = make_hypha(3, 2.0, 1000.0, 0, chem_c);

        let query = QuerySignature::new([0x42; 64]);
        let signal = make_tendril_signal(query);
        let hyphae: Vec<&Hypha> = vec![&h_a, &h_b, &h_c];

        let eval = enzyme.evaluate_with_scores(&signal, &hyphae);

        // Top picks should have at least 2 distinct agents (disagreement).
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

        let eval = enzyme.evaluate_with_scores(&signal, &hyphae);
        assert!(eval.agent_scores.is_empty());
        assert!(eval.top_picks.is_empty());
    }
}
