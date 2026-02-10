// ALPS Discovery SDK — Enzyme (Multi-Kernel Routing)
//
// Implements the Enzyme trait with multi-kernel voting for discovery.
// Each kernel evaluates independently; the enzyme picks the hypha
// with the most votes (simple majority).

use std::collections::BTreeMap;

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
                (h.id.clone(), novelty * h.state.diameter)
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
    fn evaluate_kernels(
        &self,
        signal: &Signal,
        hyphae: &[&Hypha],
    ) -> EnzymeAction {
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
        let (best_hypha, best_count) = votes
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(h, c)| (h.clone(), *c))
            .unwrap();

        // Majority or unanimity → forward to the winner.
        if best_count * 2 >= self.kernels.len() {
            EnzymeAction::Forward {
                target: best_hypha,
            }
        } else {
            // No majority → split across distinct top picks.
            let mut targets: Vec<HyphaId> = votes.keys().cloned().collect();
            targets.truncate(self.config.max_disagreement_split);
            if targets.len() == 1 {
                EnzymeAction::Forward {
                    target: targets.into_iter().next().unwrap(),
                }
            } else {
                EnzymeAction::Split { targets }
            }
        }
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
