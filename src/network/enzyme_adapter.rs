// ALPS Discovery — Enzyme Adapter
//
// Wraps the multi-kernel reasoning enzyme. OSS-simplified: no Spore or
// MembraneState (vestigial protocol constructs removed in P3.7).

use crate::core::action::EnzymeAction;
use crate::core::enzyme::{Enzyme, KernelEvaluation, SLNEnzyme, SLNEnzymeConfig, ScorerContext};
use crate::core::hyphae::Hypha;
use crate::core::signal::Signal;

/// Enzyme decision output.
/// Public when module is public (bench feature enabled).
pub struct EnzymeDecision {
    pub action: EnzymeAction,
}

/// Wraps the SLNEnzyme for use in the discovery pipeline.
pub struct EnzymeAdapter {
    enzyme: SLNEnzyme,
}

impl EnzymeAdapter {
    /// Create a new EnzymeAdapter with the given enzyme configuration.
    pub fn new(config: SLNEnzymeConfig) -> Self {
        Self {
            enzyme: SLNEnzyme::with_discovery_kernels(config),
        }
    }

    /// Run kernel evaluation with pre-computed scorer context.
    pub fn evaluate_with_scores(
        &self,
        signal: &Signal,
        hyphae: &[&Hypha],
        scorer_context: &ScorerContext,
    ) -> KernelEvaluation {
        self.enzyme
            .evaluate_with_scores(signal, hyphae, scorer_context)
    }

    /// Run enzyme process to update internal state.
    pub fn process(
        &mut self,
        signal: &Signal,
        hyphae: &[&Hypha],
        scorer_context: &ScorerContext,
    ) -> EnzymeDecision {
        let decision = self.enzyme.process(signal, hyphae, scorer_context);
        EnzymeDecision {
            action: decision.action,
        }
    }

    /// Access the enzyme configuration.
    pub fn config(&self) -> &SLNEnzymeConfig {
        self.enzyme.config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_construction() {
        let adapter = EnzymeAdapter::new(SLNEnzymeConfig::default());
        let config = adapter.config();
        assert_eq!(config.max_disagreement_split, 3);
    }
}
