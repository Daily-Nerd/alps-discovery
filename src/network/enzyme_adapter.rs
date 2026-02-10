// ALPS Discovery — Enzyme Adapter
//
// Wraps the multi-kernel reasoning enzyme with the vestigial Spore and
// MembraneState stubs required by the Enzyme trait.

use std::time::Duration;

use crate::core::action::EnzymeAction;
use crate::core::config::SporeConfig;
use crate::core::enzyme::{Enzyme, KernelEvaluation, SLNEnzyme, SLNEnzymeConfig, ScorerContext};
use crate::core::hyphae::Hypha;
use crate::core::membrane::MembraneState;
use crate::core::signal::Signal;
use crate::core::spore::tree::Spore;

/// Enzyme decision output.
pub(crate) struct EnzymeDecision {
    pub action: EnzymeAction,
}

/// Wraps the SLNEnzyme with vestigial Spore and MembraneState.
pub struct EnzymeAdapter {
    enzyme: SLNEnzyme,
    spore: Spore,
    membrane_state: MembraneState,
}

impl EnzymeAdapter {
    /// Create a new EnzymeAdapter with the given enzyme configuration.
    pub fn new(config: SLNEnzymeConfig) -> Self {
        Self {
            enzyme: SLNEnzyme::with_discovery_kernels(config),
            spore: Spore::new(SporeConfig::default()),
            membrane_state: MembraneState {
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
            },
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
        let decision = self.enzyme.process(
            signal,
            &self.spore,
            hyphae,
            &self.membrane_state,
            scorer_context,
        );
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
