// ALPS Discovery SDK — Action Types
//
// Minimal routing action types for the discovery enzyme.

use crate::core::types::{DecisionId, HyphaId};

/// Routing decision made by the discovery enzyme.
#[derive(Debug, Clone, PartialEq)]
pub enum EnzymeAction {
    /// Absorb the signal locally.
    Absorb,
    /// Forward the signal to a single target.
    Forward { target: HyphaId },
    /// Split the signal across multiple targets.
    Split { targets: Vec<HyphaId> },
    /// Dissolve the signal (drop it).
    Dissolve,
}

/// An enzyme decision pairing an action with a correlation identifier.
#[derive(Debug, Clone, PartialEq)]
pub struct EnzymeDecision {
    /// The routing action to take.
    pub action: EnzymeAction,
    /// Decision identifier.
    pub decision_id: DecisionId,
}
