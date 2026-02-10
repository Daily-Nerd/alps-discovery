// ALPS Discovery SDK — Pheromone State
//
// Minimal hypha state type for agent scoring.

use serde::{Deserialize, Serialize};

/// Pheromone-relevant state of an agent slot.
///
/// Fields and their roles:
/// - `diameter`: Structural capacity / quality weight. Adjusted by feedback
///   (success → +0.01 capped at 1.0, failure → -0.05 floored at 0.1).
///   Influences CapabilityKernel and LoadBalancingKernel scoring.
/// - `tau`: Reliability signal. Incremented on success (+0.05), decayed per
///   tick (×0.995, floor TAU_FLOOR). Used as a multiplier in CapabilityKernel
///   scoring: higher tau → higher capability score for reliable agents.
/// - `sigma`: Exploration count / popularity. Incremented on success (+0.01),
///   decayed per tick (×0.99). Inversely used by NoveltyKernel: 1/(1+sigma)
///   favors less-explored agents.
/// - `consecutive_pulse_timeouts`: Failure streak counter. Incremented on
///   failure, reset on success. Available for circuit-breaker patterns.
/// - `forwards_count`: Total signals routed through this slot. Used by
///   LoadBalancingKernel: diameter/(1+forwards_count) favors less-used agents.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HyphaState {
    /// Connection diameter (strength/capacity). Adjusted by feedback.
    pub diameter: f64,
    /// Tau pheromone — tracks reliability. Influences CapabilityKernel scoring.
    pub tau: f64,
    /// Sigma pheromone — tracks exploration count / popularity.
    pub sigma: f64,
    /// Consecutive timeouts without a response.
    pub consecutive_pulse_timeouts: u8,
    /// Number of signals forwarded through this slot (for load balancing).
    pub forwards_count: u64,
}

impl HyphaState {
    /// Minimum floor for tau pheromone (prevents zero-trap absorbing state).
    pub const TAU_FLOOR: f64 = 0.001;

    /// Creates a new HyphaState with the given initial diameter.
    pub fn new(initial_diameter: f64) -> Self {
        Self {
            diameter: initial_diameter,
            tau: Self::TAU_FLOOR,
            sigma: 0.0,
            consecutive_pulse_timeouts: 0,
            forwards_count: 0,
        }
    }
}
