// ALPS Discovery SDK — Pheromone State
//
// Minimal hypha state type for agent scoring.

use serde::{Deserialize, Serialize};

/// Pheromone-relevant state of an agent slot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HyphaState {
    /// Connection diameter (strength/capacity).
    pub diameter: f64,
    /// Tau pheromone — tracks reliability.
    pub tau: f64,
    /// Sigma pheromone — tracks popularity.
    pub sigma: f64,
    /// Omega pheromone — tracks verification quality.
    pub omega: f64,
    /// Consecutive timeouts without a response.
    pub consecutive_pulse_timeouts: u8,
    /// Number of signals forwarded through this slot (for load balancing).
    pub forwards_count: u64,
}

impl HyphaState {
    /// Creates a new HyphaState with the given initial diameter.
    pub fn new(initial_diameter: f64) -> Self {
        Self {
            diameter: initial_diameter,
            tau: 0.0,
            sigma: 0.0,
            omega: 0.0,
            consecutive_pulse_timeouts: 0,
            forwards_count: 0,
        }
    }
}
