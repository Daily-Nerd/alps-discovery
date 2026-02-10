// ALPS Discovery SDK — Hypha Record
//
// Minimal hypha struct for agent representation in the routing engine.

use std::time::Instant;

use crate::core::chemistry::Chemistry;
use crate::core::pheromone::HyphaState;
use crate::core::types::{HyphaId, PeerAddr};

/// State of a single agent slot in the discovery network.
#[derive(Debug, Clone)]
pub struct Hypha {
    /// Unique identifier for this agent slot.
    pub id: HyphaId,
    /// Address of the agent.
    pub peer: PeerAddr,
    /// Scoring state (diameter, tau, sigma, etc.).
    pub state: HyphaState,
    /// Accumulated chemistry from capability descriptions.
    pub chemistry: Chemistry,
    /// Timestamp of last activity.
    pub last_activity: Instant,
}
