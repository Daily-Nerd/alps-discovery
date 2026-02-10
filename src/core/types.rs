// ALPS Discovery SDK — Shared Identifier Types
//
// Minimal identifier types needed for local agent discovery.

use serde::{Deserialize, Serialize};

/// Unique identifier for a hypha (connection / agent slot).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default)]
pub struct HyphaId(pub [u8; 32]);

/// Unique identifier for a query trail.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default)]
pub struct TrailId(pub [u8; 32]);

/// Network address for a peer.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PeerAddr(pub String);

/// Decision identifier for enzyme correlation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct DecisionId(pub u64);

/// Kernel type discriminant for enzyme classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum KernelType {
    /// Capability matching: scores by Chemistry similarity to query.
    CapabilityMatching,
    /// Seeks novel or rare signal patterns.
    NoveltySeeking,
    /// Balances load across available hyphae.
    LoadBalancing,
}

impl std::fmt::Display for KernelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KernelType::CapabilityMatching => write!(f, "capability_matching"),
            KernelType::NoveltySeeking => write!(f, "novelty_seeking"),
            KernelType::LoadBalancing => write!(f, "load_balancing"),
        }
    }
}

// QueryConfig is defined in config.rs and re-exported here for convenience.
pub use crate::core::config::QueryConfig;
