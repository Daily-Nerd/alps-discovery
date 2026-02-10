// ALPS Discovery SDK — Signal Types
//
// Minimal signal types for the discovery routing engine.

use crate::core::chemistry::QuerySignature;
use crate::core::config::QueryConfig;
use crate::core::types::TrailId;

/// Signal enum for the discovery routing engine.
///
/// Only Tendril carries data; other variants exist so the enzyme
/// can handle them (absorb/dissolve) if encountered.
#[derive(Debug, Clone, PartialEq)]
pub enum Signal {
    /// Query signal for capability-based discovery.
    Tendril(Tendril),
    /// Data-carrying signal (not used in local discovery).
    Nutrient,
    /// Heartbeat signal (not used in local discovery).
    Pulse,
    /// State snapshot signal (not used in local discovery).
    Spore,
    /// Abstraction signal (not used in local discovery).
    Insight,
}

/// Query signal for capability-based discovery.
///
/// The `query_signature` is the MinHash signature used for
/// Chemistry similarity matching against registered agents.
#[derive(Debug, Clone, PartialEq)]
pub struct Tendril {
    /// Unique identifier for the query trail.
    pub trail_id: TrailId,
    /// MinHash query signature for chemical matching.
    pub query_signature: QuerySignature,
    /// Configuration for the query behavior.
    pub query_config: QueryConfig,
}
