// ALPS Discovery — Network Persistence
//
// Save/load network state as JSON snapshots with versioned schema.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;
use thiserror::Error;

use super::registry::{AgentRecord, FeedbackIndex, FeedbackRecord, TAU_FLOOR};

/// Structured error type for network persistence operations.
#[derive(Debug, Error)]
pub enum NetworkError {
    /// JSON serialization/deserialization failure.
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    /// File system I/O failure.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// Snapshot version is newer than what this library supports.
    #[error("unsupported snapshot version {found} (supported up to {supported})")]
    UnsupportedVersion { found: u32, supported: u32 },
}

/// Current snapshot schema version.
/// v1: original format (includes omega field).
/// v2: omega removed from HyphaState, tau wired into CapabilityKernel.
pub const SNAPSHOT_VERSION: u32 = 2;

/// Serializable snapshot of the entire network state.
#[derive(Serialize, Deserialize)]
pub(crate) struct NetworkSnapshot {
    pub version: u32,
    pub agents: Vec<AgentSnapshot>,
}

/// Serializable snapshot of a single agent.
#[derive(Serialize, Deserialize)]
pub(crate) struct AgentSnapshot {
    pub name: String,
    pub capabilities: Vec<String>,
    pub endpoint: Option<String>,
    pub metadata: HashMap<String, String>,
    pub diameter: f64,
    pub tau: f64,
    pub sigma: f64,
    /// Legacy field, always 0.0. Retained for backward-compatible deserialization
    /// of v1 snapshots. Ignored on load; omitted in v2+ snapshots.
    #[serde(default)]
    pub omega: f64,
    pub forwards_count: u64,
    pub consecutive_pulse_timeouts: u8,
    pub feedback: Vec<FeedbackSnapshot>,
}

/// Serializable snapshot of a single feedback record.
#[derive(Serialize, Deserialize)]
pub(crate) struct FeedbackSnapshot {
    #[serde(with = "BigArray")]
    pub query_minhash: [u8; 64],
    pub outcome: f64,
}

/// Data for one agent loaded from a snapshot, ready to be re-registered.
pub(crate) struct AgentLoadData {
    pub name: String,
    pub capabilities: Vec<String>,
    pub endpoint: Option<String>,
    pub metadata: HashMap<String, String>,
    pub diameter: f64,
    pub tau: f64,
    pub sigma: f64,
    pub forwards_count: u64,
    pub consecutive_pulse_timeouts: u8,
    pub feedback: FeedbackIndex,
}

/// Save agents to a JSON file.
pub(crate) fn save_snapshot(
    agents: &std::collections::BTreeMap<String, AgentRecord>,
    path: &str,
) -> Result<(), NetworkError> {
    let snapshot = NetworkSnapshot {
        version: SNAPSHOT_VERSION,
        agents: agents
            .iter()
            .map(|(name, record)| {
                let feedback: Vec<FeedbackSnapshot> = record
                    .feedback
                    .records()
                    .iter()
                    .map(|fb| FeedbackSnapshot {
                        query_minhash: fb.query_minhash,
                        outcome: fb.outcome,
                    })
                    .collect();
                AgentSnapshot {
                    name: name.clone(),
                    capabilities: record.capabilities.clone(),
                    endpoint: record.endpoint.clone(),
                    metadata: record.metadata.clone(),
                    diameter: record.hypha.state.diameter,
                    tau: record.hypha.state.tau,
                    sigma: record.hypha.state.sigma,
                    omega: 0.0,
                    forwards_count: record.hypha.state.forwards_count.get(),
                    consecutive_pulse_timeouts: record.hypha.state.consecutive_pulse_timeouts,
                    feedback,
                }
            })
            .collect(),
    };
    let json = serde_json::to_string_pretty(&snapshot)?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Load agents from a JSON file. Returns the loaded agent data.
pub(crate) fn load_snapshot(path: &str) -> Result<Vec<AgentLoadData>, NetworkError> {
    let json = std::fs::read_to_string(path)?;
    let snapshot: NetworkSnapshot = serde_json::from_str(&json)?;

    if snapshot.version > SNAPSHOT_VERSION {
        return Err(NetworkError::UnsupportedVersion {
            found: snapshot.version,
            supported: SNAPSHOT_VERSION,
        });
    }

    let agents = snapshot
        .agents
        .into_iter()
        .map(|agent| {
            let mut feedback = FeedbackIndex::new();
            for fb in agent.feedback {
                feedback.insert(FeedbackRecord {
                    query_minhash: fb.query_minhash,
                    outcome: fb.outcome,
                });
            }
            AgentLoadData {
                name: agent.name,
                capabilities: agent.capabilities,
                endpoint: agent.endpoint,
                metadata: agent.metadata,
                diameter: agent.diameter,
                tau: agent.tau.max(TAU_FLOOR),
                sigma: agent.sigma,
                forwards_count: agent.forwards_count,
                consecutive_pulse_timeouts: agent.consecutive_pulse_timeouts,
                feedback,
            }
        })
        .collect();

    Ok(agents)
}
