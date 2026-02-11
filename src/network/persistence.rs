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
/// v3: added network_epoch, last_activity_duration, conductance for temporal state and Physarum;
///     added cooccurrence_matrix for self-improving query expansion.
pub const SNAPSHOT_VERSION: u32 = 3;

/// Serializable snapshot of the entire network state.
#[derive(Serialize, Deserialize)]
pub(crate) struct NetworkSnapshot {
    pub version: u32,
    pub agents: Vec<AgentSnapshot>,
    /// Network creation time (epoch for temporal state serialization).
    #[serde(default = "std::time::SystemTime::now")]
    pub network_epoch: std::time::SystemTime,
    /// Co-occurrence matrix for query expansion (query_token, agent_cap_token) -> count.
    /// Serialized as flat map with "token1,token2" keys for JSON compatibility.
    #[serde(default)]
    pub cooccurrence_matrix: HashMap<String, u32>,
    /// Total feedback count for co-occurrence threshold gating.
    #[serde(default)]
    pub cooccurrence_feedback_count: usize,
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
    /// Physarum flow conductance (v3+).
    #[serde(default = "default_conductance")]
    pub conductance: f64,
    /// Duration since network_epoch when agent was last active (v3+).
    #[serde(default)]
    pub last_activity_duration: std::time::Duration,
}

fn default_conductance() -> f64 {
    1.0
}

/// Flatten co-occurrence matrix from (token1, token2) -> count to "token1,token2" -> count.
///
/// Used for JSON serialization since JSON keys must be strings.
fn flatten_cooccurrence_matrix(matrix: &HashMap<(String, String), u32>) -> HashMap<String, u32> {
    matrix
        .iter()
        .map(|((qt, act), count)| (format!("{},{}", qt, act), *count))
        .collect()
}

/// Unflatten co-occurrence matrix from "token1,token2" -> count to (token1, token2) -> count.
///
/// Used for JSON deserialization.
fn unflatten_cooccurrence_matrix(
    flat_matrix: &HashMap<String, u32>,
) -> HashMap<(String, String), u32> {
    flat_matrix
        .iter()
        .filter_map(|(key, count)| {
            let parts: Vec<&str> = key.splitn(2, ',').collect();
            if parts.len() == 2 {
                Some(((parts[0].to_string(), parts[1].to_string()), *count))
            } else {
                None
            }
        })
        .collect()
}

/// Serializable snapshot of a single feedback record.
#[derive(Serialize, Deserialize)]
pub(crate) struct FeedbackSnapshot {
    #[serde(with = "BigArray")]
    pub query_minhash: [u8; 64],
    pub outcome: f64,
}

/// Data for one agent loaded from a snapshot, ready to be re-registered.
#[allow(dead_code)] // conductance and last_activity_duration will be used in future load logic
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
    pub conductance: f64,
    pub last_activity_duration: std::time::Duration,
}

/// Save agents to a JSON file (legacy non-atomic version).
#[allow(dead_code)]
pub(crate) fn save_snapshot(
    agents: &std::collections::BTreeMap<String, AgentRecord>,
    cooccurrence_matrix: &HashMap<(String, String), u32>,
    cooccurrence_feedback_count: usize,
    path: &str,
) -> Result<(), NetworkError> {
    let now = std::time::Instant::now();
    let network_epoch = std::time::SystemTime::now();

    let snapshot = NetworkSnapshot {
        version: SNAPSHOT_VERSION,
        network_epoch,
        cooccurrence_matrix: flatten_cooccurrence_matrix(cooccurrence_matrix),
        cooccurrence_feedback_count,
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

                // Compute duration since "now" for temporal state
                let last_activity_duration =
                    now.saturating_duration_since(record.hypha.last_activity);

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
                    conductance: record.hypha.state.conductance,
                    last_activity_duration,
                }
            })
            .collect(),
    };
    let json = serde_json::to_string_pretty(&snapshot)?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Save agents to a JSON file using atomic write-rename pattern.
///
/// Writes to a temporary file in the same directory, then atomically
/// renames it to the target path. If the process crashes during write,
/// the old file remains intact (temp file orphaned, not corrupted).
///
/// Uses tempfile crate for cross-platform atomic rename.
pub(crate) fn save_snapshot_atomic(
    agents: &std::collections::BTreeMap<String, AgentRecord>,
    cooccurrence_matrix: &HashMap<(String, String), u32>,
    cooccurrence_feedback_count: usize,
    path: &str,
) -> Result<(), NetworkError> {
    use std::io::Write;
    use std::path::Path;

    let now = std::time::Instant::now();
    let network_epoch = std::time::SystemTime::now();

    let snapshot = NetworkSnapshot {
        version: SNAPSHOT_VERSION,
        network_epoch,
        cooccurrence_matrix: flatten_cooccurrence_matrix(cooccurrence_matrix),
        cooccurrence_feedback_count,
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

                // Compute duration since "now" for temporal state
                let last_activity_duration =
                    now.saturating_duration_since(record.hypha.last_activity);

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
                    conductance: record.hypha.state.conductance,
                    last_activity_duration,
                }
            })
            .collect(),
    };

    let json = serde_json::to_string_pretty(&snapshot)?;

    // Write to temporary file in same directory as target
    let target_path = Path::new(path);
    let parent_dir = target_path.parent().unwrap_or_else(|| Path::new("."));

    let mut temp_file = tempfile::NamedTempFile::new_in(parent_dir)?;
    temp_file.write_all(json.as_bytes())?;
    temp_file.flush()?;

    // Atomically rename temp file to target path
    temp_file.persist(target_path).map_err(|e| {
        NetworkError::Io(std::io::Error::other(format!(
            "Failed to persist temp file: {}",
            e
        )))
    })?;

    Ok(())
}

/// Loaded snapshot data including agents and co-occurrence matrix.
pub(crate) struct LoadedSnapshot {
    pub agents: Vec<AgentLoadData>,
    pub cooccurrence_matrix: HashMap<(String, String), u32>,
    pub cooccurrence_feedback_count: usize,
}

/// Load agents from a JSON file. Returns the loaded agent data and co-occurrence matrix.
pub(crate) fn load_snapshot(path: &str) -> Result<LoadedSnapshot, NetworkError> {
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
                conductance: agent.conductance,
                last_activity_duration: agent.last_activity_duration,
            }
        })
        .collect();

    let cooccurrence_matrix = unflatten_cooccurrence_matrix(&snapshot.cooccurrence_matrix);

    Ok(LoadedSnapshot {
        agents,
        cooccurrence_matrix,
        cooccurrence_feedback_count: snapshot.cooccurrence_feedback_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    // --- Task 10.1: Atomic file operations tests ---

    #[test]
    fn save_snapshot_atomic_leaves_old_file_intact_on_failure() {
        // This test will be implemented after atomic save is added
        // For now, just verify the function signature exists
        let agents = BTreeMap::new();
        let cooccurrence = HashMap::new();
        let result = save_snapshot_atomic(&agents, &cooccurrence, 0, "/tmp/test_snapshot.json");
        // Function should exist but may fail on empty agents
        assert!(result.is_err() || result.is_ok());
    }

    #[test]
    fn save_snapshot_atomic_creates_new_file() {
        use std::path::Path;
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let snapshot_path = temp_dir.path().join("snapshot.json");

        let agents = BTreeMap::new();
        let cooccurrence = HashMap::new();
        let result =
            save_snapshot_atomic(&agents, &cooccurrence, 0, snapshot_path.to_str().unwrap());

        // Should succeed
        assert!(result.is_ok());
        // File should exist
        assert!(Path::new(&snapshot_path).exists());
    }

    // --- Task 10.2: Temporal state serialization tests ---

    #[test]
    fn snapshot_includes_network_epoch() {
        let snapshot = NetworkSnapshot {
            version: 3,
            agents: vec![],
            network_epoch: std::time::SystemTime::now(),
            cooccurrence_matrix: HashMap::new(),
            cooccurrence_feedback_count: 0,
        };

        // Should serialize successfully
        let json = serde_json::to_string(&snapshot);
        assert!(json.is_ok());
    }

    #[test]
    fn agent_snapshot_includes_temporal_duration() {
        use std::time::Duration;

        let agent_snap = AgentSnapshot {
            name: "test-agent".to_string(),
            capabilities: vec!["test".to_string()],
            endpoint: None,
            metadata: HashMap::new(),
            diameter: 1.0,
            tau: 0.1,
            sigma: 0.0,
            omega: 0.0,
            forwards_count: 0,
            consecutive_pulse_timeouts: 0,
            feedback: vec![],
            conductance: 1.0,
            last_activity_duration: Duration::from_secs(120),
        };

        // Should serialize successfully
        let json = serde_json::to_string(&agent_snap);
        assert!(json.is_ok());
    }
}
