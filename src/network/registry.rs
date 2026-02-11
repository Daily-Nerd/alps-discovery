// ALPS Discovery — Agent Registry
//
// Manages agent lifecycle: registration, deregistration, feedback recording,
// and temporal decay. The agents BTreeMap lives on LocalNetwork for backward
// compatibility; these are helper functions and types.

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use crate::core::chemistry::Chemistry;
use crate::core::config::LshConfig;
use crate::core::hyphae::Hypha;
use crate::core::lsh::{compute_query_signature, compute_semantic_signature};
use crate::core::pheromone::HyphaState;
use crate::core::types::{HyphaId, PeerAddr};
use crate::scorer::Scorer;

/// Maximum number of feedback records stored per agent.
pub const MAX_FEEDBACK_RECORDS: usize = 100;

/// Minimum floor for tau pheromone. Prevents tau=0 absorbing state
/// which creates infinite first-mover advantage (zero-trap).
pub const TAU_FLOOR: f64 = 0.001;

/// A single recorded outcome for a specific query type.
pub struct FeedbackRecord {
    /// MinHash signature of the query that produced this outcome.
    pub query_minhash: [u8; 64],
    /// +1.0 for success, -1.0 for failure.
    pub outcome: f64,
}

/// Banded LSH index for fast near-neighbor feedback lookup.
///
/// Partitions 64-byte MinHash signatures into `NUM_BANDS` bands of `BAND_WIDTH`
/// bytes each. Each band is hashed into a bucket; candidate feedback records
/// are those sharing at least one band hash with the query.
///
/// This reduces per-query feedback scan from O(n) to O(k) near-neighbors
/// for large feedback histories.
pub struct FeedbackIndex {
    /// Band buckets: band_index → (band_hash → list of feedback indices).
    bands: Vec<HashMap<u64, Vec<usize>>>,
    /// All feedback records (append-only).
    records: VecDeque<FeedbackRecord>,
}

const NUM_BANDS: usize = 16;
const BAND_WIDTH: usize = 4; // 16 bands × 4 bytes = 64 bytes

impl Default for FeedbackIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl FeedbackIndex {
    pub fn new() -> Self {
        Self {
            bands: (0..NUM_BANDS).map(|_| HashMap::new()).collect(),
            records: VecDeque::new(),
        }
    }

    /// Compute the band hash for a given band index.
    fn band_hash(sig: &[u8; 64], band: usize) -> u64 {
        let start = band * BAND_WIDTH;
        let end = start + BAND_WIDTH;
        xxhash_rust::xxh3::xxh3_64_with_seed(&sig[start..end], band as u64)
    }

    /// Insert a feedback record into the index.
    pub fn insert(&mut self, record: FeedbackRecord) {
        let idx = self.records.len();
        for band in 0..NUM_BANDS {
            let hash = Self::band_hash(&record.query_minhash, band);
            self.bands[band].entry(hash).or_default().push(idx);
        }
        self.records.push_back(record);

        // Evict oldest if over capacity.
        if self.records.len() > MAX_FEEDBACK_RECORDS {
            self.records.pop_front();
            // Rebuild index after eviction (infrequent).
            self.rebuild();
        }
    }

    /// Rebuild band index from current records.
    fn rebuild(&mut self) {
        for band_map in &mut self.bands {
            band_map.clear();
        }
        for (idx, record) in self.records.iter().enumerate() {
            for band in 0..NUM_BANDS {
                let hash = Self::band_hash(&record.query_minhash, band);
                self.bands[band].entry(hash).or_default().push(idx);
            }
        }
    }

    /// Find candidate feedback records similar to the query signature.
    ///
    /// Returns an iterator over records sharing at least one band hash
    /// with the query (near-neighbors in LSH space).
    pub fn find_candidates(&self, query_minhash: &[u8; 64]) -> Vec<&FeedbackRecord> {
        let mut seen = std::collections::HashSet::new();
        let mut candidates = Vec::new();

        for band in 0..NUM_BANDS {
            let hash = Self::band_hash(query_minhash, band);
            if let Some(indices) = self.bands[band].get(&hash) {
                for &idx in indices {
                    if idx < self.records.len() && seen.insert(idx) {
                        candidates.push(&self.records[idx]);
                    }
                }
            }
        }

        candidates
    }

    /// Returns the underlying records for persistence/backward compat.
    pub fn records(&self) -> &VecDeque<FeedbackRecord> {
        &self.records
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

/// Per-agent record stored in the network.
pub struct AgentRecord {
    /// Capability descriptions retained for introspection and explain mode.
    pub capabilities: Vec<String>,
    pub endpoint: Option<String>,
    pub metadata: HashMap<String, String>,
    pub hypha: Hypha,
    /// Per-query-type feedback history with banded LSH index for O(k) lookup.
    pub feedback: FeedbackIndex,
}

/// Deterministically convert an agent name to a HyphaId.
pub fn name_to_hypha_id(name: &str) -> HyphaId {
    use xxhash_rust::xxh3::xxh3_64_with_seed;
    let mut id = [0u8; 32];
    for i in 0..4u64 {
        let hash = xxh3_64_with_seed(name.as_bytes(), i);
        let offset = (i as usize) * 8;
        id[offset..offset + 8].copy_from_slice(&hash.to_le_bytes());
    }
    HyphaId(id)
}

/// Register an agent in the agents map and scorer.
pub fn register_agent(
    agents: &mut std::collections::BTreeMap<String, AgentRecord>,
    scorer: &mut dyn Scorer,
    lsh_config: &LshConfig,
    name: &str,
    capabilities: &[&str],
    endpoint: Option<&str>,
    metadata: HashMap<String, String>,
) {
    let hypha_id = name_to_hypha_id(name);

    let mut chemistry = Chemistry::default();
    for cap in capabilities {
        let sig = compute_semantic_signature(cap.as_bytes(), lsh_config);
        chemistry.deposit(&sig);
    }

    scorer.index_capabilities(name, capabilities);

    let hypha = Hypha {
        id: hypha_id,
        peer: PeerAddr(format!("local://{}", name)),
        state: HyphaState {
            diameter: 1.0,
            tau: 0.01,
            sigma: 0.0,
            consecutive_pulse_timeouts: 0,
            forwards_count: crate::core::pheromone::AtomicCounter::new(0),
            conductance: 1.0,
            circuit_state: crate::core::pheromone::CircuitState::Closed,
        },
        chemistry,
        last_activity: Instant::now(),
    };

    agents.insert(
        name.to_string(),
        AgentRecord {
            capabilities: capabilities.iter().map(|s| s.to_string()).collect(),
            endpoint: endpoint.map(|s| s.to_string()),
            metadata,
            hypha,
            feedback: FeedbackIndex::new(),
        },
    );
}

/// Record a successful interaction with an agent.
pub fn record_success(
    agents: &mut std::collections::BTreeMap<String, AgentRecord>,
    lsh_config: &LshConfig,
    agent_name: &str,
    query: Option<&str>,
) {
    if let Some(record) = agents.get_mut(agent_name) {
        // Global boost (always). Floor prevents zero-trap absorbing state.
        record.hypha.state.tau = (record.hypha.state.tau + 0.05).max(TAU_FLOOR);
        record.hypha.state.sigma += 0.01;
        record.hypha.state.diameter = (record.hypha.state.diameter + 0.01).min(1.0);

        // Update circuit breaker state (resets consecutive_pulse_timeouts and closes circuit)
        record.hypha.state.record_circuit_success();
        record.hypha.last_activity = Instant::now();

        // Per-query-type feedback (if query provided).
        if let Some(q) = query {
            let query_sig = compute_query_signature(q.as_bytes(), lsh_config);
            record.feedback.insert(FeedbackRecord {
                query_minhash: query_sig.minhash,
                outcome: 1.0,
            });
        }
    }
}

/// Record a failed interaction with an agent.
pub fn record_failure(
    agents: &mut std::collections::BTreeMap<String, AgentRecord>,
    lsh_config: &LshConfig,
    agent_name: &str,
    query: Option<&str>,
    circuit_config: &crate::core::pheromone::CircuitBreakerConfig,
) {
    if let Some(record) = agents.get_mut(agent_name) {
        // Global penalty (always).
        record.hypha.state.diameter = (record.hypha.state.diameter - 0.05).max(0.1);

        // Update circuit breaker state (increments consecutive_pulse_timeouts and may open circuit)
        record.hypha.state.record_circuit_failure(circuit_config);

        // Per-query-type feedback (if query provided).
        if let Some(q) = query {
            let query_sig = compute_query_signature(q.as_bytes(), lsh_config);
            record.feedback.insert(FeedbackRecord {
                query_minhash: query_sig.minhash,
                outcome: -1.0,
            });
        }
    }
}

/// Apply temporal decay to all agent pheromone state.
pub fn tick(
    agents: &mut std::collections::BTreeMap<String, AgentRecord>,
    circuit_config: &crate::core::pheromone::CircuitBreakerConfig,
) {
    for record in agents.values_mut() {
        let s = &mut record.hypha.state;
        s.tau = (s.tau * 0.995).max(TAU_FLOOR);
        s.sigma *= 0.99;

        // Check if Open circuits should transition to HalfOpen for recovery probe
        s.check_recovery_probe(circuit_config);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scorer::MinHashScorer;
    use std::collections::BTreeMap;

    fn setup() -> (BTreeMap<String, AgentRecord>, MinHashScorer) {
        let mut agents = BTreeMap::new();
        let mut scorer = MinHashScorer::default();
        let config = LshConfig::default();
        register_agent(
            &mut agents,
            &mut scorer,
            &config,
            "test-agent",
            &["legal translation"],
            None,
            HashMap::new(),
        );
        (agents, scorer)
    }

    #[test]
    fn register_creates_agent() {
        let (agents, _) = setup();
        assert_eq!(agents.len(), 1);
        assert!(agents.contains_key("test-agent"));
    }

    #[test]
    fn name_to_hypha_id_deterministic() {
        let a = name_to_hypha_id("test");
        let b = name_to_hypha_id("test");
        assert_eq!(a, b);
    }

    #[test]
    fn tick_decays_tau() {
        let (mut agents, _) = setup();
        let tau_before = agents.get("test-agent").unwrap().hypha.state.tau;
        tick(
            &mut agents,
            &crate::core::pheromone::CircuitBreakerConfig::new(),
        );
        let tau_after = agents.get("test-agent").unwrap().hypha.state.tau;
        assert!(tau_after < tau_before);
        assert!(tau_after >= TAU_FLOOR);
    }

    #[test]
    fn record_success_increases_diameter() {
        let (mut agents, _) = setup();
        let config = LshConfig::default();
        // Reduce diameter first so there's room to increase (initial=1.0, cap=1.0).
        record_failure(
            &mut agents,
            &config,
            "test-agent",
            None,
            &crate::core::pheromone::CircuitBreakerConfig::new(),
        );
        let d_before = agents.get("test-agent").unwrap().hypha.state.diameter;
        record_success(&mut agents, &config, "test-agent", None);
        let d_after = agents.get("test-agent").unwrap().hypha.state.diameter;
        assert!(d_after > d_before);
    }

    #[test]
    fn record_failure_decreases_diameter() {
        let (mut agents, _) = setup();
        let config = LshConfig::default();
        let d_before = agents.get("test-agent").unwrap().hypha.state.diameter;
        record_failure(
            &mut agents,
            &config,
            "test-agent",
            None,
            &crate::core::pheromone::CircuitBreakerConfig::new(),
        );
        let d_after = agents.get("test-agent").unwrap().hypha.state.diameter;
        assert!(d_after < d_before);
    }
}
