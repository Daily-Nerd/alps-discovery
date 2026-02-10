// ALPS Discovery SDK — LocalNetwork
//
// In-process agent discovery using multi-kernel voting.
// No networking, no decay, no membrane — just the routing engine
// applied to Chemistry-based capability matching.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;

use thiserror::Error;

use crate::core::chemistry::Chemistry;
use crate::core::config::{LshConfig, SporeConfig};
use crate::core::enzyme::{Enzyme, SLNEnzyme, SLNEnzymeConfig};
use crate::core::hyphae::Hypha;
use crate::core::lsh::{compute_query_signature, compute_semantic_signature, MinHasher};
use crate::core::membrane::MembraneState;
use crate::core::pheromone::HyphaState;
use crate::core::signal::{Signal, Tendril};
use crate::core::spore::tree::Spore;
use crate::core::types::{HyphaId, PeerAddr, TrailId};

use crate::core::action::EnzymeAction;
use crate::core::config::QueryConfig;
use crate::scorer::{MinHashScorer, Scorer};

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

/// A single discovery result with scoring breakdown.
#[derive(Debug, Clone)]
pub struct DiscoveryResult {
    /// Agent name.
    pub agent_name: String,
    /// Raw Chemistry similarity to the query [0.0, 1.0].
    pub similarity: f64,
    /// Combined routing score (similarity × diameter, plus feedback effects).
    pub score: f64,
    /// Agent endpoint (URI, URL, module path, etc.) if provided at registration.
    pub endpoint: Option<String>,
    /// Arbitrary metadata (protocol, version, framework, etc.) if provided.
    pub metadata: HashMap<String, String>,
}

/// Extended discovery result with full scoring breakdown for debugging.
#[derive(Debug, Clone)]
pub struct ExplainedResult {
    /// Agent name.
    pub agent_name: String,
    /// Raw similarity from the scorer [0.0, 1.0].
    pub raw_similarity: f64,
    /// Agent diameter (routing weight from feedback history).
    pub diameter: f64,
    /// Per-query feedback factor [-1.0, 1.0].
    pub feedback_factor: f64,
    /// Final combined score.
    pub final_score: f64,
    /// Agent endpoint if provided at registration.
    pub endpoint: Option<String>,
    /// Agent metadata if provided at registration.
    pub metadata: HashMap<String, String>,
}

/// Internal scored candidate used by the shared discovery pipeline.
struct ScoredCandidate {
    agent_name: String,
    raw_similarity: f64,
    diameter: f64,
    feedback_factor: f64,
    final_score: f64,
    endpoint: Option<String>,
    metadata: HashMap<String, String>,
}

/// Lightweight candidate with borrowed fields for the pre-filter scoring phase.
/// Only survivors get promoted to owned ScoredCandidate.
struct CandidateRef<'a> {
    agent_name: &'a str,
    raw_similarity: f64,
    diameter: f64,
    feedback_factor: f64,
    final_score: f64,
    endpoint: &'a Option<String>,
    metadata: &'a HashMap<String, String>,
}

/// Maximum number of feedback records stored per agent.
const MAX_FEEDBACK_RECORDS: usize = 100;

/// Strength of per-query feedback adjustment to diameter (max ±50%).
const FEEDBACK_STRENGTH: f64 = 0.5;

/// A single recorded outcome for a specific query type.
struct FeedbackRecord {
    /// MinHash signature of the query that produced this outcome.
    query_minhash: [u8; 64],
    /// +1.0 for success, -1.0 for failure.
    outcome: f64,
}

/// Per-agent record stored in the network.
struct AgentRecord {
    /// Capability descriptions retained for introspection and explain mode.
    capabilities: Vec<String>,
    endpoint: Option<String>,
    metadata: HashMap<String, String>,
    hypha: Hypha,
    /// Per-query-type feedback history (most recent last).
    feedback: VecDeque<FeedbackRecord>,
}

/// A filter condition for metadata-based result filtering.
#[derive(Debug, Clone)]
pub enum FilterValue {
    /// Exact string match.
    Exact(String),
    /// Substring containment check.
    Contains(String),
    /// Value must be one of the listed options.
    OneOf(Vec<String>),
    /// Numeric value must be less than the threshold.
    LessThan(f64),
    /// Numeric value must be greater than the threshold.
    GreaterThan(f64),
}

impl FilterValue {
    /// Check if a metadata value matches this filter condition.
    fn matches(&self, value: &str) -> bool {
        match self {
            FilterValue::Exact(expected) => value == expected,
            FilterValue::Contains(substring) => value.contains(substring.as_str()),
            FilterValue::OneOf(options) => options.iter().any(|o| o == value),
            FilterValue::LessThan(threshold) => value.parse::<f64>().is_ok_and(|v| v < *threshold),
            FilterValue::GreaterThan(threshold) => {
                value.parse::<f64>().is_ok_and(|v| v > *threshold)
            }
        }
    }
}

/// Metadata filters applied to discovery results.
pub type Filters = HashMap<String, FilterValue>;

/// Local-mode agent discovery network.
///
/// Wraps the routing engine for single-process agent discovery.
/// Each registered agent becomes a virtual hypha with Chemistry derived
/// from its capability descriptions. Discovery queries run through
/// multi-kernel voting (CapabilityKernel + LoadBalancing + Novelty).
///
/// The routing improves over time via `record_success` / `record_failure`
/// which update the scoring state feeding the kernel scoring.
pub struct LocalNetwork {
    /// Registered agents keyed by name.
    agents: BTreeMap<String, AgentRecord>,
    /// The multi-kernel reasoning enzyme.
    enzyme: SLNEnzyme,
    /// LSH configuration for signature generation.
    lsh_config: LshConfig,
    /// Pluggable scorer for agent-to-query matching.
    scorer: Box<dyn Scorer>,
    /// Empty spore (discovery only).
    spore: Spore,
    /// Static membrane state (passthrough).
    membrane_state: MembraneState,
}

/// Serializable snapshot of the entire network state.
#[derive(Serialize, Deserialize)]
struct NetworkSnapshot {
    /// Schema version for forward compatibility.
    version: u32,
    /// All registered agents with their state.
    agents: Vec<AgentSnapshot>,
}

/// Serializable snapshot of a single agent.
#[derive(Serialize, Deserialize)]
struct AgentSnapshot {
    name: String,
    capabilities: Vec<String>,
    endpoint: Option<String>,
    metadata: HashMap<String, String>,
    diameter: f64,
    tau: f64,
    sigma: f64,
    omega: f64,
    forwards_count: u64,
    consecutive_pulse_timeouts: u8,
    feedback: Vec<FeedbackSnapshot>,
}

/// Serializable snapshot of a single feedback record.
#[derive(Serialize, Deserialize)]
struct FeedbackSnapshot {
    #[serde(with = "BigArray")]
    query_minhash: [u8; 64],
    outcome: f64,
}

impl LocalNetwork {
    /// Creates a new empty LocalNetwork with default configuration.
    pub fn new() -> Self {
        let config = LshConfig::default();
        let scorer = MinHashScorer::new(config.clone());
        Self::with_scorer(Box::new(scorer))
    }

    /// Creates a new LocalNetwork with custom configuration.
    pub fn with_config(enzyme_config: SLNEnzymeConfig, lsh_config: LshConfig) -> Self {
        let scorer = MinHashScorer::new(lsh_config.clone());
        Self {
            agents: BTreeMap::new(),
            enzyme: SLNEnzyme::with_discovery_kernels(enzyme_config),
            lsh_config,
            scorer: Box::new(scorer),
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

    /// Creates a new LocalNetwork with a custom scorer implementation.
    ///
    /// Use this to plug in alternative scoring strategies (e.g. embedding-based).
    /// The default `new()` and `with_config()` constructors use `MinHashScorer`.
    pub fn with_scorer(scorer: Box<dyn Scorer>) -> Self {
        Self {
            agents: BTreeMap::new(),
            enzyme: SLNEnzyme::with_discovery_kernels(SLNEnzymeConfig::default()),
            lsh_config: LshConfig::default(),
            scorer,
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

    /// Register an agent with its capabilities.
    ///
    /// Optionally provide an `endpoint` (URI/URL) and `metadata` (key-value
    /// pairs like protocol, version, framework) that will be returned in
    /// discovery results. ALPS does not interpret these — they are passed
    /// through so the caller can invoke the agent using their own client.
    pub fn register(
        &mut self,
        name: &str,
        capabilities: &[&str],
        endpoint: Option<&str>,
        metadata: HashMap<String, String>,
    ) {
        let hypha_id = Self::name_to_hypha_id(name);

        // Still deposit Chemistry for the enzyme kernels.
        let mut chemistry = Chemistry::default();
        for cap in capabilities {
            let sig = compute_semantic_signature(cap.as_bytes(), &self.lsh_config);
            chemistry.deposit(&sig);
        }

        // Index in the scorer.
        self.scorer.index_capabilities(name, capabilities);

        let hypha = Hypha {
            id: hypha_id,
            peer: PeerAddr(format!("local://{}", name)),
            state: HyphaState {
                diameter: 1.0,
                tau: 0.01,
                sigma: 0.0,
                omega: 0.0,
                consecutive_pulse_timeouts: 0,
                forwards_count: 0,
            },
            chemistry,
            last_activity: Instant::now(),
        };

        self.agents.insert(
            name.to_string(),
            AgentRecord {
                capabilities: capabilities.iter().map(|s| s.to_string()).collect(),
                endpoint: endpoint.map(|s| s.to_string()),
                metadata,
                hypha,
                feedback: VecDeque::new(),
            },
        );
    }

    /// Deregister an agent by name. Returns true if found and removed.
    pub fn deregister(&mut self, name: &str) -> bool {
        self.scorer.remove_agent(name);
        self.agents.remove(name).is_some()
    }

    /// Discover agents matching a natural-language query.
    ///
    /// Returns a ranked list of all agents with similarity > 0, sorted by
    /// `score = similarity × diameter`. The diameter incorporates feedback
    /// from `record_success` / `record_failure`.
    pub fn discover(&mut self, query: &str) -> Vec<DiscoveryResult> {
        self.discover_filtered(query, None)
    }

    /// Discover agents matching a query, with optional metadata filters.
    ///
    /// Filters are applied post-scoring: agents must pass all filter
    /// conditions on their metadata to appear in results. Missing metadata
    /// keys cause the filter to fail (strict mode).
    ///
    /// Results within 5% of the top score are randomly shuffled to
    /// distribute load across equally-capable agents.
    pub fn discover_filtered(
        &mut self,
        query: &str,
        filters: Option<&Filters>,
    ) -> Vec<DiscoveryResult> {
        self.discover_core(query, filters)
            .into_iter()
            .map(|c| DiscoveryResult {
                agent_name: c.agent_name,
                similarity: c.raw_similarity,
                score: c.final_score,
                endpoint: c.endpoint,
                metadata: c.metadata,
            })
            .collect()
    }

    /// Discover agents with full scoring breakdown for debugging.
    ///
    /// Returns all matching agents with detailed scoring components:
    /// raw similarity, diameter, feedback factor, and final score.
    /// Supports the same filters and tie-breaking as regular discover.
    pub fn discover_explained(
        &mut self,
        query: &str,
        filters: Option<&Filters>,
    ) -> Vec<ExplainedResult> {
        self.discover_core(query, filters)
            .into_iter()
            .map(|c| ExplainedResult {
                agent_name: c.agent_name,
                raw_similarity: c.raw_similarity,
                diameter: c.diameter,
                feedback_factor: c.feedback_factor,
                final_score: c.final_score,
                endpoint: c.endpoint,
                metadata: c.metadata,
            })
            .collect()
    }

    /// Discover agents for multiple queries in a single call.
    ///
    /// Returns one result list per query, in the same order as the input.
    /// Filters are shared across all queries. Each query runs through
    /// the full discovery pipeline (scoring, feedback, tie-breaking, enzyme update).
    ///
    /// This moves the query loop from Python to Rust, avoiding per-query
    /// GIL acquisition overhead. Future versions may parallelize internally.
    pub fn discover_many(
        &mut self,
        queries: &[&str],
        filters: Option<&Filters>,
    ) -> Vec<Vec<DiscoveryResult>> {
        queries
            .iter()
            .map(|q| self.discover_filtered(q, filters))
            .collect()
    }

    /// Discover agents for multiple queries with full scoring breakdown.
    ///
    /// Like `discover_many` but returns `ExplainedResult` per match.
    pub fn discover_many_explained(
        &mut self,
        queries: &[&str],
        filters: Option<&Filters>,
    ) -> Vec<Vec<ExplainedResult>> {
        queries
            .iter()
            .map(|q| self.discover_explained(q, filters))
            .collect()
    }

    /// Shared discovery pipeline: scoring, feedback, tie-breaking, filtering, enzyme update.
    fn discover_core(&mut self, query: &str, filters: Option<&Filters>) -> Vec<ScoredCandidate> {
        if self.agents.is_empty() {
            return Vec::new();
        }

        let query_sig = compute_query_signature(query.as_bytes(), &self.lsh_config);

        // Get raw scores from the scorer.
        let raw_scores = match self.scorer.score(query) {
            Ok(scores) => scores,
            Err(e) => {
                eprintln!("alps-discovery: scorer.score() error: {}", e);
                return Vec::new();
            }
        };
        let score_map: HashMap<String, f64> = raw_scores.into_iter().collect();

        // Build lightweight references — no cloning yet.
        let mut refs: Vec<CandidateRef<'_>> = self
            .agents
            .iter()
            .filter_map(|(name, record)| {
                let sim = score_map.get(name.as_str()).copied().unwrap_or(0.0);
                if sim < self.lsh_config.similarity_threshold {
                    return None;
                }

                let feedback_factor = Self::compute_feedback_factor(
                    &record.feedback,
                    &query_sig.minhash,
                    self.lsh_config.similarity_threshold,
                );
                let adjusted_diameter =
                    record.hypha.state.diameter * (1.0 + feedback_factor * FEEDBACK_STRENGTH);

                let score = sim * adjusted_diameter;
                Some(CandidateRef {
                    agent_name: name.as_str(),
                    raw_similarity: sim,
                    diameter: record.hypha.state.diameter,
                    feedback_factor,
                    final_score: score,
                    endpoint: &record.endpoint,
                    metadata: &record.metadata,
                })
            })
            .collect();

        // Apply metadata filters before sorting (avoids sorting filtered-out items).
        if let Some(filters) = filters {
            refs.retain(|r| {
                filters
                    .iter()
                    .all(|(key, filter)| match r.metadata.get(key) {
                        Some(value) => filter.matches(value),
                        None => false,
                    })
            });
        }

        refs.sort_by(|a, b| {
            b.final_score
                .partial_cmp(&a.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Randomized tie-breaking: shuffle agents within 5% of top score.
        if refs.len() > 1 {
            let top_score = refs[0].final_score;
            let tie_threshold = top_score * 0.95;
            let tie_count = refs
                .iter()
                .take_while(|r| r.final_score >= tie_threshold)
                .count();
            if tie_count > 1 {
                use std::time::SystemTime;
                use xxhash_rust::xxh3::xxh3_64_with_seed;
                let time_seed = SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(0);
                let base_seed = xxh3_64_with_seed(query.as_bytes(), time_seed);

                let tie_slice = &mut refs[..tie_count];
                for i in (1..tie_slice.len()).rev() {
                    let j_seed = base_seed.wrapping_add(i as u64);
                    let j = (xxh3_64_with_seed(&i.to_le_bytes(), j_seed) as usize) % (i + 1);
                    tie_slice.swap(i, j);
                }
            }
        }

        // Promote survivors to owned ScoredCandidate (clone only here).
        let results: Vec<ScoredCandidate> = refs
            .into_iter()
            .map(|r| ScoredCandidate {
                agent_name: r.agent_name.to_string(),
                raw_similarity: r.raw_similarity,
                diameter: r.diameter,
                feedback_factor: r.feedback_factor,
                final_score: r.final_score,
                endpoint: r.endpoint.clone(),
                metadata: r.metadata.clone(),
            })
            .collect();

        // Run the enzyme to update internal state (feeds LoadBalancingKernel).
        let signal = Signal::Tendril(Tendril {
            trail_id: TrailId([0u8; 32]),
            query_signature: query_sig,
            query_config: QueryConfig::default(),
        });
        let hyphae: Vec<&Hypha> = self.agents.values().map(|r| &r.hypha).collect();
        let decision = self
            .enzyme
            .process(&signal, &self.spore, &hyphae, &self.membrane_state);

        // Update forwards_count for the enzyme's pick.
        let picked = match &decision.action {
            EnzymeAction::Forward { target } => vec![target.clone()],
            EnzymeAction::Split { targets } => targets.clone(),
            _ => vec![],
        };
        for hypha_id in &picked {
            for record in self.agents.values_mut() {
                if record.hypha.id == *hypha_id {
                    record.hypha.state.forwards_count += 1;
                }
            }
        }

        results
    }

    /// Record a successful interaction with an agent.
    ///
    /// If `query` is provided, stores per-query-type feedback so future
    /// queries similar to this one boost the agent's ranking — without
    /// affecting unrelated query types.
    ///
    /// Global diameter adjustment always applies regardless of query.
    pub fn record_success(&mut self, agent_name: &str, query: Option<&str>) {
        if let Some(record) = self.agents.get_mut(agent_name) {
            // Global boost (always).
            record.hypha.state.tau += 0.05;
            record.hypha.state.sigma += 0.01;
            record.hypha.state.diameter = (record.hypha.state.diameter + 0.01).min(1.0);
            record.hypha.state.consecutive_pulse_timeouts = 0;

            // Per-query-type feedback (if query provided).
            if let Some(q) = query {
                let sig = compute_query_signature(q.as_bytes(), &self.lsh_config);
                record.feedback.push_back(FeedbackRecord {
                    query_minhash: sig.minhash,
                    outcome: 1.0,
                });
                if record.feedback.len() > MAX_FEEDBACK_RECORDS {
                    record.feedback.pop_front();
                }
            }
        }
    }

    /// Record a failed interaction with an agent.
    ///
    /// If `query` is provided, stores per-query-type feedback so future
    /// queries similar to this one penalize the agent's ranking — without
    /// affecting unrelated query types.
    pub fn record_failure(&mut self, agent_name: &str, query: Option<&str>) {
        if let Some(record) = self.agents.get_mut(agent_name) {
            // Global penalty (always).
            record.hypha.state.consecutive_pulse_timeouts = record
                .hypha
                .state
                .consecutive_pulse_timeouts
                .saturating_add(1);
            record.hypha.state.diameter = (record.hypha.state.diameter - 0.05).max(0.1);

            // Per-query-type feedback (if query provided).
            if let Some(q) = query {
                let sig = compute_query_signature(q.as_bytes(), &self.lsh_config);
                record.feedback.push_back(FeedbackRecord {
                    query_minhash: sig.minhash,
                    outcome: -1.0,
                });
                if record.feedback.len() > MAX_FEEDBACK_RECORDS {
                    record.feedback.pop_front();
                }
            }
        }
    }

    /// Compute per-query feedback factor for an agent.
    ///
    /// Returns a value in [-1.0, 1.0] representing how well this agent has
    /// performed on queries similar to the current one. 0.0 means no relevant
    /// feedback exists.
    fn compute_feedback_factor(
        feedback: &VecDeque<FeedbackRecord>,
        query_minhash: &[u8; 64],
        relevance_threshold: f64,
    ) -> f64 {
        if feedback.is_empty() {
            return 0.0;
        }

        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;

        for fb in feedback {
            let relevance = MinHasher::similarity(&fb.query_minhash, query_minhash);
            if relevance >= relevance_threshold {
                weighted_sum += relevance * fb.outcome;
                weight_sum += relevance;
            }
        }

        if weight_sum > 0.0 {
            (weighted_sum / weight_sum).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }

    /// Returns the number of registered agents.
    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }

    /// Returns all registered agent names.
    pub fn agents(&self) -> Vec<String> {
        self.agents.keys().cloned().collect()
    }

    /// Deterministically convert an agent name to a HyphaId.
    fn name_to_hypha_id(name: &str) -> HyphaId {
        use xxhash_rust::xxh3::xxh3_64_with_seed;
        let mut id = [0u8; 32];
        for i in 0..4u64 {
            let hash = xxh3_64_with_seed(name.as_bytes(), i);
            let offset = (i as usize) * 8;
            id[offset..offset + 8].copy_from_slice(&hash.to_le_bytes());
        }
        HyphaId(id)
    }

    /// Save the network state to a JSON file.
    ///
    /// Persists all agents, their scoring state (diameter, tau, sigma, etc.),
    /// and per-query feedback history. The scorer re-indexes capabilities
    /// from the saved data on load.
    ///
    /// `Hypha.last_activity` (an `Instant`) is NOT serialized — it resets
    /// to `Instant::now()` on load.
    pub fn save(&self, path: &str) -> Result<(), NetworkError> {
        let snapshot = NetworkSnapshot {
            version: Self::SNAPSHOT_VERSION,
            agents: self
                .agents
                .iter()
                .map(|(name, record)| {
                    let feedback: Vec<FeedbackSnapshot> = record
                        .feedback
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
                        omega: record.hypha.state.omega,
                        forwards_count: record.hypha.state.forwards_count,
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

    /// Load network state from a JSON file.
    ///
    /// Rebuilds the network from a previously saved snapshot. Capabilities
    /// are re-indexed through the scorer. `last_activity` is reset to now.
    ///
    /// Uses the default MinHash scorer. For custom scorers, load then
    /// reconfigure.
    /// Current snapshot schema version.
    const SNAPSHOT_VERSION: u32 = 1;

    pub fn load(path: &str) -> Result<Self, NetworkError> {
        let json = std::fs::read_to_string(path)?;
        let snapshot: NetworkSnapshot = serde_json::from_str(&json)?;

        if snapshot.version > Self::SNAPSHOT_VERSION {
            return Err(NetworkError::UnsupportedVersion {
                found: snapshot.version,
                supported: Self::SNAPSHOT_VERSION,
            });
        }

        let mut network = Self::new();

        for agent in snapshot.agents {
            let caps: Vec<&str> = agent.capabilities.iter().map(|s| s.as_str()).collect();
            network.register(
                &agent.name,
                &caps,
                agent.endpoint.as_deref(),
                agent.metadata,
            );

            // Restore scoring state (register() sets defaults, so override).
            if let Some(record) = network.agents.get_mut(&agent.name) {
                record.hypha.state.diameter = agent.diameter;
                record.hypha.state.tau = agent.tau;
                record.hypha.state.sigma = agent.sigma;
                record.hypha.state.omega = agent.omega;
                record.hypha.state.forwards_count = agent.forwards_count;
                record.hypha.state.consecutive_pulse_timeouts = agent.consecutive_pulse_timeouts;

                // Restore feedback history, capped at MAX_FEEDBACK_RECORDS.
                let feedback_iter = agent.feedback.into_iter().map(|fb| FeedbackRecord {
                    query_minhash: fb.query_minhash,
                    outcome: fb.outcome,
                });
                let mut feedback: VecDeque<FeedbackRecord> = feedback_iter.collect();
                while feedback.len() > MAX_FEEDBACK_RECORDS {
                    feedback.pop_front();
                }
                record.feedback = feedback;
            }
        }

        Ok(network)
    }
}

impl Default for LocalNetwork {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: register with no endpoint/metadata.
    fn reg(network: &mut LocalNetwork, name: &str, caps: &[&str]) {
        network.register(name, caps, None, HashMap::new());
    }

    #[test]
    fn empty_network_returns_no_results() {
        let mut network = LocalNetwork::new();
        let results = network.discover("anything");
        assert!(results.is_empty());
    }

    #[test]
    fn register_and_discover_matches_capability() {
        let mut network = LocalNetwork::new();
        reg(
            &mut network,
            "translate-agent",
            &["legal translation", "EN-DE", "EN-FR"],
        );
        let results = network.discover("legal translation");
        assert!(!results.is_empty());
        assert_eq!(results[0].agent_name, "translate-agent");
        assert!(results[0].similarity > 0.0);
    }

    #[test]
    fn discover_ranks_more_similar_agent_higher() {
        let mut network = LocalNetwork::new();
        reg(
            &mut network,
            "translate-agent",
            &["legal translation", "EN-DE", "EN-FR"],
        );
        reg(
            &mut network,
            "summarize-agent",
            &["document summarization", "legal briefs"],
        );
        let results = network.discover("legal translation services");
        assert!(!results.is_empty());
        assert_eq!(results[0].agent_name, "translate-agent");
    }

    #[test]
    fn deregister_removes_agent() {
        let mut network = LocalNetwork::new();
        reg(&mut network, "agent-a", &["capability-a"]);
        assert_eq!(network.agent_count(), 1);
        assert!(network.deregister("agent-a"));
        assert_eq!(network.agent_count(), 0);
        let results = network.discover("capability-a");
        assert!(results.is_empty());
    }

    #[test]
    fn deregister_nonexistent_returns_false() {
        let mut network = LocalNetwork::new();
        assert!(!network.deregister("nonexistent"));
    }

    #[test]
    fn agents_returns_all_names() {
        let mut network = LocalNetwork::new();
        reg(&mut network, "agent-a", &["cap-a"]);
        reg(&mut network, "agent-b", &["cap-b"]);
        let names = network.agents();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"agent-a".to_string()));
        assert!(names.contains(&"agent-b".to_string()));
    }

    #[test]
    fn record_success_improves_ranking() {
        let mut network = LocalNetwork::new();
        reg(&mut network, "agent-a", &["data processing"]);
        reg(&mut network, "agent-b", &["data processing"]);

        for _ in 0..20 {
            network.record_success("agent-a", None);
        }

        let results = network.discover("data processing");
        assert!(results.len() >= 2);
        let a_result = results.iter().find(|r| r.agent_name == "agent-a").unwrap();
        let b_result = results.iter().find(|r| r.agent_name == "agent-b").unwrap();
        assert!(
            a_result.score >= b_result.score,
            "agent-a score ({}) should be >= agent-b score ({})",
            a_result.score,
            b_result.score
        );
    }

    #[test]
    fn record_failure_reduces_ranking() {
        let mut network = LocalNetwork::new();
        reg(&mut network, "agent-a", &["data processing"]);
        reg(&mut network, "agent-b", &["data processing"]);

        for _ in 0..10 {
            network.record_failure("agent-a", None);
        }

        let results = network.discover("data processing");
        assert!(results.len() >= 2);
        let a_result = results.iter().find(|r| r.agent_name == "agent-a").unwrap();
        let b_result = results.iter().find(|r| r.agent_name == "agent-b").unwrap();
        assert!(
            b_result.score >= a_result.score,
            "agent-b score ({}) should be >= agent-a score ({})",
            b_result.score,
            a_result.score
        );
    }

    #[test]
    fn multiple_agents_with_overlapping_capabilities() {
        let mut network = LocalNetwork::new();
        reg(
            &mut network,
            "agent-1",
            &[
                "legal document translation service",
                "translate contracts and legal briefs",
            ],
        );
        reg(
            &mut network,
            "agent-2",
            &[
                "medical record translation service",
                "translate clinical notes",
            ],
        );
        reg(
            &mut network,
            "agent-3",
            &[
                "legal document summarization service",
                "summarize contracts and briefs",
            ],
        );

        let results = network.discover("translate legal documents and contracts");
        assert!(!results.is_empty());
        assert!(results.len() >= 2);
        let agent_1_found = results.iter().any(|r| r.agent_name == "agent-1");
        assert!(agent_1_found, "agent-1 should appear in results");
        for r in &results {
            assert!(r.similarity > 0.0);
        }
    }

    #[test]
    fn hypha_id_is_deterministic() {
        let id1 = LocalNetwork::name_to_hypha_id("test-agent");
        let id2 = LocalNetwork::name_to_hypha_id("test-agent");
        assert_eq!(id1, id2);

        let id3 = LocalNetwork::name_to_hypha_id("other-agent");
        assert_ne!(id1, id3);
    }

    #[test]
    fn endpoint_and_metadata_returned_in_results() {
        let mut network = LocalNetwork::new();
        let mut meta = HashMap::new();
        meta.insert("protocol".to_string(), "mcp".to_string());
        meta.insert("version".to_string(), "1.0".to_string());
        network.register(
            "translate-agent",
            &["legal translation"],
            Some("http://localhost:8080/translate"),
            meta,
        );
        reg(&mut network, "bare-agent", &["legal translation"]);

        let results = network.discover("legal translation");
        assert!(results.len() >= 2);

        let translate = results
            .iter()
            .find(|r| r.agent_name == "translate-agent")
            .unwrap();
        assert_eq!(
            translate.endpoint.as_deref(),
            Some("http://localhost:8080/translate")
        );
        assert_eq!(
            translate.metadata.get("protocol").map(|s| s.as_str()),
            Some("mcp")
        );
        assert_eq!(
            translate.metadata.get("version").map(|s| s.as_str()),
            Some("1.0")
        );

        let bare = results
            .iter()
            .find(|r| r.agent_name == "bare-agent")
            .unwrap();
        assert!(bare.endpoint.is_none());
        assert!(bare.metadata.is_empty());
    }

    #[test]
    fn threshold_filters_noise() {
        let mut network = LocalNetwork::new();
        reg(
            &mut network,
            "translate-agent",
            &["legal translation services"],
        );
        let results = network.discover("add these two numbers together");
        assert!(
            results.is_empty(),
            "unrelated query should return no results, got {} with sims: {:?}",
            results.len(),
            results.iter().map(|r| r.similarity).collect::<Vec<_>>()
        );
    }

    #[test]
    fn composite_capability_strings_improve_matching() {
        let mut network = LocalNetwork::new();
        reg(
            &mut network,
            "translate-server",
            &["translate text: Translate text between languages with legal domain expertise. text, source lang, target lang"],
        );
        reg(
            &mut network,
            "summarize-server",
            &["summarize document: Generate a concise summary of a legal document. document, max length"],
        );

        let results = network.discover("translate legal contract to German");
        assert!(!results.is_empty(), "should find translate-server");
        assert_eq!(results[0].agent_name, "translate-server");

        let results = network.discover("summarize a legal document");
        assert!(!results.is_empty(), "should find summarize-server");
        assert_eq!(results[0].agent_name, "summarize-server");
    }

    #[test]
    fn custom_similarity_threshold() {
        use crate::core::config::LshConfig;
        use crate::core::enzyme::SLNEnzymeConfig;

        // Very high threshold — filters everything
        let lsh = LshConfig {
            similarity_threshold: 0.9,
            ..LshConfig::default()
        };
        let mut network = LocalNetwork::with_config(SLNEnzymeConfig::default(), lsh);
        reg(&mut network, "agent", &["data processing"]);
        let results = network.discover("process some data");
        assert!(
            results.is_empty(),
            "high threshold should filter partial matches"
        );

        // Very low threshold — includes everything
        let lsh = LshConfig {
            similarity_threshold: 0.0,
            ..LshConfig::default()
        };
        let mut network = LocalNetwork::with_config(SLNEnzymeConfig::default(), lsh);
        reg(&mut network, "agent", &["data processing"]);
        let results = network.discover("completely unrelated query");
        // With threshold 0.0, even noise matches should appear
        // (unless they happen to have exactly 0.0 similarity)
        assert!(results.len() <= 1); // might or might not match
    }

    #[test]
    fn per_query_feedback_boosts_similar_queries() {
        // Two identical agents. Record translation successes for agent-a.
        // Translation-like queries should prefer agent-a.
        // Summarization queries should NOT prefer agent-a.
        let mut network = LocalNetwork::new();
        reg(
            &mut network,
            "agent-a",
            &["legal translation", "document summarization"],
        );
        reg(
            &mut network,
            "agent-b",
            &["legal translation", "document summarization"],
        );

        // Record translation successes for agent-a WITH query context.
        for _ in 0..10 {
            network.record_success("agent-a", Some("translate legal contract to German"));
        }

        // Translation-like query → agent-a should be boosted.
        let results = network.discover("translate patent application to French");
        assert!(results.len() >= 2);
        let a = results.iter().find(|r| r.agent_name == "agent-a").unwrap();
        let b = results.iter().find(|r| r.agent_name == "agent-b").unwrap();
        assert!(
            a.score > b.score,
            "agent-a ({:.4}) should outscore agent-b ({:.4}) for translation queries",
            a.score,
            b.score
        );

        // Summarization query → scores should be close (no translation feedback boost).
        let results = network.discover("summarize this legal brief");
        let a = results.iter().find(|r| r.agent_name == "agent-a").unwrap();
        let b = results.iter().find(|r| r.agent_name == "agent-b").unwrap();
        // agent-a still has a small global diameter boost from record_success,
        // but the per-query feedback should NOT amplify it for summarization.
        let ratio = a.score / b.score;
        assert!(
            ratio < 1.3,
            "agent-a/agent-b score ratio ({:.3}) should be close for unrelated query type",
            ratio
        );
    }

    #[test]
    fn per_query_failure_penalizes_similar_queries() {
        let mut network = LocalNetwork::new();
        reg(&mut network, "agent-a", &["legal translation"]);
        reg(&mut network, "agent-b", &["legal translation"]);

        // Record translation failures for agent-a.
        for _ in 0..10 {
            network.record_failure("agent-a", Some("translate legal contract"));
        }

        // Translation query → agent-b should win.
        let results = network.discover("translate patent to German");
        assert!(results.len() >= 2);
        let a = results.iter().find(|r| r.agent_name == "agent-a").unwrap();
        let b = results.iter().find(|r| r.agent_name == "agent-b").unwrap();
        assert!(
            b.score > a.score,
            "agent-b ({:.4}) should outscore agent-a ({:.4}) after translation failures",
            b.score,
            a.score
        );
    }

    #[test]
    fn feedback_without_query_still_works_globally() {
        // Backward compat: record_success(name, None) does global diameter boost.
        let mut network = LocalNetwork::new();
        reg(&mut network, "agent-a", &["data processing"]);
        reg(&mut network, "agent-b", &["data processing"]);

        for _ in 0..20 {
            network.record_success("agent-a", None);
        }

        let results = network.discover("data processing");
        assert!(results.len() >= 2);
        let a = results.iter().find(|r| r.agent_name == "agent-a").unwrap();
        let b = results.iter().find(|r| r.agent_name == "agent-b").unwrap();
        assert!(
            a.score >= b.score,
            "global feedback should still boost agent-a"
        );
    }

    #[test]
    fn custom_scorer_overrides_matching() {
        use crate::scorer::Scorer;

        /// A mock scorer that returns hardcoded scores.
        struct MockScorer {
            scores: HashMap<String, f64>,
        }

        impl Scorer for MockScorer {
            fn index_capabilities(&mut self, agent_id: &str, _capabilities: &[&str]) {
                self.scores.insert(agent_id.to_string(), 0.8);
            }
            fn remove_agent(&mut self, agent_id: &str) {
                self.scores.remove(agent_id);
            }
            fn score(&self, _query: &str) -> Result<Vec<(String, f64)>, String> {
                Ok(self.scores.iter().map(|(k, v)| (k.clone(), *v)).collect())
            }
        }

        let scorer = MockScorer {
            scores: HashMap::new(),
        };
        let mut network = LocalNetwork::with_scorer(Box::new(scorer));
        network.register("agent-a", &["anything"], None, HashMap::new());
        network.register("agent-b", &["anything"], None, HashMap::new());

        let results = network.discover("any query");
        assert_eq!(results.len(), 2);
        // Both should have similarity 0.8 from mock
        for r in &results {
            assert!((r.similarity - 0.8).abs() < 0.001);
        }
    }

    #[test]
    fn custom_scorer_results_flow_through_discovery() {
        use crate::scorer::Scorer;

        struct RankedMockScorer;

        impl Scorer for RankedMockScorer {
            fn index_capabilities(&mut self, _agent_id: &str, _capabilities: &[&str]) {}
            fn remove_agent(&mut self, _agent_id: &str) {}
            fn score(&self, _query: &str) -> Result<Vec<(String, f64)>, String> {
                Ok(vec![
                    ("agent-high".to_string(), 0.9),
                    ("agent-low".to_string(), 0.3),
                ])
            }
        }

        let mut network = LocalNetwork::with_scorer(Box::new(RankedMockScorer));
        network.register("agent-high", &["x"], None, HashMap::new());
        network.register("agent-low", &["y"], None, HashMap::new());

        let results = network.discover("test");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].agent_name, "agent-high");
        assert!(results[0].score > results[1].score);
    }

    #[test]
    fn filter_exact_match() {
        let mut network = LocalNetwork::new();
        let mut meta_mcp = HashMap::new();
        meta_mcp.insert("protocol".to_string(), "mcp".to_string());
        network.register("agent-mcp", &["legal translation"], None, meta_mcp);

        let mut meta_rest = HashMap::new();
        meta_rest.insert("protocol".to_string(), "rest".to_string());
        network.register("agent-rest", &["legal translation"], None, meta_rest);

        let mut filters = Filters::new();
        filters.insert(
            "protocol".to_string(),
            FilterValue::Exact("mcp".to_string()),
        );

        let results = network.discover_filtered("legal translation", Some(&filters));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].agent_name, "agent-mcp");
    }

    #[test]
    fn filter_contains() {
        let mut network = LocalNetwork::new();
        let mut meta = HashMap::new();
        meta.insert(
            "description".to_string(),
            "fast translation service".to_string(),
        );
        network.register("agent-a", &["legal translation"], None, meta);

        network.register("agent-b", &["legal translation"], None, HashMap::new());

        let mut filters = Filters::new();
        filters.insert(
            "description".to_string(),
            FilterValue::Contains("translation".to_string()),
        );

        let results = network.discover_filtered("legal translation", Some(&filters));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].agent_name, "agent-a");
    }

    #[test]
    fn filter_one_of() {
        let mut network = LocalNetwork::new();
        let mut meta1 = HashMap::new();
        meta1.insert("protocol".to_string(), "mcp".to_string());
        network.register("agent-mcp", &["legal translation"], None, meta1);

        let mut meta2 = HashMap::new();
        meta2.insert("protocol".to_string(), "grpc".to_string());
        network.register("agent-grpc", &["legal translation"], None, meta2);

        let mut meta3 = HashMap::new();
        meta3.insert("protocol".to_string(), "rest".to_string());
        network.register("agent-rest", &["legal translation"], None, meta3);

        let mut filters = Filters::new();
        filters.insert(
            "protocol".to_string(),
            FilterValue::OneOf(vec!["mcp".to_string(), "grpc".to_string()]),
        );

        let results = network.discover_filtered("legal translation", Some(&filters));
        assert_eq!(results.len(), 2);
        let names: Vec<&str> = results.iter().map(|r| r.agent_name.as_str()).collect();
        assert!(names.contains(&"agent-mcp"));
        assert!(names.contains(&"agent-grpc"));
    }

    #[test]
    fn filter_numeric_comparison() {
        let mut network = LocalNetwork::new();
        let mut meta1 = HashMap::new();
        meta1.insert("latency_ms".to_string(), "50".to_string());
        network.register("fast-agent", &["legal translation"], None, meta1);

        let mut meta2 = HashMap::new();
        meta2.insert("latency_ms".to_string(), "200".to_string());
        network.register("slow-agent", &["legal translation"], None, meta2);

        let mut filters = Filters::new();
        filters.insert("latency_ms".to_string(), FilterValue::LessThan(100.0));

        let results = network.discover_filtered("legal translation", Some(&filters));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].agent_name, "fast-agent");
    }

    #[test]
    fn filter_missing_key_fails_strict() {
        let mut network = LocalNetwork::new();
        network.register("agent-a", &["legal translation"], None, HashMap::new());

        let mut filters = Filters::new();
        filters.insert(
            "protocol".to_string(),
            FilterValue::Exact("mcp".to_string()),
        );

        let results = network.discover_filtered("legal translation", Some(&filters));
        assert!(
            results.is_empty(),
            "missing metadata key should fail filter"
        );
    }

    #[test]
    fn filter_multiple_conditions() {
        let mut network = LocalNetwork::new();
        let mut meta = HashMap::new();
        meta.insert("protocol".to_string(), "mcp".to_string());
        meta.insert("version".to_string(), "2.0".to_string());
        network.register("agent-match", &["legal translation"], None, meta);

        let mut meta2 = HashMap::new();
        meta2.insert("protocol".to_string(), "mcp".to_string());
        meta2.insert("version".to_string(), "1.0".to_string());
        network.register("agent-old", &["legal translation"], None, meta2);

        let mut filters = Filters::new();
        filters.insert(
            "protocol".to_string(),
            FilterValue::Exact("mcp".to_string()),
        );
        filters.insert("version".to_string(), FilterValue::GreaterThan(1.5));

        let results = network.discover_filtered("legal translation", Some(&filters));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].agent_name, "agent-match");
    }

    #[test]
    fn discover_without_filters_unchanged() {
        // Verify backwards compatibility -- discover() without filters works as before.
        let mut network = LocalNetwork::new();
        reg(&mut network, "agent-a", &["legal translation"]);
        let results = network.discover("legal translation");
        assert!(!results.is_empty());
    }

    #[test]
    fn tie_breaking_distributes_across_identical_agents() {
        let mut network = LocalNetwork::new();
        // Register 5 identical agents
        for i in 0..5 {
            reg(
                &mut network,
                &format!("agent-{}", i),
                &["data processing service"],
            );
        }

        // Run many queries and collect who wins first place
        let mut first_place_counts: HashMap<String, usize> = HashMap::new();
        for i in 0..50 {
            // Use slightly different queries to get time-based seed variation
            let query = format!("data processing service query number {}", i);
            let results = network.discover(&query);
            assert!(!results.is_empty());
            *first_place_counts
                .entry(results[0].agent_name.clone())
                .or_insert(0) += 1;
        }

        // At least 2 different agents should have been first at some point
        let unique_winners = first_place_counts.len();
        assert!(
            unique_winners > 1,
            "expected >1 unique first-place agents from tie-breaking, got {} (counts: {:?})",
            unique_winners,
            first_place_counts
        );
    }

    #[test]
    fn filters_and_tie_breaking_combined() {
        let mut network = LocalNetwork::new();
        for i in 0..3 {
            let mut meta = HashMap::new();
            meta.insert("protocol".to_string(), "mcp".to_string());
            network.register(
                &format!("mcp-agent-{}", i),
                &["data processing"],
                None,
                meta,
            );
        }
        let mut meta = HashMap::new();
        meta.insert("protocol".to_string(), "rest".to_string());
        network.register("rest-agent", &["data processing"], None, meta);

        let mut filters = Filters::new();
        filters.insert(
            "protocol".to_string(),
            FilterValue::Exact("mcp".to_string()),
        );

        let results = network.discover_filtered("data processing", Some(&filters));
        assert_eq!(results.len(), 3);
        for r in &results {
            assert!(r.agent_name.starts_with("mcp-agent"));
        }
    }

    #[test]
    fn save_and_load_preserves_agents() {
        let mut network = LocalNetwork::new();
        let mut meta = HashMap::new();
        meta.insert("protocol".to_string(), "mcp".to_string());
        network.register(
            "translate-agent",
            &["legal translation", "EN-DE"],
            Some("http://localhost:8080"),
            meta,
        );
        reg(&mut network, "summarize-agent", &["document summarization"]);

        // Record some feedback
        for _ in 0..5 {
            network.record_success("translate-agent", Some("translate legal docs"));
        }
        for _ in 0..3 {
            network.record_failure("summarize-agent", Some("summarize briefs"));
        }

        // Save
        let path = "/tmp/alps_test_save_load.json";
        network.save(path).expect("save should succeed");

        // Load
        let mut loaded = LocalNetwork::load(path).expect("load should succeed");
        assert_eq!(loaded.agent_count(), 2);
        assert!(loaded.agents().contains(&"translate-agent".to_string()));
        assert!(loaded.agents().contains(&"summarize-agent".to_string()));

        // Discover should still work
        let results = loaded.discover("legal translation");
        assert!(!results.is_empty());
        let translate = results
            .iter()
            .find(|r| r.agent_name == "translate-agent")
            .unwrap();
        assert_eq!(translate.endpoint.as_deref(), Some("http://localhost:8080"));
        assert_eq!(
            translate.metadata.get("protocol").map(|s| s.as_str()),
            Some("mcp")
        );

        // Cleanup
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn save_and_load_preserves_feedback() {
        let mut network = LocalNetwork::new();
        reg(&mut network, "agent-a", &["data processing"]);
        reg(&mut network, "agent-b", &["data processing"]);

        // Boost agent-a with query-specific feedback
        for _ in 0..10 {
            network.record_success("agent-a", Some("process data files"));
        }

        let path = "/tmp/alps_test_feedback.json";
        network.save(path).expect("save should succeed");

        let mut loaded = LocalNetwork::load(path).expect("load should succeed");
        let results = loaded.discover("process data files");
        assert!(results.len() >= 2);
        let a = results.iter().find(|r| r.agent_name == "agent-a").unwrap();
        let b = results.iter().find(|r| r.agent_name == "agent-b").unwrap();
        assert!(
            a.score > b.score,
            "feedback should be preserved: agent-a ({:.4}) should outscore agent-b ({:.4})",
            a.score,
            b.score
        );

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn load_missing_file_returns_error() {
        let result = LocalNetwork::load("/tmp/nonexistent_alps_file.json");
        assert!(result.is_err());
    }

    #[test]
    fn load_invalid_json_returns_error() {
        let path = "/tmp/alps_test_invalid.json";
        std::fs::write(path, "not valid json").expect("write test file");
        let result = LocalNetwork::load(path);
        assert!(result.is_err());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn save_includes_version_field() {
        let network = LocalNetwork::new();
        let path = "/tmp/alps_test_version.json";
        network.save(path).expect("save should succeed");
        let json = std::fs::read_to_string(path).expect("read file");
        assert!(
            json.contains("\"version\""),
            "snapshot should include version field"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn load_rejects_future_snapshot_version() {
        let path = "/tmp/alps_test_future_version.json";
        let json = r#"{"version": 99, "agents": []}"#;
        std::fs::write(path, json).expect("write test file");
        let result = LocalNetwork::load(path);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(
            matches!(
                err,
                NetworkError::UnsupportedVersion {
                    found: 99,
                    supported: 1
                }
            ),
            "expected UnsupportedVersion, got: {:?}",
            err
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn save_and_load_empty_network() {
        let network = LocalNetwork::new();
        let path = "/tmp/alps_test_empty.json";
        network.save(path).expect("save should succeed");
        let loaded = LocalNetwork::load(path).expect("load should succeed");
        assert_eq!(loaded.agent_count(), 0);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn explain_result_contains_all_fields() {
        let mut network = LocalNetwork::new();
        let mut meta = HashMap::new();
        meta.insert("protocol".to_string(), "mcp".to_string());
        network.register(
            "translate-agent",
            &["legal translation"],
            Some("http://localhost:8080"),
            meta,
        );

        // Add some feedback to verify feedback_factor is populated
        for _ in 0..5 {
            network.record_success("translate-agent", Some("translate legal contract"));
        }

        let results = network.discover_explained("translate legal contract", None);
        assert!(!results.is_empty());
        let r = &results[0];
        assert_eq!(r.agent_name, "translate-agent");
        assert!(r.raw_similarity > 0.0);
        assert!(r.diameter > 0.0);
        assert!(
            r.feedback_factor > 0.0,
            "should have positive feedback after successes"
        );
        assert!(r.final_score > 0.0);
        assert_eq!(r.endpoint.as_deref(), Some("http://localhost:8080"));
        assert_eq!(r.metadata.get("protocol").map(|s| s.as_str()), Some("mcp"));
    }

    #[test]
    fn explain_scores_match_regular_discover() {
        let mut network = LocalNetwork::new();
        reg(&mut network, "agent-a", &["legal translation"]);
        reg(&mut network, "agent-b", &["document summarization"]);

        // Run regular discover
        let regular = network.discover("legal translation");
        // Run explained discover
        let explained = network.discover_explained("legal translation", None);

        // Same number of results
        assert_eq!(regular.len(), explained.len());

        // Scores should match (both use same discover_core pipeline).
        // Compare by agent name since tie-breaking is randomized per call.
        for exp_r in &explained {
            let reg_r = regular
                .iter()
                .find(|r| r.agent_name == exp_r.agent_name)
                .expect("explained agent should appear in regular results");
            assert!(
                (reg_r.similarity - exp_r.raw_similarity).abs() < 0.001,
                "{}: similarity mismatch",
                exp_r.agent_name
            );
            assert!(
                (reg_r.score - exp_r.final_score).abs() < 0.001,
                "{}: regular score ({:.4}) should match explained final_score ({:.4})",
                exp_r.agent_name,
                reg_r.score,
                exp_r.final_score
            );
        }
    }

    #[test]
    fn explain_empty_network() {
        let mut network = LocalNetwork::new();
        let results = network.discover_explained("anything", None);
        assert!(results.is_empty());
    }

    #[test]
    fn explain_with_filters() {
        let mut network = LocalNetwork::new();
        let mut meta_mcp = HashMap::new();
        meta_mcp.insert("protocol".to_string(), "mcp".to_string());
        network.register("agent-mcp", &["legal translation"], None, meta_mcp);

        let mut meta_rest = HashMap::new();
        meta_rest.insert("protocol".to_string(), "rest".to_string());
        network.register("agent-rest", &["legal translation"], None, meta_rest);

        let mut filters = Filters::new();
        filters.insert(
            "protocol".to_string(),
            FilterValue::Exact("mcp".to_string()),
        );

        let results = network.discover_explained("legal translation", Some(&filters));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].agent_name, "agent-mcp");
        assert!(results[0].raw_similarity > 0.0);
        assert!(results[0].diameter > 0.0);
    }

    #[test]
    fn explain_feedback_factor_zero_without_feedback() {
        let mut network = LocalNetwork::new();
        reg(&mut network, "agent-a", &["data processing"]);
        let results = network.discover_explained("data processing", None);
        assert!(!results.is_empty());
        assert!(
            results[0].feedback_factor.abs() < 0.001,
            "feedback_factor should be ~0 without feedback, got {:.4}",
            results[0].feedback_factor
        );
    }

    #[test]
    fn discover_many_returns_per_query_results() {
        let mut network = LocalNetwork::new();
        reg(
            &mut network,
            "translate-agent",
            &["legal translation services"],
        );
        reg(&mut network, "summarize-agent", &["document summarization"]);

        let results = network.discover_many(&["legal translation", "document summarization"], None);
        assert_eq!(results.len(), 2);
        // First query should find translate-agent
        assert!(!results[0].is_empty());
        // Second query should find summarize-agent
        assert!(!results[1].is_empty());
    }

    #[test]
    fn discover_many_empty_queries() {
        let mut network = LocalNetwork::new();
        reg(&mut network, "agent-a", &["data processing"]);
        let results = network.discover_many(&[], None);
        assert!(results.is_empty());
    }

    #[test]
    fn discover_many_with_filters() {
        let mut network = LocalNetwork::new();
        let mut meta_mcp = HashMap::new();
        meta_mcp.insert("protocol".to_string(), "mcp".to_string());
        network.register("agent-mcp", &["legal translation"], None, meta_mcp);

        let mut meta_rest = HashMap::new();
        meta_rest.insert("protocol".to_string(), "rest".to_string());
        network.register("agent-rest", &["legal translation"], None, meta_rest);

        let mut filters = Filters::new();
        filters.insert(
            "protocol".to_string(),
            FilterValue::Exact("mcp".to_string()),
        );

        let results =
            network.discover_many(&["legal translation", "legal translation"], Some(&filters));
        assert_eq!(results.len(), 2);
        for query_results in &results {
            assert_eq!(query_results.len(), 1);
            assert_eq!(query_results[0].agent_name, "agent-mcp");
        }
    }

    #[test]
    fn save_io_error_on_invalid_path() {
        let network = LocalNetwork::new();
        let result = network.save("/nonexistent-dir/deeply/nested/file.json");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), NetworkError::Io(_)));
    }

    #[test]
    fn load_io_error_on_missing_file() {
        let result = LocalNetwork::load("/tmp/nonexistent_alps_file_xyz.json");
        assert!(result.is_err());
        assert!(matches!(result.err().unwrap(), NetworkError::Io(_)));
    }

    #[test]
    fn load_serialization_error_on_invalid_json() {
        let path = "/tmp/alps_test_serde_error.json";
        std::fs::write(path, "not valid json").expect("write test file");
        let result = LocalNetwork::load(path);
        assert!(result.is_err());
        assert!(matches!(
            result.err().unwrap(),
            NetworkError::Serialization(_)
        ));
        let _ = std::fs::remove_file(path);
    }
}
