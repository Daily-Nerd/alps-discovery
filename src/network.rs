// ALPS Discovery SDK — LocalNetwork
//
// In-process agent discovery using multi-kernel voting.
// No networking, no decay, no membrane — just the routing engine
// applied to Chemistry-based capability matching.

use std::collections::{BTreeMap, HashMap};
use std::time::{Duration, Instant};

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

/// Per-agent record stored in the network.
struct AgentRecord {
    capabilities: Vec<String>,
    cap_signatures: Vec<[u8; 64]>,
    endpoint: Option<String>,
    metadata: HashMap<String, String>,
    hypha: Hypha,
}

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
    /// Empty spore (discovery only).
    spore: Spore,
    /// Static membrane state (passthrough).
    membrane_state: MembraneState,
}

impl LocalNetwork {
    /// Creates a new empty LocalNetwork with default configuration.
    pub fn new() -> Self {
        Self::with_config(SLNEnzymeConfig::default())
    }

    /// Creates a new LocalNetwork with custom enzyme configuration.
    pub fn with_config(config: SLNEnzymeConfig) -> Self {
        Self {
            agents: BTreeMap::new(),
            enzyme: SLNEnzyme::with_discovery_kernels(config),
            lsh_config: LshConfig::default(),
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

        let mut chemistry = Chemistry::default();
        let mut cap_signatures = Vec::with_capacity(capabilities.len());
        for cap in capabilities {
            let sig = compute_semantic_signature(cap.as_bytes(), &self.lsh_config);
            chemistry.deposit(&sig);
            cap_signatures.push(sig);
        }

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
                cap_signatures,
                endpoint: endpoint.map(|s| s.to_string()),
                metadata,
                hypha,
            },
        );
    }

    /// Deregister an agent by name. Returns true if found and removed.
    pub fn deregister(&mut self, name: &str) -> bool {
        self.agents.remove(name).is_some()
    }

    /// Discover agents matching a natural-language query.
    ///
    /// Returns a ranked list of all agents with similarity > 0, sorted by
    /// `score = similarity × diameter`. The diameter incorporates feedback
    /// from `record_success` / `record_failure`.
    pub fn discover(&mut self, query: &str) -> Vec<DiscoveryResult> {
        if self.agents.is_empty() {
            return Vec::new();
        }

        let query_sig = compute_query_signature(query.as_bytes(), &self.lsh_config);

        // Score all agents: max per-capability similarity × diameter.
        // Using max over individual capability signatures avoids the dilution
        // problem where element-wise MIN accumulation merges unrelated capabilities.
        let mut results: Vec<DiscoveryResult> = self
            .agents
            .iter()
            .map(|(name, record)| {
                let sim = record
                    .cap_signatures
                    .iter()
                    .map(|sig| MinHasher::similarity(sig, &query_sig.minhash))
                    .fold(0.0f64, f64::max);
                let score = sim * record.hypha.state.diameter;
                DiscoveryResult {
                    agent_name: name.clone(),
                    similarity: sim,
                    score,
                    endpoint: record.endpoint.clone(),
                    metadata: record.metadata.clone(),
                }
            })
            .filter(|r| r.similarity > 0.0)
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

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
    pub fn record_success(&mut self, agent_name: &str) {
        if let Some(record) = self.agents.get_mut(agent_name) {
            record.hypha.state.tau += 0.05;
            record.hypha.state.sigma += 0.01;
            record.hypha.state.diameter = (record.hypha.state.diameter + 0.01).min(1.0);
            record.hypha.state.consecutive_pulse_timeouts = 0;
        }
    }

    /// Record a failed interaction with an agent.
    pub fn record_failure(&mut self, agent_name: &str) {
        if let Some(record) = self.agents.get_mut(agent_name) {
            record.hypha.state.consecutive_pulse_timeouts =
                record.hypha.state.consecutive_pulse_timeouts.saturating_add(1);
            record.hypha.state.diameter = (record.hypha.state.diameter - 0.05).max(0.1);
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
        reg(&mut network, "translate-agent", &["legal translation", "EN-DE", "EN-FR"]);
        let results = network.discover("legal translation");
        assert!(!results.is_empty());
        assert_eq!(results[0].agent_name, "translate-agent");
        assert!(results[0].similarity > 0.0);
    }

    #[test]
    fn discover_ranks_more_similar_agent_higher() {
        let mut network = LocalNetwork::new();
        reg(&mut network, "translate-agent", &["legal translation", "EN-DE", "EN-FR"]);
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
            network.record_success("agent-a");
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
            network.record_failure("agent-a");
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

        let translate = results.iter().find(|r| r.agent_name == "translate-agent").unwrap();
        assert_eq!(
            translate.endpoint.as_deref(),
            Some("http://localhost:8080/translate")
        );
        assert_eq!(translate.metadata.get("protocol").map(|s| s.as_str()), Some("mcp"));
        assert_eq!(translate.metadata.get("version").map(|s| s.as_str()), Some("1.0"));

        let bare = results.iter().find(|r| r.agent_name == "bare-agent").unwrap();
        assert!(bare.endpoint.is_none());
        assert!(bare.metadata.is_empty());
    }
}
