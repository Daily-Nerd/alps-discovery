// ALPS Discovery — Mycorrhizal Feedback Propagation
//
// Transitive feedback that propagates success signals to similar agents,
// enabling faster network learning through capability overlap.

use std::collections::BTreeMap;

use crate::scorer::Scorer;

use super::registry::AgentRecord;

/// Mycorrhizal feedback propagator for transitive learning.
///
/// When an agent succeeds, propagates attenuated positive feedback to agents
/// with overlapping capabilities. This enables the network to learn faster by
/// leveraging similarity: if agent A succeeds and agent B has overlapping
/// capabilities, B gets a proportional tau boost.
///
/// Inspired by mycorrhizal networks in biology where fungi connect plant roots,
/// sharing nutrients bidirectionally based on proximity and need.
#[derive(Debug, Clone)]
pub struct MycorrhizalPropagator {
    /// Attenuation factor for propagated feedback (default: 0.3).
    /// Range: [0.0, 1.0]. Set to 0.0 to disable propagation entirely.
    pub propagation_attenuation: f64,
    /// Minimum overlap threshold to trigger propagation (default: 0.3).
    /// Range: [0.0, 1.0]. Only agents with overlap >= threshold receive boosts.
    pub propagation_threshold: f64,
}

impl MycorrhizalPropagator {
    /// Create a new propagator with default configuration.
    pub fn new() -> Self {
        Self {
            propagation_attenuation: 0.3,
            propagation_threshold: 0.3,
        }
    }

    /// Create a propagator with custom configuration.
    pub fn with_config(propagation_attenuation: f64, propagation_threshold: f64) -> Self {
        Self {
            propagation_attenuation,
            propagation_threshold,
        }
    }

    /// Propagate feedback to overlapping agents.
    ///
    /// Computes capability overlap between the successful agent and all other
    /// agents using the scorer. For agents with overlap >= threshold, applies
    /// an attenuated tau boost proportional to the overlap.
    ///
    /// # Arguments
    /// * `successful_agent_id` - Name of the agent that succeeded
    /// * `successful_agent_caps` - Capabilities of the successful agent
    /// * `outcome` - Success magnitude (+1.0 for success, -1.0 for failure)
    /// * `scorer` - Scorer to compute capability overlaps
    /// * `all_agents` - Mutable reference to all agents in the network
    pub fn propagate_feedback(
        &self,
        successful_agent_id: &str,
        successful_agent_caps: &[&str],
        outcome: f64,
        scorer: &dyn Scorer,
        all_agents: &mut BTreeMap<String, AgentRecord>,
    ) {
        // Early return if propagation is disabled
        if self.propagation_attenuation == 0.0 {
            return;
        }

        // Score successful agent caps against all agents to find overlap
        let overlap_scores = match scorer.score(&successful_agent_caps.join(" ")) {
            Ok(scores) => scores,
            Err(_) => return, // If scoring fails, skip propagation
        };

        // Apply attenuated tau boost to overlapping agents
        for (agent_id, overlap) in overlap_scores {
            // Skip self
            if agent_id == successful_agent_id {
                continue;
            }

            // Apply boost if overlap exceeds threshold
            if overlap >= self.propagation_threshold {
                if let Some(record) = all_agents.get_mut(&agent_id) {
                    let boost = outcome * self.propagation_attenuation * overlap;
                    record.hypha.state.tau += boost;
                }
            }
        }
    }

    /// Check if propagation is enabled.
    pub fn is_enabled(&self) -> bool {
        self.propagation_attenuation > 0.0
    }
}

impl Default for MycorrhizalPropagator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::chemistry::Chemistry;
    use crate::core::config::LshConfig;
    use crate::core::hyphae::Hypha;
    use crate::core::pheromone::{AtomicCounter, HyphaState};
    use crate::core::types::{HyphaId, PeerAddr};
    use crate::scorer::{MinHashScorer, Scorer};
    use std::collections::BTreeMap;
    use std::time::Instant;

    fn create_test_agent(name: &str, capabilities: &[&str], tau: f64) -> AgentRecord {
        AgentRecord {
            capabilities: capabilities.iter().map(|s| s.to_string()).collect(),
            endpoint: None,
            metadata: std::collections::HashMap::new(),
            hypha: Hypha {
                id: HyphaId([0; 32]),
                peer: PeerAddr(format!("local://{}", name)),
                state: HyphaState {
                    diameter: 1.0,
                    tau,
                    sigma: 0.0,
                    consecutive_pulse_timeouts: 0,
                    forwards_count: AtomicCounter::new(0),
                    conductance: 1.0,
                },
                chemistry: Chemistry::new(),
                last_activity: Instant::now(),
            },
            feedback: super::super::registry::FeedbackIndex::new(),
        }
    }

    #[test]
    fn test_propagation_disabled_when_attenuation_zero() {
        // GIVEN: A propagator with attenuation = 0.0
        let propagator = MycorrhizalPropagator::with_config(0.0, 0.3);
        let mut agents = BTreeMap::new();

        agents.insert(
            "agent-a".to_string(),
            create_test_agent("agent-a", &["translate"], 0.5),
        );
        agents.insert(
            "agent-b".to_string(),
            create_test_agent("agent-b", &["convert"], 0.5),
        );

        let initial_tau_b = agents["agent-b"].hypha.state.tau;

        // Create a scorer (won't be used because propagation is disabled)
        let scorer = Box::new(MinHashScorer::new(LshConfig::default()));

        // WHEN: Propagating feedback
        propagator.propagate_feedback("agent-a", &["translate"], 1.0, scorer.as_ref(), &mut agents);

        // THEN: No tau changes should occur
        assert_eq!(
            agents["agent-b"].hypha.state.tau, initial_tau_b,
            "Tau should not change when attenuation is 0.0"
        );
        assert!(!propagator.is_enabled());
    }

    #[test]
    fn test_propagation_boosts_similar_agent() {
        // GIVEN: Two agents with overlapping capabilities
        let propagator = MycorrhizalPropagator::new();
        let mut agents = BTreeMap::new();

        agents.insert(
            "translate-agent".to_string(),
            create_test_agent(
                "translate-agent",
                &["translate text", "convert language"],
                0.5,
            ),
        );
        agents.insert(
            "convert-agent".to_string(),
            create_test_agent(
                "convert-agent",
                &["convert text", "transform language"],
                0.5,
            ),
        );

        let initial_tau = agents["convert-agent"].hypha.state.tau;

        // Create and index scorer
        let mut scorer = Box::new(MinHashScorer::new(LshConfig::default()));
        scorer.index_capabilities("translate-agent", &["translate text", "convert language"]);
        scorer.index_capabilities("convert-agent", &["convert text", "transform language"]);

        // WHEN: Propagating feedback from translate-agent
        propagator.propagate_feedback(
            "translate-agent",
            &["translate text", "convert language"],
            1.0,
            scorer.as_ref(),
            &mut agents,
        );

        // THEN: Similar agent should have increased tau
        assert!(
            agents["convert-agent"].hypha.state.tau > initial_tau,
            "Similar agent should receive tau boost via propagation"
        );
    }

    #[test]
    fn test_propagation_skips_self() {
        // GIVEN: A single agent
        let propagator = MycorrhizalPropagator::new();
        let mut agents = BTreeMap::new();

        agents.insert(
            "agent-a".to_string(),
            create_test_agent("agent-a", &["translate"], 0.5),
        );

        let initial_tau = agents["agent-a"].hypha.state.tau;

        // Create and index scorer
        let mut scorer = Box::new(MinHashScorer::new(LshConfig::default()));
        scorer.index_capabilities("agent-a", &["translate"]);

        // WHEN: Propagating feedback
        propagator.propagate_feedback("agent-a", &["translate"], 1.0, scorer.as_ref(), &mut agents);

        // THEN: Agent should not boost its own tau
        assert_eq!(
            agents["agent-a"].hypha.state.tau, initial_tau,
            "Agent should not propagate feedback to itself"
        );
    }

    #[test]
    fn test_propagation_respects_threshold() {
        // GIVEN: A propagator with high threshold
        let propagator = MycorrhizalPropagator::with_config(0.3, 0.9); // Very high threshold
        let mut agents = BTreeMap::new();

        agents.insert(
            "agent-a".to_string(),
            create_test_agent("agent-a", &["translate"], 0.5),
        );
        agents.insert(
            "agent-b".to_string(),
            create_test_agent("agent-b", &["somewhat different"], 0.5),
        );

        let initial_tau_b = agents["agent-b"].hypha.state.tau;

        // Create and index scorer
        let mut scorer = Box::new(MinHashScorer::new(LshConfig::default()));
        scorer.index_capabilities("agent-a", &["translate"]);
        scorer.index_capabilities("agent-b", &["somewhat different"]);

        // WHEN: Propagating feedback
        propagator.propagate_feedback("agent-a", &["translate"], 1.0, scorer.as_ref(), &mut agents);

        // THEN: Low-overlap agent should not receive boost
        assert_eq!(
            agents["agent-b"].hypha.state.tau, initial_tau_b,
            "Agent with overlap below threshold should not receive boost"
        );
    }

    #[test]
    fn test_propagation_boost_proportional_to_overlap() {
        // GIVEN: Two agents with different overlaps
        let propagator = MycorrhizalPropagator::with_config(0.5, 0.1); // Low threshold
        let mut agents = BTreeMap::new();

        agents.insert(
            "source-agent".to_string(),
            create_test_agent("source-agent", &["translate language convert"], 0.5),
        );
        agents.insert(
            "high-overlap-agent".to_string(),
            create_test_agent("high-overlap-agent", &["translate language"], 0.5),
        );
        agents.insert(
            "low-overlap-agent".to_string(),
            create_test_agent("low-overlap-agent", &["translate"], 0.5),
        );

        // Create and index scorer
        let mut scorer = Box::new(MinHashScorer::new(LshConfig::default()));
        scorer.index_capabilities("source-agent", &["translate language convert"]);
        scorer.index_capabilities("high-overlap-agent", &["translate language"]);
        scorer.index_capabilities("low-overlap-agent", &["translate"]);

        // WHEN: Propagating feedback
        propagator.propagate_feedback(
            "source-agent",
            &["translate language convert"],
            1.0,
            scorer.as_ref(),
            &mut agents,
        );

        let high_boost = agents["high-overlap-agent"].hypha.state.tau - 0.5; // Initial tau was 0.5
        let low_boost = agents["low-overlap-agent"].hypha.state.tau - 0.5;

        // THEN: Higher overlap should result in larger boost
        assert!(
            high_boost > low_boost,
            "Agent with higher overlap should receive larger boost"
        );
    }
}
