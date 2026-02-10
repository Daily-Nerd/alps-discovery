// ALPS Discovery — Discovery Replay / Time-Travel Debugging
//
// Append-only event log enabling post-hoc analysis of discovery decisions:
// "why was this agent chosen?" Addresses observability gap in agent orchestration.

use std::time::Instant;

/// A single timestamped event in the discovery replay log.
#[derive(Debug, Clone)]
pub struct DiscoveryEvent {
    /// Monotonic timestamp (relative to process start).
    pub timestamp: Instant,
    /// Event payload.
    pub kind: EventKind,
}

/// The different kinds of discovery events.
#[derive(Debug, Clone)]
pub enum EventKind {
    /// A discovery query was submitted.
    QuerySubmitted { query: String },
    /// An agent was scored during discovery.
    AgentScored {
        query: String,
        agent_name: String,
        raw_similarity: f64,
        enzyme_score: f64,
        feedback_factor: f64,
        final_score: f64,
    },
    /// Feedback was recorded for an agent.
    FeedbackRecorded {
        agent_name: String,
        query: Option<String>,
        outcome: f64,
    },
    /// Temporal tick was applied.
    TickApplied,
}

/// Append-only event log for discovery replay and debugging.
///
/// Records all discovery events with monotonic timestamps. Enables
/// post-hoc analysis: "why was agent X chosen for query Y?"
///
/// The log is in-memory and bounded by `max_events`. When the limit
/// is reached, oldest events are discarded (ring buffer behavior).
pub struct ReplayLog {
    events: Vec<DiscoveryEvent>,
    max_events: usize,
    enabled: bool,
}

impl ReplayLog {
    /// Create a new replay log with the given capacity.
    pub fn new(max_events: usize) -> Self {
        Self {
            events: Vec::with_capacity(max_events.min(1024)),
            max_events,
            enabled: true,
        }
    }

    /// Create a disabled replay log (no-op recording).
    pub fn disabled() -> Self {
        Self {
            events: Vec::new(),
            max_events: 0,
            enabled: false,
        }
    }

    /// Returns true if replay logging is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Record an event.
    pub fn record(&mut self, kind: EventKind) {
        if !self.enabled {
            return;
        }
        if self.events.len() >= self.max_events {
            // Remove oldest half to amortize cost.
            let half = self.max_events / 2;
            self.events.drain(..half);
        }
        self.events.push(DiscoveryEvent {
            timestamp: Instant::now(),
            kind,
        });
    }

    /// Returns all events in chronological order.
    pub fn events(&self) -> &[DiscoveryEvent] {
        &self.events
    }

    /// Returns events for a specific query string.
    pub fn query_history(&self, query: &str) -> Vec<&DiscoveryEvent> {
        self.events
            .iter()
            .filter(|e| match &e.kind {
                EventKind::QuerySubmitted { query: q } => q == query,
                EventKind::AgentScored { query: q, .. } => q == query,
                _ => false,
            })
            .collect()
    }

    /// Returns events involving a specific agent.
    pub fn agent_history(&self, agent_name: &str) -> Vec<&DiscoveryEvent> {
        self.events
            .iter()
            .filter(|e| match &e.kind {
                EventKind::AgentScored { agent_name: a, .. } => a == agent_name,
                EventKind::FeedbackRecorded { agent_name: a, .. } => a == agent_name,
                _ => false,
            })
            .collect()
    }

    /// Clear all events.
    pub fn clear(&mut self) {
        self.events.clear();
    }

    /// Number of recorded events.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Returns true if no events recorded.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_and_retrieve_events() {
        let mut log = ReplayLog::new(100);
        log.record(EventKind::QuerySubmitted {
            query: "legal translation".to_string(),
        });
        log.record(EventKind::AgentScored {
            query: "legal translation".to_string(),
            agent_name: "agent-a".to_string(),
            raw_similarity: 0.8,
            enzyme_score: 0.6,
            feedback_factor: 0.1,
            final_score: 0.7,
        });
        assert_eq!(log.len(), 2);
        assert!(!log.is_empty());
    }

    #[test]
    fn query_history_filters_by_query() {
        let mut log = ReplayLog::new(100);
        log.record(EventKind::QuerySubmitted {
            query: "legal translation".to_string(),
        });
        log.record(EventKind::QuerySubmitted {
            query: "data processing".to_string(),
        });
        log.record(EventKind::AgentScored {
            query: "legal translation".to_string(),
            agent_name: "agent-a".to_string(),
            raw_similarity: 0.8,
            enzyme_score: 0.6,
            feedback_factor: 0.0,
            final_score: 0.7,
        });

        let history = log.query_history("legal translation");
        assert_eq!(history.len(), 2); // QuerySubmitted + AgentScored
    }

    #[test]
    fn agent_history_filters_by_agent() {
        let mut log = ReplayLog::new(100);
        log.record(EventKind::AgentScored {
            query: "q1".to_string(),
            agent_name: "agent-a".to_string(),
            raw_similarity: 0.8,
            enzyme_score: 0.6,
            feedback_factor: 0.0,
            final_score: 0.7,
        });
        log.record(EventKind::FeedbackRecorded {
            agent_name: "agent-a".to_string(),
            query: Some("q1".to_string()),
            outcome: 1.0,
        });
        log.record(EventKind::FeedbackRecorded {
            agent_name: "agent-b".to_string(),
            query: Some("q2".to_string()),
            outcome: -1.0,
        });

        let history = log.agent_history("agent-a");
        assert_eq!(history.len(), 2); // AgentScored + FeedbackRecorded
    }

    #[test]
    fn max_events_evicts_oldest() {
        let mut log = ReplayLog::new(10);
        for i in 0..20 {
            log.record(EventKind::QuerySubmitted {
                query: format!("q{}", i),
            });
        }
        assert!(log.len() <= 10, "should evict to stay within max_events");
    }

    #[test]
    fn disabled_log_is_noop() {
        let mut log = ReplayLog::disabled();
        log.record(EventKind::QuerySubmitted {
            query: "test".to_string(),
        });
        assert_eq!(log.len(), 0);
        assert!(!log.is_enabled());
    }

    #[test]
    fn clear_removes_all_events() {
        let mut log = ReplayLog::new(100);
        log.record(EventKind::TickApplied);
        assert_eq!(log.len(), 1);
        log.clear();
        assert_eq!(log.len(), 0);
    }
}
