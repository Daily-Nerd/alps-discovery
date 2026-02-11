// ALPS Discovery SDK — Pheromone State
//
// Minimal hypha state type for agent scoring.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Atomic u64 counter that supports Clone, PartialEq, Serialize, Deserialize.
///
/// Enables interior mutability for concurrent reads: `forwards_count` can be
/// incremented during `discover(&self)` without requiring `&mut self`.
#[derive(Debug)]
pub struct AtomicCounter(AtomicU64);

impl AtomicCounter {
    /// Create a new counter with the given initial value.
    pub fn new(val: u64) -> Self {
        Self(AtomicU64::new(val))
    }

    /// Read the current value.
    pub fn get(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }

    /// Set the value.
    pub fn set(&self, val: u64) {
        self.0.store(val, Ordering::Relaxed);
    }

    /// Atomically increment by 1.
    pub fn increment(&self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

impl Clone for AtomicCounter {
    fn clone(&self) -> Self {
        Self::new(self.get())
    }
}

impl PartialEq for AtomicCounter {
    fn eq(&self, other: &Self) -> bool {
        self.get() == other.get()
    }
}

impl Serialize for AtomicCounter {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.get().serialize(s)
    }
}

impl<'de> Deserialize<'de> for AtomicCounter {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        u64::deserialize(d).map(Self::new)
    }
}

/// Circuit breaker state for agent health tracking.
///
/// Implements a three-state circuit breaker pattern to exclude failing agents
/// from discovery results and allow recovery probes after timeout.
///
/// State transitions:
/// - Closed (normal) → Open (failing) when consecutive_pulse_timeouts >= failure_threshold
/// - Open → HalfOpen when recovery_timeout elapses (allows single probe)
/// - HalfOpen → Closed on success, or HalfOpen → Open on failure
#[derive(Debug, Clone, PartialEq, Default)]
pub enum CircuitState {
    /// Normal state - agent is healthy and included in discovery.
    #[default]
    Closed,
    /// Circuit is open - agent has failed consecutively and is excluded from discovery.
    /// Recovery probe allowed after recovery_timeout elapses.
    Open { opened_at: Instant },
    /// Half-open state - testing recovery with a single probe query.
    HalfOpen,
}

/// Configuration for circuit breaker behavior.
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Number of consecutive failures before opening circuit (default: 5).
    pub failure_threshold: u8,
    /// Duration to wait before attempting recovery probe (default: 60s).
    pub recovery_timeout: Duration,
}

impl CircuitBreakerConfig {
    /// Create a new circuit breaker config with default values.
    pub fn new() -> Self {
        Self {
            failure_threshold: 5,
            recovery_timeout: Duration::from_secs(60),
        }
    }

    /// Create a circuit breaker config with custom values.
    pub fn with_threshold_and_timeout(failure_threshold: u8, recovery_timeout: Duration) -> Self {
        Self {
            failure_threshold,
            recovery_timeout,
        }
    }
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Pheromone-relevant state of an agent slot.
///
/// Fields and their roles:
/// - `diameter`: Structural capacity / quality weight. Adjusted by feedback
///   (success → +0.01 capped at 1.0, failure → -0.05 floored at 0.1).
///   Influences CapabilityKernel and LoadBalancingKernel scoring.
/// - `tau`: Reliability signal. Incremented on success (+0.05), decayed per
///   tick (×0.995, floor TAU_FLOOR). Used as a multiplier in CapabilityKernel
///   scoring: higher tau → higher capability score for reliable agents.
/// - `sigma`: Exploration count / popularity. Incremented on success (+0.01),
///   decayed per tick (×0.99). Inversely used by NoveltyKernel: 1/(1+sigma)
///   favors less-explored agents.
/// - `consecutive_pulse_timeouts`: Failure streak counter. Incremented on
///   failure, reset on success. Used by circuit breaker to detect failing agents.
/// - `forwards_count`: Total signals routed through this slot. Used by
///   LoadBalancingKernel: diameter/(1+forwards_count) favors less-used agents.
///   Uses `AtomicCounter` for interior mutability during concurrent discovery.
/// - `conductance`: Physarum-inspired flow conductance. Increases on success,
///   decays over time. Used by PhysarumFlowKernel for bio-inspired load balancing.
/// - `circuit_state`: Circuit breaker state. Excludes failing agents from discovery.
#[derive(Debug, Clone, PartialEq)]
pub struct HyphaState {
    /// Connection diameter (strength/capacity). Adjusted by feedback.
    pub diameter: f64,
    /// Tau pheromone — tracks reliability. Influences CapabilityKernel scoring.
    pub tau: f64,
    /// Sigma pheromone — tracks exploration count / popularity.
    pub sigma: f64,
    /// Consecutive timeouts without a response.
    pub consecutive_pulse_timeouts: u8,
    /// Number of signals forwarded through this slot (for load balancing).
    /// Atomic for concurrent `discover(&self)` support.
    pub forwards_count: AtomicCounter,
    /// Physarum-inspired flow conductance (for PhysarumFlowKernel).
    pub conductance: f64,
    /// Circuit breaker state for failure exclusion.
    pub circuit_state: CircuitState,
}

impl HyphaState {
    /// Minimum floor for tau pheromone (prevents zero-trap absorbing state).
    pub const TAU_FLOOR: f64 = 0.001;

    /// Creates a new HyphaState with the given initial diameter.
    pub fn new(initial_diameter: f64) -> Self {
        Self {
            diameter: initial_diameter,
            tau: Self::TAU_FLOOR,
            sigma: 0.0,
            consecutive_pulse_timeouts: 0,
            forwards_count: AtomicCounter::new(0),
            conductance: 1.0, // Default conductance for PhysarumFlowKernel
            circuit_state: CircuitState::Closed,
        }
    }

    /// Check if the circuit breaker is open (agent should be excluded from discovery).
    pub fn is_circuit_open(&self) -> bool {
        matches!(self.circuit_state, CircuitState::Open { .. })
    }

    /// Transition circuit state after a failure.
    ///
    /// Increments consecutive_pulse_timeouts and opens circuit if threshold reached.
    pub fn record_circuit_failure(&mut self, config: &CircuitBreakerConfig) {
        self.consecutive_pulse_timeouts = self.consecutive_pulse_timeouts.saturating_add(1);
        if self.consecutive_pulse_timeouts >= config.failure_threshold {
            self.circuit_state = CircuitState::Open {
                opened_at: Instant::now(),
            };
        }
    }

    /// Transition circuit state after a success.
    ///
    /// Resets consecutive_pulse_timeouts and closes circuit.
    pub fn record_circuit_success(&mut self) {
        self.consecutive_pulse_timeouts = 0;
        self.circuit_state = CircuitState::Closed;
    }

    /// Check if recovery probe should be attempted (Open → HalfOpen transition).
    ///
    /// Called by tick() or before discovery to transition Open agents to HalfOpen
    /// after recovery_timeout has elapsed.
    pub fn check_recovery_probe(&mut self, config: &CircuitBreakerConfig) {
        if let CircuitState::Open { opened_at } = self.circuit_state {
            if opened_at.elapsed() > config.recovery_timeout {
                self.circuit_state = CircuitState::HalfOpen;
            }
        }
    }
}

// Manual Serialize/Deserialize for HyphaState to handle CircuitState (non-serializable Instant)
impl Serialize for HyphaState {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        // Serialize without circuit_state (Instant is not serializable)
        // Circuit state resets to Closed on load
        use serde::ser::SerializeStruct;
        let mut state = s.serialize_struct("HyphaState", 6)?;
        state.serialize_field("diameter", &self.diameter)?;
        state.serialize_field("tau", &self.tau)?;
        state.serialize_field("sigma", &self.sigma)?;
        state.serialize_field(
            "consecutive_pulse_timeouts",
            &self.consecutive_pulse_timeouts,
        )?;
        state.serialize_field("forwards_count", &self.forwards_count)?;
        state.serialize_field("conductance", &self.conductance)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for HyphaState {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct HyphaStateHelper {
            diameter: f64,
            tau: f64,
            sigma: f64,
            consecutive_pulse_timeouts: u8,
            forwards_count: AtomicCounter,
            conductance: f64,
        }

        let helper = HyphaStateHelper::deserialize(d)?;
        Ok(HyphaState {
            diameter: helper.diameter,
            tau: helper.tau,
            sigma: helper.sigma,
            consecutive_pulse_timeouts: helper.consecutive_pulse_timeouts,
            forwards_count: helper.forwards_count,
            conductance: helper.conductance,
            circuit_state: CircuitState::Closed, // Reset to Closed on load
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Task 13.1: CircuitState tests ---

    #[test]
    fn circuit_state_defaults_to_closed() {
        let state = HyphaState::new(1.0);
        assert_eq!(state.circuit_state, CircuitState::Closed);
        assert!(!state.is_circuit_open());
    }

    #[test]
    fn circuit_breaker_config_has_sensible_defaults() {
        let config = CircuitBreakerConfig::new();
        assert_eq!(config.failure_threshold, 5);
        assert_eq!(config.recovery_timeout, Duration::from_secs(60));
    }

    // --- Task 13.2: State transition tests ---

    #[test]
    fn circuit_opens_after_threshold_failures() {
        let mut state = HyphaState::new(1.0);
        let config = CircuitBreakerConfig::new();

        // Record failures up to threshold
        for i in 0..5 {
            state.record_circuit_failure(&config);
            if i < 4 {
                // Should still be closed until we hit threshold
                assert_eq!(state.circuit_state, CircuitState::Closed);
            }
        }

        // After 5th failure, circuit should open
        assert!(state.is_circuit_open());
        assert_eq!(state.consecutive_pulse_timeouts, 5);
    }

    #[test]
    fn circuit_closes_on_success() {
        let mut state = HyphaState::new(1.0);
        let config = CircuitBreakerConfig::new();

        // Open the circuit
        for _ in 0..5 {
            state.record_circuit_failure(&config);
        }
        assert!(state.is_circuit_open());

        // Record success
        state.record_circuit_success();

        // Circuit should be closed
        assert_eq!(state.circuit_state, CircuitState::Closed);
        assert_eq!(state.consecutive_pulse_timeouts, 0);
        assert!(!state.is_circuit_open());
    }

    #[test]
    fn circuit_transitions_to_half_open_after_timeout() {
        let mut state = HyphaState::new(1.0);
        let config = CircuitBreakerConfig::with_threshold_and_timeout(
            5,
            Duration::from_millis(10), // Very short timeout for testing
        );

        // Open the circuit
        for _ in 0..5 {
            state.record_circuit_failure(&config);
        }
        assert!(state.is_circuit_open());

        // Wait for recovery timeout
        std::thread::sleep(Duration::from_millis(15));

        // Check recovery probe
        state.check_recovery_probe(&config);

        // Should transition to HalfOpen
        assert_eq!(state.circuit_state, CircuitState::HalfOpen);
        assert!(!state.is_circuit_open()); // HalfOpen is not considered "open"
    }

    #[test]
    fn half_open_closes_on_success() {
        let mut state = HyphaState::new(1.0);
        state.circuit_state = CircuitState::HalfOpen;

        state.record_circuit_success();

        assert_eq!(state.circuit_state, CircuitState::Closed);
    }

    #[test]
    fn half_open_reopens_on_failure() {
        let mut state = HyphaState::new(1.0);
        let config = CircuitBreakerConfig::new();

        // Set to HalfOpen
        state.circuit_state = CircuitState::HalfOpen;
        state.consecutive_pulse_timeouts = 5; // Already at threshold

        // Record another failure
        state.record_circuit_failure(&config);

        // Should reopen circuit
        assert!(state.is_circuit_open());
        assert_eq!(state.consecutive_pulse_timeouts, 6);
    }

    #[test]
    fn check_recovery_probe_only_affects_open_circuits() {
        let mut state = HyphaState::new(1.0);
        let config = CircuitBreakerConfig::new();

        // Closed state should remain closed
        state.check_recovery_probe(&config);
        assert_eq!(state.circuit_state, CircuitState::Closed);

        // HalfOpen state should remain half-open
        state.circuit_state = CircuitState::HalfOpen;
        state.check_recovery_probe(&config);
        assert_eq!(state.circuit_state, CircuitState::HalfOpen);
    }

    #[test]
    fn circuit_breaker_saturates_at_max_u8() {
        let mut state = HyphaState::new(1.0);
        let config = CircuitBreakerConfig::new();

        // Hammer it with 300 failures (more than u8::MAX)
        for _ in 0..300 {
            state.record_circuit_failure(&config);
        }

        // Should saturate at 255, not wrap to 0
        assert_eq!(state.consecutive_pulse_timeouts, 255);
        assert!(state.is_circuit_open());
    }
}
