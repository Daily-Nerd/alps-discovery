// ALPS Discovery SDK — Pheromone State
//
// Minimal hypha state type for agent scoring.

use std::sync::atomic::{AtomicU64, Ordering};

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
///   failure, reset on success. Available for circuit-breaker patterns.
/// - `forwards_count`: Total signals routed through this slot. Used by
///   LoadBalancingKernel: diameter/(1+forwards_count) favors less-used agents.
///   Uses `AtomicCounter` for interior mutability during concurrent discovery.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
        }
    }
}
