// ALPS Discovery SDK — PyO3 bindings
//
// Thin #[pyclass] wrappers around the Rust LocalNetwork.

use pyo3::prelude::*;

use crate::network::LocalNetwork;

/// Local agent discovery network.
///
/// Creates an in-process routing engine that discovers agents by
/// capability matching using multi-kernel voting.
///
/// Example:
///     network = LocalNetwork()
///     network.register("translate-agent", ["legal translation", "EN-DE"])
///     results = network.discover("translate legal contract")
///     print(results[0].agent_name)  # "translate-agent"
#[pyclass(name = "LocalNetwork")]
pub struct PyLocalNetwork {
    inner: LocalNetwork,
}

#[pymethods]
impl PyLocalNetwork {
    /// Create a new empty LocalNetwork.
    #[new]
    fn new() -> Self {
        Self {
            inner: LocalNetwork::new(),
        }
    }

    /// Register an agent with its capabilities.
    ///
    /// Args:
    ///     name: Unique agent identifier.
    ///     capabilities: List of capability description strings.
    fn register(&mut self, name: &str, capabilities: Vec<String>) {
        let caps: Vec<&str> = capabilities.iter().map(|s| s.as_str()).collect();
        self.inner.register(name, &caps);
    }

    /// Remove an agent from the network.
    ///
    /// Returns True if the agent was found and removed.
    fn deregister(&mut self, name: &str) -> bool {
        self.inner.deregister(name)
    }

    /// Discover agents matching a query string.
    ///
    /// Returns a ranked list of DiscoveryResult objects, best match first.
    /// The ranking uses multi-kernel voting that considers capability
    /// similarity, load balancing, and exploration.
    fn discover(&mut self, query: &str) -> Vec<PyDiscoveryResult> {
        self.inner
            .discover(query)
            .into_iter()
            .map(|r| PyDiscoveryResult {
                agent_name: r.agent_name,
                similarity: r.similarity,
                score: r.score,
            })
            .collect()
    }

    /// Record a successful interaction with an agent.
    ///
    /// Improves the agent's ranking in future queries.
    fn record_success(&mut self, agent_name: &str) {
        self.inner.record_success(agent_name);
    }

    /// Record a failed interaction with an agent.
    ///
    /// Reduces the agent's ranking in future queries.
    fn record_failure(&mut self, agent_name: &str) {
        self.inner.record_failure(agent_name);
    }

    /// Number of registered agents.
    #[getter]
    fn agent_count(&self) -> usize {
        self.inner.agent_count()
    }

    /// List of all registered agent names.
    fn agents(&self) -> Vec<String> {
        self.inner.agents()
    }
}

/// A single discovery result.
///
/// Attributes:
///     agent_name: The matched agent's name.
///     similarity: Raw capability similarity [0.0, 1.0] based on MinHash overlap.
///     score: Combined routing score from multi-kernel voting. Incorporates
///         similarity, load balancing, novelty, and feedback from record_success/failure.
#[pyclass(name = "DiscoveryResult")]
#[derive(Clone)]
pub struct PyDiscoveryResult {
    #[pyo3(get)]
    pub agent_name: String,
    #[pyo3(get)]
    pub similarity: f64,
    #[pyo3(get)]
    pub score: f64,
}

#[pymethods]
impl PyDiscoveryResult {
    fn __repr__(&self) -> String {
        format!(
            "DiscoveryResult(agent='{}', similarity={:.3}, score={:.3})",
            self.agent_name, self.similarity, self.score
        )
    }
}
