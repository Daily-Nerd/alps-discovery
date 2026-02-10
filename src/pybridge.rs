// ALPS Discovery SDK — PyO3 bindings
//
// Thin #[pyclass] wrappers around the Rust LocalNetwork.
// Stores optional `invoke` callables on the Python side (not in Rust core).

use std::collections::HashMap;

use pyo3::prelude::*;

use crate::network::LocalNetwork;

/// Local agent discovery network.
///
/// Creates an in-process routing engine that discovers agents by
/// capability matching using multi-kernel voting.
///
/// Example:
///     network = LocalNetwork()
///     network.register("translate-agent", ["legal translation", "EN-DE"],
///                       endpoint="http://localhost:8080/translate",
///                       metadata={"protocol": "mcp"})
///     results = network.discover("translate legal contract")
///     print(results[0].agent_name, results[0].endpoint)
#[pyclass(name = "LocalNetwork")]
pub struct PyLocalNetwork {
    inner: LocalNetwork,
    /// Optional invoke callables keyed by agent name (Python-only).
    invocables: HashMap<String, Py<PyAny>>,
}

#[pymethods]
impl PyLocalNetwork {
    /// Create a new empty LocalNetwork.
    #[new]
    fn new() -> Self {
        Self {
            inner: LocalNetwork::new(),
            invocables: HashMap::new(),
        }
    }

    /// Register an agent with its capabilities.
    ///
    /// Args:
    ///     name: Unique agent identifier.
    ///     capabilities: List of capability description strings.
    ///     endpoint: Optional URI/URL for invoking the agent (e.g. MCP server URL).
    ///     metadata: Optional dict of key-value pairs (protocol, version, framework, etc.).
    ///     invoke: Optional callable for local single-process invocation convenience.
    #[pyo3(signature = (name, capabilities, *, endpoint=None, metadata=None, invoke=None))]
    fn register(
        &mut self,
        name: &str,
        capabilities: Vec<String>,
        endpoint: Option<String>,
        metadata: Option<HashMap<String, String>>,
        invoke: Option<Py<PyAny>>,
    ) {
        let caps: Vec<&str> = capabilities.iter().map(|s| s.as_str()).collect();
        self.inner
            .register(name, &caps, endpoint.as_deref(), metadata.unwrap_or_default());

        if let Some(callable) = invoke {
            self.invocables.insert(name.to_string(), callable);
        } else {
            self.invocables.remove(name);
        }
    }

    /// Remove an agent from the network.
    ///
    /// Returns True if the agent was found and removed.
    fn deregister(&mut self, name: &str) -> bool {
        self.invocables.remove(name);
        self.inner.deregister(name)
    }

    /// Discover agents matching a query string.
    ///
    /// Returns a ranked list of DiscoveryResult objects, best match first.
    /// Each result includes endpoint and metadata if they were provided
    /// at registration, plus an invoke callable if one was registered.
    fn discover(&mut self, py: Python<'_>, query: &str) -> Vec<PyDiscoveryResult> {
        self.inner
            .discover(query)
            .into_iter()
            .map(|r| {
                let invoke = self
                    .invocables
                    .get(&r.agent_name)
                    .map(|f| f.clone_ref(py));
                PyDiscoveryResult {
                    agent_name: r.agent_name,
                    similarity: r.similarity,
                    score: r.score,
                    endpoint: r.endpoint,
                    metadata: r.metadata,
                    invoke,
                }
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
///     score: Combined routing score (similarity x diameter, adjusted by feedback).
///     endpoint: Agent URI/URL if provided at registration, else None.
///     metadata: Dict of key-value pairs if provided at registration, else {}.
///     invoke: Callable if provided at registration, else None.
#[pyclass(name = "DiscoveryResult")]
pub struct PyDiscoveryResult {
    #[pyo3(get)]
    pub agent_name: String,
    #[pyo3(get)]
    pub similarity: f64,
    #[pyo3(get)]
    pub score: f64,
    #[pyo3(get)]
    pub endpoint: Option<String>,
    #[pyo3(get)]
    pub metadata: HashMap<String, String>,
    #[pyo3(get)]
    pub invoke: Option<Py<PyAny>>,
}

#[pymethods]
impl PyDiscoveryResult {
    fn __repr__(&self) -> String {
        let ep = match &self.endpoint {
            Some(e) => format!(", endpoint='{}'", e),
            None => String::new(),
        };
        format!(
            "DiscoveryResult(agent='{}', similarity={:.3}, score={:.3}{})",
            self.agent_name, self.similarity, self.score, ep
        )
    }
}
