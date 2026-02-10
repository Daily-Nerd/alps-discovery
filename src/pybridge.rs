// ALPS Discovery SDK — PyO3 bindings
//
// Thin #[pyclass] wrappers around the Rust LocalNetwork.
// Stores optional `invoke` callables on the Python side (not in Rust core).

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::IntoPyObject;

use crate::core::config::LshConfig;
use crate::core::enzyme::SLNEnzymeConfig;
use crate::network::LocalNetwork;
use crate::scorer::Scorer;

/// Python scorer wrapper implementing the Rust Scorer trait.
///
/// Wraps a Python object with methods:
/// - `index_capabilities(agent_id: str, capabilities: list[str])`
/// - `remove_agent(agent_id: str)`
/// - `score(query: str) -> list[tuple[str, float]]`
struct PyScorer {
    inner: Py<PyAny>,
}

// Safety: PyScorer is Send+Sync because we only access the Python object
// while holding the GIL.
unsafe impl Send for PyScorer {}
unsafe impl Sync for PyScorer {}

impl Scorer for PyScorer {
    fn index_capabilities(&mut self, agent_id: &str, capabilities: &[&str]) {
        Python::attach(|py| {
            let caps: Vec<String> = capabilities.iter().map(|s| s.to_string()).collect();
            if let Err(e) = self
                .inner
                .call_method1(py, "index_capabilities", (agent_id, caps))
            {
                eprintln!(
                    "alps-discovery: scorer.index_capabilities() failed for '{}': {}",
                    agent_id, e
                );
            }
        });
    }

    fn remove_agent(&mut self, agent_id: &str) {
        Python::attach(|py| {
            if let Err(e) = self.inner.call_method1(py, "remove_agent", (agent_id,)) {
                eprintln!(
                    "alps-discovery: scorer.remove_agent() failed for '{}': {}",
                    agent_id, e
                );
            }
        });
    }

    fn score(&self, query: &str) -> Vec<(String, f64)> {
        Python::attach(|py| match self.inner.call_method1(py, "score", (query,)) {
            Ok(result) => match result.extract::<Vec<(String, f64)>>(py) {
                Ok(scores) => scores,
                Err(e) => {
                    eprintln!(
                        "alps-discovery: scorer.score() returned invalid type: {}",
                        e
                    );
                    Vec::new()
                }
            },
            Err(e) => {
                eprintln!("alps-discovery: scorer.score() failed: {}", e);
                Vec::new()
            }
        })
    }
}

/// Local agent discovery network.
///
/// Creates an in-process routing engine that discovers agents by
/// capability matching using multi-kernel voting.
///
/// Args:
///     similarity_threshold: Minimum similarity for results (default: 0.1).
///         Results below this are filtered as noise.
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
    ///
    /// Args:
    ///     similarity_threshold: Minimum similarity to include in results (default: 0.1).
    ///     scorer: Optional custom scorer object implementing index_capabilities(),
    ///         remove_agent(), and score() methods. Overrides the default MinHash scorer.
    #[new]
    #[pyo3(signature = (*, similarity_threshold=None, scorer=None))]
    fn new(similarity_threshold: Option<f64>, scorer: Option<Py<PyAny>>) -> Self {
        let inner = if let Some(py_scorer) = scorer {
            let scorer = PyScorer { inner: py_scorer };
            LocalNetwork::with_scorer(Box::new(scorer))
        } else {
            let mut lsh_config = LshConfig::default();
            if let Some(t) = similarity_threshold {
                lsh_config.similarity_threshold = t;
            }
            LocalNetwork::with_config(SLNEnzymeConfig::default(), lsh_config)
        };
        Self {
            inner,
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
        self.inner.register(
            name,
            &caps,
            endpoint.as_deref(),
            metadata.unwrap_or_default(),
        );

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
    ///
    /// When ``explain=True``, returns ExplainedResult objects with full
    /// scoring breakdown (raw_similarity, diameter, feedback_factor,
    /// final_score) for debugging and understanding routing decisions.
    ///
    /// Args:
    ///     query: Natural-language capability query.
    ///     filters: Optional dict of metadata filters. Values can be:
    ///         - str: exact match (e.g. ``{"protocol": "mcp"}``)
    ///         - ``{"$in": ["a", "b"]}``: value must be one of the listed options
    ///         - ``{"$lt": 100.0}``: numeric less-than
    ///         - ``{"$gt": 1.5}``: numeric greater-than
    ///         - ``{"$contains": "sub"}``: substring containment
    ///     explain: If True, return ExplainedResult with scoring breakdown.
    #[pyo3(signature = (query, *, filters=None, explain=false))]
    fn discover(
        &mut self,
        py: Python<'_>,
        query: &str,
        filters: Option<HashMap<String, Py<PyAny>>>,
        explain: bool,
    ) -> PyResult<Py<PyAny>> {
        let rust_filters = if let Some(py_filters) = filters {
            let mut f = crate::network::Filters::new();
            for (key, value) in py_filters {
                let filter_value = Self::parse_filter_value(py, &value)?;
                f.insert(key, filter_value);
            }
            Some(f)
        } else {
            None
        };

        if explain {
            let results: Vec<PyExplainedResult> = self
                .inner
                .discover_explained(query, rust_filters.as_ref())
                .into_iter()
                .map(|r| PyExplainedResult {
                    agent_name: r.agent_name,
                    raw_similarity: r.raw_similarity,
                    diameter: r.diameter,
                    feedback_factor: r.feedback_factor,
                    final_score: r.final_score,
                    endpoint: r.endpoint,
                    metadata: r.metadata,
                })
                .collect();
            Ok(results.into_pyobject(py)?.into_any().unbind())
        } else {
            let results: Vec<PyDiscoveryResult> = self
                .inner
                .discover_filtered(query, rust_filters.as_ref())
                .into_iter()
                .map(|r| {
                    let invoke = self.invocables.get(&r.agent_name).map(|f| f.clone_ref(py));
                    PyDiscoveryResult {
                        agent_name: r.agent_name,
                        similarity: r.similarity,
                        score: r.score,
                        endpoint: r.endpoint,
                        metadata: r.metadata,
                        invoke,
                    }
                })
                .collect();
            Ok(results.into_pyobject(py)?.into_any().unbind())
        }
    }

    /// Record a successful interaction with an agent.
    ///
    /// If `query` is provided, future queries similar to it will boost
    /// this agent's ranking — without affecting unrelated query types.
    /// Always applies a small global ranking improvement regardless.
    ///
    /// Args:
    ///     agent_name: The agent that succeeded.
    ///     query: The query that led to this interaction (recommended).
    #[pyo3(signature = (agent_name, *, query=None))]
    fn record_success(&mut self, agent_name: &str, query: Option<&str>) {
        self.inner.record_success(agent_name, query);
    }

    /// Record a failed interaction with an agent.
    ///
    /// If `query` is provided, future queries similar to it will penalize
    /// this agent's ranking — without affecting unrelated query types.
    ///
    /// Args:
    ///     agent_name: The agent that failed.
    ///     query: The query that led to this interaction (recommended).
    #[pyo3(signature = (agent_name, *, query=None))]
    fn record_failure(&mut self, agent_name: &str, query: Option<&str>) {
        self.inner.record_failure(agent_name, query);
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

    /// Save the network state to a JSON file.
    ///
    /// Args:
    ///     path: File path to save to (e.g. "state.json").
    fn save(&self, path: &str) -> PyResult<()> {
        self.inner
            .save(path)
            .map_err(pyo3::exceptions::PyIOError::new_err)
    }

    /// Load network state from a previously saved JSON file.
    ///
    /// Returns a new LocalNetwork with all agents, scoring state, and
    /// feedback history restored. Agent capabilities are re-indexed.
    ///
    /// Args:
    ///     path: File path to load from (e.g. "state.json").
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner = LocalNetwork::load(path).map_err(pyo3::exceptions::PyIOError::new_err)?;
        Ok(Self {
            inner,
            invocables: HashMap::new(),
        })
    }
}

impl PyLocalNetwork {
    /// Parse a Python filter value into a Rust FilterValue.
    fn parse_filter_value(
        py: Python<'_>,
        value: &Py<PyAny>,
    ) -> PyResult<crate::network::FilterValue> {
        let value = value.bind(py);

        // Try as dict first (operator syntax).
        if let Ok(dict) = value.cast::<pyo3::types::PyDict>() {
            if let Some(v) = dict.get_item("$in")? {
                let options: Vec<String> = v.extract()?;
                return Ok(crate::network::FilterValue::OneOf(options));
            }
            if let Some(v) = dict.get_item("$lt")? {
                let threshold: f64 = v.extract()?;
                return Ok(crate::network::FilterValue::LessThan(threshold));
            }
            if let Some(v) = dict.get_item("$gt")? {
                let threshold: f64 = v.extract()?;
                return Ok(crate::network::FilterValue::GreaterThan(threshold));
            }
            if let Some(v) = dict.get_item("$contains")? {
                let substring: String = v.extract()?;
                return Ok(crate::network::FilterValue::Contains(substring));
            }
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Filter dict must contain one of: $in, $lt, $gt, $contains",
            ));
        }

        // Otherwise treat as exact string match.
        let s: String = value.extract()?;
        Ok(crate::network::FilterValue::Exact(s))
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

/// Extended discovery result with full scoring breakdown.
///
/// Attributes:
///     agent_name: The matched agent's name.
///     raw_similarity: Raw capability similarity from the scorer [0.0, 1.0].
///     diameter: Agent's routing diameter (weight from feedback history).
///     feedback_factor: Per-query feedback adjustment [-1.0, 1.0].
///     final_score: Combined routing score.
///     endpoint: Agent URI/URL if provided at registration, else None.
///     metadata: Dict of key-value pairs if provided at registration, else {}.
#[pyclass(name = "ExplainedResult")]
pub struct PyExplainedResult {
    #[pyo3(get)]
    pub agent_name: String,
    #[pyo3(get)]
    pub raw_similarity: f64,
    #[pyo3(get)]
    pub diameter: f64,
    #[pyo3(get)]
    pub feedback_factor: f64,
    #[pyo3(get)]
    pub final_score: f64,
    #[pyo3(get)]
    pub endpoint: Option<String>,
    #[pyo3(get)]
    pub metadata: HashMap<String, String>,
}

#[pymethods]
impl PyExplainedResult {
    fn __repr__(&self) -> String {
        format!(
            "ExplainedResult(agent='{}', sim={:.3}, diameter={:.3}, feedback={:.3}, score={:.3})",
            self.agent_name,
            self.raw_similarity,
            self.diameter,
            self.feedback_factor,
            self.final_score
        )
    }
}
