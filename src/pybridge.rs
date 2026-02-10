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

    fn score(&self, query: &str) -> Result<Vec<(String, f64)>, String> {
        Python::attach(|py| match self.inner.call_method1(py, "score", (query,)) {
            Ok(result) => match result.extract::<Vec<(String, f64)>>(py) {
                Ok(scores) => Ok(scores),
                Err(e) => Err(format!("scorer.score() returned invalid type: {}", e)),
            },
            Err(e) => Err(format!("scorer.score() failed: {}", e)),
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
    /// When ``with_confidence=True``, returns a DiscoveryResponse with
    /// results, confidence level, and recommended_parallelism.
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
    ///     with_confidence: If True, return DiscoveryResponse with confidence signal.
    #[pyo3(signature = (query, *, filters=None, explain=false, with_confidence=false))]
    fn discover(
        &mut self,
        py: Python<'_>,
        query: &str,
        filters: Option<HashMap<String, Py<PyAny>>>,
        explain: bool,
        with_confidence: bool,
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
                    enzyme_score: r.enzyme_score,
                    feedback_factor: r.feedback_factor,
                    final_score: r.final_score,
                    endpoint: r.endpoint,
                    metadata: r.metadata,
                })
                .collect();
            Ok(results.into_pyobject(py)?.into_any().unbind())
        } else if with_confidence {
            let resp = self
                .inner
                .discover_with_confidence_filtered(query, rust_filters.as_ref());

            let stored: Vec<StoredResult> = resp
                .results
                .into_iter()
                .map(|r| {
                    let invoke = self.invocables.get(&r.agent_name).map(|f| f.clone_ref(py));
                    StoredResult {
                        agent_name: r.agent_name,
                        similarity: r.similarity,
                        score: r.score,
                        endpoint: r.endpoint,
                        metadata: r.metadata,
                        invoke,
                    }
                })
                .collect();

            let (confidence_str, dissenting_kernel, alternative_agents) = match &resp.confidence {
                crate::network::DiscoveryConfidence::Unanimous => {
                    ("unanimous".to_string(), None, Vec::new())
                }
                crate::network::DiscoveryConfidence::Majority { dissenting_kernel } => (
                    "majority".to_string(),
                    Some(dissenting_kernel.to_string()),
                    Vec::new(),
                ),
                crate::network::DiscoveryConfidence::Split { alternative_agents } => {
                    ("split".to_string(), None, alternative_agents.clone())
                }
            };

            let py_resp = PyDiscoveryResponse {
                stored_results: stored,
                confidence: confidence_str,
                dissenting_kernel,
                alternative_agents,
                recommended_parallelism: resp.recommended_parallelism,
            };
            Ok(py_resp.into_pyobject(py)?.into_any().unbind())
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

    /// Discover agents for multiple queries in a single call.
    ///
    /// Returns a list of result lists, one per query. Moves the query loop
    /// from Python to Rust for better performance (avoids per-query GIL overhead).
    ///
    /// Args:
    ///     queries: List of natural-language capability queries.
    ///     filters: Optional metadata filters (shared across all queries).
    ///     explain: If True, return ExplainedResult with scoring breakdown.
    #[pyo3(signature = (queries, *, filters=None, explain=false))]
    fn discover_many(
        &mut self,
        py: Python<'_>,
        queries: Vec<String>,
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

        let query_refs: Vec<&str> = queries.iter().map(|s| s.as_str()).collect();

        if explain {
            let batch: Vec<Vec<PyExplainedResult>> = self
                .inner
                .discover_many_explained(&query_refs, rust_filters.as_ref())
                .into_iter()
                .map(|results| {
                    results
                        .into_iter()
                        .map(|r| PyExplainedResult {
                            agent_name: r.agent_name,
                            raw_similarity: r.raw_similarity,
                            diameter: r.diameter,
                            enzyme_score: r.enzyme_score,
                            feedback_factor: r.feedback_factor,
                            final_score: r.final_score,
                            endpoint: r.endpoint,
                            metadata: r.metadata,
                        })
                        .collect()
                })
                .collect();
            Ok(batch.into_pyobject(py)?.into_any().unbind())
        } else {
            let batch: Vec<Vec<PyDiscoveryResult>> = self
                .inner
                .discover_many(&query_refs, rust_filters.as_ref())
                .into_iter()
                .map(|results| {
                    results
                        .into_iter()
                        .map(|r| {
                            let invoke =
                                self.invocables.get(&r.agent_name).map(|f| f.clone_ref(py));
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
                })
                .collect();
            Ok(batch.into_pyobject(py)?.into_any().unbind())
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

    /// Apply temporal decay to all agent pheromone state.
    ///
    /// Call periodically (e.g. once per discovery cycle or on a timer) to
    /// prevent stale agents from retaining inflated scores indefinitely.
    fn tick(&mut self) {
        self.inner.tick();
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
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
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
        let inner = LocalNetwork::load(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
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
    pub enzyme_score: f64,
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
            "ExplainedResult(agent='{}', sim={:.3}, diameter={:.3}, enzyme={:.3}, feedback={:.3}, score={:.3})",
            self.agent_name,
            self.raw_similarity,
            self.diameter,
            self.enzyme_score,
            self.feedback_factor,
            self.final_score
        )
    }
}

/// Internal storage for a discovery result (non-pyclass).
struct StoredResult {
    agent_name: String,
    similarity: f64,
    score: f64,
    endpoint: Option<String>,
    metadata: HashMap<String, String>,
    invoke: Option<Py<PyAny>>,
}

impl StoredResult {
    fn to_py(&self, py: Python<'_>) -> PyDiscoveryResult {
        PyDiscoveryResult {
            agent_name: self.agent_name.clone(),
            similarity: self.similarity,
            score: self.score,
            endpoint: self.endpoint.clone(),
            metadata: self.metadata.clone(),
            invoke: self.invoke.as_ref().map(|f| f.clone_ref(py)),
        }
    }
}

/// Discovery response with confidence signal.
///
/// Supports iteration for backwards compatibility — existing code that
/// iterates over discover() results still works when with_confidence=True.
///
/// Attributes:
///     results: Ranked list of DiscoveryResult objects.
///     confidence: "unanimous", "majority", or "split".
///     dissenting_kernel: Name of the dissenting kernel (majority only).
///     alternative_agents: Alternative agent names suggested by dissenting kernels (split only).
///     recommended_parallelism: Suggested number of agents to invoke in parallel.
#[pyclass(name = "DiscoveryResponse")]
pub struct PyDiscoveryResponse {
    stored_results: Vec<StoredResult>,
    #[pyo3(get)]
    pub confidence: String,
    #[pyo3(get)]
    pub dissenting_kernel: Option<String>,
    #[pyo3(get)]
    pub alternative_agents: Vec<String>,
    #[pyo3(get)]
    pub recommended_parallelism: usize,
}

#[pymethods]
impl PyDiscoveryResponse {
    /// Get the results as a list of DiscoveryResult objects.
    #[getter]
    fn results(&self, py: Python<'_>) -> Vec<PyDiscoveryResult> {
        self.stored_results.iter().map(|r| r.to_py(py)).collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "DiscoveryResponse(results={}, confidence='{}', parallelism={})",
            self.stored_results.len(),
            self.confidence,
            self.recommended_parallelism
        )
    }

    fn __len__(&self) -> usize {
        self.stored_results.len()
    }

    fn __getitem__(&self, py: Python<'_>, idx: isize) -> PyResult<Py<PyAny>> {
        let len = self.stored_results.len() as isize;
        let actual_idx = if idx < 0 { len + idx } else { idx };
        if actual_idx < 0 || actual_idx >= len {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "index out of range",
            ));
        }
        let result = self.stored_results[actual_idx as usize].to_py(py);
        Ok(result.into_pyobject(py)?.into_any().unbind())
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PyDiscoveryResponseIter>> {
        let py = slf.py();
        let items: Vec<Py<PyAny>> = slf
            .stored_results
            .iter()
            .map(|r| {
                let result = r.to_py(py);
                result.into_pyobject(py).unwrap().into_any().unbind()
            })
            .collect();
        let iter = PyDiscoveryResponseIter {
            inner: items.into_iter(),
        };
        Py::new(py, iter)
    }
}

/// Iterator for PyDiscoveryResponse.
#[pyclass]
pub struct PyDiscoveryResponseIter {
    inner: std::vec::IntoIter<Py<PyAny>>,
}

#[pymethods]
impl PyDiscoveryResponseIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<Py<PyAny>> {
        self.inner.next()
    }
}
