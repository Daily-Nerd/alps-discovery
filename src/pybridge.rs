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
use crate::scorer::{Scorer, TfIdfScorer};

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
                tracing::warn!(
                    agent_id = agent_id,
                    error = %e,
                    "Python scorer index_capabilities() failed"
                );
            }
        });
    }

    fn remove_agent(&mut self, agent_id: &str) {
        Python::attach(|py| {
            if let Err(e) = self.inner.call_method1(py, "remove_agent", (agent_id,)) {
                tracing::warn!(
                    agent_id = agent_id,
                    error = %e,
                    "Python scorer remove_agent() failed"
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
    ///     config: Optional DiscoveryConfig for unified configuration.
    ///     scorer: Optional custom scorer object implementing index_capabilities(),
    ///         remove_agent(), and score() methods. Overrides the default MinHash scorer.
    #[new]
    #[pyo3(signature = (*, similarity_threshold=None, scorer=None, config=None))]
    fn new(
        similarity_threshold: Option<f64>,
        scorer: Option<Py<PyAny>>,
        config: Option<PyDiscoveryConfig>,
    ) -> PyResult<Self> {
        // If config is provided, use it (ignoring similarity_threshold)
        let inner = if let Some(cfg) = config {
            if scorer.is_some() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Cannot specify both 'config' and 'scorer' parameters",
                ));
            }
            if similarity_threshold.is_some() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Cannot specify both 'config' and 'similarity_threshold' parameters. \
                     Use config.similarity_threshold instead.",
                ));
            }
            LocalNetwork::with_config(cfg.inner.enzyme.clone(), cfg.inner.lsh.clone())
        } else if let Some(py_scorer) = scorer {
            // Check if it's our built-in TfIdfScorer first.
            let is_tfidf =
                Python::attach(|py| py_scorer.bind(py).is_instance_of::<PyTfIdfScorer>());
            if is_tfidf {
                LocalNetwork::with_scorer(Box::new(TfIdfScorer::new()))
            } else {
                let scorer = PyScorer { inner: py_scorer };
                LocalNetwork::with_scorer(Box::new(scorer))
            }
        } else {
            let mut lsh_config = LshConfig::default();
            if let Some(t) = similarity_threshold {
                lsh_config.similarity_threshold = t;
            }
            LocalNetwork::with_config(SLNEnzymeConfig::default(), lsh_config)
        };
        Ok(Self {
            inner,
            invocables: HashMap::new(),
        })
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
            .register(
                name,
                &caps,
                endpoint.as_deref(),
                metadata.unwrap_or_default(),
            )
            .expect("Python bridge registration should not fail with validated inputs");

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
        query: Py<PyAny>,
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

        // Accept either a str or a Query object.
        let bound = query.bind(py);
        let is_query = bound.extract::<PyQuery>().is_ok();

        if is_query {
            let rust_query = bound.extract::<PyQuery>()?.inner;
            self.discover_with_query(py, &rust_query, rust_filters, explain, with_confidence)
        } else {
            let query_str: String = bound.extract().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err("query must be a str or Query object")
            })?;
            self.discover_with_string(py, &query_str, rust_filters, explain, with_confidence)
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
                            similarity_lower: r.similarity_ci.lower_bound,
                            similarity_upper: r.similarity_ci.upper_bound,
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

    /// Current exploration epsilon value.
    ///
    /// Starts high (default 0.8) and decays with feedback. Controls the
    /// probability of random exploration vs. deterministic ranking during
    /// tie-breaking.
    #[getter]
    fn exploration_epsilon(&self) -> f64 {
        self.inner.exploration_epsilon()
    }

    /// Enable the discovery replay log for post-hoc analysis.
    ///
    /// After enabling, all discover(), record_success/failure(), and tick()
    /// operations are recorded. Use replay_events() to retrieve the log.
    ///
    /// Args:
    ///     max_events: Maximum number of events to keep (default: 10000).
    #[pyo3(signature = (max_events=10000))]
    fn enable_replay(&mut self, max_events: usize) {
        self.inner.with_replay_log_mut(|log| {
            *log = crate::network::replay::ReplayLog::new(max_events);
        });
    }

    /// Number of events in the replay log.
    #[getter]
    fn replay_event_count(&self) -> usize {
        self.inner.with_replay_log(|log| log.len())
    }

    /// Whether the replay log is enabled.
    #[getter]
    fn replay_enabled(&self) -> bool {
        self.inner.with_replay_log(|log| log.is_enabled())
    }

    /// Retrieve replay events as a list of dicts.
    ///
    /// Each event is a dict with keys: "type", "query", "agent_name",
    /// "raw_similarity", "enzyme_score", "feedback_factor", "final_score",
    /// "outcome" (depending on event type).
    fn replay_events(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.inner.with_replay_log(|log| {
            let events: Vec<Py<PyAny>> = log
                .events()
                .iter()
                .map(|event| {
                    let dict = pyo3::types::PyDict::new(py);
                    match &event.kind {
                        crate::network::replay::EventKind::QuerySubmitted { query } => {
                            dict.set_item("type", "query_submitted").unwrap();
                            dict.set_item("query", query).unwrap();
                        }
                        crate::network::replay::EventKind::AgentScored {
                            query,
                            agent_name,
                            raw_similarity,
                            enzyme_score,
                            feedback_factor,
                            final_score,
                        } => {
                            dict.set_item("type", "agent_scored").unwrap();
                            dict.set_item("query", query).unwrap();
                            dict.set_item("agent_name", agent_name).unwrap();
                            dict.set_item("raw_similarity", raw_similarity).unwrap();
                            dict.set_item("enzyme_score", enzyme_score).unwrap();
                            dict.set_item("feedback_factor", feedback_factor).unwrap();
                            dict.set_item("final_score", final_score).unwrap();
                        }
                        crate::network::replay::EventKind::FeedbackRecorded {
                            agent_name,
                            query,
                            outcome,
                        } => {
                            dict.set_item("type", "feedback_recorded").unwrap();
                            dict.set_item("agent_name", agent_name).unwrap();
                            dict.set_item("query", query.as_deref().unwrap_or(""))
                                .unwrap();
                            dict.set_item("outcome", outcome).unwrap();
                        }
                        crate::network::replay::EventKind::TickApplied => {
                            dict.set_item("type", "tick_applied").unwrap();
                        }
                    }
                    dict.into_any().unbind()
                })
                .collect();
            Ok(events)
        })
    }

    /// Clear the replay log.
    fn replay_clear(&mut self) {
        self.inner.with_replay_log_mut(|log| log.clear());
    }

    /// Detect capability drift for registered agents.
    ///
    /// Returns a list of dicts, one per agent with enough feedback data.
    /// Each dict has: agent_name, alignment, sample_count, drifted.
    ///
    /// Args:
    ///     threshold: Alignment below this triggers drift flag (default: 0.3).
    ///     min_samples: Minimum feedback records required (default: 5).
    #[pyo3(signature = (*, threshold=0.3, min_samples=5))]
    fn detect_drift(
        &self,
        py: Python<'_>,
        threshold: f64,
        min_samples: usize,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let reports = self.inner.detect_drift(threshold, min_samples);
        let result: Vec<Py<PyAny>> = reports
            .into_iter()
            .map(|r| {
                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("agent_name", r.agent_name).unwrap();
                dict.set_item("alignment", r.alignment).unwrap();
                dict.set_item("sample_count", r.sample_count).unwrap();
                dict.set_item("drifted", r.drifted).unwrap();
                dict.into_any().unbind()
            })
            .collect();
        Ok(result)
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

        // Try PyFilterValue first (direct FilterValue object).
        if let Ok(py_filter) = value.extract::<PyFilterValue>() {
            return Ok(py_filter.inner);
        }

        // Try as dict second (operator syntax).
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

    /// Discovery implementation for string queries.
    fn discover_with_string(
        &mut self,
        py: Python<'_>,
        query: &str,
        rust_filters: Option<crate::network::Filters>,
        explain: bool,
        with_confidence: bool,
    ) -> PyResult<Py<PyAny>> {
        if explain {
            let results: Vec<PyExplainedResult> = self
                .inner
                .discover_explained(query, rust_filters.as_ref())
                .into_iter()
                .map(|r| PyExplainedResult {
                    agent_name: r.agent_name,
                    raw_similarity: r.raw_similarity,
                    similarity_lower: r.similarity_ci.lower_bound,
                    similarity_upper: r.similarity_ci.upper_bound,
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
            self.build_confidence_response(py, resp)
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

    /// Discovery implementation for Query algebra objects.
    fn discover_with_query(
        &mut self,
        py: Python<'_>,
        query: &crate::query::Query,
        rust_filters: Option<crate::network::Filters>,
        explain: bool,
        with_confidence: bool,
    ) -> PyResult<Py<PyAny>> {
        if explain {
            let results: Vec<PyExplainedResult> = self
                .inner
                .discover_query_explained(query, rust_filters.as_ref())
                .into_iter()
                .map(|r| PyExplainedResult {
                    agent_name: r.agent_name,
                    raw_similarity: r.raw_similarity,
                    similarity_lower: r.similarity_ci.lower_bound,
                    similarity_upper: r.similarity_ci.upper_bound,
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
                .discover_query_with_confidence(query, rust_filters.as_ref());
            self.build_confidence_response(py, resp)
        } else {
            let results: Vec<PyDiscoveryResult> = self
                .inner
                .discover_query(query, rust_filters.as_ref())
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

    /// Build a PyDiscoveryResponse from a Rust DiscoveryResponse.
    fn build_confidence_response(
        &self,
        py: Python<'_>,
        resp: crate::network::DiscoveryResponse,
    ) -> PyResult<Py<PyAny>> {
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
            Some(crate::network::DiscoveryConfidence::Unanimous) => {
                ("unanimous".to_string(), None, Vec::new())
            }
            Some(crate::network::DiscoveryConfidence::Majority { dissenting_kernel }) => (
                "majority".to_string(),
                Some(dissenting_kernel.to_string()),
                Vec::new(),
            ),
            Some(crate::network::DiscoveryConfidence::Split { alternative_agents }) => {
                ("split".to_string(), None, alternative_agents.clone())
            }
            Some(crate::network::DiscoveryConfidence::NoViableAgents) => {
                ("no_viable_agents".to_string(), None, Vec::new())
            }
            None => ("none".to_string(), None, Vec::new()),
        };

        let py_resp = PyDiscoveryResponse {
            stored_results: stored,
            confidence: confidence_str,
            dissenting_kernel,
            alternative_agents,
            recommended_parallelism: resp.recommended_parallelism,
            best_below_threshold: resp.best_below_threshold,
        };
        Ok(py_resp.into_pyobject(py)?.into_any().unbind())
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
    pub similarity_lower: f64,
    #[pyo3(get)]
    pub similarity_upper: f64,
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
            "ExplainedResult(agent='{}', sim={:.3} [{:.3}, {:.3}], diameter={:.3}, enzyme={:.3}, feedback={:.3}, score={:.3})",
            self.agent_name,
            self.raw_similarity,
            self.similarity_lower,
            self.similarity_upper,
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
    /// When results are empty, reports the highest similarity score that was filtered out.
    /// Format: (agent_name, similarity_score) or None if results are not empty.
    #[pyo3(get)]
    pub best_below_threshold: Option<(String, f64)>,
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

/// Composable query expression for agent discovery.
///
/// Supports set-theoretic composition of text queries:
///
/// - ``Query.all("legal translation", "German language")``
///   — agent must match ALL terms (min similarity)
/// - ``Query.any("translate", "interpret")``
///   — agent can match ANY term (max similarity)
/// - ``Query.all("translate").exclude("medical")``
///   — match but penalise unwanted matches
/// - ``Query.weighted({"translate": 2.0, "legal": 1.0})``
///   — weighted combination
///
/// Pass a Query to ``network.discover()`` in place of a string.
#[pyclass(name = "Query")]
#[derive(Clone)]
pub struct PyQuery {
    pub(crate) inner: crate::query::Query,
}

#[pymethods]
impl PyQuery {
    /// Create an All query (AND semantics).
    ///
    /// Agent must match ALL sub-queries. Score = min across sub-queries.
    ///
    /// Args:
    ///     *queries: Text strings or Query objects.
    #[staticmethod]
    #[pyo3(signature = (*queries))]
    fn all(queries: Vec<Py<PyAny>>, py: Python<'_>) -> PyResult<Self> {
        let inner_queries = Self::parse_query_args(queries, py)?;
        Ok(PyQuery {
            inner: crate::query::Query::All(inner_queries),
        })
    }

    /// Create an Any query (OR semantics).
    ///
    /// Agent can match ANY sub-query. Score = max across sub-queries.
    ///
    /// Args:
    ///     *queries: Text strings or Query objects.
    #[staticmethod]
    #[pyo3(signature = (*queries))]
    fn any(queries: Vec<Py<PyAny>>, py: Python<'_>) -> PyResult<Self> {
        let inner_queries = Self::parse_query_args(queries, py)?;
        Ok(PyQuery {
            inner: crate::query::Query::Any(inner_queries),
        })
    }

    /// Create a Weighted query (boosted combination).
    ///
    /// Score = weighted average of sub-query similarities.
    ///
    /// Args:
    ///     mapping: Dict mapping query strings to weight floats.
    #[staticmethod]
    fn weighted(mapping: HashMap<String, f64>) -> Self {
        let entries: Vec<(crate::query::Query, f64)> = mapping
            .into_iter()
            .map(|(text, weight)| (crate::query::Query::Text(text), weight))
            .collect();
        PyQuery {
            inner: crate::query::Query::Weighted(entries),
        }
    }

    /// Chain an exclusion onto this query.
    ///
    /// Agents matching the exclusion have their score reduced proportionally.
    ///
    /// Args:
    ///     query: Text string or Query object to exclude.
    fn exclude(&self, query: Py<PyAny>, py: Python<'_>) -> PyResult<Self> {
        let exclusion = Self::parse_single_query(query, py)?;
        Ok(PyQuery {
            inner: self.inner.clone().exclude(exclusion),
        })
    }

    fn __repr__(&self) -> String {
        format!("Query({:?})", self.inner)
    }
}

/// Built-in TF-IDF scorer for semantic matching.
///
/// Weights rare domain terms higher than common words. Pass to
/// ``LocalNetwork(scorer=TfIdfScorer())`` to use instead of the default MinHash.
///
/// Example:
///     scorer = TfIdfScorer()
///     network = LocalNetwork(scorer=scorer)
#[pyclass(name = "TfIdfScorer")]
pub struct PyTfIdfScorer {
    _inner: TfIdfScorer,
}

#[pymethods]
impl PyTfIdfScorer {
    #[new]
    fn new() -> Self {
        Self {
            _inner: TfIdfScorer::new(),
        }
    }

    fn __repr__(&self) -> String {
        "TfIdfScorer()".to_string()
    }
}

impl PyQuery {
    /// Parse a single Python argument into a Rust Query.
    fn parse_single_query(arg: Py<PyAny>, py: Python<'_>) -> PyResult<crate::query::Query> {
        let bound = arg.bind(py);
        if let Ok(pyq) = bound.extract::<PyQuery>() {
            return Ok(pyq.inner);
        }
        if let Ok(s) = bound.extract::<String>() {
            return Ok(crate::query::Query::Text(s));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Query argument must be a str or Query object",
        ))
    }

    /// Parse a list of Python arguments into Rust Query objects.
    fn parse_query_args(
        args: Vec<Py<PyAny>>,
        py: Python<'_>,
    ) -> PyResult<Vec<crate::query::Query>> {
        args.into_iter()
            .map(|a| Self::parse_single_query(a, py))
            .collect()
    }
}

/// Mycorrhizal feedback propagator for transitive learning.
///
/// Propagates success feedback to similar agents for faster network learning.
#[pyclass(name = "MycorrhizalPropagator")]
#[derive(Clone)]
pub struct PyMycorrhizalPropagator {
    inner: crate::network::mycorrhizal::MycorrhizalPropagator,
}

#[pymethods]
impl PyMycorrhizalPropagator {
    #[new]
    fn new() -> Self {
        Self {
            inner: crate::network::mycorrhizal::MycorrhizalPropagator::new(),
        }
    }

    #[staticmethod]
    fn with_config(propagation_attenuation: f64, propagation_threshold: f64) -> Self {
        Self {
            inner: crate::network::mycorrhizal::MycorrhizalPropagator::with_config(
                propagation_attenuation,
                propagation_threshold,
            ),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "MycorrhizalPropagator(attenuation={}, threshold={})",
            self.inner.propagation_attenuation, self.inner.propagation_threshold
        )
    }
}

impl PyMycorrhizalPropagator {
    pub fn inner(&self) -> crate::network::mycorrhizal::MycorrhizalPropagator {
        self.inner.clone()
    }
}

/// Circuit breaker configuration for failure exclusion.
#[pyclass(name = "CircuitBreakerConfig")]
#[derive(Clone)]
pub struct PyCircuitBreakerConfig {
    inner: crate::core::pheromone::CircuitBreakerConfig,
}

#[pymethods]
impl PyCircuitBreakerConfig {
    #[new]
    fn new() -> Self {
        Self {
            inner: crate::core::pheromone::CircuitBreakerConfig::new(),
        }
    }

    #[staticmethod]
    fn with_threshold_and_timeout(failure_threshold: u8, recovery_timeout_secs: u64) -> Self {
        Self {
            inner: crate::core::pheromone::CircuitBreakerConfig::with_threshold_and_timeout(
                failure_threshold,
                std::time::Duration::from_secs(recovery_timeout_secs),
            ),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "CircuitBreakerConfig(threshold={}, timeout={}s)",
            self.inner.failure_threshold,
            self.inner.recovery_timeout.as_secs()
        )
    }
}

impl PyCircuitBreakerConfig {
    pub fn inner(&self) -> crate::core::pheromone::CircuitBreakerConfig {
        self.inner.clone()
    }
}

/// Unified configuration for ALPS Discovery.
///
/// Consolidates all tuning parameters into a single configuration object.
/// Pass this to LocalNetwork() constructor to customize discovery behavior.
#[pyclass(name = "DiscoveryConfig")]
#[derive(Clone)]
pub struct PyDiscoveryConfig {
    inner: crate::core::config::DiscoveryConfig,
}

#[pymethods]
impl PyDiscoveryConfig {
    /// Create a new DiscoveryConfig with default values.
    #[new]
    #[pyo3(signature = (
        similarity_threshold=0.1,
        feedback_relevance_threshold=0.3,
        tie_epsilon=1e-4,
        tau_floor=0.001,
        max_feedback_records=100,
        diameter_initial=0.5,
        diameter_min=0.01,
        diameter_max=2.0,
        epsilon_initial=0.8,
        epsilon_floor=0.05,
        epsilon_decay_rate=0.99,
        max_disagreement_split=3
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        similarity_threshold: f64,
        feedback_relevance_threshold: f64,
        tie_epsilon: f64,
        tau_floor: f64,
        max_feedback_records: usize,
        diameter_initial: f64,
        diameter_min: f64,
        diameter_max: f64,
        epsilon_initial: f64,
        epsilon_floor: f64,
        epsilon_decay_rate: f64,
        max_disagreement_split: usize,
    ) -> PyResult<Self> {
        let config = crate::core::config::DiscoveryConfig {
            lsh: crate::core::config::LshConfig {
                similarity_threshold,
                ..Default::default()
            },
            enzyme: crate::core::enzyme::SLNEnzymeConfig {
                max_disagreement_split,
                quorum: crate::core::enzyme::Quorum::Majority,
            },
            exploration: crate::core::config::ExplorationConfig {
                epsilon_initial,
                epsilon_floor,
                epsilon_decay_rate,
            },
            feedback_relevance_threshold,
            tie_epsilon,
            tau_floor,
            max_feedback_records,
            diameter_initial,
            diameter_min,
            diameter_max,
        };

        // Validate at construction time
        config.validate().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid configuration: {}", e))
        })?;

        Ok(Self { inner: config })
    }

    /// Get similarity threshold.
    #[getter]
    fn similarity_threshold(&self) -> f64 {
        self.inner.lsh.similarity_threshold
    }

    /// Get feedback relevance threshold.
    #[getter]
    fn feedback_relevance_threshold(&self) -> f64 {
        self.inner.feedback_relevance_threshold
    }

    /// Get tie epsilon.
    #[getter]
    fn tie_epsilon(&self) -> f64 {
        self.inner.tie_epsilon
    }

    /// Get tau floor.
    #[getter]
    fn tau_floor(&self) -> f64 {
        self.inner.tau_floor
    }

    /// Get max feedback records.
    #[getter]
    fn max_feedback_records(&self) -> usize {
        self.inner.max_feedback_records
    }

    /// Get diameter initial.
    #[getter]
    fn diameter_initial(&self) -> f64 {
        self.inner.diameter_initial
    }

    /// Get diameter min.
    #[getter]
    fn diameter_min(&self) -> f64 {
        self.inner.diameter_min
    }

    /// Get diameter max.
    #[getter]
    fn diameter_max(&self) -> f64 {
        self.inner.diameter_max
    }

    /// Get epsilon initial.
    #[getter]
    fn epsilon_initial(&self) -> f64 {
        self.inner.exploration.epsilon_initial
    }

    /// Get epsilon floor.
    #[getter]
    fn epsilon_floor(&self) -> f64 {
        self.inner.exploration.epsilon_floor
    }

    /// Get epsilon decay rate.
    #[getter]
    fn epsilon_decay_rate(&self) -> f64 {
        self.inner.exploration.epsilon_decay_rate
    }

    /// Get max disagreement split.
    #[getter]
    fn max_disagreement_split(&self) -> usize {
        self.inner.enzyme.max_disagreement_split
    }

    fn __repr__(&self) -> String {
        format!(
            "DiscoveryConfig(similarity_threshold={}, feedback_relevance_threshold={}, \
             tau_floor={}, diameter_initial={})",
            self.inner.lsh.similarity_threshold,
            self.inner.feedback_relevance_threshold,
            self.inner.tau_floor,
            self.inner.diameter_initial
        )
    }
}

impl PyDiscoveryConfig {
    pub fn inner(&self) -> crate::core::config::DiscoveryConfig {
        self.inner.clone()
    }
}

/// Filter condition for metadata-based filtering.
#[pyclass(name = "FilterValue")]
#[derive(Clone)]
pub struct PyFilterValue {
    inner: crate::network::FilterValue,
}

#[pymethods]
impl PyFilterValue {
    /// Create an exact match filter.
    #[staticmethod]
    fn exact(value: String) -> Self {
        Self {
            inner: crate::network::FilterValue::Exact(value),
        }
    }

    /// Create a contains filter (substring match).
    #[staticmethod]
    fn contains(substring: String) -> Self {
        Self {
            inner: crate::network::FilterValue::Contains(substring),
        }
    }

    /// Create a one-of filter (value must be in list).
    #[staticmethod]
    fn one_of(options: Vec<String>) -> Self {
        Self {
            inner: crate::network::FilterValue::OneOf(options),
        }
    }

    /// Create a less-than filter (for numeric values).
    #[staticmethod]
    fn less_than(threshold: f64) -> Self {
        Self {
            inner: crate::network::FilterValue::LessThan(threshold),
        }
    }

    /// Create a greater-than filter (for numeric values).
    #[staticmethod]
    fn greater_than(threshold: f64) -> Self {
        Self {
            inner: crate::network::FilterValue::GreaterThan(threshold),
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            crate::network::FilterValue::Exact(v) => format!("FilterValue.Exact('{}')", v),
            crate::network::FilterValue::Contains(v) => format!("FilterValue.Contains('{}')", v),
            crate::network::FilterValue::OneOf(opts) => {
                format!("FilterValue.OneOf({:?})", opts)
            }
            crate::network::FilterValue::LessThan(t) => format!("FilterValue.LessThan({})", t),
            crate::network::FilterValue::GreaterThan(t) => {
                format!("FilterValue.GreaterThan({})", t)
            }
        }
    }
}

impl PyFilterValue {
    pub fn inner(&self) -> crate::network::FilterValue {
        self.inner.clone()
    }
}

/// Extract capability strings from a Google A2A AgentCard.
///
/// Parses an A2A AgentCard JSON value and extracts capability descriptions
/// from the agent name, description, skills, and tags. This enables ALPS
/// to discover A2A agents using the same local discovery mechanism as MCP tools.
///
/// Args:
///     agent_card: JSON value (dict-like) with optional fields:
///         - name: Agent name
///         - description: Agent description
///         - skills: List of skill objects with name, description, tags
///
/// Returns:
///     List of capability strings for use with register().
///
/// Example:
///     ```python
///     agent_card = {
///         "name": "legal-assistant",
///         "description": "Legal document analysis",
///         "skills": [{
///             "name": "analyze_contract",
///             "description": "Analyze legal contracts",
///             "tags": ["legal", "contracts"]
///         }]
///     }
///     caps = capabilities_from_a2a_rust(agent_card)
///     network.register("legal-agent", caps)
///     ```
#[pyfunction]
pub fn capabilities_from_a2a_rust(agent_card: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    let mut caps = Vec::new();

    // Extract agent-level name and description
    let name = agent_card
        .get_item("name")
        .ok()
        .and_then(|v| v.extract::<String>().ok())
        .unwrap_or_default()
        .replace(['-', '_'], " ")
        .trim()
        .to_string();

    let desc = agent_card
        .get_item("description")
        .ok()
        .and_then(|v| v.extract::<String>().ok())
        .unwrap_or_default()
        .trim()
        .to_string();

    // Build agent-level capability
    let mut agent_parts = Vec::new();
    if !name.is_empty() && !desc.is_empty() {
        agent_parts.push(format!("{}: {}", name, desc));
    } else if !name.is_empty() {
        agent_parts.push(name.clone());
    } else if !desc.is_empty() {
        agent_parts.push(desc.clone());
    }

    if !agent_parts.is_empty() {
        caps.push(agent_parts.join(". "));
    }

    // Extract skill-level capabilities
    if let Ok(skills) = agent_card.get_item("skills") {
        if let Ok(skills_list) = skills.extract::<Vec<Bound<'_, PyAny>>>() {
            for skill in skills_list {
                let skill_name = skill
                    .get_item("name")
                    .ok()
                    .and_then(|v| v.extract::<String>().ok())
                    .unwrap_or_default()
                    .replace('_', " ")
                    .trim()
                    .to_string();

                let skill_desc = skill
                    .get_item("description")
                    .ok()
                    .and_then(|v| v.extract::<String>().ok())
                    .unwrap_or_default()
                    .trim()
                    .to_string();

                let skill_tags = skill
                    .get_item("tags")
                    .ok()
                    .and_then(|v| v.extract::<Vec<String>>().ok())
                    .unwrap_or_default();

                // Build skill capability
                let mut skill_parts = Vec::new();
                if !skill_name.is_empty() && !skill_desc.is_empty() {
                    skill_parts.push(format!("{}: {}", skill_name, skill_desc));
                } else if !skill_name.is_empty() {
                    skill_parts.push(skill_name);
                } else if !skill_desc.is_empty() {
                    skill_parts.push(skill_desc);
                }

                // Add tags
                if !skill_tags.is_empty() {
                    skill_parts.push(skill_tags.join(", "));
                }

                if !skill_parts.is_empty() {
                    caps.push(skill_parts.join(". "));
                }
            }
        }
    }

    Ok(caps)
}
