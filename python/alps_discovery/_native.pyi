"""Type stubs for alps_discovery._native (Rust extension module)."""

from collections.abc import Iterator
from typing import Any, Literal, overload

class LocalNetwork:
    """Local agent discovery network using multi-kernel routing.

    Args:
        similarity_threshold: Minimum similarity for results (default: 0.1).
        scorer: Optional custom scorer object with index_capabilities(),
            remove_agent(), and score() methods.
        config: Optional DiscoveryConfig for unified configuration.
            Cannot be combined with similarity_threshold or scorer.
    """

    def __init__(
        self,
        *,
        similarity_threshold: float | None = None,
        scorer: Any | None = None,
        config: DiscoveryConfig | None = None,
    ) -> None: ...
    def register(
        self,
        name: str,
        capabilities: list[str],
        *,
        endpoint: str | None = None,
        metadata: dict[str, str] | None = None,
        invoke: Any | None = None,
    ) -> None:
        """Register an agent with its capabilities.

        Args:
            name: Unique agent identifier.
            capabilities: List of capability description strings.
            endpoint: Optional URI/URL for invoking the agent.
            metadata: Optional key-value pairs (protocol, version, etc.).
            invoke: Optional callable for local invocation.
        """
        ...

    def deregister(self, name: str) -> bool:
        """Remove an agent. Returns True if found."""
        ...

    @overload
    def discover(
        self,
        query: str | Query,
        *,
        filters: dict[str, Any] | None = None,
        explain: Literal[False] = False,
        with_confidence: Literal[False] = False,
    ) -> list[DiscoveryResult]:
        """Discover agents matching a query (basic mode)."""
        ...

    @overload
    def discover(
        self,
        query: str | Query,
        *,
        filters: dict[str, Any] | None = None,
        explain: Literal[True],
        with_confidence: Literal[False] = False,
    ) -> list[ExplainedResult]:
        """Discover agents matching a query (explain mode)."""
        ...

    @overload
    def discover(
        self,
        query: str | Query,
        *,
        filters: dict[str, Any] | None = None,
        explain: Literal[False] = False,
        with_confidence: Literal[True],
    ) -> DiscoveryResponse:
        """Discover agents matching a query (with confidence signal)."""
        ...

    def discover_many(
        self,
        queries: list[str],
        *,
        filters: dict[str, Any] | None = None,
        explain: bool = False,
    ) -> list[list[DiscoveryResult]] | list[list[ExplainedResult]]:
        """Discover agents for multiple queries in a single call.

        Returns one result list per query, in input order. Moves the
        query loop from Python to Rust for better performance.

        Args:
            queries: List of natural-language capability queries.
            filters: Optional metadata filters (shared across all queries).
            explain: If True, returns ExplainedResult with scoring breakdown.

        Returns:
            List of ranked result lists, one per query.
        """
        ...

    def record_success(self, agent_name: str, *, query: str | None = None) -> None:
        """Record a successful interaction. Boosts agent ranking."""
        ...

    def record_failure(self, agent_name: str, *, query: str | None = None) -> None:
        """Record a failed interaction. Reduces agent ranking."""
        ...

    def tick(self) -> None:
        """Apply temporal decay to all agent pheromone state.

        Call periodically to prevent stale agents from retaining
        inflated scores indefinitely.
        """
        ...

    @property
    def agent_count(self) -> int:
        """Number of registered agents."""
        ...

    def agents(self) -> list[str]:
        """List of all registered agent names."""
        ...

    def save(self, path: str) -> None:
        """Save network state to a JSON file."""
        ...

    @staticmethod
    def load(path: str) -> LocalNetwork:
        """Load network state from a JSON file."""
        ...

class Query:
    """Composable query expression for agent discovery.

    Supports set-theoretic composition of text queries:

    - ``Query.all("legal translation", "German language")``
      — agent must match ALL terms (min similarity)
    - ``Query.any("translate", "interpret")``
      — agent can match ANY term (max similarity)
    - ``Query.all("translate").exclude("medical")``
      — match but penalise unwanted matches
    - ``Query.weighted({"translate": 2.0, "legal": 1.0})``
      — weighted combination

    Pass a Query to ``network.discover()`` in place of a string.
    """

    @staticmethod
    def all(*queries: str | Query) -> Query:
        """Create an All query (AND semantics).

        Agent must match ALL sub-queries. Score = min across sub-queries.
        """
        ...

    @staticmethod
    def any(*queries: str | Query) -> Query:
        """Create an Any query (OR semantics).

        Agent can match ANY sub-query. Score = max across sub-queries.
        """
        ...

    @staticmethod
    def weighted(mapping: dict[str, float]) -> Query:
        """Create a Weighted query.

        Score = weighted average of sub-query similarities.

        Args:
            mapping: Dict mapping query strings to weight floats.
        """
        ...

    def exclude(self, query: str | Query) -> Query:
        """Chain an exclusion onto this query.

        Agents matching the exclusion have their score reduced proportionally.
        """
        ...

class DiscoveryResult:
    """A single discovery result.

    Attributes:
        agent_name: The matched agent's name.
        similarity: Raw capability similarity [0.0, 1.0].
        score: Combined routing score (similarity x enzyme x feedback).
        endpoint: Agent URI/URL if provided at registration, else None.
        metadata: Dict of key-value pairs if provided, else {}.
        invoke: Callable if provided at registration, else None.
    """

    agent_name: str
    similarity: float
    score: float
    endpoint: str | None
    metadata: dict[str, str]
    invoke: Any | None

class ExplainedResult:
    """Extended discovery result with full scoring breakdown.

    Attributes:
        agent_name: The matched agent's name.
        raw_similarity: Raw capability similarity from the scorer [0.0, 1.0].
        similarity_lower: Lower bound of 95% confidence interval.
        similarity_upper: Upper bound of 95% confidence interval.
        diameter: Agent's routing diameter (weight from feedback history).
        enzyme_score: Normalized composite enzyme score [0.0, 1.0].
        feedback_factor: Per-query feedback adjustment [-1.0, 1.0].
        final_score: Combined routing score.
        endpoint: Agent URI/URL if provided at registration, else None.
        metadata: Dict of key-value pairs if provided, else {}.
    """

    agent_name: str
    raw_similarity: float
    similarity_lower: float
    similarity_upper: float
    diameter: float
    enzyme_score: float
    feedback_factor: float
    final_score: float
    endpoint: str | None
    metadata: dict[str, str]

class DiscoveryResponse:
    """Discovery response with confidence signal.

    Supports iteration and indexing for backwards compatibility.

    Attributes:
        results: Ranked list of DiscoveryResult objects.
        confidence: "unanimous", "majority", or "split".
        dissenting_kernel: Name of the dissenting kernel (majority only).
        alternative_agents: Alternative agent names (split only).
        recommended_parallelism: Suggested number of agents to invoke in parallel.
        best_below_threshold: When results are empty, reports (agent_name, similarity)
            of the highest scoring agent that was filtered out. None if results exist.
    """

    results: list[DiscoveryResult]
    confidence: str
    dissenting_kernel: str | None
    alternative_agents: list[str]
    recommended_parallelism: int
    best_below_threshold: tuple[str, float] | None

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> DiscoveryResult: ...
    def __iter__(self) -> Iterator[DiscoveryResult]: ...

class TfIdfScorer:
    """TF-IDF cosine similarity scorer.

    Alternative to MinHash for exact token matching use cases.
    Implements the scorer protocol: index_capabilities(), remove_agent(), score().
    """

    def __init__(self) -> None: ...

class DiscoveryConfig:
    """Unified configuration for ALPS Discovery.

    Consolidates all tuning parameters into a single configuration object.

    Args:
        similarity_threshold: Minimum similarity for results (default: 0.1).
        feedback_relevance_threshold: Threshold for feedback matching (default: 0.3).
        tie_epsilon: Epsilon for strict score equality detection (default: 1e-4).
        tau_floor: Minimum tau to prevent zero-trap (default: 0.001).
        max_feedback_records: Max feedback records per agent (default: 100).
        diameter_initial: Initial diameter for new agents (default: 0.5).
        diameter_min: Minimum diameter value (default: 0.01).
        diameter_max: Maximum diameter value (default: 2.0).
        epsilon_initial: Starting exploration probability (default: 0.8).
        epsilon_floor: Minimum exploration probability (default: 0.05).
        epsilon_decay_rate: Exponential decay rate per feedback (default: 0.99).
        max_disagreement_split: Max agents to split on disagreement (default: 3).
    """

    def __init__(
        self,
        *,
        similarity_threshold: float = 0.1,
        feedback_relevance_threshold: float = 0.3,
        tie_epsilon: float = 1e-4,
        tau_floor: float = 0.001,
        max_feedback_records: int = 100,
        diameter_initial: float = 0.5,
        diameter_min: float = 0.01,
        diameter_max: float = 2.0,
        epsilon_initial: float = 0.8,
        epsilon_floor: float = 0.05,
        epsilon_decay_rate: float = 0.99,
        max_disagreement_split: int = 3,
    ) -> None: ...
    @property
    def similarity_threshold(self) -> float: ...
    @property
    def feedback_relevance_threshold(self) -> float: ...
    @property
    def tie_epsilon(self) -> float: ...
    @property
    def tau_floor(self) -> float: ...
    @property
    def max_feedback_records(self) -> int: ...
    @property
    def diameter_initial(self) -> float: ...
    @property
    def diameter_min(self) -> float: ...
    @property
    def diameter_max(self) -> float: ...
    @property
    def epsilon_initial(self) -> float: ...
    @property
    def epsilon_floor(self) -> float: ...
    @property
    def epsilon_decay_rate(self) -> float: ...
    @property
    def max_disagreement_split(self) -> int: ...

class FilterValue:
    """Metadata filter condition.

    Use static methods to create filter instances:
    - FilterValue.exact("value") - Exact string match
    - FilterValue.contains("substring") - Substring containment
    - FilterValue.one_of(["a", "b"]) - Value in list
    - FilterValue.less_than(10.0) - Numeric less than
    - FilterValue.greater_than(1.0) - Numeric greater than
    """

    @staticmethod
    def exact(value: str) -> FilterValue:
        """Create an exact match filter."""
        ...

    @staticmethod
    def contains(substring: str) -> FilterValue:
        """Create a substring containment filter."""
        ...

    @staticmethod
    def one_of(options: list[str]) -> FilterValue:
        """Create a one-of filter (value must be in list)."""
        ...

    @staticmethod
    def less_than(threshold: float) -> FilterValue:
        """Create a less-than filter (for numeric values)."""
        ...

    @staticmethod
    def greater_than(threshold: float) -> FilterValue:
        """Create a greater-than filter (for numeric values)."""
        ...

class MycorrhizalPropagator:
    """Configuration for mycorrhizal feedback propagation.

    Controls transitive feedback: success signals propagate to similar agents.

    Args:
        propagation_attenuation: Strength of propagated signal (default: 0.3).
        propagation_threshold: Min similarity for propagation (default: 0.3).
    """

    def __init__(
        self,
        *,
        propagation_attenuation: float = 0.3,
        propagation_threshold: float = 0.3,
    ) -> None: ...

class CircuitBreakerConfig:
    """Configuration for circuit breaker (failure exclusion).

    Args:
        failure_threshold: Consecutive failures before opening circuit (default: 5).
        recovery_timeout_secs: Seconds before recovery probe (default: 60).
    """

    def __init__(self) -> None: ...
    @staticmethod
    def with_threshold_and_timeout(
        failure_threshold: int,
        recovery_timeout_secs: int,
    ) -> CircuitBreakerConfig: ...

def capabilities_from_a2a_rust(agent_card: dict[str, Any]) -> list[str]:
    """Extract capability strings from Google A2A AgentCard.

    Rust implementation of A2A parsing (identical to Python version).

    Args:
        agent_card: A2A AgentCard dict with name, description, skills, tags.

    Returns:
        List of capability strings for use with register().
    """
    ...
