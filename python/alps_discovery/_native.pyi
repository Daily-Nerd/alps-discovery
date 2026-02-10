"""Type stubs for alps_discovery._native (Rust extension module)."""

from collections.abc import Iterator
from typing import Any, overload

class LocalNetwork:
    """Local agent discovery network using multi-kernel routing.

    Args:
        similarity_threshold: Minimum similarity for results (default: 0.1).
        scorer: Optional custom scorer object with index_capabilities(),
            remove_agent(), and score() methods.
    """

    def __init__(
        self,
        *,
        similarity_threshold: float | None = None,
        scorer: Any | None = None,
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
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        explain: bool = False,
        with_confidence: bool = False,
    ) -> list[DiscoveryResult] | list[ExplainedResult] | DiscoveryResponse:
        """Discover agents matching a query.

        Args:
            query: Natural-language capability query.
            filters: Optional metadata filters. String values for exact match,
                or dicts with $in, $lt, $gt, $contains operators.
            explain: If True, returns ExplainedResult with scoring breakdown.
            with_confidence: If True, returns DiscoveryResponse with confidence signal.

        Returns:
            Ranked list of results (default), ExplainedResult list (explain=True),
            or DiscoveryResponse (with_confidence=True).
        """
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
        diameter: Agent's routing diameter (weight from feedback history).
        enzyme_score: Normalized composite enzyme score [0.0, 1.0].
        feedback_factor: Per-query feedback adjustment [-1.0, 1.0].
        final_score: Combined routing score.
        endpoint: Agent URI/URL if provided at registration, else None.
        metadata: Dict of key-value pairs if provided, else {}.
    """

    agent_name: str
    raw_similarity: float
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
    """

    results: list[DiscoveryResult]
    confidence: str
    dissenting_kernel: str | None
    alternative_agents: list[str]
    recommended_parallelism: int

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> DiscoveryResult: ...
    def __iter__(self) -> Iterator[DiscoveryResult]: ...
