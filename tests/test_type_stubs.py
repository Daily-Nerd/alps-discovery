"""Type checking validation for _native.pyi stub file (Requirement 21, Task 21.2).

This file is checked with mypy to ensure type stubs are consistent with the
actual Rust extension module. Run with: uv run mypy tests/test_type_stubs.py
"""

import alps_discovery as alps


def test_type_stubs_local_network() -> None:
    """Type check LocalNetwork methods."""
    # Basic constructor
    network1: alps.LocalNetwork = alps.LocalNetwork()

    # With similarity_threshold
    _network2: alps.LocalNetwork = alps.LocalNetwork(similarity_threshold=0.2)

    # With custom scorer
    scorer: alps.TfIdfScorer = alps.TfIdfScorer()
    _network3: alps.LocalNetwork = alps.LocalNetwork(scorer=scorer)

    # With config
    config: alps.DiscoveryConfig = alps.DiscoveryConfig(
        similarity_threshold=0.15,
        diameter_initial=0.6,
    )
    _network4: alps.LocalNetwork = alps.LocalNetwork(config=config)

    # Register agent
    network1.register("test-agent", ["capability"])
    network1.register(
        "test-agent-2",
        ["capability A", "capability B"],
        endpoint="http://localhost:8080",
        metadata={"protocol": "mcp"},
    )

    # Discover (string query)
    _results1: list[alps.DiscoveryResult] = network1.discover("test query")

    # Discover with filters
    filters: dict[str, alps.FilterValue] = {"protocol": alps.FilterValue.exact("mcp")}
    _results2: list[alps.DiscoveryResult] = network1.discover(
        "test query",
        filters=filters,
    )

    # Discover with confidence
    response: alps.DiscoveryResponse = network1.discover(
        "test query",
        with_confidence=True,
    )

    # Check response attributes
    _results: list[alps.DiscoveryResult] = response.results
    _confidence: str = response.confidence
    _parallelism: int = response.recommended_parallelism
    _best_below: tuple[str, float] | None = response.best_below_threshold

    # Query algebra
    query: alps.Query = alps.Query.all("translate", "legal")
    _results3: list[alps.DiscoveryResult] = network1.discover(query)

    # Feedback
    network1.record_success("test-agent", query="test query")
    network1.record_failure("test-agent", query="test query")
    network1.tick()


def test_type_stubs_discovery_config() -> None:
    """Type check DiscoveryConfig."""
    # Default config
    config1: alps.DiscoveryConfig = alps.DiscoveryConfig()

    # Custom values
    _config2: alps.DiscoveryConfig = alps.DiscoveryConfig(
        similarity_threshold=0.2,
        feedback_relevance_threshold=0.4,
        tie_epsilon=2e-4,
        tau_floor=0.002,
        max_feedback_records=200,
        diameter_initial=0.6,
        diameter_min=0.02,
        diameter_max=1.5,
        epsilon_initial=0.7,
        epsilon_floor=0.1,
        epsilon_decay_rate=0.95,
        max_disagreement_split=4,
    )

    # Property access
    _threshold: float = config1.similarity_threshold
    _feedback_threshold: float = config1.feedback_relevance_threshold
    _tau: float = config1.tau_floor
    _diameter: float = config1.diameter_initial


def test_type_stubs_filter_value() -> None:
    """Type check FilterValue."""
    _filter1: alps.FilterValue = alps.FilterValue.exact("value")
    _filter2: alps.FilterValue = alps.FilterValue.contains("substring")
    _filter3: alps.FilterValue = alps.FilterValue.one_of(["a", "b", "c"])
    _filter4: alps.FilterValue = alps.FilterValue.less_than(10.0)
    _filter5: alps.FilterValue = alps.FilterValue.greater_than(1.0)


def test_type_stubs_query() -> None:
    """Type check Query algebra."""
    q1: alps.Query = alps.Query.all("translate", "legal")
    _q2: alps.Query = alps.Query.any("translate", "interpret")
    _q3: alps.Query = alps.Query.weighted({"translate": 2.0, "legal": 1.0})
    _q4: alps.Query = q1.exclude("medical")


def test_type_stubs_capabilities_from_a2a() -> None:
    """Type check capabilities_from_a2a functions."""
    agent_card: dict[str, object] = {
        "name": "test-agent",
        "description": "Test agent",
        "skills": [
            {
                "name": "test_skill",
                "description": "Test skill",
                "tags": ["tag1", "tag2"],
            }
        ],
    }

    # Python implementation
    _caps1: list[str] = alps.capabilities_from_a2a(agent_card)

    # Rust implementation
    _caps2: list[str] = alps._native.capabilities_from_a2a_rust(agent_card)


def test_type_stubs_mcp_integration() -> None:
    """Type check capabilities_from_mcp."""
    tools: list[dict[str, object]] = [
        {
            "name": "translate_text",
            "description": "Translate text between languages",
            "inputSchema": {
                "properties": {
                    "source_text": {},
                    "target_language": {},
                }
            },
        }
    ]

    _caps: list[str] = alps.capabilities_from_mcp(tools)
