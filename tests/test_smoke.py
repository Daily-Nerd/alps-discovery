"""Smoke tests for alps_discovery Python bindings."""

import os
import tempfile

import alps_discovery as alps


def test_register_and_discover():
    """Basic round-trip: register agents and discover by query."""
    network = alps.LocalNetwork()
    network.register("translate-agent", ["legal translation", "EN-DE"])
    network.register("summarize-agent", ["document summarization"])

    results = network.discover("legal translation")
    assert len(results) >= 1
    assert results[0].agent_name == "translate-agent"
    assert results[0].similarity > 0.0
    assert results[0].score > 0.0


def test_discover_with_metadata_filter():
    """Metadata filters narrow results correctly."""
    network = alps.LocalNetwork()
    network.register(
        "agent-mcp",
        ["legal translation"],
        metadata={"protocol": "mcp"},
    )
    network.register(
        "agent-rest",
        ["legal translation"],
        metadata={"protocol": "rest"},
    )

    results = network.discover(
        "legal translation",
        filters={"protocol": "mcp"},
    )
    assert len(results) == 1
    assert results[0].agent_name == "agent-mcp"


def test_explain_mode():
    """Explain mode returns ExplainedResult with scoring breakdown."""
    network = alps.LocalNetwork()
    network.register("agent-a", ["data processing"])

    results = network.discover("data processing", explain=True)
    assert len(results) >= 1
    r = results[0]
    assert hasattr(r, "raw_similarity")
    assert hasattr(r, "diameter")
    assert hasattr(r, "feedback_factor")
    assert hasattr(r, "final_score")
    assert r.raw_similarity > 0.0
    assert r.diameter > 0.0


def test_save_and_load():
    """Persistence round-trip preserves agents and scores."""
    network = alps.LocalNetwork()
    network.register(
        "translate-agent",
        ["legal translation"],
        endpoint="http://localhost:8080",
        metadata={"protocol": "mcp"},
    )

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        network.save(path)
        loaded = alps.LocalNetwork.load(path)
        assert loaded.agent_count == 1
        assert "translate-agent" in loaded.agents()

        results = loaded.discover("legal translation")
        assert len(results) >= 1
        assert results[0].agent_name == "translate-agent"
        assert results[0].endpoint == "http://localhost:8080"
    finally:
        os.unlink(path)


def test_capabilities_from_mcp():
    """MCP tool extraction produces composite capability strings."""
    tools = [
        {
            "name": "translate_text",
            "description": "Translate between languages",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "target_lang": {"type": "string"},
                },
            },
        },
    ]
    caps = alps.capabilities_from_mcp(tools)
    assert len(caps) == 1
    assert "translate text" in caps[0]
    assert "Translate between languages" in caps[0]
    assert "text" in caps[0]
    assert "target lang" in caps[0]


def test_discover_many():
    """Batch discovery returns one result list per query."""
    network = alps.LocalNetwork()
    network.register("translate-agent", ["legal translation services"])
    network.register("summarize-agent", ["document summarization"])

    results = network.discover_many(["legal translation", "document summarization"])
    assert len(results) == 2
    assert len(results[0]) >= 1
    assert len(results[1]) >= 1
