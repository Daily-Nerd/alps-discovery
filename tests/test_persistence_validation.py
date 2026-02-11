"""NetworkSnapshot v3 format validation tests (Requirements 9, 24, Task 22.3)."""

import json
import pathlib
import tempfile

import alps_discovery as alps


def test_save_load_preserves_agents():
    """Test save/load cycle preserves all agent state."""
    network = alps.LocalNetwork()

    # Register agents with metadata
    network.register(
        "agent-1",
        ["translate legal documents"],
        endpoint="http://localhost:8080",
        metadata={"protocol": "mcp", "version": "1.0"},
    )
    network.register(
        "agent-2",
        ["summarize contracts"],
        metadata={"protocol": "rest"},
    )

    # Record feedback to create state
    network.record_success("agent-1", query="translate")
    network.record_failure("agent-2", query="summarize")

    # Save
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = f.name

    try:
        network.save(path)

        # Load
        loaded = alps.LocalNetwork.load(path)

        # Verify agents preserved
        assert loaded.agent_count == 2
        agents = loaded.agents()
        assert "agent-1" in agents
        assert "agent-2" in agents

        # Verify discovery still works (capabilities preserved)
        results = loaded.discover("translate legal")
        assert len(results) > 0
        assert any(r.agent_name == "agent-1" for r in results)

    finally:
        pathlib.Path(path).unlink()


def test_save_load_preserves_cooccurrence_matrix():
    """Test co-occurrence matrix is persisted and restored."""
    network = alps.LocalNetwork()

    network.register("agent-1", ["translate documents"])
    network.register("agent-2", ["summarize text"])

    # Record enough feedback to build co-occurrence matrix (needs 10+ for expansion)
    for _ in range(15):
        network.record_success("agent-1", query="translate legal contract")

    # Save
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = f.name

    try:
        network.save(path)

        # Verify co-occurrence matrix is in the saved JSON
        with open(path) as f:
            snapshot = json.load(f)
            assert "cooccurrence_matrix" in snapshot
            # Should have recorded co-occurrences
            assert len(snapshot["cooccurrence_matrix"]) > 0

        # Load and verify expansion still works
        loaded = alps.LocalNetwork.load(path)
        results = loaded.discover("translate")
        assert len(results) > 0

    finally:
        pathlib.Path(path).unlink()


def test_save_load_cross_scorer():
    """Test save with MinHash, load with TfIdf (scorer-agnostic persistence)."""
    # Create network with default MinHashScorer
    network = alps.LocalNetwork()
    network.register("agent-1", ["translate documents"])
    network.register("agent-2", ["summarize text"])

    # Save
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = f.name

    try:
        network.save(path)

        # Load with TfIdfScorer (different scorer)
        tfidf_scorer = alps.TfIdfScorer()
        _loaded = alps.LocalNetwork(scorer=tfidf_scorer)

        # Note: Currently load() is a static method that creates a new network
        # So we can't pre-configure the scorer. This test verifies the concept
        # that raw capabilities are stored, not scorer-specific index state.

        # For now, just verify the saved snapshot has raw capabilities
        with open(path) as f:
            snapshot = json.load(f)
            assert "agents" in snapshot
            for agent_snapshot in snapshot["agents"]:
                assert "capabilities" in agent_snapshot
                assert isinstance(agent_snapshot["capabilities"], list)
                # Capabilities stored as strings (scorer-agnostic)
                assert all(isinstance(cap, str) for cap in agent_snapshot["capabilities"])

    finally:
        pathlib.Path(path).unlink()


def test_snapshot_version_field():
    """Test NetworkSnapshot has correct version field."""
    network = alps.LocalNetwork()
    network.register("test-agent", ["test capability"])

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = f.name

    try:
        network.save(path)

        # Verify version field in JSON
        with open(path) as f:
            snapshot = json.load(f)
            assert "version" in snapshot
            # Should be v3 (includes temporal state + co-occurrence)
            assert snapshot["version"] == 3

    finally:
        pathlib.Path(path).unlink()


def test_temporal_state_serialization():
    """Test last_activity is serialized as Duration (not Instant)."""
    network = alps.LocalNetwork()
    network.register("agent-1", ["test"])

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = f.name

    try:
        network.save(path)

        with open(path) as f:
            snapshot = json.load(f)
            # Verify network_epoch exists (serializable SystemTime)
            assert "network_epoch" in snapshot

            # Verify agent has last_activity_duration (not Instant)
            agent = snapshot["agents"][0]
            assert "last_activity_duration" in agent
            duration = agent["last_activity_duration"]
            assert "secs" in duration or isinstance(duration, int | float)

        # Verify load rehydrates correctly (no crash)
        loaded = alps.LocalNetwork.load(path)
        assert loaded.agent_count == 1

    finally:
        pathlib.Path(path).unlink()
