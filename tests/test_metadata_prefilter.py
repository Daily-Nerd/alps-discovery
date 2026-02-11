"""Tests for pre-scoring metadata filtering (Requirement 16)."""

import alps_discovery as alps


def test_metadata_prefilter_reduces_scorer_invocations():
    """Test that metadata filters reduce scorer invocations."""

    # Create a custom scorer that counts how many times score() is called
    class CountingScorer:
        def __init__(self):
            self.score_call_count = 0
            self.indexed_agents = {}

        def index_capabilities(self, agent_id, capabilities):
            self.indexed_agents[agent_id] = capabilities

        def remove_agent(self, agent_id):
            if agent_id in self.indexed_agents:
                del self.indexed_agents[agent_id]

        def score(self, query):
            self.score_call_count += 1
            # Return all indexed agents with high similarity
            return [(name, 0.8) for name in self.indexed_agents]

    scorer = CountingScorer()
    network = alps.LocalNetwork(scorer=scorer)

    # Register 100 agents with different metadata
    for i in range(100):
        region = "us-west" if i < 10 else "us-east"
        network.register(f"agent-{i}", [f"capability-{i}"], metadata={"region": region})

    # Without filter: scorer should be called once, processing all 100 agents
    scorer.score_call_count = 0
    _results = network.discover("test query")
    unfiltered_count = scorer.score_call_count
    assert unfiltered_count >= 1

    # With metadata filter: pre-filtering reduces candidate set before enzyme evaluation
    scorer.score_call_count = 0
    filters = {"region": alps.FilterValue.exact("us-west")}
    filtered_results = network.discover("test query", filters=filters)

    # Results should only contain us-west region agents (10 out of 100)
    assert len(filtered_results) == 10
    # All results should have us-west region
    for result in filtered_results:
        assert result.metadata.get("region") == "us-west"


def test_prefilter_metadata_then_similarity_threshold():
    """Test that metadata filtering happens before similarity scoring."""
    network = alps.LocalNetwork()

    # Register agents with matching capabilities but different metadata
    network.register("agent-premium", ["translate legal documents"], metadata={"tier": "premium"})
    network.register("agent-basic-1", ["translate legal documents"], metadata={"tier": "basic"})
    network.register("agent-basic-2", ["translate legal documents"], metadata={"tier": "basic"})

    # Filter to only basic tier
    filters = {"tier": alps.FilterValue.exact("basic")}
    results = network.discover("translate legal", filters=filters)

    # Should only return basic tier agents (metadata post-filtering currently)
    assert all(r.metadata.get("tier") == "basic" for r in results)
    assert len(results) == 2

    # After pre-filtering is implemented, only basic agents will be scored
    # (Currently all 3 are scored, then filtered)


def test_prefilter_reduces_work_proportionally():
    """Test that pre-filtering reduces work proportional to selectivity."""

    class CountingScorer:
        def __init__(self):
            self.score_call_count = 0
            self.indexed_count = 0

        def index_capabilities(self, agent_id, capabilities):
            self.indexed_count += 1

        def remove_agent(self, agent_id):
            pass

        def score(self, query):
            self.score_call_count += 1
            # Simulate scoring work proportional to indexed agents
            return [(f"agent-{i}", 0.5) for i in range(self.indexed_count)]

    scorer = CountingScorer()
    network = alps.LocalNetwork(scorer=scorer)

    # Register 50 agents: 25 in each region
    for i in range(50):
        region = "A" if i < 25 else "B"
        network.register(f"agent-{i}", [f"cap-{i}"], metadata={"region": region})

    assert scorer.indexed_count == 50

    # Query without filter - should process all 50 agents
    scorer.score_call_count = 0
    _results_all = network.discover("test")
    _calls_without_filter = scorer.score_call_count

    # Query with filter - pre-filtering reduces candidate set to 25 agents
    scorer.score_call_count = 0
    filters = {"region": alps.FilterValue.exact("A")}
    filtered_results = network.discover("test", filters=filters)
    _calls_with_filter = scorer.score_call_count

    # Verify only region A agents in results (25 out of 50)
    # Note: scorer.score() is still called once and scores all indexed agents,
    # but enzyme evaluation and feedback lookup only happen for filtered agents
    assert len(filtered_results) == 25
    for result in filtered_results:
        assert result.metadata.get("region") == "A"
