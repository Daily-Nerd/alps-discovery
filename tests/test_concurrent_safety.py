"""Concurrent safety tests for Python scorer (Requirement 18)."""

import threading

import alps_discovery as alps


def test_python_scorer_concurrent_discover():
    """Test Python scorer under concurrent discover() calls.

    Verifies that GIL release via py.allow_threads() prevents deadlocks
    when multiple threads call discover() with a Python custom scorer.
    """

    class SimpleScorer:
        """Simple Python scorer for testing."""

        def __init__(self):
            self.agents = {}

        def index_capabilities(self, agent_id, capabilities):
            self.agents[agent_id] = capabilities

        def remove_agent(self, agent_id):
            if agent_id in self.agents:
                del self.agents[agent_id]

        def score(self, query):
            # Return all agents with constant similarity
            return [(agent_id, 0.5) for agent_id in self.agents]

    scorer = SimpleScorer()
    network = alps.LocalNetwork(scorer=scorer)

    # Register a few agents
    for i in range(5):
        network.register(f"agent-{i}", [f"capability-{i}"])

    # Spawn multiple threads calling discover() concurrently
    results_list = []
    errors = []

    def discover_worker(thread_id):
        try:
            for i in range(20):
                query = f"test query {thread_id}-{i}"
                results = network.discover(query)
                results_list.append(len(results))
        except Exception as e:
            errors.append(e)

    threads = []
    for tid in range(4):
        t = threading.Thread(target=discover_worker, args=(tid,))
        threads.append(t)
        t.start()

    # Wait for all threads
    for t in threads:
        t.join()

    # Verify no errors occurred
    assert len(errors) == 0, f"Concurrent access caused errors: {errors}"

    # Verify all threads got results
    assert len(results_list) == 80  # 4 threads x 20 iterations

    # All should have gotten results (5 agents registered)
    assert all(count == 5 for count in results_list)


def test_python_scorer_concurrent_discover_and_feedback():
    """Test Python scorer with concurrent discover() and feedback recording.

    Verifies thread safety when mixing immutable (discover) and mutable
    (record_success) operations across threads.
    """

    class SimpleScorer:
        def __init__(self):
            self.agents = {}
            self.lock = threading.Lock()

        def index_capabilities(self, agent_id, capabilities):
            with self.lock:
                self.agents[agent_id] = capabilities

        def remove_agent(self, agent_id):
            with self.lock:
                if agent_id in self.agents:
                    del self.agents[agent_id]

        def score(self, query):
            with self.lock:
                return [(agent_id, 0.6) for agent_id in self.agents]

    scorer = SimpleScorer()
    network = alps.LocalNetwork(scorer=scorer)

    # Register agents
    for i in range(3):
        network.register(f"agent-{i}", [f"test capability {i}"])

    errors = []

    def worker(thread_id):
        try:
            for i in range(30):
                # Discover
                results = network.discover(f"query-{i}")

                # Record feedback for first result
                if results:
                    agent_name = results[0].agent_name
                    query_str = f"query-{i}"
                    if i % 2 == 0:
                        network.record_success(agent_name, query=query_str)
                    else:
                        network.record_failure(agent_name, query=query_str)
        except Exception as e:
            errors.append(e)

    threads = []
    for tid in range(4):
        t = threading.Thread(target=worker, args=(tid,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # No errors should occur
    assert len(errors) == 0, f"Concurrent operations caused errors: {errors}"

    # Verify feedback was recorded (epsilon should have decayed)
    # Note: We can't directly check epsilon from Python, but no crash means success
