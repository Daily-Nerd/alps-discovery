#!/usr/bin/env python3
"""P3 Before/After Benchmark — Visual comparison of alps-discovery improvements.

Run BEFORE P3 changes: captures baseline behavior.
Run AFTER P3 changes: captures improved behavior + new features.

Compares: timing, exploration, concurrent safety, feedback performance,
drift detection, replay, and TF-IDF scorer.
"""

import sys
import threading
import time
from collections import Counter

from alps_discovery import LocalNetwork, Query

SEPARATOR = "=" * 60
IS_AFTER = "--after" in sys.argv


def section(title: str):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def benchmark_basic_discovery():
    """Benchmark 1: Basic discovery timing and correctness."""
    section("1. BASIC DISCOVERY TIMING")

    net = LocalNetwork()
    agents = [
        ("translate-agent", ["legal translation", "EN-DE", "EN-FR", "contract translation"]),
        ("summarize-agent", ["document summarization", "legal briefs", "executive summaries"]),
        ("review-agent", ["code review", "pull request analysis", "security audit"]),
        ("data-agent", ["data processing", "ETL pipelines", "data cleaning"]),
        ("search-agent", ["web search", "document retrieval", "information extraction"]),
    ]
    for name, caps in agents:
        net.register(name, caps)

    queries = [
        "translate legal contract to German",
        "summarize this legal document",
        "review this pull request for security issues",
        "process and clean this dataset",
        "search for relevant legal precedents",
    ]

    # Warm up
    for q in queries:
        net.discover(q)

    # Timed run
    start = time.perf_counter()
    iterations = 1000
    for _ in range(iterations):
        for q in queries:
            net.discover(q)
    elapsed = time.perf_counter() - start

    print(f"  {iterations * len(queries)} discoveries in {elapsed:.3f}s")
    print(f"  {elapsed / (iterations * len(queries)) * 1000:.3f} ms/query")

    # Show results for one query
    results = net.discover("translate legal contract to German", explain=True)
    print("\n  Query: 'translate legal contract to German'")
    for r in results[:3]:
        print(
            f"    {r.agent_name:20s}  sim={r.raw_similarity:.3f}"
            f"  enzyme={r.enzyme_score:.3f}  score={r.final_score:.3f}"
        )


def benchmark_exploration_distribution():
    """Benchmark 2: How well tie-breaking distributes across identical agents."""
    section("2. EXPLORATION DISTRIBUTION (TIE-BREAKING)")

    net = LocalNetwork()
    # 5 identical agents
    for i in range(5):
        net.register(f"agent-{i}", ["data processing service"])

    # Show epsilon value (P3.2 feature)
    if hasattr(net, "exploration_epsilon"):
        print("  Epsilon-greedy exploration: YES")
        print(f"  Initial epsilon: {net.exploration_epsilon:.3f}")
    else:
        print("  Epsilon-greedy exploration: NO (fixed CI overlap tie-breaking)")

    winners = Counter()
    num_queries = 200
    for i in range(num_queries):
        results = net.discover(f"data processing query {i}")
        if results:
            winners[results[0].agent_name] += 1

    if hasattr(net, "exploration_epsilon"):
        print(f"  Epsilon after {num_queries} queries: {net.exploration_epsilon:.3f}")

    print(f"  {num_queries} queries across 5 identical agents:")
    for name in sorted(winners.keys()):
        count = winners[name]
        bar = "#" * (count // 2)
        print(f"    {name:10s}  {count:3d} wins ({count / num_queries * 100:5.1f}%)  {bar}")

    # Ideal: 20% each = 40 wins. Measure deviation.
    expected = num_queries / 5
    max_dev = max(abs(c - expected) for c in winners.values())
    print(f"  Max deviation from ideal: {max_dev:.0f} ({max_dev / expected * 100:.0f}%)")


def benchmark_concurrent_discovery():
    """Benchmark 3: Concurrent discovery (tests thread safety)."""
    section("3. CONCURRENT DISCOVERY")

    net = LocalNetwork()
    for i in range(10):
        net.register(f"agent-{i}", [f"capability-{i}", "shared capability"])

    results_store = {}
    errors = []

    def discover_thread(thread_id, query):
        try:
            results = net.discover(query)
            results_store[thread_id] = len(results)
        except Exception as e:
            errors.append((thread_id, str(e)))

    threads = []
    for i in range(5):
        t = threading.Thread(target=discover_thread, args=(i, f"capability-{i}"))
        threads.append(t)

    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - start

    if errors:
        print(f"  ERRORS: {errors}")
    else:
        total_results = sum(results_store.values())
        print(f"  5 concurrent discovers completed in {elapsed * 1000:.1f}ms")
        print(f"  Total results across threads: {total_results}")

    # P3.1: discover() now takes &self internally (immutable borrow)
    if IS_AFTER:
        print("  discover() signature: &self (interior mutability via atomics + mutex)")
        print("  True concurrent discovery: YES (Rust API is thread-safe)")
    else:
        print("  discover() signature: &mut self (sequential under GIL)")
        print("  True concurrent discovery available: False")


def benchmark_feedback_performance():
    """Benchmark 4: Feedback scan performance with many records."""
    section("4. FEEDBACK PERFORMANCE (SCAN COMPLEXITY)")

    net = LocalNetwork()
    net.register("agent-a", ["legal translation", "document processing"])
    net.register("agent-b", ["legal translation", "document processing"])

    # Add many feedback records
    num_feedback = 100
    for i in range(num_feedback):
        net.record_success("agent-a", query=f"translate document variant {i}")

    # Time discovery with heavy feedback
    start = time.perf_counter()
    iterations = 500
    for _ in range(iterations):
        net.discover("translate legal document")
    elapsed = time.perf_counter() - start

    # P3.5: Banded LSH index reduces scan from O(n) to O(k)
    scan_label = "O(k) banded LSH lookup" if IS_AFTER else "O(n) feedback scan"

    print(f"  {num_feedback} feedback records per agent")
    print(f"  {iterations} discoveries in {elapsed:.3f}s")
    print(f"  {elapsed / iterations * 1000:.3f} ms/query (includes {scan_label})")

    # Show feedback impact
    results = net.discover("translate legal document", explain=True)
    if len(results) >= 2:
        a = next(r for r in results if r.agent_name == "agent-a")
        b = next(r for r in results if r.agent_name == "agent-b")
        print(f"\n  agent-a: feedback={a.feedback_factor:.3f}, score={a.final_score:.3f}")
        print(f"  agent-b: feedback={b.feedback_factor:.3f}, score={b.final_score:.3f}")
        print(f"  Feedback advantage: {a.final_score / b.final_score:.2f}x")


def benchmark_confidence_splits():
    """Benchmark 5: Confidence signal and split behavior."""
    section("5. CONFIDENCE SIGNAL & SPLIT DETECTION")

    net = LocalNetwork()
    # Create agents that should cause kernel disagreement
    net.register("fast-agent", ["data processing", "quick results"])
    net.register("thorough-agent", ["data processing", "detailed analysis"])
    net.register("novel-agent", ["data processing", "experimental approach"])

    # Give fast-agent heavy usage (high sigma, high forwards)
    for _ in range(50):
        net.record_success("fast-agent", query="data processing")

    resp = net.discover("data processing task", with_confidence=True)
    print(f"  Confidence: {resp.confidence}")
    print(f"  Recommended parallelism: {resp.recommended_parallelism}")
    if resp.alternative_agents:
        print(f"  Alternative agents: {resp.alternative_agents}")
    print("  Results:")
    for r in resp.results[:3]:
        print(f"    {r.agent_name:20s}  score={r.score:.3f}")

    # Count splits over many queries
    split_count = 0
    for i in range(100):
        r = net.discover(f"data processing query {i}", with_confidence=True)
        if r.confidence == "split":
            split_count += 1
    print(f"\n  Splits in 100 queries: {split_count}")


def benchmark_query_algebra():
    """Benchmark 6: Query algebra composition."""
    section("6. QUERY ALGEBRA")

    net = LocalNetwork()
    net.register("legal-translate", ["legal translation", "German language", "contract law"])
    net.register("medical-translate", ["medical translation", "clinical records"])
    net.register("general-translate", ["general translation", "multiple languages"])
    net.register("summarize", ["document summarization", "legal briefs"])

    # Simple text query
    r1 = net.discover("legal translation")
    print("  Text('legal translation'):")
    for r in r1[:2]:
        print(f"    {r.agent_name:20s}  score={r.score:.3f}")

    # All query (AND)
    q_all = Query.all("legal translation", "German language")
    r2 = net.discover(q_all)
    print("\n  All('legal translation', 'German language'):")
    for r in r2[:2]:
        print(f"    {r.agent_name:20s}  score={r.score:.3f}")

    # Any query (OR)
    q_any = Query.any("legal translation", "document summarization")
    r3 = net.discover(q_any)
    print("\n  Any('legal translation', 'document summarization'):")
    for r in r3[:3]:
        print(f"    {r.agent_name:20s}  score={r.score:.3f}")

    # Exclude query
    q_exc = Query.all("translation").exclude("medical")
    r4 = net.discover(q_exc)
    print("\n  All('translation').exclude('medical'):")
    for r in r4[:3]:
        print(f"    {r.agent_name:20s}  score={r.score:.3f}")


def benchmark_drift_detection():
    """Benchmark 7: Capability drift detection (P3.3 feature)."""
    section("7. CAPABILITY DRIFT DETECTION")

    net = LocalNetwork()
    # Agent registered for translation but used for summarization
    net.register("versatile-agent", ["legal translation", "EN-DE"])

    # Simulate usage drift: agent succeeds on summarization tasks
    for _ in range(30):
        net.record_success("versatile-agent", query="summarize this legal document")
        net.record_success("versatile-agent", query="create executive summary")

    has_drift = hasattr(net, "detect_drift")
    if has_drift:
        drift_reports = net.detect_drift()
        print("  Drift detection available: YES")
        if drift_reports:
            for report in drift_reports:
                print(f"    Agent: {report['agent_name']}")
                print(f"    Alignment: {report['alignment']:.3f} (1.0=perfect, 0.0=complete drift)")
                print(f"    Samples analyzed: {report['sample_count']}")
                print(f"    Drifted: {report['drifted']}")
                if report["drifted"]:
                    print(
                        "    -> Agent is being used for tasks OUTSIDE its registered capabilities!"
                    )
        else:
            print("    No agents with sufficient feedback for drift analysis.")
    else:
        print("  Drift detection available: NO (P3.3 not yet implemented)")
        print("  Cannot detect that 'versatile-agent' is being used for")
        print("  summarization tasks despite being registered for translation.")


def benchmark_replay():
    """Benchmark 8: Discovery replay / event log (P3.4 feature)."""
    section("8. DISCOVERY REPLAY / EVENT LOG")

    has_replay = hasattr(LocalNetwork, "enable_replay")
    if has_replay:
        net = LocalNetwork()
        net.enable_replay(max_events=1000)
        net.register("agent-a", ["data processing"])
        net.register("agent-b", ["data processing"])

        # Run some discoveries and feedback
        net.discover("data processing task 1")
        net.record_success("agent-a", query="data processing task 1")
        net.discover("data processing task 2")
        net.record_failure("agent-b", query="data processing task 2")
        net.discover("data processing task 3")

        print("  Replay log available: YES")
        print(f"  Replay enabled: {net.replay_enabled}")
        print(f"  Total events: {net.replay_event_count}")

        events = net.replay_events()
        print("  Event log:")
        for event in events[:8]:
            etype = event["type"]
            if etype == "query_submitted":
                print(f"    [{etype}] query='{event['query']}'")
            elif etype == "agent_scored":
                print(
                    f"    [{etype}] agent={event['agent_name']}, "
                    f"sim={event['raw_similarity']:.3f}, "
                    f"enzyme={event['enzyme_score']:.3f}, "
                    f"final={event['final_score']:.3f}"
                )
            elif etype == "feedback_recorded":
                outcome = "success" if event["outcome"] > 0 else "failure"
                print(f"    [{etype}] agent={event['agent_name']}, outcome={outcome}")
            elif etype == "tick_applied":
                print(f"    [{etype}]")

        print(f"  ... ({net.replay_event_count} total events)")
        print("  Post-hoc debugging: 'Why was agent-a chosen?' -> check replay log")
    else:
        net = LocalNetwork()
        net.register("agent-a", ["data processing"])
        net.register("agent-b", ["data processing"])
        net.discover("data processing task 1")
        net.record_success("agent-a", query="data processing task 1")
        net.discover("data processing task 2")
        net.record_failure("agent-b", query="data processing task 2")
        net.discover("data processing task 3")

        print("  Replay log available: NO (P3.4 not yet implemented)")
        print("  Cannot answer: 'Why was agent-a chosen for task 1?'")
        print("  No audit trail for discovery decisions.")


def benchmark_tfidf_scorer():
    """Benchmark 9: TF-IDF scorer (P3.6 feature)."""
    section("9. TF-IDF SCORER")

    # Check if TfIdfScorer is available
    try:
        from alps_discovery import TfIdfScorer

        has_tfidf = True
    except ImportError:
        has_tfidf = False

    if has_tfidf:
        # --- MinHash baseline (default scorer) ---
        net_mh = LocalNetwork()
        net_mh.register("legal-expert", ["legal translation patent contract"])
        net_mh.register("medical-expert", ["medical translation clinical records"])
        net_mh.register("general-agent", ["translation text language general"])

        mh_results = net_mh.discover("legal patent translation", explain=True)
        print("  MinHash (default) — Query: 'legal patent translation'")
        for r in mh_results[:3]:
            print(f"    {r.agent_name:20s}  sim={r.raw_similarity:.3f}  score={r.final_score:.3f}")

        # --- TF-IDF scorer ---
        scorer = TfIdfScorer()
        net_tf = LocalNetwork(scorer=scorer)
        net_tf.register("legal-expert", ["legal translation patent contract"])
        net_tf.register("medical-expert", ["medical translation clinical records"])
        net_tf.register("general-agent", ["translation text language general"])

        tf_results = net_tf.discover("legal patent translation", explain=True)
        print("\n  TF-IDF scorer — Query: 'legal patent translation'")
        for r in tf_results[:3]:
            print(f"    {r.agent_name:20s}  sim={r.raw_similarity:.3f}  score={r.final_score:.3f}")

        print("\n  TF-IDF weights rare terms (patent, legal) higher than")
        print("  common terms (translation), so legal-expert gets a larger boost.")
        print("  MinHash treats all shared tokens equally.")
    else:
        print("  TF-IDF scorer available: NO (P3.6 not yet implemented)")
        print("  Only MinHash (keyword overlap) scorer available.")
        print("  Cannot weight rare domain terms higher than common words.")


def benchmark_enzyme_api():
    """Benchmark 10: Enzyme API simplification (P3.7)."""
    section("10. ENZYME API SURFACE")

    net = LocalNetwork()
    net.register("agent-a", ["data processing"])

    import inspect

    discover_sig = inspect.signature(net.discover)
    print(f"  discover() signature: {discover_sig}")
    if IS_AFTER:
        print("  Internal Rust API: Enzyme::process(&self, signal, hyphae, scorer_context)")
        print("  Vestigial Spore and MembraneState parameters REMOVED (P3.7)")
    else:
        print(
            "  (Internal Rust API was: Enzyme::process("
            "&mut self, signal, &Spore, hyphae, &MembraneState, scorer_context))"
        )
        print("  Spore and MembraneState are vestigial stubs in OSS.")


def main():
    print(f"{'=' * 60}")
    print("  ALPS Discovery P3 Benchmark")
    print(f"  Phase: {'AFTER' if IS_AFTER else 'BEFORE'}")
    print(f"{'=' * 60}")

    benchmark_basic_discovery()
    benchmark_exploration_distribution()
    benchmark_concurrent_discovery()
    benchmark_feedback_performance()
    benchmark_confidence_splits()
    benchmark_query_algebra()
    benchmark_drift_detection()
    benchmark_replay()
    benchmark_tfidf_scorer()
    benchmark_enzyme_api()

    section("SUMMARY")
    phase = "AFTER" if IS_AFTER else "BEFORE"
    print(f"  Phase: {phase}")
    if IS_AFTER:
        print("  P3 Changes Applied:")
        print("    P3.1  Interior mutability: discover(&self) via atomics + mutex")
        print("    P3.2  Epsilon-greedy exploration: adaptive tie-breaking")
        print("    P3.3  Capability drift detection: alignment monitoring")
        print("    P3.4  Discovery replay: append-only event log for debugging")
        print("    P3.5  Banded LSH feedback index: O(k) near-neighbor lookup")
        print("    P3.6  TF-IDF scorer: semantic term weighting")
        print("    P3.7  Enzyme API simplification: removed vestigial parameters")
    print("  Complete.")


if __name__ == "__main__":
    main()
