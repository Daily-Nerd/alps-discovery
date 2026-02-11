#!/usr/bin/env python3
"""ALPS Discovery Convergence Benchmark Suite

Measures feedback convergence rate: how many queries until the correct agent
ranks first, starting from a cold network.

Usage:
    uv run python benchmarks/run.py [--json] [--trials N]

Requirements: Task 20.3, Requirement 22
"""

import argparse
import json
import random
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from alps_discovery import LocalNetwork


@dataclass
class ConvergenceResult:
    """Result of a single convergence trial."""

    queries_until_correct: int
    converged: bool


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results."""

    name: str
    mean_convergence: float
    median_convergence: float
    convergence_rate: float  # % of trials that converged
    min_queries: int
    max_queries: int


# Test dataset: queries mapped to correct agent
TEST_DATASET = [
    ("translate legal contract to German", "translate-agent"),
    ("translate patent application to French", "translate-agent"),
    ("convert this document to Spanish", "translate-agent"),
    ("summarize this legal brief", "summarize-agent"),
    ("create executive summary of report", "summarize-agent"),
    ("condense this research paper", "summarize-agent"),
    ("review pull request for security", "review-agent"),
    ("audit this code for vulnerabilities", "review-agent"),
    ("analyze code quality", "review-agent"),
    ("process ETL pipeline data", "data-agent"),
    ("clean and normalize dataset", "data-agent"),
    ("transform data schema", "data-agent"),
]

# Agent definitions
AGENTS = [
    ("translate-agent", ["legal translation", "document conversion", "language services"]),
    ("summarize-agent", ["document summarization", "executive summaries", "content condensation"]),
    ("review-agent", ["code review", "security audit", "quality analysis"]),
    ("data-agent", ["data processing", "ETL pipelines", "data transformation"]),
]


def run_convergence_trial(
    query: str,
    correct_agent: str,
    max_queries: int = 50,
    seed: int = 42,
) -> ConvergenceResult:
    """Run a single convergence trial.

    Measures how many queries it takes for the correct agent to rank first,
    starting from a cold network (no feedback).
    """
    random.seed(seed)
    network = LocalNetwork()

    # Register all agents
    for name, caps in AGENTS:
        network.register(name, caps)

    # Run queries until correct agent ranks first
    for i in range(1, max_queries + 1):
        results = network.discover(query)

        if results and results[0].agent_name == correct_agent:
            # Converged!
            return ConvergenceResult(queries_until_correct=i, converged=True)

        # Record feedback for learning
        if results:
            selected_agent = results[0].agent_name
            if selected_agent == correct_agent:
                network.record_success(selected_agent, query=query)
            else:
                network.record_failure(selected_agent, query=query)
                # Also boost the correct agent
                network.record_success(correct_agent, query=query)

    # Did not converge within max_queries
    return ConvergenceResult(queries_until_correct=max_queries, converged=False)


def benchmark_alps(trials: int = 10, seed_start: int = 42) -> BenchmarkResult:
    """Benchmark ALPS Discovery with all features enabled."""
    results = []

    for trial_idx in range(trials):
        query, correct_agent = TEST_DATASET[trial_idx % len(TEST_DATASET)]
        seed = seed_start + trial_idx

        result = run_convergence_trial(query, correct_agent, seed=seed)
        results.append(result)

    # Aggregate results
    converged_results = [r for r in results if r.converged]
    convergence_queries = [r.queries_until_correct for r in converged_results]

    if convergence_queries:
        mean_conv = statistics.mean(convergence_queries)
        median_conv = statistics.median(convergence_queries)
        min_q = min(convergence_queries)
        max_q = max(convergence_queries)
    else:
        mean_conv = float("inf")
        median_conv = float("inf")
        min_q = 0
        max_q = 0

    return BenchmarkResult(
        name="ALPS Discovery (Full)",
        mean_convergence=mean_conv,
        median_convergence=median_conv,
        convergence_rate=len(converged_results) / len(results),
        min_queries=min_q,
        max_queries=max_q,
    )


def baseline_random(trials: int = 10, seed_start: int = 42) -> BenchmarkResult:
    """Baseline: random routing (no learning)."""
    results = []
    num_agents = len(AGENTS)

    for trial_idx in range(trials):
        random.seed(seed_start + trial_idx)
        # Expected queries until random hits correct agent: E[Geometric(1/n)] = n
        queries = 0
        while queries < 50:
            queries += 1
            if random.random() < 1.0 / num_agents:
                results.append(ConvergenceResult(queries, True))
                break
        else:
            results.append(ConvergenceResult(50, False))

    converged = [r for r in results if r.converged]
    conv_queries = [r.queries_until_correct for r in converged]

    return BenchmarkResult(
        name="Baseline: Random Routing",
        mean_convergence=statistics.mean(conv_queries) if conv_queries else float("inf"),
        median_convergence=statistics.median(conv_queries) if conv_queries else float("inf"),
        convergence_rate=len(converged) / len(results),
        min_queries=min(conv_queries) if conv_queries else 0,
        max_queries=max(conv_queries) if conv_queries else 0,
    )


def baseline_round_robin(trials: int = 10) -> BenchmarkResult:
    """Baseline: round-robin (deterministic cycling, no learning)."""
    num_agents = len(AGENTS)
    avg_queries = (num_agents + 1) / 2  # Expected value

    return BenchmarkResult(
        name="Baseline: Round-Robin",
        mean_convergence=avg_queries,
        median_convergence=avg_queries,
        convergence_rate=1.0,
        min_queries=1,
        max_queries=num_agents,
    )


def main():
    parser = argparse.ArgumentParser(description="ALPS Discovery Convergence Benchmark")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--trials", type=int, default=10, help="Number of trials")
    parser.add_argument("--quick", action="store_true", help="Quick mode (5 trials)")
    args = parser.parse_args()

    trials = 5 if args.quick else args.trials

    print(f"ALPS Discovery Convergence Benchmark ({trials} trials)")
    print("=" * 70)

    # Benchmark ALPS with all features
    print("\n[1/3] Benchmarking: ALPS Discovery (co-occurrence + mycorrhizal + circuit breaker)...")
    alps_result = benchmark_alps(trials=trials)

    # Baseline 1: Random routing
    print("[2/3] Benchmarking: Random routing baseline...")
    random_baseline = baseline_random(trials=trials)

    # Baseline 2: Round-robin
    print("[3/3] Benchmarking: Round-robin baseline...")
    roundrobin_baseline = baseline_round_robin(trials=trials)

    # Collect results
    results = [alps_result, random_baseline, roundrobin_baseline]

    # Output results
    if args.json:
        output = {
            "trials": trials,
            "deterministic_seed": 42,
            "results": [asdict(r) for r in results],
        }
        print(json.dumps(output, indent=2))
    else:
        print("\n" + "=" * 70)
        print("CONVERGENCE RESULTS")
        print("=" * 70)
        print(f"{'Configuration':<30} {'Mean':<8} {'Median':<8} {'Rate':<10} {'Min':<6} {'Max':<6}")
        print("-" * 70)

        for r in results:
            mean_str = f"{r.mean_convergence:.1f}" if r.mean_convergence != float("inf") else "∞"
            median_str = (
                f"{r.median_convergence:.1f}" if r.median_convergence != float("inf") else "∞"
            )
            rate_str = f"{r.convergence_rate * 100:.0f}%"

            print(
                f"{r.name:<30} {mean_str:<8} {median_str:<8} {rate_str:<10} "
                f"{r.min_queries:<6} {r.max_queries:<6}"
            )

        print("\n" + "=" * 70)
        print("KEY INSIGHTS:")
        print("=" * 70)

        # Compare ALPS vs Random
        if alps_result.mean_convergence < random_baseline.mean_convergence:
            speedup = random_baseline.mean_convergence / alps_result.mean_convergence
            print(f"✓ ALPS converges {speedup:.1f}x faster than random routing")

        # Compare ALPS vs Round-robin
        if alps_result.mean_convergence < roundrobin_baseline.mean_convergence:
            speedup = roundrobin_baseline.mean_convergence / alps_result.mean_convergence
            print(f"✓ ALPS converges {speedup:.1f}x faster than round-robin")

        print(f"✓ Convergence rate: {alps_result.convergence_rate * 100:.0f}% of trials converged")
        print(f"✓ Mean queries to convergence: {alps_result.mean_convergence:.1f}")
        print(f"✓ Median queries to convergence: {alps_result.median_convergence:.1f}")

        if alps_result.mean_convergence < 10:
            print("\n🎯 EXCELLENT: ALPS converges in <10 queries on average")
        elif alps_result.mean_convergence < 20:
            print("\n✓ GOOD: ALPS converges in <20 queries on average")
        else:
            print("\n⚠ Consider tuning: convergence is slower than expected")


if __name__ == "__main__":
    main()
