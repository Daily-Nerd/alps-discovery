"""Feedback loop: how success/failure shifts rankings over time.

Demonstrates global feedback (diameter adjustment) and per-query
feedback (query-specific boosting/penalizing). Per-query feedback
lets an agent be "good at X" without being boosted for unrelated "Y".
"""

import alps_discovery as alps

network = alps.LocalNetwork()

# Two agents with identical capabilities — start equal
network.register("agent-alpha", ["data processing", "analytics"], endpoint="http://alpha:8000")
network.register("agent-beta", ["data processing", "analytics"], endpoint="http://beta:8000")

query = "data processing pipeline"

print("=== Initial state (identical agents) ===")
results = network.discover(query)
for r in results:
    print(f"  {r.agent_name}: similarity={r.similarity:.3f}, score={r.score:.3f}")

# Simulate: alpha succeeds consistently, beta fails
print("\n=== Simulating 20 rounds: alpha succeeds, beta fails ===")
for _i in range(20):
    network.record_success("agent-alpha")
    network.record_failure("agent-beta")

results = network.discover(query)
for r in results:
    print(f"  {r.agent_name}: similarity={r.similarity:.3f}, score={r.score:.3f}")

alpha = next(r for r in results if r.agent_name == "agent-alpha")
beta = next(r for r in results if r.agent_name == "agent-beta")
print(f"\n  Alpha/Beta score ratio: {alpha.score / beta.score:.1f}x")

# Can beta recover?
print("\n=== Recovery: beta succeeds 40 rounds ===")
for _i in range(40):
    network.record_success("agent-beta")

results = network.discover(query)
for r in results:
    print(f"  {r.agent_name}: similarity={r.similarity:.3f}, score={r.score:.3f}")

alpha = next(r for r in results if r.agent_name == "agent-alpha")
beta = next(r for r in results if r.agent_name == "agent-beta")
print(f"\n  Alpha/Beta score ratio: {alpha.score / beta.score:.1f}x")

print("\n  Note: diameter clamps to [0.1, 1.0], so agents converge")
print("  toward 1.0 after enough successes. Prevents runaway.")

# ---------------------------------------------------------------------------
print(f"\n{'=' * 60}")
print("  PER-QUERY FEEDBACK")
print(f"{'=' * 60}\n")
# ---------------------------------------------------------------------------

network2 = alps.LocalNetwork()
network2.register("multi-agent", ["translation", "summarization"], endpoint="http://multi:8000")
network2.register(
    "translate-only", ["translation", "legal translation"], endpoint="http://translate:8000"
)

translate_q = "translate legal contract"
summarize_q = "summarize a document"


def score_for(results, name):
    return next((r.score for r in results if r.agent_name == name), None)


print("Before feedback:")
t_score = score_for(network2.discover(translate_q), "multi-agent")
s_score = score_for(network2.discover(summarize_q), "multi-agent")
print(f"  multi-agent on '{translate_q}': score={t_score:.4f}")
print(f"  multi-agent on '{summarize_q}':  score={s_score:.4f}")

# Record 10 translation successes WITH the query context
for _ in range(10):
    network2.record_success("multi-agent", query=translate_q)

print("\nAfter 10 translation successes (with query= kwarg):")
t_score_after = score_for(network2.discover(translate_q), "multi-agent")
s_score_after = score_for(network2.discover(summarize_q), "multi-agent")
print(f"  multi-agent on '{translate_q}': score={t_score_after:.4f}")
print(f"  multi-agent on '{summarize_q}':  score={s_score_after:.4f}")

if t_score and t_score_after and s_score and s_score_after:
    t_boost = (t_score_after - t_score) / t_score * 100
    s_boost = (s_score_after - s_score) / s_score * 100
    print(f"\n  Translation query boost: {t_boost:+.1f}%")
    print(f"  Summarize query boost:   {s_boost:+.1f}% (smaller — different query type)")

print("\n  Per-query feedback targets similar queries, not all queries.")
print("  Always pass query= to record_success/failure for best results.")
