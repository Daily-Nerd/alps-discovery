"""Feedback loop: how success/failure shifts rankings over time.

Demonstrates that record_success/record_failure actually change
agent scores, and explores how many iterations it takes to
meaningfully reorder agents.
"""

import alps_discovery as alps

network = alps.LocalNetwork()

# Two agents with identical capabilities — start equal
network.register("agent-alpha", ["data processing", "analytics"],
                  endpoint="http://alpha:8000")
network.register("agent-beta", ["data processing", "analytics"],
                  endpoint="http://beta:8000")

query = "data processing pipeline"

print("=== Initial state (identical agents) ===")
results = network.discover(query)
for r in results:
    print(f"  {r.agent_name}: similarity={r.similarity:.3f}, score={r.score:.3f}")

# Simulate: alpha succeeds consistently, beta fails
print("\n=== Simulating 20 rounds: alpha succeeds, beta fails ===")
for i in range(20):
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
for i in range(40):
    network.record_success("agent-beta")

results = network.discover(query)
for r in results:
    print(f"  {r.agent_name}: similarity={r.similarity:.3f}, score={r.score:.3f}")

alpha = next(r for r in results if r.agent_name == "agent-alpha")
beta = next(r for r in results if r.agent_name == "agent-beta")
print(f"\n  Alpha/Beta score ratio: {alpha.score / beta.score:.1f}x")

# NOTE: diameter is clamped to [0.1, 1.0], so after enough successes
# both agents converge toward 1.0 and become equal again.
print("\n=== Note: diameter clamps to [0.1, 1.0] ===")
print("  After enough successes, agents converge toward diameter=1.0.")
print("  Feedback saturates — prevents runaway but limits differentiation.")
