"""Quickstart: register agents, discover, give feedback."""

import alps_discovery as alps

network = alps.LocalNetwork()

# Register agents with capabilities and endpoints
network.register(
    "translate-agent",
    ["legal translation", "EN-DE", "EN-FR"],
    endpoint="http://localhost:8080/translate",
    metadata={"protocol": "mcp", "version": "1.0"},
)
network.register(
    "summarize-agent",
    ["document summarization", "legal briefs"],
    endpoint="http://localhost:9090/summarize",
)
network.register("classify-agent", ["document classification", "contract type detection"])

# Discover the best agent for a task
results = network.discover("translate legal contract")
print("=== Discovery: 'translate legal contract' ===")
for r in results:
    print(f"  {r}")

# The top result includes the endpoint to invoke
best = results[0]
print(f"\nBest match: {best.agent_name}")
print(f"  endpoint: {best.endpoint}")
print(f"  metadata: {best.metadata}")

# Feedback loop: tell the network what worked
network.record_success("translate-agent")
network.record_success("translate-agent")
network.record_failure("classify-agent")

# Re-discover — feedback shifts rankings
results = network.discover("translate legal contract")
print("\n=== After feedback ===")
for r in results:
    print(f"  {r}")

# Inspect network state
print(f"\nRegistered agents ({network.agent_count}): {network.agents()}")
