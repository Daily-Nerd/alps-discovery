"""Persistence: save and load network state across restarts.

Network state — agents, capabilities, feedback history, and scoring
weights — is saved to a JSON file and restored on load. This means
you don't need to re-register agents or lose feedback on restart.
"""

import os

import alps_discovery as alps

STATE_FILE = "/tmp/alps_demo_state.json"

# --- Build a network with agents and feedback ---
print("=== Building network ===")
network = alps.LocalNetwork()

network.register(
    "translate-agent",
    ["legal translation", "EN-DE", "EN-FR"],
    endpoint="http://localhost:8080/translate",
    metadata={"protocol": "mcp"},
)
network.register(
    "summarize-agent",
    ["document summarization", "legal briefs"],
    endpoint="http://localhost:9090/summarize",
    metadata={"protocol": "rest"},
)

# Build some feedback history
for _ in range(15):
    network.record_success("translate-agent", query="translate legal contract")
for _ in range(5):
    network.record_failure("summarize-agent", query="summarize legal brief")

# Check scores before saving
results_before = network.discover("translate legal contract")
print(f"Agents: {network.agents()}")
print(f"Agent count: {network.agent_count}")
for r in results_before:
    print(f"  {r.agent_name}: score={r.score:.4f}")

# --- Save ---
print(f"\n=== Saving to {STATE_FILE} ===")
network.save(STATE_FILE)
print(f"  Saved ({os.path.getsize(STATE_FILE)} bytes)")

# --- Load into a new network ---
print(f"\n=== Loading from {STATE_FILE} ===")
loaded = alps.LocalNetwork.load(STATE_FILE)
print(f"Agents: {loaded.agents()}")
print(f"Agent count: {loaded.agent_count}")

# --- Verify scores match ---
results_after = loaded.discover("translate legal contract")
print("\nScores after load:")
for r in results_after:
    print(f"  {r.agent_name}: score={r.score:.4f}")

# Compare (note: result counts may differ due to temporal state rehydration)
print("\n=== Verification ===")
# Build lookups for comparison
before_by_name = {r.agent_name: r for r in results_before}
after_by_name = {r.agent_name: r for r in results_after}

for agent_name in sorted(set(before_by_name.keys()) | set(after_by_name.keys())):
    before_score = before_by_name[agent_name].score if agent_name in before_by_name else None
    after_score = after_by_name[agent_name].score if agent_name in after_by_name else None

    if before_score and after_score:
        match = "MATCH" if abs(before_score - after_score) < 0.1 else "DIFF"
        print(f"  {agent_name}: {before_score:.4f} -> {after_score:.4f} [{match}]")
    elif before_score:
        print(f"  {agent_name}: {before_score:.4f} -> (not in results after load)")
    else:
        print(f"  {agent_name}: (not in results before) -> {after_score:.4f}")

# --- Continue using the loaded network ---
print("\n=== Continuing with loaded network ===")
loaded.register(
    "classify-agent",
    ["document classification"],
    endpoint="http://localhost:7070/classify",
)
print(f"Added classify-agent. Agent count: {loaded.agent_count}")

results = loaded.discover("classify this contract")
if results:
    print(f"  '{results[0].agent_name}' found for classification query")

# Clean up
os.remove(STATE_FILE)
print(f"\nCleaned up {STATE_FILE}")
