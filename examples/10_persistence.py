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

# Compare
print("\n=== Verification ===")
for before, after in zip(results_before, results_after, strict=True):
    match = "MATCH" if abs(before.score - after.score) < 0.001 else "DIFF"
    print(f"  {before.agent_name}: {before.score:.4f} -> {after.score:.4f} [{match}]")

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
