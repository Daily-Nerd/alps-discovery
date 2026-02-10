"""Explain mode: see exactly how routing decisions are made.

discover(query, explain=True) returns ExplainedResult objects with the
full scoring breakdown: raw similarity, agent diameter (health weight),
per-query feedback factor, and the final combined score.

Use this to debug why an agent ranks higher or lower than expected.
"""

import alps_discovery as alps

network = alps.LocalNetwork()

network.register(
    "translate-agent",
    ["legal translation", "EN-DE", "EN-FR"],
    endpoint="http://localhost:8080/translate",
)
network.register(
    "summarize-agent",
    ["document summarization", "legal briefs"],
    endpoint="http://localhost:9090/summarize",
)
network.register(
    "classify-agent",
    ["document classification", "contract type detection"],
)

# --- Basic explain ---
print("=== Explain: 'translate legal contract' ===\n")
results = network.discover("translate legal contract", explain=True)
for r in results:
    print(f"  {r.agent_name}:")
    print(f"    raw_similarity = {r.raw_similarity:.4f}  (MinHash overlap)")
    print(f"    diameter       = {r.diameter:.4f}  (agent health weight)")
    print(f"    feedback_factor= {r.feedback_factor:.4f}  (per-query adjustment)")
    print(f"    final_score    = {r.final_score:.4f}  (combined routing score)")
    print(f"    endpoint       = {r.endpoint}")
    print()

# --- Show how feedback changes the breakdown ---
print("=" * 60)
print("  AFTER FEEDBACK")
print("=" * 60)
print()

# Boost translate-agent with query-specific feedback
for _ in range(15):
    network.record_success("translate-agent", query="translate legal contract")

# Penalize summarize-agent globally
for _ in range(10):
    network.record_failure("summarize-agent")

results = network.discover("translate legal contract", explain=True)
for r in results:
    print(f"  {r.agent_name}:")
    print(f"    raw_similarity = {r.raw_similarity:.4f}")
    print(f"    diameter       = {r.diameter:.4f}")
    print(f"    feedback_factor= {r.feedback_factor:.4f}")
    print(f"    final_score    = {r.final_score:.4f}")
    print()

# --- Compare explained vs regular discover ---
print("=== Consistency check: explain scores match regular discover ===\n")
explained = network.discover("translate legal contract", explain=True)
regular = network.discover("translate legal contract")

for e, r in zip(explained, regular, strict=True):
    match = "OK" if abs(e.final_score - r.score) < 0.001 else "MISMATCH"
    print(f"  {e.agent_name}: explain={e.final_score:.4f}, regular={r.score:.4f} [{match}]")
