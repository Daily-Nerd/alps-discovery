"""Internal: comprehensive limitations test for planning purposes.

Run this to see every known limitation demonstrated with concrete examples.
Use this for roadmap planning — not intended as user-facing documentation.
"""

import alps_discovery as alps


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
section("1. LEXICAL MATCHING — no semantic understanding")
# ---------------------------------------------------------------------------

network = alps.LocalNetwork()
network.register("translator", ["language translation services"],
                  endpoint="http://translator:8000")
network.register("localizer", ["localization and internationalization"],
                  endpoint="http://localizer:8000")

for query in ["translate document", "localize content", "convert to another language"]:
    results = network.discover(query)
    print(f"  '{query}':")
    for r in results:
        print(f"    {r.agent_name}: sim={r.similarity:.3f}")
    print()

print("  MinHash sees character 3-grams, not meaning.")
print("  Workaround: use descriptive, overlapping capability strings.")
print("  Or use capabilities_from_mcp() for MCP servers.")

# ---------------------------------------------------------------------------
section("2. FEEDBACK IS GLOBAL — not per-query-type")
# ---------------------------------------------------------------------------

network = alps.LocalNetwork()
network.register("multi-agent", ["translation", "summarization"],
                  endpoint="http://multi:8000")

for _ in range(10):
    network.record_success("multi-agent")   # from translation queries
for _ in range(10):
    network.record_failure("multi-agent")   # from summarization queries

results = network.discover("summarize a document")
if results:
    print(f"  'summarize a document' -> {results[0].agent_name}: score={results[0].score:.3f}")
print()
print("  record_success/failure adjusts GLOBAL diameter.")
print("  Can't distinguish 'good at X' from 'bad at Y'.")

# ---------------------------------------------------------------------------
section("3. NO CAPABILITY SCHEMA — flat strings only")
# ---------------------------------------------------------------------------

network = alps.LocalNetwork()
network.register("fast-agent", ["translation", "low-latency"],
                  endpoint="http://fast:8000", metadata={"latency_ms": "50"})
network.register("quality-agent", ["translation", "high-accuracy"],
                  endpoint="http://quality:8000", metadata={"accuracy": "0.98"})

results = network.discover("translate with high accuracy")
print("  Query: 'translate with high accuracy'")
for r in results:
    print(f"    {r.agent_name}: sim={r.similarity:.3f}")
print()
print("  Metadata is passthrough only — not used in matching.")
print("  Can't filter by latency < 100ms or accuracy > 0.95.")

# ---------------------------------------------------------------------------
section("4. IN-MEMORY ONLY — no persistence")
# ---------------------------------------------------------------------------

print("  LocalNetwork is in-memory. No save()/load().")
print("  Re-register agents on startup from your own store.")

# ---------------------------------------------------------------------------
section("5. DETERMINISTIC TIE-BREAKING — no load balancing for identical agents")
# ---------------------------------------------------------------------------

network = alps.LocalNetwork()
for i in range(5):
    network.register(f"worker-{i}", ["data processing"],
                      endpoint=f"http://worker-{i}:8000")

from collections import Counter
picks = Counter()
for _ in range(50):
    results = network.discover("data processing")
    picks[results[0].agent_name] += 1

print("  50 queries across 5 identical agents:")
for agent, count in picks.most_common():
    print(f"    {agent}: {count}")
print()
print("  Same agent wins every time due to deterministic enzyme.")
print("  Real workloads have different capabilities — this is synthetic.")

# ---------------------------------------------------------------------------
section("SUMMARY")
# ---------------------------------------------------------------------------

print("""  What matters now:
    1. Lexical matching — workaround: descriptive capabilities
    2. Global feedback — acceptable for v0.1
    3. No schema filtering — metadata passthrough only
    4. In-memory — re-register on startup
    5. Deterministic ties — only for identical agents
""")
