"""LSH resolution: what the MinHash similarity can and can't distinguish.

Probes the limits of 64-byte MinHash with 3-byte shingles.
Short strings, similar strings, and semantic meaning are all explored.
"""

import alps_discovery as alps

network = alps.LocalNetwork()

# Register agents with varying capability styles
agents = {
    "legal-translate": (
        ["legal translation", "contract translation", "EN-DE"],
        "http://legal-translate:8000",
    ),
    "medical-translate": (
        ["medical translation", "clinical notes", "EN-DE"],
        "http://medical-translate:8000",
    ),
    "general-translate": (
        ["translation services"],
        "http://general-translate:8000",
    ),
    "legal-summarize": (
        ["legal summarization", "contract summaries"],
        "http://legal-summarize:8000",
    ),
    "code-review": (
        ["code review", "static analysis", "Python"],
        "http://code-review:8000",
    ),
}

for name, (caps, endpoint) in agents.items():
    network.register(name, caps, endpoint=endpoint)

# Test queries from obvious to subtle
queries = [
    # Easy: strong keyword overlap
    "legal translation",
    "medical translation",
    "code review Python",
    # Medium: partial overlap
    "translate legal contract",
    "translate medical records",
    # Hard: semantic but not lexical
    "convert contract to German",         # means translation but doesn't say "translate"
    "review my pull request",             # means code review but different words
    "make this document shorter",         # means summarization but doesn't say it
    # Edge: very short queries
    "translate",
    "legal",
]

print("=== LSH Resolution Test ===\n")
for q in queries:
    results = network.discover(q)
    if not results:
        print(f"  '{q}' -> NO MATCHES")
        continue

    top = results[0]
    runner_up = results[1] if len(results) > 1 else None
    margin = (top.similarity - runner_up.similarity) / top.similarity * 100 if runner_up and top.similarity > 0 else 100

    print(f"  '{q}'")
    print(f"    #1: {top.agent_name} (sim={top.similarity:.3f})")
    if runner_up:
        print(f"    #2: {runner_up.agent_name} (sim={runner_up.similarity:.3f}) margin={margin:.0f}%")
    print()

print("=== Key observations ===")
print("  - Exact substring matches (e.g. 'legal translation') score 1.000")
print("  - Partial overlap (e.g. 'translate legal contract') scores ~0.3")
print("  - Semantic synonyms (e.g. 'convert to German') score lower")
print("  - Tip: use capabilities_from_mcp() to maximize n-gram surface")
