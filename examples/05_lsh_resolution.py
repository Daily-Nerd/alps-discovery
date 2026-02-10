"""LSH resolution: what the MinHash similarity can and can't distinguish.

This example probes the limits of 64-byte MinHash with 3-byte shingles.
Short strings, similar strings, and semantic meaning are all explored.
"""

import alps_discovery as alps

network = alps.LocalNetwork()

# Register agents with varying capability styles
agents = {
    "legal-translate": ["legal translation", "contract translation", "EN-DE"],
    "medical-translate": ["medical translation", "clinical notes", "EN-DE"],
    "general-translate": ["translation services"],
    "legal-summarize": ["legal summarization", "contract summaries"],
    "code-review": ["code review", "static analysis", "Python"],
}

for name, caps in agents.items():
    network.register(name, caps)

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
    "DE",
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

print("=== LIMITATIONS ===")
print()
print("1. NO SEMANTIC UNDERSTANDING:")
print("   'convert contract to German' != 'legal translation'")
print("   MinHash matches character n-grams, not meaning.")
print("   Synonyms, paraphrases, and intent are invisible.")
print()
print("2. SHORT STRINGS ARE NOISY:")
print("   'DE' or 'legal' have few 3-byte shingles, producing")
print("   low-resolution signatures with unreliable similarity.")
print()
print("3. CAPABILITIES MUST SHARE VOCABULARY WITH QUERIES:")
print("   If users say 'translate' but capabilities say 'localization',")
print("   similarity will be low. No embedding-based semantic matching.")
