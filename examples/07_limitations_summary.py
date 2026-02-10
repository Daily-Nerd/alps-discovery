"""Known limitations of MinHash-based capability matching.

Run this to see the remaining limitations demonstrated with concrete
examples. These are inherent to the lexical matching approach — solvable
by plugging in a custom scorer (e.g. embedding-based).
"""

import alps_discovery as alps


def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
section("1. LEXICAL MATCHING — no semantic understanding")
# ---------------------------------------------------------------------------

network = alps.LocalNetwork()
network.register("translator", ["language translation services"], endpoint="http://translator:8000")
network.register(
    "localizer", ["localization and internationalization"], endpoint="http://localizer:8000"
)

for query in ["translate document", "localize content", "convert to another language"]:
    results = network.discover(query)
    print(f"  '{query}':")
    for r in results:
        print(f"    {r.agent_name}: sim={r.similarity:.3f}")
    print()

print("  MinHash sees character n-grams, not meaning.")
print("  'convert to another language' has low overlap with 'translation'.")
print("  Workaround: use descriptive, overlapping capability strings,")
print("  or plug in an embedding-based scorer (see 12_custom_scorer.py).")

# ---------------------------------------------------------------------------
section("2. SHORT STRINGS — limited n-gram surface")
# ---------------------------------------------------------------------------

network = alps.LocalNetwork()
network.register("agent-a", ["NLP"], endpoint="http://a:8000")
network.register("agent-b", ["natural language processing"], endpoint="http://b:8000")

for query in ["NLP", "natural language processing", "text analysis"]:
    results = network.discover(query)
    print(f"  '{query}':")
    for r in results:
        print(f"    {r.agent_name}: sim={r.similarity:.3f}")
    print()

print("  Very short strings (e.g. 'NLP') produce few shingles,")
print("  making similarity unreliable. Use longer, descriptive capabilities.")

# ---------------------------------------------------------------------------
section("3. ACRONYMS AND ABBREVIATIONS — no expansion")
# ---------------------------------------------------------------------------

network = alps.LocalNetwork()
network.register("ml-agent", ["machine learning model training"], endpoint="http://ml:8000")

for query in ["ML training", "machine learning", "train a model"]:
    results = network.discover(query)
    if results:
        print(f"  '{query}' -> {results[0].agent_name}: sim={results[0].similarity:.3f}")
    else:
        print(f"  '{query}' -> NO MATCHES")

print()
print("  'ML' doesn't match 'machine learning' lexically.")
print("  Include both forms in capabilities to handle this.")

# ---------------------------------------------------------------------------
section("WHAT'S SOLVED")
# ---------------------------------------------------------------------------

print("""  These former limitations are now addressed:

    - Per-query feedback: record_success/failure accept query= kwarg
      so feedback only affects similar query types (see 04_feedback_loop.py)

    - Metadata filtering: discover(query, filters={...}) supports
      $in, $lt, $gt, $contains operators (see 09_metadata_filters.py)

    - Persistence: save()/load() preserves agents, scores, and
      feedback history across restarts (see 10_persistence.py)

    - Randomized tie-breaking: identical agents are shuffled so
      traffic distributes naturally (see 06_scale_and_load_balancing.py)

    - Explain mode: discover(query, explain=True) shows full scoring
      breakdown for debugging (see 11_explain_mode.py)

    - Pluggable scorer: swap MinHash for any scorer (e.g. embeddings)
      via the scorer= kwarg (see 12_custom_scorer.py)
""")
