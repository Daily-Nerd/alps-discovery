"""Custom scorer: plug in your own matching engine.

alps-discovery uses MinHash (lexical n-gram similarity) by default,
but you can replace it with any scorer — embeddings, BM25, hybrid,
or even a remote API call. Just implement three methods:

    index_capabilities(agent_id: str, capabilities: list[str])
    remove_agent(agent_id: str)
    score(query: str) -> list[tuple[str, float]]

The scorer handles matching; alps-discovery handles routing (diameter,
feedback, tie-breaking, filtering).
"""

import alps_discovery as alps

# --- Example 1: Keyword scorer (simple, no dependencies) ---


class KeywordScorer:
    """Scores agents by counting keyword overlaps.

    A minimal scorer to demonstrate the protocol. In production,
    you'd use something like sentence-transformers for semantic matching.
    """

    def __init__(self):
        self.agents: dict[str, set[str]] = {}

    def index_capabilities(self, agent_id: str, capabilities: list[str]) -> None:
        keywords = set()
        for cap in capabilities:
            keywords.update(cap.lower().split())
        self.agents[agent_id] = keywords

    def remove_agent(self, agent_id: str) -> None:
        self.agents.pop(agent_id, None)

    def score(self, query: str) -> list[tuple[str, float]]:
        query_words = set(query.lower().split())
        results = []
        for agent_id, keywords in self.agents.items():
            if not query_words or not keywords:
                continue
            overlap = len(query_words & keywords)
            similarity = overlap / max(len(query_words), len(keywords))
            if similarity > 0.0:
                results.append((agent_id, similarity))
        return results


# Use the custom scorer
print("=== KeywordScorer (word overlap) ===\n")
network = alps.LocalNetwork(scorer=KeywordScorer())

network.register(
    "translate-agent", ["legal translation EN-DE EN-FR"], endpoint="http://localhost:8080/translate"
)
network.register(
    "summarize-agent",
    ["document summarization legal briefs"],
    endpoint="http://localhost:9090/summarize",
)
network.register(
    "classify-agent",
    ["document classification contract type"],
    endpoint="http://localhost:7070/classify",
)

queries = [
    "legal translation",
    "summarize document",
    "classify contract type",
    "translate legal contract",
]

for q in queries:
    results = network.discover(q)
    if results:
        best = results[0]
        print(f"  '{q}'")
        print(f"    -> {best.agent_name} (score={best.score:.3f})")
    else:
        print(f"  '{q}' -> NO MATCHES")
print()

# Feedback still works with custom scorers
print("=== Feedback works with any scorer ===\n")
before = network.discover("legal translation")
print(f"Before feedback: {before[0].agent_name} score={before[0].score:.4f}")

for _ in range(10):
    network.record_success("translate-agent", query="legal translation")

after = network.discover("legal translation")
print(f"After feedback:  {after[0].agent_name} score={after[0].score:.4f}")

# --- Example 2: Stub for embedding scorer ---
print()
print("=" * 60)
print("  EMBEDDING SCORER STUB")
print("=" * 60)
print()

print("""To use sentence-transformers for semantic matching:

    from sentence_transformers import SentenceTransformer
    import numpy as np

    class EmbeddingScorer:
        def __init__(self, model="all-MiniLM-L6-v2"):
            self.model = SentenceTransformer(model)
            self.agents = {}

        def index_capabilities(self, agent_id, capabilities):
            self.agents[agent_id] = self.model.encode(capabilities)

        def remove_agent(self, agent_id):
            self.agents.pop(agent_id, None)

        def score(self, query):
            q = self.model.encode([query])[0]
            results = []
            for aid, embs in self.agents.items():
                sim = max(np.dot(q, e) / (np.linalg.norm(q) * np.linalg.norm(e))
                          for e in embs)
                if sim > 0.3:
                    results.append((aid, float(sim)))
            return results

    network = alps.LocalNetwork(scorer=EmbeddingScorer())
""")
print("  This solves the synonym problem: 'convert to German' matches")
print("  'language translation' semantically, not just lexically.")
