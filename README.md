# alps-discovery

Local agent discovery powered by bio-inspired multi-kernel routing.

Register agents with capability descriptions, then discover the best match for any natural-language query. Ranking improves over time via success/failure feedback.

## Install

Requires Python 3.11+ and [Rust](https://rustup.rs/).

```bash
# Eventually: pip install alps-discovery
# For now (development):
pip install maturin
git clone https://github.com/Daily-Nerd/alps-discovery.git
cd alps-discovery
maturin develop
```

## Quick start

```python
import alps_discovery as alps

network = alps.LocalNetwork()

network.register("translate-agent", ["legal translation", "EN-DE", "EN-FR"],
                  endpoint="http://localhost:8080/translate",
                  metadata={"protocol": "mcp", "version": "1.0"})

network.register("summarize-agent", ["document summarization", "legal briefs"],
                  endpoint="http://localhost:9090/summarize")

results = network.discover("translate legal contract")
best = results[0]
print(best.agent_name)  # "translate-agent"
print(best.endpoint)    # "http://localhost:8080/translate"
print(best.metadata)    # {"protocol": "mcp", "version": "1.0"}
```

## Using discovery results

ALPS is DNS for agents — it resolves capability queries to endpoints. You invoke agents using your own client:

```python
results = network.discover("translate legal contract")
best = results[0]

# Use the endpoint with your existing client
if best.endpoint:
    response = mcp_client.call(best.endpoint, {"text": doc, "target": "DE"})
    network.record_success(best.agent_name)
```

For local single-process agents, register an optional `invoke` callable:

```python
def my_translate(request):
    return {"translated": do_translation(request["text"])}

network.register("local-translator", ["legal translation"],
                  invoke=my_translate)

results = network.discover("translate legal contract")
if results[0].invoke:
    output = results[0].invoke({"text": "Vertragsbedingungen..."})
    network.record_success(results[0].agent_name)
```

## MCP server integration

Register agents directly from MCP tool definitions — no manual capability transcription:

```python
import alps_discovery as alps

tools = mcp_client.list_tools()  # MCP tool definitions
network.register("my-mcp-server", alps.capabilities_from_mcp(tools),
                  endpoint="http://localhost:3001",
                  metadata={"protocol": "mcp"})
```

`capabilities_from_mcp()` extracts tool names, descriptions, and parameter descriptions into capability strings. The richer your tool schemas, the better the matching.

## Feedback loop

Ranking adapts to real-world outcomes:

```python
# Agent performed well — boost its ranking
network.record_success("translate-agent")

# Agent failed — reduce its ranking
network.record_failure("summarize-agent")

# Future queries reflect the feedback
results = network.discover("translate legal contract")
```

## Metadata filtering

Filter discovery results by agent metadata:

```python
network.register("fast-mcp", ["translation"],
                  endpoint="http://localhost:8080",
                  metadata={"protocol": "mcp", "latency_ms": "50"})

results = network.discover("translate contract",
                           filters={"protocol": "mcp"})

# Operator syntax:
results = network.discover("translate contract",
                           filters={"latency_ms": {"$lt": 100}})
```

## Persistence

Save and restore network state (agents, feedback, scoring):

```python
# Save state (agents, feedback, scoring)
network.save("state.json")

# Load on restart
network = alps.LocalNetwork.load("state.json")
```

## Explain mode

Inspect how scores are calculated:

```python
results = network.discover("translate contract", explain=True)
for r in results:
    # Explain mode provides full scoring breakdown
    # Note: raw_similarity and final_score also work in regular mode (as aliases)
    print(f"{r.agent_name}: sim={r.raw_similarity:.3f}, "
          f"diameter={r.diameter:.3f}, feedback={r.feedback_factor:.3f}, "
          f"score={r.final_score:.3f}")
```

## API

### `LocalNetwork(*, similarity_threshold=None, scorer=None)`

Create a new discovery network.

- `similarity_threshold` — optional float `[0.0, 1.0]` to filter low-similarity matches (default: 0.1)
- `scorer` — optional custom scorer object implementing the scorer protocol (see [Community Scorers](#community-scorers))

### `.register(name, capabilities, *, endpoint=None, metadata=None, invoke=None)`

Register an agent.

- `name` — unique agent identifier
- `capabilities` — list of capability description strings (converted to MinHash signatures)
- `endpoint` — optional URI/URL for the agent (MCP server, REST API, etc.)
- `metadata` — optional dict of key-value pairs (protocol, version, framework, etc.)
- `invoke` — optional callable for local single-process invocation

### `.deregister(name: str) -> bool`

Remove an agent. Returns `True` if found.

### `.discover(query: str, *, filters=None, explain=False) -> list[DiscoveryResult]`

Find agents matching a query. Returns results sorted by score (best first).

- `filters` — optional dict of metadata key-value pairs to filter results (supports operators like `{"latency_ms": {"$lt": 100}}`)
- `explain` — if `True`, returns `ExplainedResult` objects with detailed scoring breakdown

Each `DiscoveryResult` has:
- `agent_name` — the matched agent
- `similarity` — capability match strength `[0.0, 1.0]`
- `score` — combined routing score (similarity x diameter, adjusted by feedback)
- `endpoint` — agent URI/URL if provided at registration, else `None`
- `metadata` — dict of key-value pairs if provided, else `{}`
- `invoke` — callable if provided at registration, else `None`

Each `ExplainedResult` has all `DiscoveryResult` fields plus:
- `raw_similarity` — base similarity before feedback
- `diameter` — agent's routing weight (adjusted by feedback)
- `feedback_factor` — multiplier from success/failure history
- `final_score` — `raw_similarity * diameter * feedback_factor`

### `.discover_filtered(query: str, filters: dict) -> list[DiscoveryResult]`

Direct filtered discovery (Rust API). Equivalent to `.discover(query, filters=filters)`.

### `.save(path: str)`

Save network state (agents, feedback, scoring) to a JSON file.

### `LocalNetwork.load(path: str) -> LocalNetwork`

Load network state from a JSON file. Returns a new `LocalNetwork` instance.

### `.record_success(agent_name)` / `.record_failure(agent_name)`

Update routing state based on interaction outcomes.

### `.agent_count` (property) / `.agents() -> list[str]`

Inspect registered agents.

### `capabilities_from_mcp(tools: list[dict]) -> list[str]`

Extract capability strings from MCP tool definitions. Flattens tool names, descriptions, and parameter descriptions into strings suitable for `register()`. See [MCP server integration](#mcp-server-integration).

## How it works

Each agent's capabilities become MinHash signatures (locality-sensitive hashing). Queries are converted to the same signature space. Similarity is the best match across individual capabilities (max per-capability Jaccard estimate).

Three independent reasoning kernels vote on routing:
- **CapabilityKernel** — ranks by chemistry similarity to the query
- **LoadBalancingKernel** — distributes traffic across equally-capable agents
- **NoveltyKernel** — favors less-explored agents for diversity

Success/failure feedback adjusts the agent's diameter (routing weight), creating an adaptive system that learns which agents deliver.

## Community Scorers

The MinHash default handles lexical matching well. For semantic similarity (synonyms, paraphrases), implement the scorer protocol and pass it to `LocalNetwork(scorer=...)`:

```python
class EmbeddingScorer:
    """Example: plug in sentence-transformers for semantic matching."""
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.agents = {}

    def index_capabilities(self, agent_id, capabilities):
        self.agents[agent_id] = self.model.encode(capabilities)

    def remove_agent(self, agent_id):
        self.agents.pop(agent_id, None)

    def score(self, query):
        import numpy as np
        q_emb = self.model.encode([query])[0]
        results = []
        for agent_id, cap_embs in self.agents.items():
            sims = np.dot(cap_embs, q_emb) / (
                np.linalg.norm(cap_embs, axis=1) * np.linalg.norm(q_emb)
            )
            sim = float(np.max(sims))
            if sim > 0.3:
                results.append((agent_id, sim))
        return results

# Use it:
network = alps.LocalNetwork(scorer=EmbeddingScorer())
```

## Current limitations

- **Lexical matching by default.** Similarity is based on character n-grams (MinHash), not meaning. Use descriptive capability strings for best results. For semantic matching (synonyms, paraphrases), use a pluggable scorer (see [Community Scorers](#community-scorers)).

- **Single-process.** Network discovery across machines is in development.

## License

Apache-2.0
