# alps-discovery

Local agent discovery powered by bio-inspired multi-kernel routing.

Register agents with capability descriptions, then discover the best match for any natural-language query. Ranking improves over time via success/failure feedback.

## Install

Requires Python 3.11+ and [Rust](https://rustup.rs/).

```bash
pip install maturin
git clone https://github.com/Daily-Nerd/alps-discovery.git
cd alps-discovery
maturin develop
```

## Quick start

```python
import alps_discovery as alps

network = alps.LocalNetwork()
network.register("translate-agent", ["legal translation", "EN-DE", "EN-FR"])
network.register("summarize-agent", ["document summarization", "legal briefs"])
network.register("classify-agent", ["document classification", "contract type detection"])

results = network.discover("translate legal contract")
for r in results:
    print(f"{r.agent_name}: similarity={r.similarity:.3f}, score={r.score:.3f}")
# translate-agent: similarity=0.344, score=0.344
# summarize-agent: similarity=0.188, score=0.188
# classify-agent:  similarity=0.094, score=0.094
```

## Feedback loop

Ranking adapts to real-world outcomes:

```python
# Agent performed well — boost its ranking
network.record_success("translate-agent")

# Agent failed — reduce its ranking
network.record_failure("classify-agent")

# Future queries reflect the feedback
results = network.discover("translate legal contract")
```

## API

### `LocalNetwork()`

Create a new discovery network.

### `.register(name: str, capabilities: list[str])`

Register an agent. Each capability string is converted to a MinHash signature for similarity matching.

### `.deregister(name: str) -> bool`

Remove an agent. Returns `True` if found.

### `.discover(query: str) -> list[DiscoveryResult]`

Find agents matching a query. Returns results sorted by score (best first). Each result has:
- `agent_name` — the matched agent
- `similarity` — capability match strength `[0.0, 1.0]`
- `score` — combined routing score (similarity x diameter, adjusted by feedback)

### `.record_success(agent_name: str)` / `.record_failure(agent_name: str)`

Update routing state based on interaction outcomes.

### `.agent_count` (property) / `.agents() -> list[str]`

Inspect registered agents.

## How it works

Each agent's capabilities become MinHash signatures (locality-sensitive hashing). Queries are converted to the same signature space. Similarity is the best match across individual capabilities (max per-capability Jaccard estimate).

Three independent reasoning kernels vote on routing:
- **CapabilityKernel** — ranks by chemistry similarity to the query
- **LoadBalancingKernel** — distributes traffic across equally-capable agents
- **NoveltyKernel** — favors less-explored agents for diversity

Success/failure feedback adjusts the agent's diameter (routing weight), creating an adaptive system that learns which agents deliver.

## License

Apache-2.0
