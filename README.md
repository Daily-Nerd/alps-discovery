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

## API

### `LocalNetwork()`

Create a new discovery network.

### `.register(name, capabilities, *, endpoint=None, metadata=None, invoke=None)`

Register an agent.

- `name` — unique agent identifier
- `capabilities` — list of capability description strings (converted to MinHash signatures)
- `endpoint` — optional URI/URL for the agent (MCP server, REST API, etc.)
- `metadata` — optional dict of key-value pairs (protocol, version, framework, etc.)
- `invoke` — optional callable for local single-process invocation

### `.deregister(name: str) -> bool`

Remove an agent. Returns `True` if found.

### `.discover(query: str) -> list[DiscoveryResult]`

Find agents matching a query. Returns results sorted by score (best first). Each result has:
- `agent_name` — the matched agent
- `similarity` — capability match strength `[0.0, 1.0]`
- `score` — combined routing score (similarity x diameter, adjusted by feedback)
- `endpoint` — agent URI/URL if provided at registration, else `None`
- `metadata` — dict of key-value pairs if provided, else `{}`
- `invoke` — callable if provided at registration, else `None`

### `.record_success(agent_name)` / `.record_failure(agent_name)`

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
