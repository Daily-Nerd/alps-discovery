"""DNS for agents: endpoint + metadata passthrough.

ALPS resolves capability queries to endpoints. You invoke agents
using your own client (MCP, HTTP, gRPC, etc.). This example shows
how endpoint and metadata flow through discovery.
"""

import alps_discovery as alps

network = alps.LocalNetwork()

# MCP server agent
network.register(
    "translate-mcp",
    ["legal translation", "EN-DE", "EN-FR"],
    endpoint="http://localhost:8080/translate",
    metadata={"protocol": "mcp", "version": "2025-03-26"},
)

# REST API agent
network.register(
    "summarize-api",
    ["document summarization", "legal briefs"],
    endpoint="https://api.example.com/v1/summarize",
    metadata={"protocol": "rest", "auth": "bearer-token"},
)

# Local agent (no endpoint)
network.register(
    "classify-local",
    ["document classification", "contract type detection"],
)

# Discover
results = network.discover("translate legal contract")
print("=== Discovery results ===")
for r in results:
    print(f"  agent: {r.agent_name}")
    print(f"  similarity: {r.similarity:.3f}, score: {r.score:.3f}")
    print(f"  endpoint: {r.endpoint}")
    print(f"  metadata: {r.metadata}")
    print()

# The caller uses endpoint + metadata to invoke
best = results[0]
if best.endpoint:
    proto = best.metadata.get("protocol", "unknown")
    print(f"Would invoke {best.agent_name} via {proto} at {best.endpoint}")
else:
    print(f"{best.agent_name} has no endpoint — local agent only")
