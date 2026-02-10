"""Metadata filtering: narrow discovery results by agent properties.

Use filters to select agents by protocol, version, latency, or any
metadata you provide at registration. Filters are applied AFTER scoring,
so you get the best-matching agents that also meet your constraints.

Filter operators:
    - str: exact match            {"protocol": "mcp"}
    - $in: one of several values  {"protocol": {"$in": ["mcp", "rest"]}}
    - $contains: substring match  {"name": {"$contains": "legal"}}
    - $lt: numeric less-than      {"latency_ms": {"$lt": 100.0}}
    - $gt: numeric greater-than   {"accuracy": {"$gt": 0.95}}
"""

import alps_discovery as alps

network = alps.LocalNetwork()

# Register agents with rich metadata
network.register(
    "translate-mcp",
    ["legal translation", "EN-DE", "EN-FR"],
    endpoint="http://localhost:3001",
    metadata={"protocol": "mcp", "version": "2025-03-26", "latency_ms": "45"},
)

network.register(
    "translate-rest",
    ["legal translation", "EN-DE"],
    endpoint="https://api.example.com/translate",
    metadata={"protocol": "rest", "version": "1.2.0", "latency_ms": "120"},
)

network.register(
    "summarize-mcp",
    ["document summarization", "legal briefs"],
    endpoint="http://localhost:3002",
    metadata={"protocol": "mcp", "version": "2025-03-26", "latency_ms": "80"},
)

network.register(
    "summarize-grpc",
    ["document summarization", "contract analysis"],
    endpoint="grpc://localhost:50051",
    metadata={"protocol": "grpc", "version": "3.0.0", "latency_ms": "30"},
)

query = "translate legal contract"

# --- Exact match ---
print("=== Exact match: protocol = 'mcp' ===")
results = network.discover(query, filters={"protocol": "mcp"})
for r in results:
    print(f"  {r.agent_name} (sim={r.similarity:.3f}) protocol={r.metadata.get('protocol')}")
print()

# --- $in: one of several values ---
print("=== $in: protocol in ['mcp', 'grpc'] ===")
results = network.discover(query, filters={"protocol": {"$in": ["mcp", "grpc"]}})
for r in results:
    print(f"  {r.agent_name} (sim={r.similarity:.3f}) protocol={r.metadata.get('protocol')}")
print()

# --- $lt: numeric less-than ---
print("=== $lt: latency_ms < 100 ===")
results = network.discover(query, filters={"latency_ms": {"$lt": 100.0}})
for r in results:
    print(f"  {r.agent_name} (sim={r.similarity:.3f}) latency={r.metadata.get('latency_ms')}ms")
print()

# --- $gt: numeric greater-than ---
print("=== $gt: latency_ms > 50 ===")
results = network.discover(query, filters={"latency_ms": {"$gt": 50.0}})
for r in results:
    print(f"  {r.agent_name} (sim={r.similarity:.3f}) latency={r.metadata.get('latency_ms')}ms")
print()

# --- $contains: substring match ---
print("=== $contains: version contains '2025' ===")
results = network.discover(query, filters={"version": {"$contains": "2025"}})
for r in results:
    print(f"  {r.agent_name} (sim={r.similarity:.3f}) version={r.metadata.get('version')}")
print()

# --- Multiple filters (AND logic) ---
print("=== Combined: protocol='mcp' AND latency_ms < 100 ===")
results = network.discover(
    query,
    filters={"protocol": "mcp", "latency_ms": {"$lt": 100.0}},
)
for r in results:
    proto = r.metadata.get("protocol")
    lat = r.metadata.get("latency_ms")
    print(f"  {r.agent_name} (sim={r.similarity:.3f}) proto={proto} lat={lat}ms")
print()

# --- No matches ---
print("=== No matches: protocol='websocket' ===")
results = network.discover(query, filters={"protocol": "websocket"})
print(f"  Results: {len(results)} (filters that match nothing return empty list)")
