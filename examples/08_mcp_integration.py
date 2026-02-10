"""MCP integration: register agents from MCP tool definitions.

Instead of manually writing capability strings, extract them
from your MCP server's tool schemas. capabilities_from_mcp() builds
one composite capability string per tool from the name, description,
and parameter names — giving the MinHash engine maximum n-gram surface.
"""

import alps_discovery as alps

# Simulated MCP tool definitions (what list_tools() returns)
translate_tools = [
    {
        "name": "translate_text",
        "description": "Translate text between languages with legal domain expertise",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Source text to translate"},
                "source_lang": {
                    "type": "string",
                    "description": "Source language code like EN, DE, FR",
                },
                "target_lang": {
                    "type": "string",
                    "description": "Target language code like EN, DE, FR",
                },
            },
        },
    },
    {
        "name": "detect_language",
        "description": "Detect the language of input text",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to analyze for language detection"},
            },
        },
    },
]

summarize_tools = [
    {
        "name": "summarize_document",
        "description": "Generate a concise summary of a legal document",
        "inputSchema": {
            "type": "object",
            "properties": {
                "document": {"type": "string", "description": "Full document text to summarize"},
                "max_length": {"type": "integer", "description": "Maximum summary length in words"},
            },
        },
    },
]

# Extract capabilities from MCP tool schemas
# Each tool becomes one composite string: "name: description. param1, param2"
translate_caps = alps.capabilities_from_mcp(translate_tools)
summarize_caps = alps.capabilities_from_mcp(summarize_tools)

print("Extracted capabilities from MCP tools:")
print(f"  translate server: {translate_caps}")
print(f"  summarize server: {summarize_caps}")
print()

# Register with extracted capabilities + endpoint
network = alps.LocalNetwork()

network.register(
    "translate-mcp-server",
    translate_caps,
    endpoint="http://localhost:3001",
    metadata={"protocol": "mcp"},
)

network.register(
    "summarize-mcp-server",
    summarize_caps,
    endpoint="http://localhost:3002",
    metadata={"protocol": "mcp"},
)

# Discovery uses hybrid word+character shingling with a similarity threshold
# (default 0.1) to filter noise matches.
test_queries = [
    "translate legal contract to German",
    "detect language of this text",
    "summarize a legal document",
    "what language is this written in",
    "source text to translate",
    "Add these two numbers together",  # should not match anything
]

print("Discovery results:")
for q in test_queries:
    results = network.discover(q)
    if results:
        best = results[0]
        print(f"  '{q}'")
        print(f"    -> {best.agent_name} (sim={best.similarity:.3f}) at {best.endpoint}")
    else:
        print(f"  '{q}' -> NO MATCHES")
    print()

# --- Per-query-type feedback ---
# Feedback is query-specific: recording success for a translation query
# boosts the agent for future translation-like queries, NOT for summarization.

print("=" * 60)
print("Per-query-type feedback demo")
print("=" * 60)
print()

# Use two query types that both match translate-mcp-server.
translate_query = "translate legal contract to German"
detect_query = "detect language of this text"


# Snapshot scores before feedback.
def score_for(results, agent_name):
    for r in results:
        if r.agent_name == agent_name:
            return r.score
    return None


trans_score_before = score_for(network.discover(translate_query), "translate-mcp-server")
detect_score_before = score_for(network.discover(detect_query), "translate-mcp-server")

print("Before feedback (translate-mcp-server scores):")
print(f"  '{translate_query}' -> score={trans_score_before:.4f}")
print(f"  '{detect_query}'       -> score={detect_score_before:.4f}")
print()

# Record 10 translation successes WITH the query that triggered them.
# This tells the routing engine: "this agent is good at translation-like queries."
for _ in range(10):
    network.record_success("translate-mcp-server", query=translate_query)

trans_score_after = score_for(network.discover(translate_query), "translate-mcp-server")
detect_score_after = score_for(network.discover(detect_query), "translate-mcp-server")

print("After 10 translation successes (with query context):")
print(f"  '{translate_query}' -> score={trans_score_after:.4f}")
print(f"  '{detect_query}'       -> score={detect_score_after:.4f}")
print()

trans_boost = (trans_score_after - trans_score_before) / trans_score_before * 100
detect_boost = (detect_score_after - detect_score_before) / detect_score_before * 100
print(f"  Translation query boost: +{trans_boost:.1f}%")
print(f"  Detect query boost:     +{detect_boost:.1f}% (much smaller — different query type)")
print()

# You can also customize the threshold:
# network = alps.LocalNetwork(similarity_threshold=0.2)  # stricter filtering
