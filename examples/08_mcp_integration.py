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
                "source_lang": {"type": "string", "description": "Source language code like EN, DE, FR"},
                "target_lang": {"type": "string", "description": "Target language code like EN, DE, FR"},
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

# You can also customize the threshold:
# network = alps.LocalNetwork(similarity_threshold=0.2)  # stricter filtering
