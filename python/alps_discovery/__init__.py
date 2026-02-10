"""ALPS Discovery SDK — local agent discovery via bio-inspired routing.

Usage:
    import alps_discovery as alps

    network = alps.LocalNetwork()
    network.register("translate-agent", ["legal translation", "EN-DE", "EN-FR"])
    network.register("summarize-agent", ["document summarization", "legal briefs"])

    results = network.discover("translate legal contract from English to German")
    for r in results:
        print(f"{r.agent_name}: similarity={r.similarity:.3f}, score={r.score:.3f}")
"""

try:
    from ._native import LocalNetwork, DiscoveryResult
except ImportError as e:
    raise ImportError(
        "Failed to import alps_discovery native module. "
        "Build with: cd alps-discovery && maturin develop"
    ) from e


def capabilities_from_mcp(tools: list[dict]) -> list[str]:
    """Extract capability strings from MCP tool definitions.

    Takes the output of an MCP server's list_tools() and builds one
    composite capability string per tool, combining the tool name,
    description, and parameter names into a single string with high
    n-gram surface area for similarity matching.

    Args:
        tools: List of MCP tool dicts, each with 'name', 'description',
               and optionally 'inputSchema' with property descriptions.

    Returns:
        List of capability strings (one per tool) for use with register().

    Example:
        tools = mcp_client.list_tools()
        caps = capabilities_from_mcp(tools)
        # ["translate text: Translate text between languages. source text, target language"]
        network.register("translate-server", caps, endpoint="http://localhost:8080")
    """
    caps = []
    for tool in tools:
        name = tool.get("name", "").replace("_", " ").strip()
        desc = tool.get("description", "").strip()

        # Build composite: "name: description. param1, param2, ..."
        parts = []
        if name and desc:
            parts.append(f"{name}: {desc}")
        elif name:
            parts.append(name)
        elif desc:
            parts.append(desc)

        # Extract parameter names (lean nouns, not verbose descriptions)
        schema = tool.get("inputSchema", {})
        props = schema.get("properties", {})
        param_names = [p.replace("_", " ") for p in props.keys()]
        if param_names:
            parts.append(", ".join(param_names))

        composite = ". ".join(parts)
        if composite:
            caps.append(composite)

    return caps


__all__ = ["LocalNetwork", "DiscoveryResult", "capabilities_from_mcp"]
