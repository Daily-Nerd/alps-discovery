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

    Takes the output of an MCP server's list_tools() and flattens
    tool names, descriptions, and parameter descriptions into
    capability strings suitable for register().

    Args:
        tools: List of MCP tool dicts, each with 'name', 'description',
               and optionally 'inputSchema' with property descriptions.

    Returns:
        List of capability strings for use with register().

    Example:
        tools = mcp_client.list_tools()
        # [{"name": "translate", "description": "Translate text between languages",
        #   "inputSchema": {"properties": {"target_lang": {"description": "Target language"}}}}]

        network.register("translate-server", capabilities_from_mcp(tools),
                          endpoint="http://localhost:8080")
    """
    caps = []
    for tool in tools:
        name = tool.get("name", "")
        desc = tool.get("description", "")

        if name:
            caps.append(name)
        if desc:
            caps.append(desc)

        # Extract parameter descriptions from inputSchema
        schema = tool.get("inputSchema", {})
        props = schema.get("properties", {})
        for prop_name, prop_def in props.items():
            prop_desc = prop_def.get("description", "") if isinstance(prop_def, dict) else ""
            if prop_desc:
                caps.append(prop_desc)

    return caps


__all__ = ["LocalNetwork", "DiscoveryResult", "capabilities_from_mcp"]
