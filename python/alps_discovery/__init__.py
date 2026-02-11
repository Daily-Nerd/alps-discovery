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
    from ._native import (
        DiscoveryConfig,
        DiscoveryResponse,
        DiscoveryResult,
        ExplainedResult,
        FilterValue,
        LocalNetwork,
        Query,
        TfIdfScorer,
    )
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
        param_names = [p.replace("_", " ") for p in props]
        if param_names:
            parts.append(", ".join(param_names))

        composite = ". ".join(parts)
        if composite:
            caps.append(composite)

    return caps


def capabilities_from_a2a(agent_card: dict) -> list[str]:
    """Extract capability strings from Google A2A AgentCard.

    Parses an A2A AgentCard JSON object and extracts capability descriptions
    from the agent name, description, skills, and tags. This enables ALPS
    to discover A2A agents using the same local discovery mechanism as MCP tools.

    Args:
        agent_card: A2A AgentCard dict with optional fields:
            - name: Agent name
            - description: Agent description
            - skills: List of skill objects with name, description, tags

    Returns:
        List of capability strings for use with register().

    Example:
        agent_card = {
            "name": "legal-assistant",
            "description": "Legal document analysis",
            "skills": [
                {
                    "name": "analyze_contract",
                    "description": "Analyze legal contracts",
                    "tags": ["legal", "contracts", "risk-analysis"]
                }
            ]
        }
        caps = capabilities_from_a2a(agent_card)
        network.register("legal-agent", caps)
    """
    caps = []

    # Extract agent-level name and description
    name = agent_card.get("name", "").replace("-", " ").replace("_", " ").strip()
    desc = agent_card.get("description", "").strip()

    # Build agent-level capability
    agent_parts = []
    if name and desc:
        agent_parts.append(f"{name}: {desc}")
    elif name:
        agent_parts.append(name)
    elif desc:
        agent_parts.append(desc)

    if agent_parts:
        caps.append(". ".join(agent_parts))

    # Extract skill-level capabilities
    skills = agent_card.get("skills", [])
    for skill in skills:
        skill_name = skill.get("name", "").replace("_", " ").strip()
        skill_desc = skill.get("description", "").strip()
        skill_tags = skill.get("tags", [])

        # Build skill capability: "name: description. tag1, tag2, tag3"
        skill_parts = []
        if skill_name and skill_desc:
            skill_parts.append(f"{skill_name}: {skill_desc}")
        elif skill_name:
            skill_parts.append(skill_name)
        elif skill_desc:
            skill_parts.append(skill_desc)

        # Add tags as additional tokens
        if skill_tags:
            tags_str = ", ".join(str(tag) for tag in skill_tags)
            skill_parts.append(tags_str)

        if skill_parts:
            caps.append(". ".join(skill_parts))

    return caps


__all__ = [
    "DiscoveryConfig",
    "DiscoveryResponse",
    "DiscoveryResult",
    "ExplainedResult",
    "FilterValue",
    "LocalNetwork",
    "Query",
    "TfIdfScorer",
    "capabilities_from_a2a",
    "capabilities_from_mcp",
]
