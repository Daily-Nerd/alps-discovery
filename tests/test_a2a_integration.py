"""Tests for Google A2A AgentCard capability extraction."""

import alps_discovery as alps


def test_capabilities_from_a2a_basic():
    """Test basic A2A AgentCard parsing with name and description."""
    agent_card = {
        "name": "translation-agent",
        "description": "Translates text between languages",
    }

    capabilities = alps.capabilities_from_a2a(agent_card)

    # Should extract name and description
    assert len(capabilities) >= 1
    assert any("translation" in cap.lower() for cap in capabilities)
    assert any("translates" in cap.lower() for cap in capabilities)


def test_capabilities_from_a2a_with_skills():
    """Test A2A AgentCard parsing with skills array."""
    agent_card = {
        "name": "multi-lingual-agent",
        "description": "Multi-language support agent",
        "skills": [
            {"name": "translate_text", "description": "Translate text between EN, FR, DE"},
            {"name": "detect_language", "description": "Automatically detect input language"},
        ],
    }

    capabilities = alps.capabilities_from_a2a(agent_card)

    # Should include agent name, description, and all skill names/descriptions
    assert len(capabilities) >= 3
    # Check that skill information is included
    caps_text = " ".join(capabilities).lower()
    assert "translate" in caps_text
    assert "detect" in caps_text
    assert "language" in caps_text


def test_capabilities_from_a2a_with_tags():
    """Test A2A AgentCard parsing with skill tags."""
    agent_card = {
        "name": "data-agent",
        "description": "Data processing agent",
        "skills": [
            {
                "name": "process_data",
                "description": "Process and transform data",
                "tags": ["ETL", "CSV", "JSON", "transformation"],
            }
        ],
    }

    capabilities = alps.capabilities_from_a2a(agent_card)

    # Should include tags as capability tokens
    caps_text = " ".join(capabilities).lower()
    assert "etl" in caps_text
    assert "csv" in caps_text
    assert "json" in caps_text
    assert "transformation" in caps_text


def test_capabilities_from_a2a_comprehensive():
    """Test comprehensive A2A AgentCard with all fields."""
    agent_card = {
        "name": "legal-assistant",
        "description": "Legal document analysis and contract review",
        "skills": [
            {
                "name": "analyze_contract",
                "description": "Analyze legal contracts for risks",
                "tags": ["legal", "contracts", "risk-analysis"],
            },
            {
                "name": "draft_agreement",
                "description": "Draft standard legal agreements",
                "tags": ["legal", "drafting", "NDA", "MSA"],
            },
        ],
    }

    capabilities = alps.capabilities_from_a2a(agent_card)

    # Verify all components are extracted
    caps_text = " ".join(capabilities).lower()

    # Agent level
    assert "legal" in caps_text
    assert "assistant" in caps_text
    assert "document" in caps_text

    # Skill level
    assert "analyze" in caps_text
    assert "contract" in caps_text
    assert "draft" in caps_text

    # Tags
    assert "nda" in caps_text
    assert "msa" in caps_text
    assert "risk" in caps_text


def test_capabilities_from_a2a_empty():
    """Test A2A AgentCard with minimal/empty fields."""
    agent_card = {}

    capabilities = alps.capabilities_from_a2a(agent_card)

    # Should return empty list or handle gracefully
    assert isinstance(capabilities, list)


def test_capabilities_from_a2a_name_only():
    """Test A2A AgentCard with only name field."""
    agent_card = {"name": "simple-agent"}

    capabilities = alps.capabilities_from_a2a(agent_card)

    # Should extract at least the name
    assert len(capabilities) >= 1
    caps_text = " ".join(capabilities).lower()
    assert "simple" in caps_text or "agent" in caps_text
