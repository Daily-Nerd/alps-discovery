"""Tests for DiscoveryConfig unified configuration."""

import alps_discovery as alps
import pytest


def test_discovery_config_default():
    """Test DiscoveryConfig with default values."""
    config = alps.DiscoveryConfig()

    # Verify default values
    assert config.similarity_threshold == 0.1
    assert config.feedback_relevance_threshold == 0.3
    assert config.tie_epsilon == 1e-4
    assert config.tau_floor == 0.001
    assert config.max_feedback_records == 100
    assert config.diameter_initial == 0.5
    assert config.diameter_min == 0.01
    assert config.diameter_max == 2.0
    assert config.epsilon_initial == 0.8
    assert config.epsilon_floor == 0.05
    assert config.epsilon_decay_rate == 0.99
    assert config.max_disagreement_split == 3


def test_discovery_config_custom_values():
    """Test DiscoveryConfig with custom values."""
    config = alps.DiscoveryConfig(
        similarity_threshold=0.2,
        feedback_relevance_threshold=0.4,
        tie_epsilon=2e-4,
        tau_floor=0.002,
        max_feedback_records=200,
        diameter_initial=0.6,
        diameter_min=0.02,
        diameter_max=1.5,
        epsilon_initial=0.7,
        epsilon_floor=0.1,
        epsilon_decay_rate=0.95,
        max_disagreement_split=4,
    )

    # Verify custom values
    assert config.similarity_threshold == 0.2
    assert config.feedback_relevance_threshold == 0.4
    assert config.tie_epsilon == 2e-4
    assert config.tau_floor == 0.002
    assert config.max_feedback_records == 200
    assert config.diameter_initial == 0.6
    assert config.diameter_min == 0.02
    assert config.diameter_max == 1.5
    assert config.epsilon_initial == 0.7
    assert config.epsilon_floor == 0.1
    assert config.epsilon_decay_rate == 0.95
    assert config.max_disagreement_split == 4


def test_discovery_config_validation_feedback_threshold():
    """Test DiscoveryConfig validates feedback threshold."""
    with pytest.raises(ValueError, match="feedback_relevance_threshold"):
        alps.DiscoveryConfig(feedback_relevance_threshold=1.5)

    with pytest.raises(ValueError, match="feedback_relevance_threshold"):
        alps.DiscoveryConfig(feedback_relevance_threshold=-0.1)


def test_discovery_config_validation_tie_epsilon():
    """Test DiscoveryConfig validates tie epsilon."""
    with pytest.raises(ValueError, match="tie_epsilon"):
        alps.DiscoveryConfig(tie_epsilon=0.0)

    with pytest.raises(ValueError, match="tie_epsilon"):
        alps.DiscoveryConfig(tie_epsilon=-0.001)


def test_discovery_config_validation_tau_floor():
    """Test DiscoveryConfig validates tau floor."""
    with pytest.raises(ValueError, match="tau_floor"):
        alps.DiscoveryConfig(tau_floor=0.0)

    with pytest.raises(ValueError, match="tau_floor"):
        alps.DiscoveryConfig(tau_floor=-0.001)


def test_discovery_config_validation_diameter_range():
    """Test DiscoveryConfig validates diameter range."""
    # Invalid: diameter_min <= 0
    with pytest.raises(ValueError, match="diameter_min"):
        alps.DiscoveryConfig(diameter_min=0.0)

    # Invalid: diameter_max <= diameter_min
    with pytest.raises(ValueError, match="diameter_max"):
        alps.DiscoveryConfig(diameter_min=0.5, diameter_max=0.3)

    # Invalid: diameter_initial < diameter_min
    with pytest.raises(ValueError, match="diameter_initial"):
        alps.DiscoveryConfig(diameter_initial=0.005, diameter_min=0.01)

    # Invalid: diameter_initial > diameter_max
    with pytest.raises(ValueError, match="diameter_initial"):
        alps.DiscoveryConfig(diameter_initial=3.0, diameter_max=2.0)


def test_local_network_with_config():
    """Test LocalNetwork accepts DiscoveryConfig."""
    config = alps.DiscoveryConfig(
        similarity_threshold=0.15,
        diameter_initial=0.7,
    )

    network = alps.LocalNetwork(config=config)

    # Register and discover to verify network works with custom config
    network.register("test-agent", ["test capability"])
    results = network.discover("test query")

    # Should work without errors
    assert isinstance(results, list)


def test_local_network_config_exclusive_with_similarity_threshold():
    """Test LocalNetwork rejects config + similarity_threshold."""
    config = alps.DiscoveryConfig()

    with pytest.raises(ValueError, match="Cannot specify both"):
        alps.LocalNetwork(config=config, similarity_threshold=0.2)


def test_local_network_config_exclusive_with_scorer():
    """Test LocalNetwork rejects config + scorer."""
    config = alps.DiscoveryConfig()
    scorer = alps.TfIdfScorer()

    with pytest.raises(ValueError, match="Cannot specify both"):
        alps.LocalNetwork(config=config, scorer=scorer)


def test_discovery_config_repr():
    """Test DiscoveryConfig has readable repr."""
    config = alps.DiscoveryConfig()
    repr_str = repr(config)

    assert "DiscoveryConfig" in repr_str
    assert "similarity_threshold" in repr_str
