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
        "Build with: cd crates/alps-discovery && maturin develop"
    ) from e

__all__ = ["LocalNetwork", "DiscoveryResult"]
