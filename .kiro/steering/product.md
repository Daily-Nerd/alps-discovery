# Product Overview

ALPS Discovery is an open-source SDK for **local agent discovery** — a DNS-like system that resolves natural-language capability queries to the best-matching agent. It uses bio-inspired multi-kernel routing to rank agents, with rankings that improve over time through success/failure feedback.

## Core Capabilities

- **Capability-based agent registration** — agents register with natural-language capability descriptions, optional endpoints, metadata, and invoke callables
- **Natural-language discovery** — queries are matched against registered agents using locality-sensitive hashing (MinHash) for set-similarity scoring
- **Adaptive routing** — three independent reasoning kernels (Capability, LoadBalancing, Novelty) vote on routing; feedback adjusts agent diameter over time
- **Pluggable scoring** — default MinHash scorer can be swapped for custom implementations (e.g., embedding-based semantic matching) via the Scorer trait/protocol
- **MCP integration** — `capabilities_from_mcp()` extracts capability strings directly from MCP tool definitions

## Target Use Cases

- Multi-agent orchestration systems that need runtime agent selection
- MCP server ecosystems requiring dynamic tool/agent routing
- Applications that want adaptive, feedback-driven agent selection without hardcoded routing rules

## Value Proposition

- **Zero-config matching** — register capabilities as plain text, no schema definitions or ontologies required
- **Bio-inspired adaptivity** — the system learns from outcomes, not just static similarity
- **Extensible by design** — pluggable scorer interface lets users bring their own similarity engine (embeddings, hybrid, etc.)
- **Open-source** — Apache-2.0 licensed, self-contained SDK with no external protocol dependencies

---
_Focus on patterns and purpose, not exhaustive feature lists_
