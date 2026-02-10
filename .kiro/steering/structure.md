# Project Structure

## Organization Philosophy

Layered architecture with a clear Rust-core / Python-surface separation. The Rust side uses a bio-inspired domain vocabulary for internal modules, with a clean public API surface in `network.rs` and `scorer.rs`. Python wraps native types and adds SDK-level conveniences.

## Directory Patterns

### Rust Core (`src/core/`)
**Purpose**: Bio-inspired primitives that power discovery — LSH, enzymes, signals, membranes, etc.
**Pattern**: One module per biological concept, each with co-located `#[cfg(test)]` tests
**Example**: `lsh.rs` (MinHash + shingling + ConfidenceInterval), `enzyme.rs` (SLNEnzyme with 4 kernels + Quorum voting), `config.rs` (LshConfig, QueryConfig, ExplorationConfig)

### Rust Public API (`src/`)
**Purpose**: Top-level modules that compose core primitives into the user-facing API
**Pattern**: `network/` (decomposed module directory), `scorer.rs` (Scorer trait + MinHashScorer + TfIdfScorer), `query.rs` (Query algebra), `pybridge.rs` (PyO3 bindings)
**Example**: `network/mod.rs` is a thin facade; sub-modules own specific concerns (pipeline, registry, replay, persistence, filter, adapters)

### Network Module (`src/network/`)
**Purpose**: Decomposed LocalNetwork — each concern in its own file behind a facade
**Pattern**: `mod.rs` re-exports public types and provides the LocalNetwork API; sub-modules are `pub(crate)` or private
**Sub-modules**: `pipeline.rs` (scoring + enzyme + tie-breaking), `registry.rs` (agent registry + feedback index + tick), `replay.rs` (event log), `persistence.rs` (save/load), `filter.rs` (metadata matching), `scorer_adapter.rs` / `enzyme_adapter.rs` (trait wrappers)

### Python Package (`python/alps_discovery/`)
**Purpose**: Thin SDK layer — re-exports native types, adds Python-only utilities
**Pattern**: `__init__.py` re-exports from `_native` and defines helper functions; `_native.pyi` provides type stubs
**Example**: `capabilities_from_mcp()` is pure Python, everything else delegates to Rust

### Examples (`examples/`)
**Purpose**: Numbered, progressive examples demonstrating SDK usage
**Pattern**: `NN_topic.py` — numbered for learning order, self-contained scripts
**Example**: `01_quickstart.py` → `12_custom_scorer.py` (plus `p3_benchmark.py` for performance)

## Naming Conventions

- **Rust files**: `snake_case.rs`, one concept per file
- **Rust types**: `PascalCase` structs/enums, `snake_case` functions
- **Bio-domain names**: Core modules use biological metaphors (enzyme, spore, membrane, pheromone, hyphae, chemistry, signal)
- **Python files**: `snake_case.py`, underscored private modules (`_native`)
- **Examples**: `NN_descriptive_name.py` (zero-padded number prefix)

## Import Organization

```rust
// Rust: standard → external → crate-internal
use std::collections::HashMap;
use pyo3::prelude::*;
use crate::core::config::LshConfig;
use crate::core::lsh::MinHasher;
```

```python
# Python: re-export from native, define helpers
from ._native import LocalNetwork, DiscoveryResult, ExplainedResult
```

## Code Organization Principles

- **Rust owns all logic** — Python never reimplements discovery, scoring, or filtering
- **PyO3 bridge is a translation layer** — `pybridge.rs` maps between Rust and Python types, does not contain business logic
- **Facade + sub-module decomposition** — `network/mod.rs` is a thin facade; heavy logic lives in dedicated sub-modules (pipeline, registry, replay) to keep files focused
- **Trait-based extension** — `Scorer` trait in Rust, scorer protocol in Python; both follow `index_capabilities / remove_agent / score` interface
- **Tests live with code** — Rust tests are `#[cfg(test)] mod tests` blocks at the bottom of each file; no separate test directory
- **Examples are documentation** — numbered scripts serve as the primary usage guide alongside README

---
_Document patterns, not file trees. New files following patterns shouldn't require updates_
