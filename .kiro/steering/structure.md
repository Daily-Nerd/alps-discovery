# Project Structure

## Organization Philosophy

Layered architecture with a clear Rust-core / Python-surface separation. The Rust side uses a bio-inspired domain vocabulary for internal modules, with a clean public API surface in `network.rs` and `scorer.rs`. Python wraps native types and adds SDK-level conveniences.

## Directory Patterns

### Rust Core (`src/core/`)
**Purpose**: Bio-inspired primitives that power discovery — LSH, enzymes, signals, membranes, etc.
**Pattern**: One module per biological concept, each with co-located `#[cfg(test)]` tests
**Example**: `lsh.rs` (MinHash + shingling), `enzyme.rs` (SLNEnzyme with 3 kernels), `config.rs` (LshConfig, QueryConfig)

### Rust Public API (`src/`)
**Purpose**: Top-level modules that compose core primitives into the user-facing API
**Pattern**: `network.rs` (LocalNetwork), `scorer.rs` (Scorer trait + MinHashScorer), `pybridge.rs` (PyO3 bindings)
**Example**: `network.rs` owns registration, discovery, filtering, persistence, and explain mode

### Python Package (`python/alps_discovery/`)
**Purpose**: Thin SDK layer — re-exports native types, adds Python-only utilities
**Pattern**: `__init__.py` re-exports from `_native` and defines helper functions; `_native.pyi` provides type stubs
**Example**: `capabilities_from_mcp()` is pure Python, everything else delegates to Rust

### Examples (`examples/`)
**Purpose**: Numbered, progressive examples demonstrating SDK usage
**Pattern**: `NN_topic.py` — numbered for learning order, self-contained scripts
**Example**: `01_quickstart.py` → `08_mcp_integration.py`

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
- **Trait-based extension** — `Scorer` trait in Rust, scorer protocol in Python; both follow `index_capabilities / remove_agent / score` interface
- **Tests live with code** — Rust tests are `#[cfg(test)] mod tests` blocks at the bottom of each file; no separate test directory
- **Examples are documentation** — numbered scripts serve as the primary usage guide alongside README

---
_Document patterns, not file trees. New files following patterns shouldn't require updates_
