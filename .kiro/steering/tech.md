# Technology Stack

## Architecture

Dual-language library: Rust core with Python bindings via PyO3. All discovery logic lives in Rust; Python is a thin ergonomic layer that re-exports native types and adds helper utilities (e.g., MCP capability extraction).

## Core Technologies

- **Language**: Rust (2021 edition) + Python 3.11+
- **FFI Bridge**: PyO3 0.27 (Rust ↔ Python)
- **Build System**: Maturin (Rust→Python wheel), Cargo (Rust)
- **Hashing**: xxhash (xxh3) for MinHash signatures — deterministic, no `rand` dependency

## Key Libraries

- `pyo3` — Python bindings and module definition
- `xxhash-rust` — Fast, deterministic hashing for MinHash LSH
- `serde` / `serde_json` — Serialization for network persistence (JSON snapshots)
- `serde-big-array` — Support for fixed-size arrays in serde (MinHash signatures)

## Development Standards

### Type Safety
- Rust: strong typing, no `unsafe` blocks in application code
- Python: type stubs (`_native.pyi`) for IDE autocompletion; `__all__` exports

### Code Quality
- Pre-commit hooks require `PYO3_PYTHON` and `VIRTUAL_ENV` env vars
- Bio-inspired naming convention for core types (enzyme, spore, membrane, pheromone, hyphae, chemistry)

### Testing
- Rust unit tests co-located with implementation (`#[cfg(test)]` modules)
- ~57 tests across LSH (13), Scorer (5), Network (39)
- Python integration via example scripts in `examples/`

## Development Environment

### Required Tools
- Rust toolchain (via rustup)
- Python 3.11+ with `uv` for dependency management
- Maturin for building the Python extension

### Common Commands
```bash
# Test:  PYO3_PYTHON=.venv/bin/python cargo test
# Build: VIRTUAL_ENV=.venv maturin develop
# Run:   uv run python examples/01_quickstart.py
```

## Key Technical Decisions

- **Self-contained core types** — all types are vendored locally for independent evolution and zero external crate dependencies beyond PyO3/serde/xxhash
- **No `rand` dependency** — deterministic tie-breaking uses xxhash-seeded Fisher-Yates shuffle (5% threshold for randomization)
- **Pluggable Scorer trait** — `Send + Sync` bound enables thread-safe custom scorers; Python scorers go through PyO3 bridge
- **NetworkSnapshot persistence** — JSON format with `version` field for forward compatibility
- **Post-scoring metadata filters** — filtering happens after similarity scoring, not during indexing

---
_Document standards and patterns, not every dependency_
