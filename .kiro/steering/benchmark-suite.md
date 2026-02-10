# Benchmark Suite Strategy

## Overview

ALPS Discovery uses a dual-layer benchmarking approach:
- **Rust Criterion benchmarks** for core algorithm performance (LSH, scoring, enzyme, pipeline, query, persistence)
- **Python P3 integration benchmark** for end-to-end before/after comparison

Benchmarks are feature-gated (`bench` feature) to keep them invisible in normal builds but readily available for performance analysis and regression detection.

---

## Rust Benchmarks (Criterion)

### Organization

Benchmarks live in `benches/` as individual Rust files, each focused on a single component:

```
benches/
  lsh.rs              # MinHash signature generation & similarity computation
  scoring.rs          # Scorer trait implementations (MinHash, TfIdf)
  enzyme.rs           # SLNEnzyme kernel evaluation & voting
  pipeline.rs         # Full discovery pipeline (end-to-end)
  query.rs            # Query enum evaluation & text extraction
  persistence.rs      # NetworkSnapshot save/load operations
  common/             # Shared test data & utilities
```

Each benchmark is declared in `Cargo.toml` with `harness = false` to use Criterion's custom harness instead of the default test runner.

### Measurement Patterns

#### Black Box & Input Isolation
```rust
// Prevent compiler optimizations from skewing results
b.iter(|| hasher.hash_key(black_box(bytes), black_box(&config)))
```

Use `black_box()` around inputs and configuration to prevent the compiler from optimizing away the work being measured.

#### Parameterized Benchmarks
```rust
// Benchmark with multiple input sizes or configurations
let mut group = c.benchmark_group("group_name");
for length in &[Short, Medium, Long] {
    let input = capability_text(*length);
    group.bench_with_input(
        BenchmarkId::from_parameter(format!("{:?}", length)),
        length,
        |b, _| b.iter(|| /* measure work */)
    );
}
group.finish();
```

Parameterized benchmarks show how performance scales with input size or configuration. Use `BenchmarkId` to label each variant.

#### Test Data Hierarchy
Common test data in `benches/common/`:
- `TextLength::Short` — Minimal capability descriptions (~20 bytes)
- `TextLength::Medium` — Typical descriptions (~200 bytes)
- `TextLength::Long` — Complex multi-capability sets (~2KB)

This ensures realistic but reproducible measurement across different benchmark scenarios.

### Running Criterion Benchmarks

```bash
# Run all benchmarks with HTML report generation
cargo bench --features bench

# Run specific benchmark group
cargo bench --features bench -- lsh

# Generate baseline for regression detection
cargo bench --features bench -- --baseline main
```

Criterion automatically generates HTML reports in `target/criterion/` with timing distributions, confidence intervals, and regression warnings.

---

## Python Integration Benchmark (P3)

### Purpose

The P3 benchmark (`examples/p3_benchmark.py`) provides a "before/after" view of improvements across a feature cycle. It measures:
- Basic discovery timing and correctness
- Exploration behavior (epsilon-greedy decay)
- Concurrent safety (threaded discovery)
- Feedback performance impact
- Drift detection accuracy
- Replay log event processing
- TF-IDF scorer behavior

### Running P3 Benchmark

```bash
# Capture baseline (before changes)
uv run python examples/p3_benchmark.py

# Run after changes to compare
uv run python examples/p3_benchmark.py --after
```

The `--after` flag indicates improved behavior; output is human-readable comparison of timings, exploration paths, and feature completeness.

### Pattern: Scenario-Based Testing

Each benchmark function follows this pattern:
1. **Setup** — Register agents with realistic capabilities
2. **Query execution** — Run queries and collect results
3. **Measurement** — Timing, correctness, or behavior observation
4. **Output** — Human-readable section with key metrics

Example:
```python
def benchmark_basic_discovery():
    section("1. BASIC DISCOVERY TIMING")
    net = LocalNetwork()
    agents = [("agent-1", ["cap1", "cap2"]), ...]
    for query in queries:
        # Measure discovery time and correctness
```

---

## CI/CD Integration

### GitHub Actions

Benchmarks run in the `benchmarks` job of `.github/workflows/ci.yml`:

```yaml
benchmarks:
  name: Benchmarks
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: dtolnay/rust-toolchain@master
    - run: cargo bench --features bench
```

**Key points**:
- Feature gate ensures benchmarks only run when explicitly requested
- Runs after the main `rust` job (dependency: `needs: [rust]`)
- Output goes to GitHub Actions logs (check run summary)

### Regression Detection

Store baseline results for comparison:
```bash
cargo bench --features bench -- --baseline <name>
cargo bench --features bench -- --compare <name>
```

Criterion flags regressions with statistical confidence intervals.

---

## Benchmark Naming & Scope

### Naming Convention
- **Group names** — Lowercase, describe what's measured: `hash_key`, `similarity`, `multiple_capabilities`
- **Benchmark functions** — `bench_*` prefix, focus on a single operation: `bench_similarity()`, `bench_query_evaluation()`
- **Parameters** — Use `BenchmarkId` with clear labels: `"short"`, `"medium"`, `"1000-agents"`

### Scope Boundaries
- **Unit benchmarks** — Micro-benchmarks of individual components (LSH hashing, similarity computation)
- **Pipeline benchmarks** — Full discovery flow with setup/teardown overhead
- **Integration benchmarks** — Python-layer behavior and end-to-end performance

Keep unit benchmarks fast (sub-millisecond) for quick iteration; pipeline/integration may take seconds.

---

## Best Practices

### 1. **Isolate Variables**
Each benchmark measures one thing. Control inputs and configuration to avoid confounding factors.

```rust
// ✓ Good: Only varies input size
for length in &[Short, Medium, Long] { /* measure */ }

// ✗ Bad: Varies multiple things at once
for config in configs { for input in inputs { /* measure */ } }
```

### 2. **Use Realistic Data**
Test data should reflect production scenarios. Common library provides typical capability descriptions.

### 3. **Document Intent**
Add comments explaining what you're measuring and why:

```rust
// Measures the cost of computing similarity between signatures.
// Important for understanding query latency with large agent networks.
c.bench_function("similarity", |b| {
    b.iter(|| MinHasher::similarity(black_box(&sig1), black_box(&sig2)))
});
```

### 4. **Monitor Regression**
Run benchmarks before and after changes:
```bash
cargo bench --features bench -- --baseline before
# [make changes]
cargo bench --features bench -- --baseline after
```

Criterion will flag suspicious slowdowns.

### 5. **Feature Gate Visibility**
Benchmarks are `#[cfg(all(test, feature = "bench"))]` to keep them out of normal builds. This prevents:
- Accidental performance impact from benchmark compilation
- Confusion between unit tests and performance tests

---

## Adding a New Benchmark

1. Create `benches/{component}.rs` with a `bench_*` function
2. Declare in `Cargo.toml`:
   ```toml
   [[bench]]
   name = "component"
   harness = false
   ```
3. Use `mod common;` to access shared test data
4. Apply black_box and BenchmarkId patterns from existing benchmarks
5. Run with `cargo bench --features bench -- component`

---

_Document patterns, not exhaustive lists. New benchmarks should follow the established Criterion + black_box + parameterization style._
