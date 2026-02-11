# Changelog

All notable changes to ALPS Discovery will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-10

### Fixed

- **Circuit breaker overflow**: Changed `consecutive_pulse_timeouts` increment to use `saturating_add(1)` instead of `+= 1` to prevent wrapping at 256 failures and silently resetting the circuit breaker. Failing agents now correctly stay excluded after 255+ consecutive failures. ([#5](https://github.com/kyvo-org/alps-discovery/issues/5))

- **Empty top picks confidence**: Fixed `derive_confidence()` returning misleading `Unanimous` confidence when all agents are circuit-open or below threshold. Added new `NoViableAgents` confidence variant to accurately signal when no agents are available. Parallelism is now correctly set to 0 in this case. ([#6](https://github.com/kyvo-org/alps-discovery/issues/6))

- **TfIdfScorer quadratic registration**: Eliminated O(N²) registration cost in `TfIdfScorer` by implementing incremental document frequency (DF) maintenance. Registration now takes O(T) time where T is the term count per agent, regardless of existing agent count. For 1000 agents, registration time improved from ~1000ms to ~0.02ms per agent (50,000x speedup). ([#7](https://github.com/kyvo-org/alps-discovery/issues/7))

### Added

- Test coverage for circuit breaker saturation behavior (`circuit_breaker_saturates_at_max_u8`)
- Test coverage for NoViableAgents confidence variant (`test_no_viable_agents_when_all_circuit_open`)
- Test coverage for TfIdfScorer incremental DF maintenance:
  - `tfidf_incremental_df_matches_full_rebuild`
  - `tfidf_remove_agent_decrements_df`
  - `tfidf_reindex_same_agent_updates_df_correctly`
  - `tfidf_incremental_performance_is_constant_time`

### Performance

- TfIdfScorer registration is now O(T) instead of O(N²), enabling efficient registration of 1000+ agents
- Verified discovery latency remains ~5ms for 1000 agents

## [0.0.1] - 2026-01-15

### Added

- Initial release of ALPS Discovery
- MinHash LSH-based agent discovery
- Multi-kernel enzyme routing (Capability, LoadBalancing, Novelty, TemporalRecency)
- Pheromone feedback system (diameter, tau, sigma)
- Circuit breaker for failing agents
- TfIdf scorer as alternative to MinHash
- Query algebra (All, Any, Exclude, Weighted)
- Metadata filtering
- Co-occurrence query expansion
- Mycorrhizal feedback propagation
- Confidence signals (Unanimous, Majority, Split)
- Replay log for debugging
- Persistence (save/load network snapshots)
- Python bindings via PyO3
- 177 Rust unit tests
- 12 Python examples
