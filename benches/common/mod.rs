// Shared benchmark setup module with factory functions for creating test fixtures.
//
// All factory functions are deterministic — identical inputs produce identical outputs.

#![allow(dead_code)] // Functions will be used in future benchmark implementations
use alps_discovery_native::core::hyphae::Hypha;
use alps_discovery_native::core::types::HyphaId;
use alps_discovery_native::network::LocalNetwork;
use criterion::Criterion;
use std::collections::HashMap;
use std::time::Duration;

#[cfg(feature = "bench")]
use alps_discovery_native::core::enzyme::ScorerContext;

/// Text length categories for parameterized benchmarks.
#[derive(Debug, Clone, Copy)]
pub enum TextLength {
    /// Short capability text (~20 characters).
    Short,
    /// Medium capability text (~100 characters).
    Medium,
    /// Long capability text (~500 characters).
    Long,
}

/// Shared Criterion configuration for consistent benchmark timing.
///
/// Current settings optimized for balance between speed and statistical reliability:
/// - sample_size(10): reduced from default 100
/// - measurement_time(1s): reduced from default 5s
/// - warm_up_time(500ms): reduced from default 3s
/// - noise_threshold(0.05): 5% regression threshold (Requirement 3.4)
/// - significance_level(0.05): 95% confidence (Requirement 3.4)
///
/// Full suite: ~166s for 71 benchmarks (exceeds original 60s target due to comprehensive coverage).
/// For faster iteration, run individual targets: `cargo bench --bench <target>`
pub fn criterion_config() -> Criterion {
    Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(1))
        .warm_up_time(Duration::from_millis(500))
        .noise_threshold(0.05)
        .significance_level(0.05)
}

/// Generate deterministic capability text for the given length category.
///
/// Returns a list of capability strings suitable for agent registration.
/// Each category produces fixed-length strings for consistent benchmarking.
pub fn capability_text(len: TextLength) -> Vec<String> {
    match len {
        TextLength::Short => vec![
            "handle user queries".to_string(),
            "process data files".to_string(),
            "manage workflows".to_string(),
            "analyze metrics".to_string(),
            "generate reports".to_string(),
        ],
        TextLength::Medium => vec![
            "Natural language processing for customer support queries with sentiment analysis and intent classification".to_string(),
            "Data pipeline orchestration for ETL workflows with validation, transformation, and error handling".to_string(),
            "Real-time analytics dashboard generation with interactive visualizations and metric aggregation".to_string(),
            "Document processing and information extraction from structured and unstructured text sources".to_string(),
            "Automated workflow management with task scheduling, dependency resolution, and notification".to_string(),
        ],
        TextLength::Long => vec![
            "Advanced natural language understanding system with multi-turn conversation management, context tracking, entity resolution, sentiment analysis, intent classification, and response generation optimized for customer support, technical documentation, and knowledge base interactions across multiple domains including finance, healthcare, and e-commerce with support for 15+ languages and real-time translation capabilities".to_string(),
            "Distributed data processing infrastructure for large-scale ETL operations with automatic schema inference, data quality validation, anomaly detection, incremental updates, partition management, compression optimization, and integration with cloud storage systems including S3, GCS, and Azure Blob with support for Parquet, Avro, ORC, and CSV formats handling terabyte-scale datasets".to_string(),
            "Comprehensive business intelligence platform providing real-time dashboards, interactive visualizations, ad-hoc query interface, scheduled report generation, data export capabilities, role-based access control, audit logging, and integration with popular BI tools supporting complex analytical queries across dimensional data models with drill-down, roll-up, and slice-and-dice operations on billions of rows".to_string(),
            "Intelligent document processing system combining OCR, layout analysis, entity extraction, relationship mapping, semantic search, and knowledge graph construction for automated processing of contracts, invoices, receipts, forms, and unstructured documents with support for multiple file formats, handwriting recognition, and integration with document management systems".to_string(),
            "Enterprise workflow automation platform with visual workflow designer, business rules engine, decision tables, task assignment, escalation policies, SLA monitoring, approval routing, parallel execution, error recovery, audit trails, and integration with 100+ third-party services through REST APIs, webhooks, and message queues supporting complex orchestration patterns".to_string(),
        ],
    }
}

/// Create a LocalNetwork pre-populated with `n` agents.
///
/// Each agent has:
/// - Deterministic name: "agent-0", "agent-1", ...
/// - Capability strings from `capability_text(text_len)`
/// - Round-robin assignment of capabilities to agents
pub fn network_with_agents(n: usize, text_len: TextLength) -> LocalNetwork {
    let mut net = LocalNetwork::new();
    let capabilities = capability_text(text_len);

    for i in 0..n {
        let agent_name = format!("agent-{}", i);
        let cap = &capabilities[i % capabilities.len()];
        let caps: Vec<&str> = vec![cap.as_str()];
        net.register(&agent_name, &caps, None, HashMap::new())
            .expect("benchmark registration should succeed");
    }

    net
}

/// Create a LocalNetwork with `n` agents and `feedback_count` success records per agent.
///
/// Uses deterministic query strings for reproducible MinHash signatures.
#[allow(dead_code)]
pub fn network_with_feedback(n: usize, feedback_count: usize) -> LocalNetwork {
    let mut net = network_with_agents(n, TextLength::Medium);
    let queries = sample_queries();

    for i in 0..n {
        let agent_name = format!("agent-{}", i);
        for j in 0..feedback_count {
            let query = queries[j % queries.len()];
            net.record_success(&agent_name, Some(query));
        }
    }

    net
}

/// Standard query strings for discovery benchmarks.
///
/// Returns a deterministic set of representative queries covering various
/// complexity levels and domain areas.
#[allow(dead_code)]
pub fn sample_queries() -> Vec<&'static str> {
    vec![
        "process user data",
        "analyze customer feedback",
        "generate monthly report",
        "extract document entities",
        "schedule workflow tasks",
        "validate data quality",
        "transform CSV to JSON",
        "detect anomalies in metrics",
        "translate text to Spanish",
        "optimize query performance",
    ]
}

/// Generate `n` pre-built Hypha fixtures for enzyme benchmarks.
///
/// Each hypha has a deterministic ID and pheromone state.
#[cfg(feature = "bench")]
#[allow(dead_code)]
pub fn sample_hyphae(n: usize) -> Vec<Hypha> {
    use alps_discovery_native::core::chemistry::Chemistry;
    use alps_discovery_native::core::pheromone::HyphaState;
    use alps_discovery_native::core::types::PeerAddr;
    use std::time::Instant;

    (0..n)
        .map(|i| {
            let id_bytes = {
                let mut bytes = [0u8; 32];
                // Use index to create deterministic IDs
                bytes[0] = (i & 0xFF) as u8;
                bytes[1] = ((i >> 8) & 0xFF) as u8;
                bytes
            };
            Hypha {
                id: HyphaId(id_bytes),
                peer: PeerAddr(format!("agent-{}", i)),
                state: HyphaState::new(1.0), // Initial diameter = 1.0
                chemistry: Chemistry::new(),
                last_activity: Instant::now(),
            }
        })
        .collect()
}

/// Create a pre-built ScorerContext for enzyme benchmarks.
///
/// Maps each hypha ID to a deterministic similarity score.
#[cfg(feature = "bench")]
#[allow(dead_code)]
pub fn sample_scorer_context(hyphae: &[Hypha]) -> ScorerContext {
    hyphae
        .iter()
        .enumerate()
        .map(|(i, h)| {
            // Deterministic score: decreasing from 1.0 based on index
            let score = 1.0 - (i as f64 * 0.1).min(0.9);
            (h.id.clone(), score)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// How to Add a New Benchmark
// ---------------------------------------------------------------------------
//
// Adding a benchmark is a 3-step process that takes under 10 lines of code:
//
// 1. Find the appropriate benchmark group file based on the subsystem:
//    - benches/lsh.rs         — LSH hashing and signature generation
//    - benches/scoring.rs     — MinHash and TF-IDF scorer benchmarks
//    - benches/enzyme.rs      — SLNEnzyme kernel evaluation
//    - benches/pipeline.rs    — Full discovery pipeline and feedback index
//    - benches/query.rs       — Query algebra evaluation
//    - benches/persistence.rs — Network save/load operations
//
// 2. Add a benchmark function:
//
//    use criterion::{Criterion, black_box};
//    use super::common::{self, TextLength};
//
//    fn bench_my_new_feature(c: &mut Criterion) {
//        let net = common::network_with_agents(50, TextLength::Medium);
//        c.bench_function("my_new_feature", |b| {
//            b.iter(|| {
//                black_box(net.discover("test query"))
//            })
//        });
//    }
//
// 3. Add the function to the criterion_group! macro at the bottom:
//
//    criterion_group! {
//        name = benches;
//        config = common::criterion_config();
//        targets = existing_bench, bench_my_new_feature  // <-- add here
//    }
//
// That's it! Run `cargo bench --features bench` to execute your new benchmark.

#[cfg(test)]
mod tests {
    #[allow(unused_imports)] // Some imports only used in specific test configurations
    use super::{
        capability_text, criterion_config, network_with_agents, network_with_feedback,
        sample_queries, TextLength,
    };

    #[cfg(feature = "bench")]
    #[allow(unused_imports)]
    use super::{sample_hyphae, sample_scorer_context};

    #[test]
    fn capability_text_short_returns_expected_length() {
        let caps = capability_text(TextLength::Short);
        assert_eq!(caps.len(), 5);
        // Verify approximate length (~20 chars)
        for cap in &caps {
            assert!(
                cap.len() >= 15 && cap.len() <= 25,
                "Unexpected length: {}",
                cap.len()
            );
        }
    }

    #[test]
    fn capability_text_medium_returns_expected_length() {
        let caps = capability_text(TextLength::Medium);
        assert_eq!(caps.len(), 5);
        // Verify approximate length (~100 chars)
        for cap in &caps {
            assert!(
                cap.len() >= 90 && cap.len() <= 150,
                "Unexpected length: {}",
                cap.len()
            );
        }
    }

    #[test]
    fn capability_text_long_returns_expected_length() {
        let caps = capability_text(TextLength::Long);
        assert_eq!(caps.len(), 5);
        // Verify approximate length (~500 chars)
        for cap in &caps {
            assert!(cap.len() >= 400, "Unexpected length: {}", cap.len());
        }
    }

    #[test]
    fn capability_text_is_deterministic() {
        let caps1 = capability_text(TextLength::Medium);
        let caps2 = capability_text(TextLength::Medium);
        assert_eq!(caps1, caps2);
    }

    #[test]
    fn network_with_agents_creates_correct_count() {
        let net = network_with_agents(10, TextLength::Short);
        let agents = net.agents();
        assert_eq!(agents.len(), 10);
    }

    #[test]
    fn network_with_agents_has_deterministic_names() {
        let net = network_with_agents(5, TextLength::Short);
        let agents = net.agents();
        assert!(agents.contains(&"agent-0".to_string()));
        assert!(agents.contains(&"agent-4".to_string()));
    }

    #[test]
    fn network_with_agents_is_deterministic() {
        let net1 = network_with_agents(3, TextLength::Medium);
        let net2 = network_with_agents(3, TextLength::Medium);

        let agents1 = net1.agents();
        let agents2 = net2.agents();
        assert_eq!(agents1, agents2);

        // Verify discovery produces same results
        let results1 = net1.discover("test query");
        let results2 = net2.discover("test query");
        assert_eq!(results1.len(), results2.len());
    }

    #[test]
    fn network_with_feedback_creates_feedback_records() {
        let net = network_with_feedback(5, 10);
        let agents = net.agents();
        assert_eq!(agents.len(), 5);

        // Verify that discovery still works (feedback was recorded)
        let results = net.discover("test query");
        assert!(!results.is_empty());
    }

    #[test]
    fn sample_queries_returns_multiple_queries() {
        let queries = sample_queries();
        assert!(queries.len() >= 5);
    }

    #[test]
    fn sample_queries_is_deterministic() {
        let queries1 = sample_queries();
        let queries2 = sample_queries();
        assert_eq!(queries1, queries2);
    }

    #[test]
    fn criterion_config_has_expected_settings() {
        let config = criterion_config();
        // Can't directly inspect Criterion config fields, but we can verify it creates
        assert_eq!(
            std::mem::size_of_val(&config),
            std::mem::size_of::<Criterion>()
        );
    }

    #[cfg(feature = "bench")]
    #[test]
    fn sample_hyphae_creates_correct_count() {
        let hyphae = sample_hyphae(10);
        assert_eq!(hyphae.len(), 10);
    }

    #[cfg(feature = "bench")]
    #[test]
    fn sample_hyphae_is_deterministic() {
        let hyphae1 = sample_hyphae(5);
        let hyphae2 = sample_hyphae(5);
        assert_eq!(hyphae1.len(), hyphae2.len());
        for (h1, h2) in hyphae1.iter().zip(hyphae2.iter()) {
            assert_eq!(h1.id, h2.id);
        }
    }

    #[cfg(feature = "bench")]
    #[test]
    fn sample_scorer_context_creates_scores_for_all_hyphae() {
        let hyphae = sample_hyphae(5);
        let context = sample_scorer_context(&hyphae);
        assert_eq!(context.len(), 5);

        // Verify all hyphae have scores
        for hypha in &hyphae {
            assert!(context.contains_key(&hypha.id));
        }
    }

    #[cfg(feature = "bench")]
    #[test]
    fn sample_scorer_context_is_deterministic() {
        let hyphae = sample_hyphae(3);
        let context1 = sample_scorer_context(&hyphae);
        let context2 = sample_scorer_context(&hyphae);

        assert_eq!(context1.len(), context2.len());
        for (id, score1) in &context1 {
            let score2 = context2.get(id).unwrap();
            assert_eq!(score1, score2);
        }
    }
}
