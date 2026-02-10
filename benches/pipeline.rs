// Pipeline Benchmarks — Full discovery pipeline and feedback index performance
//
// Benchmarks the end-to-end discovery flow and feedback lookup optimization.

use alps_discovery_native::bench_internals::{FeedbackIndex, FeedbackRecord};
use alps_discovery_native::core::config::LshConfig;
use alps_discovery_native::core::lsh::MinHasher;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

mod common;

fn bench_discover(c: &mut Criterion) {
    let mut group = c.benchmark_group("discover");

    for &agent_count in &[10, 50, 100, 500] {
        // Use smaller sample size for 500-agent configuration
        if agent_count >= 500 {
            group.sample_size(10);
        }

        let net = common::network_with_agents(agent_count, common::TextLength::Medium);

        group.bench_with_input(
            BenchmarkId::from_parameter(agent_count),
            &agent_count,
            |b, _| b.iter(|| net.discover(black_box("test query"))),
        );
    }
    group.finish();
}

fn bench_discover_with_feedback(c: &mut Criterion) {
    let mut group = c.benchmark_group("discover_with_feedback");

    for &feedback_count in &[0, 50, 200] {
        let net = common::network_with_feedback(50, feedback_count);

        group.bench_with_input(
            BenchmarkId::from_parameter(feedback_count),
            &feedback_count,
            |b, _| b.iter(|| net.discover(black_box("test query"))),
        );
    }
    group.finish();
}

fn bench_run_pipeline(c: &mut Criterion) {
    // This benchmark requires bench_internals access to run_pipeline directly.
    // For now, we'll use discover as a proxy since run_pipeline is internal.
    let net = common::network_with_agents(50, common::TextLength::Medium);

    c.bench_function("run_pipeline", |b| {
        b.iter(|| net.discover(black_box("test query")))
    });
}

fn bench_feedback_index_insert(c: &mut Criterion) {
    let config = LshConfig::default();
    let hasher = MinHasher::new(config.dimensions);
    let mut index = FeedbackIndex::new();
    let query = "test query";
    let query_sig = hasher.hash_key(query.as_bytes(), &config.shingle_mode);

    c.bench_function("feedback_index_insert", |b| {
        b.iter(|| {
            let record = FeedbackRecord {
                query_minhash: black_box(query_sig),
                outcome: black_box(1.0),
            };
            index.insert(black_box(record));
        })
    });
}

fn bench_feedback_index_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("feedback_index_lookup");
    let config = LshConfig::default();
    let hasher = MinHasher::new(config.dimensions);

    for &record_count in &[10, 50, 200, 500] {
        let mut index = FeedbackIndex::new();
        let queries = common::sample_queries();

        // Pre-populate index
        for i in 0..record_count {
            let query = queries[i % queries.len()];
            let sig = hasher.hash_key(query.as_bytes(), &config.shingle_mode);
            index.insert(FeedbackRecord {
                query_minhash: sig,
                outcome: 1.0,
            });
        }

        let test_query_sig = hasher.hash_key("test query".as_bytes(), &config.shingle_mode);

        group.bench_with_input(
            BenchmarkId::from_parameter(record_count),
            &record_count,
            |b, _| b.iter(|| index.find_candidates(black_box(&test_query_sig))),
        );
    }
    group.finish();
}

fn bench_discover_with_confidence(c: &mut Criterion) {
    let net = common::network_with_agents(50, common::TextLength::Medium);

    c.bench_function("discover_with_confidence", |b| {
        b.iter(|| net.discover_with_confidence(black_box("test query")))
    });
}

criterion_group! {
    name = benches;
    config = common::criterion_config();
    targets = bench_discover, bench_discover_with_feedback, bench_run_pipeline,
              bench_feedback_index_insert, bench_feedback_index_lookup, bench_discover_with_confidence
}
criterion_main!(benches);
