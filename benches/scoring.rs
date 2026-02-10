// Scoring Benchmarks — MinHash and TF-IDF scorer performance
//
// Benchmarks the Scorer trait implementations used for capability matching.

use alps_discovery_native::core::config::LshConfig;
use alps_discovery_native::scorer::{MinHashScorer, Scorer, TfIdfScorer};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

mod common;

fn bench_minhash_scorer_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("minhash_scorer_index");
    let config = LshConfig::default();

    for &agent_count in &[10, 50, 100] {
        for length in &[
            common::TextLength::Short,
            common::TextLength::Medium,
            common::TextLength::Long,
        ] {
            let caps = common::capability_text(*length);

            group.bench_with_input(
                BenchmarkId::new(format!("{:?}", length), agent_count),
                &(agent_count, length),
                |b, _| {
                    b.iter(|| {
                        let mut scorer = MinHashScorer::new(config.clone());
                        for i in 0..agent_count {
                            let agent_name = format!("agent-{}", i);
                            let cap = &caps[i % caps.len()];
                            scorer.index_capabilities(
                                black_box(&agent_name),
                                black_box(&[cap.as_str()]),
                            );
                        }
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_minhash_scorer_score(c: &mut Criterion) {
    let mut group = c.benchmark_group("minhash_scorer_score");
    let config = LshConfig::default();
    let caps = common::capability_text(common::TextLength::Medium);

    for &agent_count in &[10, 50, 100] {
        let mut scorer = MinHashScorer::new(config.clone());

        for i in 0..agent_count {
            let agent_name = format!("agent-{}", i);
            let cap = &caps[i % caps.len()];
            scorer.index_capabilities(&agent_name, &[cap.as_str()]);
        }

        group.bench_with_input(
            BenchmarkId::from_parameter(agent_count),
            &agent_count,
            |b, _| b.iter(|| scorer.score(black_box("test query"))),
        );
    }
    group.finish();
}

fn bench_tfidf_scorer_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("tfidf_scorer_index");
    let caps = common::capability_text(common::TextLength::Medium);

    for &agent_count in &[10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::from_parameter(agent_count),
            &agent_count,
            |b, _| {
                b.iter(|| {
                    let mut scorer = TfIdfScorer::new();
                    for i in 0..agent_count {
                        let agent_name = format!("agent-{}", i);
                        let cap = &caps[i % caps.len()];
                        scorer
                            .index_capabilities(black_box(&agent_name), black_box(&[cap.as_str()]));
                    }
                })
            },
        );
    }
    group.finish();
}

fn bench_tfidf_scorer_score(c: &mut Criterion) {
    let mut group = c.benchmark_group("tfidf_scorer_score");
    let caps = common::capability_text(common::TextLength::Medium);

    for &agent_count in &[10, 50, 100] {
        let mut scorer = TfIdfScorer::new();

        for i in 0..agent_count {
            let agent_name = format!("agent-{}", i);
            let cap = &caps[i % caps.len()];
            scorer.index_capabilities(&agent_name, &[cap.as_str()]);
        }

        group.bench_with_input(
            BenchmarkId::from_parameter(agent_count),
            &agent_count,
            |b, _| b.iter(|| scorer.score(black_box("test query"))),
        );
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = common::criterion_config();
    targets = bench_minhash_scorer_index, bench_minhash_scorer_score, bench_tfidf_scorer_index, bench_tfidf_scorer_score
}
criterion_main!(benches);
