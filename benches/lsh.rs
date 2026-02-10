// LSH Benchmarks — MinHash signature generation and similarity computation
//
// Benchmarks the core LSH hashing primitives used throughout the discovery pipeline.

use alps_discovery_native::core::config::LshConfig;
use alps_discovery_native::core::lsh::MinHasher;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

mod common;

fn bench_hash_key(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_key");
    let config = LshConfig::default();
    let hasher = MinHasher::new(config.dimensions);

    for length in &[
        common::TextLength::Short,
        common::TextLength::Medium,
        common::TextLength::Long,
    ] {
        let caps = common::capability_text(*length);
        let text = &caps[0];
        let bytes = text.as_bytes();
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", length)),
            length,
            |b, _| b.iter(|| hasher.hash_key(black_box(bytes), black_box(&config.shingle_mode))),
        );
    }
    group.finish();
}

fn bench_similarity(c: &mut Criterion) {
    let config = LshConfig::default();
    let hasher = MinHasher::new(config.dimensions);
    let caps = common::capability_text(common::TextLength::Medium);

    let sig1 = hasher.hash_key(caps[0].as_bytes(), &config.shingle_mode);
    let sig2 = hasher.hash_key(caps[1].as_bytes(), &config.shingle_mode);

    c.bench_function("similarity", |b| {
        b.iter(|| MinHasher::similarity(black_box(&sig1), black_box(&sig2)))
    });
}

fn bench_similarity_with_confidence(c: &mut Criterion) {
    let config = LshConfig::default();
    let hasher = MinHasher::new(config.dimensions);
    let caps = common::capability_text(common::TextLength::Medium);

    let sig1 = hasher.hash_key(caps[0].as_bytes(), &config.shingle_mode);
    let sig2 = hasher.hash_key(caps[1].as_bytes(), &config.shingle_mode);

    c.bench_function("similarity_with_confidence", |b| {
        b.iter(|| MinHasher::similarity_with_confidence(black_box(&sig1), black_box(&sig2)))
    });
}

fn bench_multiple_capabilities(c: &mut Criterion) {
    let mut group = c.benchmark_group("multiple_capabilities");
    let config = LshConfig::default();
    let hasher = MinHasher::new(config.dimensions);

    for length in &[
        common::TextLength::Short,
        common::TextLength::Medium,
        common::TextLength::Long,
    ] {
        let caps = common::capability_text(*length);
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", length)),
            length,
            |b, _| {
                b.iter(|| {
                    for cap in &caps {
                        black_box(hasher.hash_key(cap.as_bytes(), &config.shingle_mode));
                    }
                })
            },
        );
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = common::criterion_config();
    targets = bench_hash_key, bench_similarity, bench_similarity_with_confidence, bench_multiple_capabilities
}
criterion_main!(benches);
