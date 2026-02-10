// Enzyme Benchmarks — SLNEnzyme kernel evaluation performance
//
// Benchmarks the multi-kernel routing engine at the heart of the discovery system.

use alps_discovery_native::core::chemistry::QuerySignature;
use alps_discovery_native::core::config::QueryConfig;
use alps_discovery_native::core::enzyme::{Enzyme, ReasoningKernel, SLNEnzyme, SLNEnzymeConfig};
use alps_discovery_native::core::signal::{Signal, Tendril};
use alps_discovery_native::core::types::TrailId;
use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};

mod common;

fn bench_enzyme_evaluate(c: &mut Criterion) {
    let mut group = c.benchmark_group("enzyme_evaluate");

    for &hypha_count in &[10, 50, 100] {
        let hyphae = common::sample_hyphae(hypha_count);
        let hypha_refs: Vec<_> = hyphae.iter().collect();
        let context = common::sample_scorer_context(&hyphae);
        let enzyme = SLNEnzyme::with_discovery_kernels(SLNEnzymeConfig::default());
        let signal = Signal::Tendril(Tendril {
            trail_id: TrailId([0u8; 32]),
            query_signature: QuerySignature { minhash: [0u8; 64] },
            query_config: QueryConfig,
        });

        group.bench_with_input(
            BenchmarkId::from_parameter(hypha_count),
            &hypha_count,
            |b, _| {
                b.iter(|| {
                    enzyme.evaluate_with_scores(
                        black_box(&signal),
                        black_box(&hypha_refs),
                        black_box(&context),
                    )
                })
            },
        );
    }
    group.finish();
}

fn bench_enzyme_process(c: &mut Criterion) {
    let mut group = c.benchmark_group("enzyme_process");

    for &hypha_count in &[10, 50, 100] {
        let hyphae = common::sample_hyphae(hypha_count);
        let hypha_refs: Vec<_> = hyphae.iter().collect();
        let context = common::sample_scorer_context(&hyphae);
        let signal = Signal::Tendril(Tendril {
            trail_id: TrailId([0u8; 32]),
            query_signature: QuerySignature { minhash: [0u8; 64] },
            query_config: QueryConfig,
        });

        group.bench_with_input(
            BenchmarkId::from_parameter(hypha_count),
            &hypha_count,
            |b, _| {
                b.iter_batched(
                    || SLNEnzyme::with_discovery_kernels(SLNEnzymeConfig::default()),
                    |mut enzyme| {
                        enzyme.process(
                            black_box(&signal),
                            black_box(&hypha_refs),
                            black_box(&context),
                        )
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

fn bench_individual_kernels(c: &mut Criterion) {
    use alps_discovery_native::core::enzyme::{
        CapabilityKernel, LoadBalancingKernel, NoveltyKernel, TemporalRecencyKernel,
    };

    let hyphae = common::sample_hyphae(50);
    let hypha_refs: Vec<_> = hyphae.iter().collect();
    let context = common::sample_scorer_context(&hyphae);
    let signal = Signal::Tendril(Tendril {
        trail_id: TrailId([0u8; 32]),
        query_signature: QuerySignature { minhash: [0u8; 64] },
        query_config: QueryConfig,
    });

    c.bench_function("capability_kernel", |b| {
        let kernel = CapabilityKernel;
        b.iter(|| {
            kernel.evaluate(
                black_box(&signal),
                black_box(&hypha_refs),
                black_box(&context),
            )
        })
    });

    c.bench_function("load_balancing_kernel", |b| {
        let kernel = LoadBalancingKernel;
        b.iter(|| {
            kernel.evaluate(
                black_box(&signal),
                black_box(&hypha_refs),
                black_box(&context),
            )
        })
    });

    c.bench_function("novelty_kernel", |b| {
        let kernel = NoveltyKernel;
        b.iter(|| {
            kernel.evaluate(
                black_box(&signal),
                black_box(&hypha_refs),
                black_box(&context),
            )
        })
    });

    c.bench_function("temporal_recency_kernel", |b| {
        let kernel = TemporalRecencyKernel;
        b.iter(|| {
            kernel.evaluate(
                black_box(&signal),
                black_box(&hypha_refs),
                black_box(&context),
            )
        })
    });
}

criterion_group! {
    name = benches;
    config = common::criterion_config();
    targets = bench_enzyme_evaluate, bench_enzyme_process, bench_individual_kernels
}
criterion_main!(benches);
