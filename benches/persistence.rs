// Persistence Benchmarks — Network save and load performance
//
// Benchmarks the JSON-based snapshot serialization system.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::fs;

mod common;

fn bench_save(c: &mut Criterion) {
    let mut group = c.benchmark_group("save");
    let temp_dir = std::env::temp_dir();

    for &agent_count in &[10, 50, 100] {
        let net = common::network_with_agents(agent_count, common::TextLength::Medium);
        let save_path_buf = temp_dir.join(format!("alps_bench_save_{}.json", agent_count));
        let save_path = save_path_buf.to_str().unwrap();

        group.bench_with_input(
            BenchmarkId::from_parameter(agent_count),
            &agent_count,
            |b, _| {
                b.iter(|| {
                    net.save(black_box(save_path)).unwrap();
                })
            },
        );

        // Cleanup
        let _ = fs::remove_file(&save_path_buf);
    }
    group.finish();
}

fn bench_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("load");
    let temp_dir = std::env::temp_dir();

    for &agent_count in &[10, 50, 100] {
        // Pre-create snapshot file
        let net = common::network_with_agents(agent_count, common::TextLength::Medium);
        let load_path_buf = temp_dir.join(format!("alps_bench_load_{}.json", agent_count));
        let load_path = load_path_buf.to_str().unwrap();
        net.save(load_path).unwrap();

        group.bench_with_input(
            BenchmarkId::from_parameter(agent_count),
            &agent_count,
            |b, _| {
                b.iter(|| {
                    use alps_discovery_native::network::LocalNetwork;
                    LocalNetwork::load(black_box(load_path)).unwrap()
                })
            },
        );

        // Cleanup
        let _ = fs::remove_file(&load_path_buf);
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = common::criterion_config();
    targets = bench_save, bench_load
}
criterion_main!(benches);
