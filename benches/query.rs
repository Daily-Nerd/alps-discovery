// Query Algebra Benchmarks — Composite query evaluation performance
//
// Benchmarks the Query enum's set-theoretic capability matching operators.

use alps_discovery_native::query::Query;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

mod common;

fn bench_text_query(c: &mut Criterion) {
    let net = common::network_with_agents(50, common::TextLength::Medium);
    let query = Query::from("test query");

    c.bench_function("text_query", |b| {
        b.iter(|| net.discover_query(black_box(&query), None))
    });
}

fn bench_all_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("all_query");
    let net = common::network_with_agents(50, common::TextLength::Medium);

    for &sub_query_count in &[2, 3, 4, 5] {
        let sub_queries: Vec<Query> = (0..sub_query_count)
            .map(|i| Query::from(format!("query {}", i).as_str()))
            .collect();
        let query = Query::All(sub_queries);

        group.bench_with_input(
            BenchmarkId::from_parameter(sub_query_count),
            &sub_query_count,
            |b, _| b.iter(|| net.discover_query(black_box(&query), None)),
        );
    }
    group.finish();
}

fn bench_any_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("any_query");
    let net = common::network_with_agents(50, common::TextLength::Medium);

    for &sub_query_count in &[2, 3, 4, 5] {
        let sub_queries: Vec<Query> = (0..sub_query_count)
            .map(|i| Query::from(format!("query {}", i).as_str()))
            .collect();
        let query = Query::Any(sub_queries);

        group.bench_with_input(
            BenchmarkId::from_parameter(sub_query_count),
            &sub_query_count,
            |b, _| b.iter(|| net.discover_query(black_box(&query), None)),
        );
    }
    group.finish();
}

fn bench_exclude_query(c: &mut Criterion) {
    let net = common::network_with_agents(50, common::TextLength::Medium);
    let query = Query::from("test query").exclude("unwanted");

    c.bench_function("exclude_query", |b| {
        b.iter(|| net.discover_query(black_box(&query), None))
    });
}

fn bench_weighted_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("weighted_query");
    let net = common::network_with_agents(50, common::TextLength::Medium);

    for &weight_count in &[2, 3, 4, 5] {
        let weighted_queries: Vec<(Query, f64)> = (0..weight_count)
            .map(|i| {
                let weight = 1.0 / (i + 1) as f64;
                (Query::from(format!("query {}", i).as_str()), weight)
            })
            .collect();
        let query = Query::Weighted(weighted_queries);

        group.bench_with_input(
            BenchmarkId::from_parameter(weight_count),
            &weight_count,
            |b, _| b.iter(|| net.discover_query(black_box(&query), None)),
        );
    }
    group.finish();
}

fn bench_nested_query(c: &mut Criterion) {
    let net = common::network_with_agents(50, common::TextLength::Medium);

    // Create deeply nested query: All(Any(Text, Text), Exclude(Text))
    let query = Query::All(vec![
        Query::Any(vec![
            Query::from("capability A"),
            Query::from("capability B"),
        ]),
        Query::from("capability C").exclude("unwanted"),
    ]);

    c.bench_function("nested_query", |b| {
        b.iter(|| net.discover_query(black_box(&query), None))
    });
}

criterion_group! {
    name = benches;
    config = common::criterion_config();
    targets = bench_text_query, bench_all_query, bench_any_query,
              bench_exclude_query, bench_weighted_query, bench_nested_query
}
criterion_main!(benches);
