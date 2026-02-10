use criterion::{criterion_group, criterion_main, Criterion};

mod common;

// Stub benchmark function - implementation in Task 3
// Verify common module is accessible
fn stub(c: &mut Criterion) {
    let _net = common::network_with_agents(10, common::TextLength::Short);
    c.bench_function("stub", |b| b.iter(|| 1 + 1));
}

criterion_group! {
    name = benches;
    config = common::criterion_config();
    targets = stub
}
criterion_main!(benches);
