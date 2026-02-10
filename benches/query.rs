use criterion::{criterion_group, criterion_main, Criterion};

// Stub benchmark function - implementation in Task 7
fn stub(_c: &mut Criterion) {}

criterion_group!(benches, stub);
criterion_main!(benches);
