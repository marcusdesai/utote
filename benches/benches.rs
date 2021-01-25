use criterion::{black_box, criterion_group, criterion_main, Criterion};
use packed_simd::*;
use multiset::Multiset;

type MS4u32 = Multiset<u32, typenum::U4>;
type MS0u32x4 = Multiset<u32x4, typenum::U0>;
type MS2u32x2 = Multiset<u32x2, typenum::U2>;
type MS1u32x4 = Multiset<u32x4, typenum::U1>;


fn intersection(c: &mut Criterion) {
    let slice1 = &[2, 1, 4, 5];
    let slice2 = &[4, 6, 1, 2];

    let set1 = MS4u32::from_slice(slice1);
    let set2 = MS4u32::from_slice(slice2);
    c.bench_function("intersection MS4u32", |b| b.iter(|| black_box(set1.intersection(&set2))));

    let set1 = MS0u32x4::from_slice(slice1);
    let set2 = MS0u32x4::from_slice(slice2);
    c.bench_function("intersection MS0u32x4", |b| b.iter(|| black_box(set1.intersection(&set2))));

    let set1 = MS2u32x2::from_slice(slice1);
    let set2 = MS2u32x2::from_slice(slice2);
    c.bench_function("intersection MS2u32x2", |b| b.iter(|| black_box(set1.intersection(&set2))));

    let set1 = MS1u32x4::from_slice(slice1);
    let set2 = MS1u32x4::from_slice(slice2);
    c.bench_function("intersection MS1u32x4", |b| b.iter(|| black_box(set1.intersection(&set2))));
}


fn union(c: &mut Criterion) {
    let slice1 = &[2, 1, 4, 5];
    let slice2 = &[4, 6, 1, 2];

    let set1 = MS4u32::from_slice(slice1);
    let set2 = MS4u32::from_slice(slice2);
    c.bench_function("union MS4u32", |b| b.iter(|| black_box(set1.union(&set2))));

    let set1 = MS0u32x4::from_slice(slice1);
    let set2 = MS0u32x4::from_slice(slice2);
    c.bench_function("union MS0u32x4", |b| b.iter(|| black_box(set1.union(&set2))));

    let set1 = MS2u32x2::from_slice(slice1);
    let set2 = MS2u32x2::from_slice(slice2);
    c.bench_function("union MS2u32x2", |b| b.iter(|| black_box(set1.union(&set2))));

    let set1 = MS1u32x4::from_slice(slice1);
    let set2 = MS1u32x4::from_slice(slice2);
    c.bench_function("union MS1u32x4", |b| b.iter(|| black_box(set1.union(&set2))));
}


fn collision_entropy(c: &mut Criterion) {
    let slice1 = &[2, 1, 4, 5];
    let mut result: f64 = 0.0;

    let set1 = MS4u32::from_slice(slice1);
    c.bench_function("collision_entropy MS4u32", |b| b.iter(|| result = black_box(set1.collision_entropy())));

    let set1 = MS0u32x4::from_slice(slice1);
    c.bench_function("collision_entropy MS0u32x4", |b| b.iter(|| result = black_box(set1.collision_entropy())));

    let set1 = MS2u32x2::from_slice(slice1);
    c.bench_function("collision_entropy MS2u32x2", |b| b.iter(|| result = black_box(set1.collision_entropy())));

    let set1 = MS1u32x4::from_slice(slice1);
    c.bench_function("collision_entropy MS1u32x4", |b| b.iter(|| result = black_box(set1.collision_entropy())));

    println!("{}", result)
}


fn shannon_entropy(c: &mut Criterion) {
    let slice1 = &[2, 1, 4, 5];
    let mut result: f64 = 0.0;

    let set1 = MS4u32::from_slice(slice1);
    c.bench_function("shannon_entropy MS4u32", |b| b.iter(|| result = black_box(set1.shannon_entropy())));

    let set1 = MS0u32x4::from_slice(slice1);
    c.bench_function("shannon_entropy MS0u32x4", |b| b.iter(|| result = black_box(set1.shannon_entropy())));

    let set1 = MS2u32x2::from_slice(slice1);
    c.bench_function("shannon_entropy MS2u32x2", |b| b.iter(|| result = black_box(set1.shannon_entropy())));

    let set1 = MS1u32x4::from_slice(slice1);
    c.bench_function("shannon_entropy MS1u32x4", |b| b.iter(|| result = black_box(set1.shannon_entropy())));

    println!("{}", result)
}


fn count_non_zero(c: &mut Criterion) {
    let slice1 = &[2, 1, 4, 5];
    let mut result: u32 = 0;

    let set1 = MS4u32::from_slice(slice1);
    c.bench_function("count_non_zero MS4u32", |b| b.iter(|| result = black_box(set1.count_non_zero())));

    let set1 = MS0u32x4::from_slice(slice1);
    c.bench_function("count_non_zero MS0u32x4", |b| b.iter(|| result = black_box(set1.count_non_zero())));

    let set1 = MS2u32x2::from_slice(slice1);
    c.bench_function("count_non_zero MS2u32x2", |b| b.iter(|| result = black_box(set1.count_non_zero())));

    let set1 = MS1u32x4::from_slice(slice1);
    c.bench_function("count_non_zero MS1u32x4", |b| b.iter(|| result = black_box(set1.count_non_zero())));

    assert_eq!(result, 4);
}


criterion_group!(
    benches,
    intersection,
    union,
    collision_entropy,
    shannon_entropy,
    count_non_zero,
);
criterion_main!(benches);
