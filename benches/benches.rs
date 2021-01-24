use criterion::{black_box, criterion_group, criterion_main, Criterion};
use packed_simd::*;
use multiset::Multiset;

type MS4u32 = Multiset<u32, typenum::U4>;
type MS0u32x4 = Multiset<u32x4, typenum::U0>;
type MS2u32x2 = Multiset<u32x2, typenum::U2>;


fn union_benchmark(c: &mut Criterion) {
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
}


criterion_group!(benches, union_benchmark);
criterion_main!(benches);
