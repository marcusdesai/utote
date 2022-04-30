use criterion::{criterion_group, criterion_main, Criterion};
use rand::distributions::Standard;
use rand::prelude::*;
use utote::counter::Counter;
use utote::simd_iter::Multiset;


fn rand_ms<T: Counter, const SIZE: usize, R: RngCore>(rng: &mut R) -> Multiset<T, SIZE>
where
    Standard: Distribution<T>,
{
    let mut arr: [T; SIZE] = [T::ZERO; SIZE];
    arr.iter_mut().for_each(|elem| *elem = rng.gen());
    Multiset::from_array(arr)
}

fn bench_is_disjoint(c: &mut Criterion) {
    let mut group = c.benchmark_group("is_disjoint/worst case");
    group.bench_function("u8x8", |b| {
        let ms1 = Multiset::<u8, 8>::from_array([1; 8]);
        let ms2 = Multiset::<u8, 8>::from_array([0; 8]);
        b.iter(|| ms1.is_disjoint(&ms2))
    });
    group.bench_function("u8x64", |b| {
        let ms1 = Multiset::<u8, 64>::from_array([1; 64]);
        let ms2 = Multiset::<u8, 64>::from_array([0; 64]);
        b.iter(|| ms1.is_disjoint(&ms2))
    });
    group.bench_function("u8x512", |b| {
        let ms1 = Multiset::<u8, 512>::from_array([1; 512]);
        let ms2 = Multiset::<u8, 512>::from_array([0; 512]);
        b.iter(|| ms1.is_disjoint(&ms2))
    });
    group.bench_function("u32x8", |b| {
        let ms1 = Multiset::<u32, 8>::from_array([1; 8]);
        let ms2 = Multiset::<u32, 8>::from_array([0; 8]);
        b.iter(|| ms1.is_disjoint(&ms2))
    });
    group.bench_function("u32x64", |b| {
        let ms1 = Multiset::<u32, 64>::from_array([1; 64]);
        let ms2 = Multiset::<u32, 64>::from_array([0; 64]);
        b.iter(|| ms1.is_disjoint(&ms2))
    });
    group.bench_function("u32x512", |b| {
        let ms1 = Multiset::<u32, 512>::from_array([1; 512]);
        let ms2 = Multiset::<u32, 512>::from_array([0; 512]);
        b.iter(|| ms1.is_disjoint(&ms2))
    });
    group.finish();
}

fn bench_partial_cmp(c: &mut Criterion) {
    let mut group = c.benchmark_group("partial_cmp/worst case");
    group.bench_function("u8x8", |b| {
        let ms1 = Multiset::<u8, 8>::from_array([1; 8]);
        let ms2 = Multiset::<u8, 8>::from_array([0; 8]);
        b.iter(|| ms1.partial_cmp(&ms2))
    });
    group.bench_function("u8x64", |b| {
        let ms1 = Multiset::<u8, 64>::from_array([1; 64]);
        let ms2 = Multiset::<u8, 64>::from_array([0; 64]);
        b.iter(|| ms1.partial_cmp(&ms2))
    });
    group.bench_function("u8x512", |b| {
        let ms1 = Multiset::<u8, 512>::from_array([1; 512]);
        let ms2 = Multiset::<u8, 512>::from_array([0; 512]);
        b.iter(|| ms1.partial_cmp(&ms2))
    });
    group.bench_function("u32x8", |b| {
        let ms1 = Multiset::<u32, 8>::from_array([1; 8]);
        let ms2 = Multiset::<u32, 8>::from_array([0; 8]);
        b.iter(|| ms1.partial_cmp(&ms2))
    });
    group.bench_function("u32x64", |b| {
        let ms1 = Multiset::<u32, 64>::from_array([1; 64]);
        let ms2 = Multiset::<u32, 64>::from_array([0; 64]);
        b.iter(|| ms1.partial_cmp(&ms2))
    });
    group.bench_function("u32x512", |b| {
        let ms1 = Multiset::<u32, 512>::from_array([1; 512]);
        let ms2 = Multiset::<u32, 512>::from_array([0; 512]);
        b.iter(|| ms1.partial_cmp(&ms2))
    });
    group.finish();
}

// const IS_SUBSET_SIZE: usize = 59;
// type IsSubsetTy = u8;
//
// fn bench_is_subset(c: &mut Criterion) {
//     let mut group = c.benchmark_group("is_subset");
//     group.bench_function("scalar", |b| {
//         let ms1 = Multiset::<IsSubsetTy, IS_SUBSET_SIZE>::from_array([1; IS_SUBSET_SIZE]);
//         let ms2 = Multiset::<IsSubsetTy, IS_SUBSET_SIZE>::from_array([2; IS_SUBSET_SIZE]);
//         b.iter(|| ms1.is_subset(&ms2))
//     });
//     group.bench_function("simd", |b| {
//         let ms1 = Multiset::<IsSubsetTy, IS_SUBSET_SIZE>::from_array([1; IS_SUBSET_SIZE]);
//         let ms2 = Multiset::<IsSubsetTy, IS_SUBSET_SIZE>::from_array([2; IS_SUBSET_SIZE]);
//         b.iter(|| ms1.is_subset(&ms2))
//     });
//     group.finish();
// }

// const COLLISION_ENTROPY_SIZE: usize = 64;
// type CollisionEntropyTy = u32;
//
// fn bench_collision_entropy(c: &mut Criterion) {
//     let mut group = c.benchmark_group("collision_entropy");
//     let mut rng = StdRng::seed_from_u64(0);
//     let ms = rand_ms::<CollisionEntropyTy, COLLISION_ENTROPY_SIZE, _>(&mut rng);
//     group.bench_function("scalar", |b| {
//         b.iter(|| ms.collision_entropy_scalar())
//     });
//     group.bench_function("simd", |b| {
//         b.iter(|| ms.collision_entropy())
//     });
//     group.finish();
// }

criterion_group!(
    benches,
    bench_is_disjoint,
    // bench_is_subset,
    // bench_collision_entropy,
    // bench_partial_cmp,
);
criterion_main!(benches);
