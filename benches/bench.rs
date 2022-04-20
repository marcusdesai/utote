use criterion::{criterion_group, criterion_main, Criterion};
use num_traits::Unsigned;
use rand::distributions::Standard;
use rand::prelude::*;
use utote::simd_iter::Multiset;


fn rand_ms<T, const SIZE: usize, R: RngCore>(rng: &mut R) -> Multiset<T, SIZE>
where
    T: Copy + Unsigned,
    Standard: Distribution<T>,
{
    let mut arr: [T; SIZE] = [T::zero(); SIZE];
    arr.iter_mut().for_each(|elem| *elem = rng.gen());
    Multiset::from_array(arr)
}

const IS_DISJOINT_SIZE: usize = 59;
type IsDisjointTy = u8;

fn bench_is_disjoint(c: &mut Criterion) {
    let mut group = c.benchmark_group("is_disjoint");
    group.bench_function("scalar", |b| {
        let ms1 = Multiset::<IsDisjointTy, IS_DISJOINT_SIZE>::from_array([1; IS_DISJOINT_SIZE]);
        let ms2 = Multiset::<IsDisjointTy, IS_DISJOINT_SIZE>::from_array([0; IS_DISJOINT_SIZE]);
        b.iter(|| ms1.is_disjoint_scalar(&ms2))
    });
    group.bench_function("simd", |b| {
        let ms1 = Multiset::<IsDisjointTy, IS_DISJOINT_SIZE>::from_array([1; IS_DISJOINT_SIZE]);
        let ms2 = Multiset::<IsDisjointTy, IS_DISJOINT_SIZE>::from_array([0; IS_DISJOINT_SIZE]);
        b.iter(|| ms1.is_disjoint(&ms2))
    });
    group.finish();
}

const IS_SUBSET_SIZE: usize = 59;
type IsSubsetTy = u8;

fn bench_is_subset(c: &mut Criterion) {
    let mut group = c.benchmark_group("is_subset");
    group.bench_function("scalar", |b| {
        let ms1 = Multiset::<IsSubsetTy, IS_SUBSET_SIZE>::from_array([1; IS_SUBSET_SIZE]);
        let ms2 = Multiset::<IsSubsetTy, IS_SUBSET_SIZE>::from_array([2; IS_SUBSET_SIZE]);
        b.iter(|| ms1.is_subset_scalar(&ms2))
    });
    group.bench_function("simd", |b| {
        let ms1 = Multiset::<IsSubsetTy, IS_SUBSET_SIZE>::from_array([1; IS_SUBSET_SIZE]);
        let ms2 = Multiset::<IsSubsetTy, IS_SUBSET_SIZE>::from_array([2; IS_SUBSET_SIZE]);
        b.iter(|| ms1.is_subset(&ms2))
    });
    group.finish();
}

const COLLISION_ENTROPY_SIZE: usize = 59;
type CollisionEntropyTy = u8;

fn bench_collision_entropy(c: &mut Criterion) {
    let mut group = c.benchmark_group("collision_entropy");
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let ms = rand_ms::<CollisionEntropyTy, COLLISION_ENTROPY_SIZE, _>(&mut rng);
    group.bench_function("scalar", |b| {
        b.iter(|| ms.collision_entropy_scalar())
    });
    group.bench_function("simd", |b| {
        b.iter(|| ms.collision_entropy())
    });
    group.finish();
}

// const CHOOSE_SIZE: usize = 1024;
// type ChooseVal = u32;
//
// fn bench_choose_random(c: &mut Criterion) {
//     let mut group = c.benchmark_group("choose_random");
//     let mut rng = rand::rngs::StdRng::seed_from_u64(0);
//     group.bench_function("one", |b| {
//         b.iter(|| {
//             let mut ms1 = rand_ms::<ChooseVal, CHOOSE_SIZE, _>(&mut rng);
//             ms1.choose_random(&mut rng)
//         })
//     });
//     group.bench_function("two", |b| {
//         b.iter(|| {
//             let mut ms2 = rand_ms::<ChooseVal, CHOOSE_SIZE, _>(&mut rng);
//             ms2.choose_random_2(&mut rng)
//         })
//     });
//     group.finish();
// }

criterion_group!(
    benches,
    bench_is_disjoint,
    bench_is_subset,
    bench_collision_entropy,
    // bench_choose_random,
);
criterion_main!(benches);
