use criterion::{criterion_group, criterion_main, Criterion};
use rand::distributions::Standard;
use rand::prelude::*;
use utote::{Counter, Multiset};

fn rand_ms<T: Counter, const SIZE: usize, R: RngCore>(rng: &mut R) -> Multiset<T, SIZE>
where
    Standard: Distribution<T>,
{
    let mut arr: [T; SIZE] = [T::ZERO; SIZE];
    arr.iter_mut().for_each(|elem| *elem = rng.gen());
    Multiset::from_array(arr)
}

fn bench_intersection(c: &mut Criterion) {
    let mut group = c.benchmark_group("intersection");
    let mut rng = StdRng::seed_from_u64(0);
    group.bench_function("u8x8", |b| {
        let ms1 = rand_ms::<u8, 8, _>(&mut rng);
        let ms2 = rand_ms::<u8, 8, _>(&mut rng);
        b.iter(|| ms1.intersection(&ms2))
    });
    group.bench_function("u8x64", |b| {
        let ms1 = rand_ms::<u8, 64, _>(&mut rng);
        let ms2 = rand_ms::<u8, 64, _>(&mut rng);
        b.iter(|| ms1.intersection(&ms2))
    });
    group.bench_function("u8x512", |b| {
        let ms1 = rand_ms::<u8, 512, _>(&mut rng);
        let ms2 = rand_ms::<u8, 512, _>(&mut rng);
        b.iter(|| ms1.intersection(&ms2))
    });
    group.bench_function("u32x8", |b| {
        let ms1 = rand_ms::<u32, 8, _>(&mut rng);
        let ms2 = rand_ms::<u32, 8, _>(&mut rng);
        b.iter(|| ms1.intersection(&ms2))
    });
    group.bench_function("u32x64", |b| {
        let ms1 = rand_ms::<u32, 64, _>(&mut rng);
        let ms2 = rand_ms::<u32, 64, _>(&mut rng);
        b.iter(|| ms1.intersection(&ms2))
    });
    group.bench_function("u32x512", |b| {
        let ms1 = rand_ms::<u32, 512, _>(&mut rng);
        let ms2 = rand_ms::<u32, 512, _>(&mut rng);
        b.iter(|| ms1.intersection(&ms2))
    });
    group.finish();
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

fn bench_is_subset(c: &mut Criterion) {
    let mut group = c.benchmark_group("is_subset/worst case");
    group.bench_function("u8x8", |b| {
        let ms1 = Multiset::<u8, 8>::from_array([1; 8]);
        let ms2 = Multiset::<u8, 8>::from_array([2; 8]);
        b.iter(|| ms1.is_subset(&ms2))
    });
    group.bench_function("u8x64", |b| {
        let ms1 = Multiset::<u8, 64>::from_array([1; 64]);
        let ms2 = Multiset::<u8, 64>::from_array([2; 64]);
        b.iter(|| ms1.is_subset(&ms2))
    });
    group.bench_function("u8x512", |b| {
        let ms1 = Multiset::<u8, 512>::from_array([1; 512]);
        let ms2 = Multiset::<u8, 512>::from_array([2; 512]);
        b.iter(|| ms1.is_subset(&ms2))
    });
    group.bench_function("u32x8", |b| {
        let ms1 = Multiset::<u32, 8>::from_array([1; 8]);
        let ms2 = Multiset::<u32, 8>::from_array([2; 8]);
        b.iter(|| ms1.is_subset(&ms2))
    });
    group.bench_function("u32x64", |b| {
        let ms1 = Multiset::<u32, 64>::from_array([1; 64]);
        let ms2 = Multiset::<u32, 64>::from_array([2; 64]);
        b.iter(|| ms1.is_subset(&ms2))
    });
    group.bench_function("u32x512", |b| {
        let ms1 = Multiset::<u32, 512>::from_array([1; 512]);
        let ms2 = Multiset::<u32, 512>::from_array([2; 512]);
        b.iter(|| ms1.is_subset(&ms2))
    });
    group.finish();
}

fn bench_collision_entropy(c: &mut Criterion) {
    let mut group = c.benchmark_group("collision_entropy");
    let mut rng = StdRng::seed_from_u64(0);
    group.bench_function("u8x8", |b| {
        let ms1 = rand_ms::<u8, 8, _>(&mut rng);
        b.iter(|| ms1.collision_entropy())
    });
    group.bench_function("u8x64", |b| {
        let ms1 = rand_ms::<u8, 64, _>(&mut rng);
        b.iter(|| ms1.collision_entropy())
    });
    group.bench_function("u8x512", |b| {
        let ms1 = rand_ms::<u8, 512, _>(&mut rng);
        b.iter(|| ms1.collision_entropy())
    });
    group.bench_function("u32x8", |b| {
        let ms1 = rand_ms::<u32, 8, _>(&mut rng);
        b.iter(|| ms1.collision_entropy())
    });
    group.bench_function("u32x64", |b| {
        let ms1 = rand_ms::<u32, 64, _>(&mut rng);
        b.iter(|| ms1.collision_entropy())
    });
    group.bench_function("u32x512", |b| {
        let ms1 = rand_ms::<u32, 512, _>(&mut rng);
        b.iter(|| ms1.collision_entropy())
    });
    group.finish();
}

fn bench_shannon_entropy(c: &mut Criterion) {
    let mut group = c.benchmark_group("shannon_entropy");
    let mut rng = StdRng::seed_from_u64(0);
    group.bench_function("u8x8", |b| {
        let ms1 = rand_ms::<u8, 8, _>(&mut rng);
        b.iter(|| ms1.shannon_entropy())
    });
    group.bench_function("u8x64", |b| {
        let ms1 = rand_ms::<u8, 64, _>(&mut rng);
        b.iter(|| ms1.shannon_entropy())
    });
    group.bench_function("u8x512", |b| {
        let ms1 = rand_ms::<u8, 512, _>(&mut rng);
        b.iter(|| ms1.shannon_entropy())
    });
    group.bench_function("u32x8", |b| {
        let ms1 = rand_ms::<u32, 8, _>(&mut rng);
        b.iter(|| ms1.shannon_entropy())
    });
    group.bench_function("u32x64", |b| {
        let ms1 = rand_ms::<u32, 64, _>(&mut rng);
        b.iter(|| ms1.shannon_entropy())
    });
    group.bench_function("u32x512", |b| {
        let ms1 = rand_ms::<u32, 512, _>(&mut rng);
        b.iter(|| ms1.shannon_entropy())
    });
    group.finish();
}

criterion_group!(
    benches,
    // bench_intersection,
    // bench_is_disjoint,
    bench_is_subset,
    // bench_collision_entropy,
    // bench_partial_cmp,
    // bench_shannon_entropy,
);
criterion_main!(benches);
