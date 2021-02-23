use criterion::{black_box, criterion_group, criterion_main, Criterion};
use packed_simd::*;
use rand::prelude::*;
use utote::Multiset;

type MS4u32 = Multiset<u32, typenum::U4>;
type MS0u32x4 = Multiset<u32x4, typenum::U0>;
type MS2u32x2 = Multiset<u32x2, typenum::U2>;
type MS1u32x4 = Multiset<u32x4, typenum::U1>;

fn _random_array(rng: &mut SmallRng) -> [u32; 4] {
    [rng.gen(), rng.gen(), rng.gen(), rng.gen()]
}

fn from_slice(c: &mut Criterion) {
    let rng = &mut SmallRng::seed_from_u64(thread_rng().next_u64());

    c.bench_function("from_slice MS4u32", |b| {
        b.iter(|| black_box(MS4u32::from_slice(&_random_array(rng))))
    });

    c.bench_function("from_slice MS0u32x4", |b| {
        b.iter(|| black_box(MS0u32x4::from_slice(&_random_array(rng))))
    });

    c.bench_function("from_slice MS2u32x2", |b| {
        b.iter(|| black_box(MS2u32x2::from_slice(&_random_array(rng))))
    });

    c.bench_function("from_slice MS1u32x4", |b| {
        b.iter(|| black_box(MS1u32x4::from_slice(&_random_array(rng))))
    });
}

fn intersection(c: &mut Criterion) {
    let rng = &mut SmallRng::seed_from_u64(thread_rng().next_u64());

    c.bench_function("intersection MS4u32", |b| {
        b.iter(|| {
            let set1 = MS4u32::from_slice(&_random_array(rng));
            let set2 = MS4u32::from_slice(&_random_array(rng));
            black_box(set1.intersection(&set2))
        })
    });

    c.bench_function("intersection MS0u32x4", |b| {
        let set1 = MS0u32x4::from_slice(&_random_array(rng));
        let set2 = MS0u32x4::from_slice(&_random_array(rng));
        b.iter(|| black_box(set1.intersection(&set2)))
    });

    c.bench_function("intersection MS2u32x2", |b| {
        let set1 = MS2u32x2::from_slice(&_random_array(rng));
        let set2 = MS2u32x2::from_slice(&_random_array(rng));
        b.iter(|| black_box(set1.intersection(&set2)))
    });

    c.bench_function("intersection MS1u32x4", |b| {
        let set1 = MS1u32x4::from_slice(&_random_array(rng));
        let set2 = MS1u32x4::from_slice(&_random_array(rng));
        b.iter(|| black_box(set1.intersection(&set2)))
    });
}

fn union(c: &mut Criterion) {
    let rng = &mut SmallRng::seed_from_u64(thread_rng().next_u64());

    c.bench_function("union MS4u32", |b| {
        b.iter(|| {
            let set1 = MS4u32::from_slice(&_random_array(rng));
            let set2 = MS4u32::from_slice(&_random_array(rng));
            black_box(set1.union(&set2))
        })
    });

    c.bench_function("union MS0u32x4", |b| {
        let set1 = MS0u32x4::from_slice(&_random_array(rng));
        let set2 = MS0u32x4::from_slice(&_random_array(rng));
        b.iter(|| black_box(set1.union(&set2)))
    });

    c.bench_function("union MS2u32x2", |b| {
        let set1 = MS2u32x2::from_slice(&_random_array(rng));
        let set2 = MS2u32x2::from_slice(&_random_array(rng));
        b.iter(|| black_box(set1.union(&set2)))
    });

    c.bench_function("union MS1u32x4", |b| {
        let set1 = MS1u32x4::from_slice(&_random_array(rng));
        let set2 = MS1u32x4::from_slice(&_random_array(rng));
        b.iter(|| black_box(set1.union(&set2)))
    });
}

fn collision_entropy(c: &mut Criterion) {
    let mut result: f64 = 0.0;
    let mut sign: f64 = 1.0;
    let rng = &mut SmallRng::seed_from_u64(thread_rng().next_u64());

    c.bench_function("collision_entropy MS4u32", |b| {
        b.iter(|| {
            let set = MS4u32::from_slice(&_random_array(rng));
            result += sign * black_box(set.collision_entropy());
            sign *= -1.0;
        })
    });

    c.bench_function("collision_entropy MS0u32x4", |b| {
        b.iter(|| {
            let set = MS4u32::from_slice(&_random_array(rng));
            result += sign * black_box(set.collision_entropy());
            sign *= -1.0;
        })
    });

    c.bench_function("collision_entropy MS2u32x2", |b| {
        b.iter(|| {
            let set = MS4u32::from_slice(&_random_array(rng));
            result += sign * black_box(set.collision_entropy());
            sign *= -1.0;
        })
    });

    c.bench_function("collision_entropy MS1u32x4", |b| {
        b.iter(|| {
            let set = MS4u32::from_slice(&_random_array(rng));
            result += sign * black_box(set.collision_entropy());
            sign *= -1.0;
        })
    });

    println!("{}", result)
}

fn shannon_entropy(c: &mut Criterion) {
    let mut result: f64 = 0.0;
    let mut sign: f64 = 1.0;
    let rng = &mut SmallRng::seed_from_u64(thread_rng().next_u64());

    c.bench_function("shannon_entropy MS4u32", |b| {
        b.iter(|| {
            let set = MS4u32::from_slice(&_random_array(rng));
            result += sign * black_box(set.shannon_entropy());
            sign *= -1.0;
        })
    });

    c.bench_function("shannon_entropy MS0u32x4", |b| {
        b.iter(|| {
            let set = MS4u32::from_slice(&_random_array(rng));
            result += sign * black_box(set.shannon_entropy());
            sign *= -1.0;
        })
    });

    c.bench_function("shannon_entropy MS2u32x2", |b| {
        b.iter(|| {
            let set = MS4u32::from_slice(&_random_array(rng));
            result += sign * black_box(set.shannon_entropy());
            sign *= -1.0;
        })
    });

    c.bench_function("shannon_entropy MS1u32x4", |b| {
        b.iter(|| {
            let set = MS4u32::from_slice(&_random_array(rng));
            result += sign * black_box(set.shannon_entropy());
            sign *= -1.0;
        })
    });

    println!("{}", result)
}

fn count_non_zero(c: &mut Criterion) {
    let slice1 = &[2, 1, 4, 5];
    let mut result: u32 = 0;

    let set1 = MS4u32::from_slice(slice1);
    c.bench_function("count_non_zero MS4u32", |b| {
        b.iter(|| result = black_box(set1.count_non_zero()))
    });

    let set1 = MS0u32x4::from_slice(slice1);
    c.bench_function("count_non_zero MS0u32x4", |b| {
        b.iter(|| result = black_box(set1.count_non_zero()))
    });

    let set1 = MS2u32x2::from_slice(slice1);
    c.bench_function("count_non_zero MS2u32x2", |b| {
        b.iter(|| result = black_box(set1.count_non_zero()))
    });

    let set1 = MS1u32x4::from_slice(slice1);
    c.bench_function("count_non_zero MS1u32x4", |b| {
        b.iter(|| result = black_box(set1.count_non_zero()))
    });

    assert_eq!(result, 4);
}

criterion_group!(
    benches,
    from_slice,
    intersection,
    union,
    collision_entropy,
    shannon_entropy,
    count_non_zero,
);
criterion_main!(benches);
