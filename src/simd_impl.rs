use crate::chunks::ChunkUtils;
use crate::small_num::SmallNumConsts;
use crate::{Counter, Multiset2};
use num_traits::AsPrimitive;
use packed_simd::*;
use paste::paste;
#[cfg(feature = "rand")]
use rand::{Rng, RngCore};
use std::mem::MaybeUninit;
use std::ops::{Add, Div};

pub trait SimdTypes<N> {
    type SIMD128: SimdBasic<N> + Add<N> + Add<Self::SIMD128> + Div<N> + Copy + PartialEq;
    type SIMD256: SimdBasic<N> + Add<N> + Add<Self::SIMD256> + Div<N> + Copy + PartialEq;
    type SIMDFloat: SimdBasic<f64> + SimdFloat<f64>;
}

macro_rules! impl_simd_types {
    ($scalar:ty, $simd128:ty, $simd256:ty) => {
        impl SimdTypes<$scalar> for $scalar {
            type SIMD128 = $simd128;
            type SIMD256 = $simd256;
            type SIMDFloat = f64x4;
        }
    };
}

impl_simd_types!(u8, u8x16, u8x32);
impl_simd_types!(u16, u16x8, u16x16);
impl_simd_types!(u32, u32x4, u32x8);
impl_simd_types!(u64, u64x4, u64x4);

pub trait SimdBool<N> {
    type Select: SimdBasic<N>;
    fn all(self) -> bool;
    fn any(self) -> bool;
    fn count_true(self) -> usize;
    fn select(self, a: Self::Select, b: Self::Select) -> Self::Select;
}

macro_rules! impl_simd_bool {
    ($scalar:ty, $simd:ty, $simd_bool:ty) => {
        impl SimdBool<$scalar> for $simd_bool {
            type Select = $simd;

            #[inline]
            fn all(self) -> bool {
                Self::all(self)
            }

            #[inline]
            fn any(self) -> bool {
                Self::any(self)
            }

            #[inline]
            fn count_true(self) -> usize {
                Self::bitmask(self).count_ones() as usize
            }

            #[inline]
            fn select(self, a: Self::Select, b: Self::Select) -> Self::Select {
                Self::select(self, a, b)
            }
        }
    };
}

impl_simd_bool!(u8, u8x16, m8x16);
impl_simd_bool!(u8, u8x32, m8x32);
impl_simd_bool!(u16, u16x8, m16x8);
impl_simd_bool!(u16, u16x16, m16x16);
impl_simd_bool!(u32, u32x4, m32x4);
impl_simd_bool!(u32, u32x8, m32x8);
impl_simd_bool!(u64, u64x4, m64x4);
impl_simd_bool!(f64, f64x4, m64x4);

pub trait SimdBasic<N> {
    const LANES: usize;
    type SIMDBool: SimdBool<N>;
    fn splat(value: N) -> Self;
    unsafe fn from_slice_unaligned_unchecked(slice: &[N]) -> Self;
    unsafe fn write_to_slice_unaligned_unchecked(self, slice: &mut [N]);
    fn max(self, other: Self) -> Self;
    fn min(self, other: Self) -> Self;
    fn ge(self, other: Self) -> Self::SIMDBool;
    fn gt(self, other: Self) -> Self::SIMDBool;
    fn le(self, other: Self) -> Self::SIMDBool;
    fn lt(self, other: Self) -> Self::SIMDBool;
}

macro_rules! impl_simd_basic {
    ($scalar:ty, $simd:ty, $simd_bool:ty) => {
        impl SimdBasic<$scalar> for $simd {
            const LANES: usize = <$simd>::lanes();

            type SIMDBool = $simd_bool;

            #[inline]
            fn splat(value: $scalar) -> Self {
                Self::splat(value)
            }

            #[inline]
            unsafe fn from_slice_unaligned_unchecked(slice: &[$scalar]) -> Self {
                Self::from_slice_unaligned_unchecked(slice)
            }

            #[inline]
            unsafe fn write_to_slice_unaligned_unchecked(self, slice: &mut [$scalar]) {
                Self::write_to_slice_unaligned_unchecked(self, slice);
            }

            //noinspection RsUnresolvedReference
            #[inline]
            fn max(self, other: Self) -> Self {
                Self::max(self, other)
            }

            //noinspection RsUnresolvedReference
            #[inline]
            fn min(self, other: Self) -> Self {
                Self::min(self, other)
            }

            #[inline]
            fn ge(self, other: Self) -> Self::SIMDBool {
                Self::ge(self, other)
            }

            #[inline]
            fn gt(self, other: Self) -> Self::SIMDBool {
                Self::gt(self, other)
            }

            #[inline]
            fn le(self, other: Self) -> Self::SIMDBool {
                Self::le(self, other)
            }

            #[inline]
            fn lt(self, other: Self) -> Self::SIMDBool {
                Self::lt(self, other)
            }
        }
    };
}

impl_simd_basic!(u8, u8x16, m8x16);
impl_simd_basic!(u8, u8x32, m8x32);
impl_simd_basic!(u16, u16x8, m16x8);
impl_simd_basic!(u16, u16x16, m16x16);
impl_simd_basic!(u32, u32x4, m32x4);
impl_simd_basic!(u32, u32x8, m32x8);
impl_simd_basic!(u64, u64x4, m64x4);
impl_simd_basic!(f64, f64x4, m64x4);

pub trait SimdFloat<N> {
    type SIMDBool: SimdBool<N>;
    //noinspection RsSelfConvention
    fn is_nan(self) -> Self::SIMDBool;
    fn ln(self) -> Self;
    fn powf(self, other: Self) -> Self;
}

impl SimdFloat<f64> for f64x4 {
    type SIMDBool = m64x4;

    #[inline]
    fn is_nan(self) -> Self::SIMDBool {
        Self::is_nan(self)
    }

    //noinspection RsUnresolvedReference
    #[inline]
    fn ln(self) -> Self {
        Self::ln(self)
    }

    //noinspection RsUnresolvedReference
    #[inline]
    fn powf(self, other: Self) -> Self {
        Self::powf(self, other)
    }
}

macro_rules! intersection_simd {
    ($name:ident, $simd:ty) => {
        #[inline]
        unsafe fn $name(&self, other: &Self) -> Self {
            let mut data = std::mem::MaybeUninit::<[N; SIZE]>::uninit().assume_init();
            self.data.zip_map_chunks::<_, { <$simd>::LANES }>(
                &other.data,
                &mut data,
                |a, b, out| {
                    let simd_a = <$simd>::from_slice_unaligned_unchecked(a);
                    let simd_b = <$simd>::from_slice_unaligned_unchecked(b);
                    simd_a.min(simd_b).write_to_slice_unaligned_unchecked(out);
                },
            );
            Multiset2 { data }
        }
    };
}

macro_rules! union_simd {
    ($name:ident, $simd:ty) => {
        #[inline]
        unsafe fn $name(&self, other: &Self) -> Self {
            let mut data = std::mem::MaybeUninit::<[N; SIZE]>::uninit().assume_init();
            self.data.zip_map_chunks::<_, { <$simd>::LANES }>(
                &other.data,
                &mut data,
                |a, b, out| {
                    let simd_a = <$simd>::from_slice_unaligned_unchecked(a);
                    let simd_b = <$simd>::from_slice_unaligned_unchecked(b);
                    simd_a.max(simd_b).write_to_slice_unaligned_unchecked(out);
                },
            );
            Multiset2 { data }
        }
    };
}

macro_rules! count_non_zero_simd {
    ($name:ident, $simd:ty) => {
        #[inline]
        unsafe fn $name(&self) -> usize {
            self.data.fold_chunks::<_, _, { <$simd>::LANES }>(0, |acc, slice| {
                let vec = <$simd>::from_slice_unaligned_unchecked(slice);
                acc + vec.gt(<$simd>::splat(N::zero())).count_true()
            })
        }
    };
}

macro_rules! is_disjoint_simd {
    ($name:ident, $simd:ty) => {
        #[inline]
        unsafe fn $name(&self, other: &Self) -> bool {
            self.data.zip_all_chunks::<_, { <$simd>::LANES }>(&other.data, |a, b| {
                let simd_a = <$simd>::from_slice_unaligned_unchecked(a);
                let simd_b = <$simd>::from_slice_unaligned_unchecked(b);
                simd_a.min(simd_b) == <$simd>::splat(N::zero())
            })
        }
    };
}

macro_rules! simd_dispatch2 {
    (pub fn $name:ident (&$self_:ty $(, $arg:ident: $typ:ty)*) -> $ret:ty $body:block) => {
        paste! {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            #[target_feature(enable = "avx2,fma")]
            [<$name _simd>]! { [<_ $name _avx2>], N::SIMD256 }

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            #[target_feature(enable = "avx")]
            [<$name _simd>]! { [<_ $name _avx>], N::SIMD256 }

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            #[target_feature(enable = "sse4.2")]
            [<$name _simd>]! { [<_ $name _sse42>], N::SIMD128 }

            #[inline]
            fn [<_ $name _default>](&$self_, $($arg: $typ),*) -> $ret $body

            #[inline]
            pub fn $name(&self, $($arg: $typ),*) -> $ret {
                unsafe {
                    if is_x86_feature_detected!("avx2") {
                        self.[<_ $name _avx2>]($($arg),*)
                    } else if is_x86_feature_detected!("avx") {
                        self.[<_ $name _avx>]($($arg),*)
                    } else if is_x86_feature_detected!("sse4.2") {
                        self.[<_ $name _sse42>]($($arg),*)
                    } else {
                        self.[<_ $name _default>]($($arg),*)
                    }
                }
            }
        }
    };
}

impl<N: SimdTypes<N> + Counter + SmallNumConsts, const SIZE: usize> Multiset2<N, SIZE>
where
    [(); N::SIMD128::LANES]: Sized,
    [(); N::SIMD256::LANES]: Sized,
    [(); N::SIMDFloat::LANES]: Sized,
{
    simd_dispatch2!{
        pub fn intersection(&self, other: &Self) -> Self {
            self.zip_map(other, |s1, s2| s1.min(s2))
        }
    }

    simd_dispatch2!{
        pub fn union(&self, other: &Self) -> Self {
            self.zip_map(other, |s1, s2| s1.max(s2))
        }
    }

    simd_dispatch2!{
        pub fn count_non_zero(&self) -> usize {
            self.fold(0, |acc, elem| acc + <N as AsPrimitive<usize>>::as_(elem.min(N::zero())))
        }
    }

    #[inline]
    pub fn count_zero(&self) -> usize {
        SIZE - self.count_non_zero()
    }

    #[inline]
    pub fn is_singleton(&self) -> bool {
        self.count_non_zero() == 1
    }

    simd_dispatch2!{
        pub fn is_disjoint(&self, other: &Self) -> bool {
            self.data
                .iter()
                .zip(other.data.iter())
                .all(|(a, b)| a.min(b) == &N::zero())
        }
    }
}

macro_rules! simd_variants {
    ($name:ty, $fn_macro:ident, $lanes128:expr, $lanes256:expr, $simd128:ty, $simd256:ty
    $(, $simd_f128:ty, $simd_f256:ty)*) => {
        paste! {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            #[target_feature(enable = "avx2,fma")]
            $fn_macro! { [<_ $name _avx2>], $simd256, $lanes256 $(, $simd_f256)* }

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            #[target_feature(enable = "avx")]
            $fn_macro! { [<_ $name _avx>], $simd256, $lanes256 $(, $simd_f256)* }

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            #[target_feature(enable = "sse4.2")]
            $fn_macro! { [<_ $name _sse42>], $simd128, $lanes128 $(, $simd_f128)* }
        }
    };
}

macro_rules! simd_dispatch {
    ($(#[$outer:meta])*
    pub fn $name:ident (&self $(, $arg:ident: $typ:ty)*) -> $ret:ty;) => {
        paste! {
            $(#[$outer])*
            #[inline]
            pub fn $name(&self, $($arg: $typ),*) -> $ret {
                unsafe {
                    if is_x86_feature_detected!("avx2") {
                        self.[<_ $name _avx2>]($($arg),*)
                    } else if is_x86_feature_detected!("avx") {
                        self.[<_ $name _avx>]($($arg),*)
                    } else if is_x86_feature_detected!("sse4.2") {
                        self.[<_ $name _sse42>]($($arg),*)
                    } else {
                        self.[<_ $name _default>]($($arg),*)
                    }
                }
            }
        }
    };
}

macro_rules! is_subset_simd {
    ($name:ident, $simd:ty, $lanes:expr) => {
        #[inline]
        unsafe fn $name(&self, other: &Self) -> bool {
            self.data.zip_all_chunks::<_, $lanes>(&other.data, |a, b| {
                let simd_a = <$simd>::from_slice_unaligned_unchecked(a);
                let simd_b = <$simd>::from_slice_unaligned_unchecked(b);
                simd_a.le(simd_b).all()
            })
        }
    };
}

macro_rules! is_superset_simd {
    ($name:ident, $simd:ty, $lanes:expr) => {
        #[inline]
        unsafe fn $name(&self, other: &Self) -> bool {
            self.data.zip_all_chunks::<_, $lanes>(&other.data, |a, b| {
                let simd_a = <$simd>::from_slice_unaligned_unchecked(a);
                let simd_b = <$simd>::from_slice_unaligned_unchecked(b);
                simd_a.ge(simd_b).all()
            })
        }
    };
}

macro_rules! is_any_lesser_simd {
    ($name:ident, $simd:ty, $lanes:expr) => {
        #[inline]
        unsafe fn $name(&self, other: &Self) -> bool {
            self.data.zip_all_chunks::<_, $lanes>(&other.data, |a, b| {
                let simd_a = <$simd>::from_slice_unaligned_unchecked(a);
                let simd_b = <$simd>::from_slice_unaligned_unchecked(b);
                simd_a.lt(simd_b).any()
            })
        }
    };
}

macro_rules! is_any_greater_simd {
    ($name:ident, $simd:ty, $lanes:expr) => {
        #[inline]
        unsafe fn $name(&self, other: &Self) -> bool {
            self.data.zip_all_chunks::<_, $lanes>(&other.data, |a, b| {
                let simd_a = <$simd>::from_slice_unaligned_unchecked(a);
                let simd_b = <$simd>::from_slice_unaligned_unchecked(b);
                simd_a.gt(simd_b).any()
            })
        }
    };
}

macro_rules! total_simd {
    ($name:ident, $simd:ty, $lanes:expr) => {
        #[inline]
        unsafe fn $name(&self) -> usize {
            if SIZE < $lanes {
                self.data.iter().map(|e| *e as usize).sum()
            } else {
                let mut out = [0; $lanes];
                let sum_vec = self
                    .data
                    .fold_chunks::<_, _, $lanes>(<$simd>::splat(0), |acc, a| {
                        let simd_a = <$simd>::from_slice_unaligned_unchecked(a);
                        acc + simd_a
                    });
                sum_vec.write_to_slice_unaligned_unchecked(&mut out);
                out.iter().map(|e| *e as usize).sum()
            }
        }
    };
}

macro_rules! collision_entropy_simd {
    ($name:ident, $simd:ty, $lanes:expr) => {
        #[inline]
        unsafe fn $name(&self) -> f64 {
            let total: f64 = self.total() as f64;
            -self
                .data
                .fold_chunks::<_, _, $lanes>(<$simd>::splat(0.0), |acc, slice| {
                    let mut f64_slice = MaybeUninit::<[f64; $lanes]>::uninit().assume_init();
                    for i in 0..$lanes {
                        *f64_slice.get_unchecked_mut(i) = *slice.get_unchecked(i) as f64;
                    }
                    let data = <$simd>::from_slice_unaligned_unchecked(&f64_slice);
                    acc + (data / total).powf(<$simd>::splat(2.0))
                })
                .sum()
                .log2()
        }
    };
}

macro_rules! shannon_entropy_simd {
    ($name:ident, $simd:ty, $lanes:expr) => {
        #[inline]
        unsafe fn $name(&self) -> f64 {
            let total: f64 = self.total() as f64;
            -self
                .data
                .fold_chunks::<_, _, $lanes>(<$simd>::splat(0.0), |acc, slice| {
                    let mut f64_slice = MaybeUninit::<[f64; $lanes]>::uninit().assume_init();
                    for i in 0..$lanes {
                        *f64_slice.get_unchecked_mut(i) = *slice.get_unchecked(i) as f64;
                    }
                    let data = <$simd>::from_slice_unaligned_unchecked(&f64_slice);
                    let prob = data / total;
                    let prob_log = prob * prob.ln();
                    acc + prob_log.is_nan().select(<$simd>::splat(0.0), prob_log)
                })
                .sum()
        }
    };
}

impl<const SIZE: usize> Multiset2<u16, SIZE> {

    #[inline]
    fn _is_subset_default(&self, other: &Self) -> bool {
        self.data.iter().zip(other.data.iter()).all(|(a, b)| a <= b)
    }

    simd_variants!(is_subset, is_subset_simd, 8, 16, u16x8, u16x16);
    simd_dispatch! { pub fn is_subset(&self, other: &Self) -> bool; }

    #[inline]
    fn _is_superset_default(&self, other: &Self) -> bool {
        self.data.iter().zip(other.data.iter()).all(|(a, b)| a >= b)
    }

    simd_variants!(is_superset, is_superset_simd, 8, 16, u16x8, u16x16);
    simd_dispatch! { pub fn is_superset(&self, other: &Self) -> bool; }

    #[inline]
    pub fn is_proper_subset(&self, other: &Self) -> bool {
        self != other && self.is_subset(other)
    }

    #[inline]
    pub fn is_proper_superset(&self, other: &Self) -> bool {
        self != other && self.is_superset(other)
    }

    #[inline]
    pub fn _is_any_lesser_default(&self, other: &Self) -> bool {
        self.data.iter().zip(other.data.iter()).any(|(a, b)| a < b)
    }

    simd_variants!(is_any_lesser, is_any_lesser_simd, 8, 16, u16x8, u16x16);
    simd_dispatch! { pub fn is_any_lesser(&self, other: &Self) -> bool; }

    #[inline]
    fn _is_any_greater_default(&self, other: &Self) -> bool {
        self.data.iter().zip(other.data.iter()).any(|(a, b)| a > b)
    }

    simd_variants!(is_any_greater, is_any_greater_simd, 8, 16, u16x8, u16x16);
    simd_dispatch! { pub fn is_any_greater(&self, other: &Self) -> bool; }

    #[inline]
    fn _total_default(&self) -> usize {
        self.data.iter().map(|e| *e as usize).sum()
    }

    simd_variants!(total, total_simd, 8, 16, u16x8, u16x16);
    simd_dispatch! { pub fn total(&self) -> usize; }

    #[cfg(feature = "rand")]
    #[inline]
    pub fn choose_random<T: RngCore>(&mut self, rng: &mut T) {
        let total = self.total();
        if total == 0 {
            return;
        }
        let choice_value = rng.gen_range(1..=total);
        let mut res = [0; SIZE];
        let mut acc = 0;
        for (i, elem) in self.data.iter().enumerate() {
            acc += *elem as usize;
            if acc >= choice_value {
                // Safety: `i` cannot be outside of `res`.
                unsafe { *res.get_unchecked_mut(i) = *elem }
                break;
            }
        }
        self.data = res
    }

    #[inline]
    pub fn _collision_entropy_default(&self) -> f64 {
        let total: f64 = self.total().as_(); // todo: note use of .as_()
        -self
            .fold(0.0, |acc, frequency| {
                let freq_f64: f64 = frequency.as_();
                acc + (freq_f64 / total).powf(2.0)
            })
            .log2()
    }

    simd_variants!(
        collision_entropy,
        collision_entropy_simd,
        4,
        4,
        f64x4,
        f64x4
    );
    simd_dispatch! { pub fn collision_entropy(&self) -> f64; }

    #[inline]
    pub fn _shannon_entropy_default(&self) -> f64 {
        let total: f64 = self.total().as_();
        -self.fold(0.0, |acc, frequency| {
            if frequency > 0 {
                let freq_f64: f64 = frequency.as_();
                let prob = freq_f64 / total;
                acc + prob * prob.ln()
            } else {
                acc
            }
        })
    }

    simd_variants!(shannon_entropy, shannon_entropy_simd, 4, 4, f64x4, f64x4);
    simd_dispatch! { pub fn shannon_entropy(&self) -> f64; }
}
