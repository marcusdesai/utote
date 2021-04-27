use crate::chunks::ChunkUtils;
use crate::{Counter, Multiset2};
use lazy_static::lazy_static;
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

impl SimdTypes<u16> for u16 {
    type SIMD128 = u16x8;
    type SIMD256 = u16x16;
    type SIMDFloat = f64x4;
}

pub trait SimdBasic<N> {
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
}

impl<N: SimdTypes<N> + Counter, const SIZE: usize> Multiset2<N, SIZE> {
    pub fn _d(t: &[N], out: &mut [N]) {
        let vec: N::SIMD128 = unsafe { N::SIMD128::from_slice_unaligned_unchecked(t) };
        let aa = vec.min(vec);
        unsafe { aa.write_to_slice_unaligned_unchecked(out) };
    }
}

#[allow(clippy::upper_case_acronyms)]
enum CPUFeature {
    AVX2,
    AVX,
    SSE42,
    DEF,
}

impl CPUFeature {
    #[inline]
    fn val(&self) -> &CPUFeature {
        &self
    }
}

lazy_static! {
    static ref CPU_FEATURE: CPUFeature = {
        if is_x86_feature_detected!("avx2") {
            CPUFeature::AVX2
        } else if is_x86_feature_detected!("avx") {
            CPUFeature::AVX
        } else if is_x86_feature_detected!("sse4.2") {
            CPUFeature::SSE42
        } else {
            CPUFeature::DEF
        }
    };
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
                    match CPU_FEATURE.val() {
                        CPUFeature::AVX2 => self.[<_ $name _avx2>]($($arg),*),
                        CPUFeature::AVX => self.[<_ $name _avx>]($($arg),*),
                        CPUFeature::SSE42 => self.[<_ $name _sse42>]($($arg),*),
                        CPUFeature::DEF => self.[<_ $name _default>]($($arg),*),
                    }
                }
            }
        }
    };
}

macro_rules! intersection_simd {
    ($name:ident, $simd:ty, $lanes:expr) => {
        #[doc(hidden)]
        #[inline]
        unsafe fn $name(&self, other: &Self) -> Self {
            let mut data = std::mem::MaybeUninit::<[u16; SIZE]>::uninit().assume_init();
            self.data
                .zip_map_chunks::<_, $lanes>(&other.data, &mut data, |a, b, out| {
                    let simd_a = <$simd>::from_slice_unaligned_unchecked(a);
                    let simd_b = <$simd>::from_slice_unaligned_unchecked(b);
                    simd_a.min(simd_b).write_to_slice_unaligned_unchecked(out);
                });
            Multiset2 { data }
        }
    };
}

macro_rules! union_simd {
    ($name:ident, $simd:ty, $lanes:expr) => {
        #[doc(hidden)]
        #[inline]
        unsafe fn $name(&self, other: &Self) -> Self {
            let mut data = std::mem::MaybeUninit::<[u16; SIZE]>::uninit().assume_init();
            self.data
                .zip_map_chunks::<_, $lanes>(&other.data, &mut data, |a, b, out| {
                    let simd_a = <$simd>::from_slice_unaligned_unchecked(a);
                    let simd_b = <$simd>::from_slice_unaligned_unchecked(b);
                    simd_a.max(simd_b).write_to_slice_unaligned_unchecked(out);
                });
            Multiset2 { data }
        }
    };
}

macro_rules! count_non_zero_simd {
    ($name:ident, $simd:ty, $lanes:expr) => {
        #[doc(hidden)]
        #[inline]
        unsafe fn $name(&self) -> usize {
            self.data.fold_chunks::<_, _, $lanes>(0, |acc, slice| {
                let vec = <$simd>::from_slice_unaligned_unchecked(slice);
                acc + vec.gt(<$simd>::splat(0)).bitmask().count_ones() as usize
            })
        }
    };
}

macro_rules! is_disjoint_simd {
    ($name:ident, $simd:ty, $lanes:expr) => {
        #[doc(hidden)]
        #[inline]
        unsafe fn $name(&self, other: &Self) -> bool {
            self.data.zip_all_chunks::<_, $lanes>(&other.data, |a, b| {
                let simd_a = <$simd>::from_slice_unaligned_unchecked(a);
                let simd_b = <$simd>::from_slice_unaligned_unchecked(b);
                simd_a.min(simd_b) == <$simd>::splat(0)
            })
        }
    };
}

macro_rules! is_subset_simd {
    ($name:ident, $simd:ty, $lanes:expr) => {
        #[doc(hidden)]
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
        #[doc(hidden)]
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
        #[doc(hidden)]
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
        #[doc(hidden)]
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
        #[doc(hidden)]
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
        #[doc(hidden)]
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
        #[doc(hidden)]
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
    #[doc(hidden)]
    #[inline]
    fn _intersection_default(&self, other: &Self) -> Self {
        self.zip_map(other, |s1, s2| s1.min(s2))
    }

    simd_variants!(intersection, intersection_simd, 8, 16, u16x8, u16x16);
    simd_dispatch! { pub fn intersection(&self, other: &Self) -> Self; }

    #[doc(hidden)]
    #[inline]
    fn _union_default(&self, other: &Self) -> Self {
        self.zip_map(other, |s1, s2| s1.max(s2))
    }

    simd_variants!(union, union_simd, 8, 16, u16x8, u16x16);
    simd_dispatch! { pub fn union(&self, other: &Self) -> Self; }

    #[doc(hidden)]
    #[inline]
    pub fn count_zero(&self) -> usize {
        SIZE - self.count_non_zero()
    }

    #[doc(hidden)]
    #[inline]
    fn _count_non_zero_default(&self) -> usize {
        self.fold(0, |acc, elem| acc + elem.min(1) as usize)
    }

    simd_variants!(count_non_zero, count_non_zero_simd, 8, 16, u16x8, u16x16);
    simd_dispatch! { pub fn count_non_zero(&self) -> usize; }

    #[doc(hidden)]
    #[inline]
    pub fn is_singleton(&self) -> bool {
        self.count_non_zero() == 1
    }

    #[doc(hidden)]
    #[inline]
    fn _is_disjoint_default(&self, other: &Self) -> bool {
        self.data
            .iter()
            .zip(other.data.iter())
            .all(|(a, b)| a.min(b) == &0)
    }

    simd_variants!(is_disjoint, is_disjoint_simd, 8, 16, u16x8, u16x16);
    simd_dispatch! { pub fn is_disjoint(&self, other: &Self) -> bool; }

    #[doc(hidden)]
    #[inline]
    fn _is_subset_default(&self, other: &Self) -> bool {
        self.data.iter().zip(other.data.iter()).all(|(a, b)| a <= b)
    }

    simd_variants!(is_subset, is_subset_simd, 8, 16, u16x8, u16x16);
    simd_dispatch! { pub fn is_subset(&self, other: &Self) -> bool; }

    #[doc(hidden)]
    #[inline]
    fn _is_superset_default(&self, other: &Self) -> bool {
        self.data.iter().zip(other.data.iter()).all(|(a, b)| a >= b)
    }

    simd_variants!(is_superset, is_superset_simd, 8, 16, u16x8, u16x16);
    simd_dispatch! { pub fn is_superset(&self, other: &Self) -> bool; }

    #[doc(hidden)]
    #[inline]
    pub fn is_proper_subset(&self, other: &Self) -> bool {
        self != other && self.is_subset(other)
    }

    #[doc(hidden)]
    #[inline]
    pub fn is_proper_superset(&self, other: &Self) -> bool {
        self != other && self.is_superset(other)
    }

    #[doc(hidden)]
    #[inline]
    pub fn _is_any_lesser_default(&self, other: &Self) -> bool {
        self.data.iter().zip(other.data.iter()).any(|(a, b)| a < b)
    }

    simd_variants!(is_any_lesser, is_any_lesser_simd, 8, 16, u16x8, u16x16);
    simd_dispatch! { pub fn is_any_lesser(&self, other: &Self) -> bool; }

    #[doc(hidden)]
    #[inline]
    fn _is_any_greater_default(&self, other: &Self) -> bool {
        self.data.iter().zip(other.data.iter()).any(|(a, b)| a > b)
    }

    simd_variants!(is_any_greater, is_any_greater_simd, 8, 16, u16x8, u16x16);
    simd_dispatch! { pub fn is_any_greater(&self, other: &Self) -> bool; }

    #[doc(hidden)]
    #[inline]
    fn _total_default(&self) -> usize {
        self.data.iter().map(|e| *e as usize).sum()
    }

    simd_variants!(total, total_simd, 8, 16, u16x8, u16x16);
    simd_dispatch! { pub fn total(&self) -> usize; }

    #[cfg(feature = "rand")]
    #[doc(hidden)]
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

    #[doc(hidden)]
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

    #[doc(hidden)]
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
