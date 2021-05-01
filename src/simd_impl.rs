use crate::chunks::ChunkUtils;
use crate::{Counter, Multiset2};
use num_traits::AsPrimitive;
use packed_simd::*;
use paste::paste;
#[cfg(feature = "rand")]
use rand::{Rng, RngCore};
use std::fmt::Debug;
use std::mem::MaybeUninit;
use std::ops::{Add, Div, Mul};

#[doc(hidden)]
pub trait SimdTypes: Sized {
    type SIMD128: SimdBasic<Self>;
    type SIMD256: SimdBasic<Self>;
    type SIMDFloat: SimdBasic<f64> + SimdFloat<f64>;
}

macro_rules! impl_simd_types {
    ($scalar:ty, $simd128:ty, $simd256:ty) => {
        impl SimdTypes for $scalar {
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
impl_simd_types!(usize, usizex4, usizex4);

#[doc(hidden)]
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
impl_simd_bool!(usize, usizex4, msizex4);
impl_simd_bool!(f64, f64x4, m64x4);

#[doc(hidden)]
pub trait SimdBasic<N>: Copy + PartialEq + Add<Self, Output = Self> + Debug {
    const LANES: usize;
    type SIMDBool: SimdBool<N, Select = Self>;
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
impl_simd_basic!(usize, usizex4, msizex4);
impl_simd_basic!(f64, f64x4, m64x4);

#[doc(hidden)]
pub trait SimdFloat<N>
where
    Self: Copy + Add<Self, Output = Self> + Mul<Self, Output = Self> + Div<f64, Output = Self>,
{
    type SIMDBool: SimdBool<N, Select = Self>;
    //noinspection RsSelfConvention
    fn is_nan(self) -> Self::SIMDBool;
    fn ln(self) -> Self;
    fn powf(self, other: Self) -> Self;
    fn sum(self) -> N;
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

    #[inline]
    fn sum(self) -> f64 {
        Self::sum(self)
    }
}

macro_rules! intersection_simd {
    ($name:ident, $simd:ty) => {
        #[doc(hidden)]
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
        #[doc(hidden)]
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
        #[doc(hidden)]
        #[inline]
        unsafe fn $name(&self) -> usize {
            self.data
                .fold_chunks::<_, _, { <$simd>::LANES }>(0, |acc, slice| {
                    let vec = <$simd>::from_slice_unaligned_unchecked(slice);
                    acc + vec.gt(<$simd>::splat(N::zero())).count_true()
                })
        }
    };
}

macro_rules! is_disjoint_simd {
    ($name:ident, $simd:ty) => {
        #[doc(hidden)]
        #[inline]
        unsafe fn $name(&self, other: &Self) -> bool {
            self.data
                .zip_all_chunks::<_, { <$simd>::LANES }>(&other.data, |a, b| {
                    let simd_a = <$simd>::from_slice_unaligned_unchecked(a);
                    let simd_b = <$simd>::from_slice_unaligned_unchecked(b);
                    simd_a.min(simd_b) == <$simd>::splat(N::zero())
                })
        }
    };
}

macro_rules! is_subset_simd {
    ($name:ident, $simd:ty) => {
        #[doc(hidden)]
        #[inline]
        unsafe fn $name(&self, other: &Self) -> bool {
            self.data
                .zip_all_chunks::<_, { <$simd>::LANES }>(&other.data, |a, b| {
                    let simd_a = <$simd>::from_slice_unaligned_unchecked(a);
                    let simd_b = <$simd>::from_slice_unaligned_unchecked(b);
                    simd_a.le(simd_b).all()
                })
        }
    };
}

macro_rules! is_superset_simd {
    ($name:ident, $simd:ty) => {
        #[doc(hidden)]
        #[inline]
        unsafe fn $name(&self, other: &Self) -> bool {
            self.data
                .zip_all_chunks::<_, { <$simd>::LANES }>(&other.data, |a, b| {
                    let simd_a = <$simd>::from_slice_unaligned_unchecked(a);
                    let simd_b = <$simd>::from_slice_unaligned_unchecked(b);
                    simd_a.ge(simd_b).all()
                })
        }
    };
}

macro_rules! is_any_lesser_simd {
    ($name:ident, $simd:ty) => {
        #[doc(hidden)]
        #[inline]
        unsafe fn $name(&self, other: &Self) -> bool {
            self.data
                .zip_all_chunks::<_, { <$simd>::LANES }>(&other.data, |a, b| {
                    let simd_a = <$simd>::from_slice_unaligned_unchecked(a);
                    let simd_b = <$simd>::from_slice_unaligned_unchecked(b);
                    simd_a.lt(simd_b).any()
                })
        }
    };
}

macro_rules! is_any_greater_simd {
    ($name:ident, $simd:ty) => {
        #[doc(hidden)]
        #[inline]
        unsafe fn $name(&self, other: &Self) -> bool {
            self.data
                .zip_all_chunks::<_, { <$simd>::LANES }>(&other.data, |a, b| {
                    let simd_a = <$simd>::from_slice_unaligned_unchecked(a);
                    let simd_b = <$simd>::from_slice_unaligned_unchecked(b);
                    simd_a.gt(simd_b).any()
                })
        }
    };
}

macro_rules! total_simd {
    ($name:ident, $simd:ty) => {
        #[doc(hidden)]
        #[inline]
        unsafe fn $name(&self) -> usize {
            if SIZE < <$simd>::LANES {
                self.data
                    .iter()
                    .map(|e| <N as AsPrimitive<usize>>::as_(*e))
                    .sum()
            } else {
                let mut out = [N::zero(); { <$simd>::LANES }];
                let sum_vec = self.data.fold_chunks::<_, _, { <$simd>::LANES }>(
                    <$simd>::splat(N::zero()),
                    |acc, a| {
                        let simd_a = <$simd>::from_slice_unaligned_unchecked(a);
                        acc + simd_a
                    },
                );
                sum_vec.write_to_slice_unaligned_unchecked(&mut out);
                out.iter().map(|e| <N as AsPrimitive<usize>>::as_(*e)).sum()
            }
        }
    };
}

macro_rules! collision_entropy_simd {
    ($name:ident, $simd:ty) => {
        #[doc(hidden)]
        #[inline]
        unsafe fn $name(&self) -> f64 {
            let total: f64 = self.total() as f64;
            -self
                .data
                .fold_chunks::<_, _, { <$simd>::LANES }>(<$simd>::splat(0.0), |acc, slice| {
                    let mut f64_slice =
                        MaybeUninit::<[f64; { <$simd>::LANES }]>::uninit().assume_init();
                    for i in 0..<$simd>::LANES {
                        *f64_slice.get_unchecked_mut(i) =
                            <N as AsPrimitive<f64>>::as_(*slice.get_unchecked(i));
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
    ($name:ident, $simd:ty) => {
        #[doc(hidden)]
        #[inline]
        unsafe fn $name(&self) -> f64 {
            let total: f64 = self.total() as f64;
            -self
                .data
                .fold_chunks::<_, _, { <$simd>::LANES }>(<$simd>::splat(0.0), |acc, slice| {
                    let mut f64_slice =
                        MaybeUninit::<[f64; { <$simd>::LANES }]>::uninit().assume_init();
                    for i in 0..<$simd>::LANES {
                        *f64_slice.get_unchecked_mut(i) =
                            <N as AsPrimitive<f64>>::as_(*slice.get_unchecked(i));
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

macro_rules! simd_dispatch {
    (simd128 = $simd128:ty, simd256 = $simd256:ty |
    pub fn $name:ident (&$self_:ty $(, $arg:ident: $typ:ty)*) -> $ret:ty $body:block) => {
        paste! {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            #[target_feature(enable = "avx2,fma")]
            [<$name _simd>]! { [<_ $name _avx2>], $simd256 }

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            #[target_feature(enable = "avx")]
            [<$name _simd>]! { [<_ $name _avx>], $simd256 }

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            #[target_feature(enable = "sse4.2")]
            [<$name _simd>]! { [<_ $name _sse42>], $simd128 }

            #[doc(hidden)]
            #[inline]
            fn [<_ $name _default>](&$self_, $($arg: $typ),*) -> $ret $body

            #[doc(hidden)]
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

impl<N: Counter, const SIZE: usize> Multiset2<N, SIZE>
where
    [(); N::SIMD128::LANES]: Sized,
    [(); N::SIMD256::LANES]: Sized,
    [(); N::SIMDFloat::LANES]: Sized,
{
    simd_dispatch! {
        simd128 = N::SIMD128, simd256 = N::SIMD256 |
        pub fn intersection(&self, other: &Self) -> Self {
            self.zip_map(other, |s1, s2| s1.min(s2))
        }
    }

    simd_dispatch! {
        simd128 = N::SIMD128, simd256 = N::SIMD256 |
        pub fn union(&self, other: &Self) -> Self {
            self.zip_map(other, |s1, s2| s1.max(s2))
        }
    }

    simd_dispatch! {
        simd128 = N::SIMD128, simd256 = N::SIMD256 |
        pub fn count_non_zero(&self) -> usize {
            self.iter().fold(0, |acc, &elem| {
                acc + <N as AsPrimitive<usize>>::as_(elem.min(N::zero()))
            })
        }
    }

    #[doc(hidden)]
    #[inline]
    pub fn count_zero(&self) -> usize {
        SIZE - self.count_non_zero()
    }

    #[doc(hidden)]
    #[inline]
    pub fn is_singleton(&self) -> bool {
        self.count_non_zero() == 1
    }

    simd_dispatch! {
        simd128 = N::SIMD128, simd256 = N::SIMD256 |
        pub fn is_disjoint(&self, other: &Self) -> bool {
            self.iter()
                .zip(other.data.iter())
                .all(|(a, b)| a.min(b) == &N::zero())
        }
    }

    simd_dispatch! {
        simd128 = N::SIMD128, simd256 = N::SIMD256 |
        pub fn is_subset(&self, other: &Self) -> bool {
            self.iter().zip(other.data.iter()).all(|(a, b)| a <= b)
        }
    }

    simd_dispatch! {
        simd128 = N::SIMD128, simd256 = N::SIMD256 |
        pub fn is_superset(&self, other: &Self) -> bool {
            self.iter().zip(other.data.iter()).all(|(a, b)| a >= b)
        }
    }

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

    simd_dispatch! {
        simd128 = N::SIMD128, simd256 = N::SIMD256 |
        pub fn is_any_lesser(&self, other: &Self) -> bool {
            self.iter().zip(other.data.iter()).any(|(a, b)| a < b)
        }
    }

    simd_dispatch! {
        simd128 = N::SIMD128, simd256 = N::SIMD256 |
        pub fn is_any_greater(&self, other: &Self) -> bool {
            self.iter().zip(other.data.iter()).any(|(a, b)| a > b)
        }
    }

    simd_dispatch! {
        simd128 = N::SIMD128, simd256 = N::SIMD256 |
        pub fn total(&self) -> usize {
            self.iter().map(|e| <N as AsPrimitive<usize>>::as_(*e)).sum()
        }
    }

    #[cfg(feature = "rand")]
    #[doc(hidden)]
    #[inline]
    pub fn choose_random<T: RngCore>(&mut self, rng: &mut T) {
        let total = self.total();
        if total == 0 {
            return;
        }
        let choice_value = rng.gen_range(1..=total);
        let mut res = [N::zero(); SIZE];
        let mut acc = 0;
        for (i, elem) in self.iter().enumerate() {
            acc += <N as AsPrimitive<usize>>::as_(*elem);
            if acc >= choice_value {
                // Safety: `i` cannot be outside of `res`.
                unsafe { *res.get_unchecked_mut(i) = *elem }
                break;
            }
        }
        self.data = res
    }

    simd_dispatch! {
        simd128 = N::SIMDFloat, simd256 = N::SIMDFloat |
        pub fn collision_entropy(&self) -> f64 {
            let total: f64 = self.total().as_();
            -self.into_iter()
                .fold(0.0, |acc, &frequency| {
                    let freq_f64: f64 = <N as AsPrimitive<f64>>::as_(frequency);
                    acc + (freq_f64 / total).powf(2.0)
                })
                .log2()
        }
    }

    simd_dispatch! {
        simd128 = N::SIMDFloat, simd256 = N::SIMDFloat |
        pub fn shannon_entropy(&self) -> f64 {
            let total: f64 = self.total().as_();
            -self.into_iter().fold(0.0, |acc, &frequency| {
                if frequency > N::zero() {
                    let freq_f64: f64 = <N as AsPrimitive<f64>>::as_(frequency);
                    let prob = freq_f64 / total;
                    acc + prob * prob.ln()
                } else {
                    acc
                }
            })
        }
    }
}
