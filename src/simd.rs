use crate::chunks::ChunkUtils;
use crate::{Counter, Multiset};
use num_traits::AsPrimitive;
use packed_simd::*;
use paste::paste;
#[cfg(feature = "rand")]
use rand::{Rng, RngCore};
use std::fmt::Debug;
use std::mem::MaybeUninit;
use std::ops::{Add, Div, Mul};
use std::cmp::Ordering;

mod sealed {
    use packed_simd::*;

    pub trait Sealed {}

    macro_rules! impl_sealed {
        ($($t:ty),*) => {$(impl Sealed for $t {})*};
    }

    impl_sealed!(
        u8, u16, u32, u64, usize, f64, m8x16, m8x32, m16x8, m16x16, m32x4, m32x8, msizex4, m64x4,
        u8x16, u8x32, u16x8, u16x16, u32x4, u32x8, u64x4, usizex4, f64x4
    );
}

#[doc(hidden)]
pub trait SimdTypes: sealed::Sealed + Sized {
    type SIMD128: SimdBasic<Self>;
    type SIMD256: SimdBasic<Self>;
    type SIMDFloat: SimdBasic<f64> + SimdFloat<f64>;

    const L128: usize;
    const L256: usize;
    const LF: usize;
}

macro_rules! impl_simd_types {
    ($scalar:ty, $simd128:ty, $simd256:ty) => {
        impl SimdTypes for $scalar {
            type SIMD128 = $simd128;
            type SIMD256 = $simd256;
            type SIMDFloat = f64x4;

            const L128: usize = <$simd128>::lanes();
            const L256: usize = <$simd256>::lanes();
            const LF: usize = f64x4::lanes();
        }
    };
}

impl_simd_types!(u8, u8x16, u8x32);
impl_simd_types!(u16, u16x8, u16x16);
impl_simd_types!(u32, u32x4, u32x8);
impl_simd_types!(u64, u64x4, u64x4);
impl_simd_types!(usize, usizex4, usizex4);

#[doc(hidden)]
pub trait SimdBool<N>: sealed::Sealed {
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
pub trait SimdBasic<N>:
    sealed::Sealed + Copy + PartialEq + Add<Self, Output = Self> + Debug
{
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
    Self: sealed::Sealed
        + Copy
        + Add<Self, Output = Self>
        + Mul<Self, Output = Self>
        + Div<f64, Output = Self>,
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
    ($name:ident, $simd:ty, $lanes:expr) => {
        #[doc(hidden)]
        #[inline]
        unsafe fn $name(&self, other: &Self) -> Self {
            let mut data = std::mem::MaybeUninit::<[N; SIZE]>::uninit().assume_init();
            self.data
                .zip_map_chunks::<_, $lanes>(&other.data, &mut data, |a, b, out| {
                    let simd_a = <$simd>::from_slice_unaligned_unchecked(a);
                    let simd_b = <$simd>::from_slice_unaligned_unchecked(b);
                    simd_a.min(simd_b).write_to_slice_unaligned_unchecked(out);
                });
            Multiset { data }
        }
    };
}

macro_rules! union_simd {
    ($name:ident, $simd:ty, $lanes:expr) => {
        #[doc(hidden)]
        #[inline]
        unsafe fn $name(&self, other: &Self) -> Self {
            let mut data = std::mem::MaybeUninit::<[N; SIZE]>::uninit().assume_init();
            self.data
                .zip_map_chunks::<_, $lanes>(&other.data, &mut data, |a, b, out| {
                    let simd_a = <$simd>::from_slice_unaligned_unchecked(a);
                    let simd_b = <$simd>::from_slice_unaligned_unchecked(b);
                    simd_a.max(simd_b).write_to_slice_unaligned_unchecked(out);
                });
            Multiset { data }
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
                acc + vec.gt(<$simd>::splat(N::zero())).count_true()
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
                simd_a.min(simd_b) == <$simd>::splat(N::zero())
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
            self.data.zip_any_chunks::<_, $lanes>(&other.data, |a, b| {
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
            self.data.zip_any_chunks::<_, $lanes>(&other.data, |a, b| {
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
            if SIZE < <$simd>::LANES {
                self.iter()
                    .map(|e| <N as AsPrimitive<usize>>::as_(*e))
                    .sum()
            } else {
                let mut out = [N::zero(); $lanes];
                let sum_vec =
                    self.data
                        .fold_chunks::<_, _, $lanes>(<$simd>::splat(N::zero()), |acc, a| {
                            let simd_a = <$simd>::from_slice_unaligned_unchecked(a);
                            acc + simd_a
                        });
                sum_vec.write_to_slice_unaligned_unchecked(&mut out);
                out.iter().map(|e| <N as AsPrimitive<usize>>::as_(*e)).sum()
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
    ($name:ident, $simd:ty, $lanes:expr) => {
        #[doc(hidden)]
        #[inline]
        unsafe fn $name(&self) -> f64 {
            let total: f64 = self.total() as f64;
            -self
                .data
                .fold_chunks::<_, _, $lanes>(<$simd>::splat(0.0), |acc, slice| {
                    let mut f64_slice = MaybeUninit::<[f64; $lanes]>::uninit().assume_init();
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
    (simd128 = $simd128:ty, simd256 = $simd256:ty, lanes128 = $lanes128:expr, lanes256 = $lanes256:expr;
    pub fn $name:ident (&$self_:ty $(, $arg:ident: $typ:ty)*) -> $ret:ty $body:block) => {
        paste! {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            #[target_feature(enable = "avx2,fma")]
            [<$name _simd>]! { [<_ $name _avx2>], $simd256, $lanes256 }

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            #[target_feature(enable = "avx")]
            [<$name _simd>]! { [<_ $name _avx>], $simd256, $lanes256 }

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            #[target_feature(enable = "sse4.2")]
            [<$name _simd>]! { [<_ $name _sse42>], $simd128, $lanes128 }

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

// pub trait MultisetSimdFn<N> {
//     unsafe fn _intersection_simd<S, const C: usize>(&self, other: &Self, out: &mut Self)
//     where
//         S: SimdBasic<N>;
//     unsafe fn _union_simd<S, const C: usize>(&self, other: &Self, out: &mut Self)
//     where
//         S: SimdBasic<N>;
//     unsafe fn _count_non_zero_simd<S, const C: usize>(&self) -> usize
//     where
//         S: SimdBasic<N>;
//     unsafe fn _is_disjoint_simd<S, const C: usize>(&self, other: &Self) -> bool
//     where
//         S: SimdBasic<N>;
//     unsafe fn _is_subset_simd<S, const C: usize>(&self, other: &Self) -> bool
//     where
//         S: SimdBasic<N>;
//     unsafe fn _is_superset_simd<S, const C: usize>(&self, other: &Self) -> bool
//     where
//         S: SimdBasic<N>;
//     unsafe fn _is_any_lesser_simd<S, const C: usize>(&self, other: &Self) -> bool
//     where
//         S: SimdBasic<N>;
//     unsafe fn _is_any_greater_simd<S, const C: usize>(&self, other: &Self) -> bool
//     where
//         S: SimdBasic<N>;
//     unsafe fn _total_simd<S, const C: usize>(&self) -> usize
//     where
//         S: SimdBasic<N>;
//     unsafe fn _collision_entropy_simd<S, F, const SC: usize, const FC: usize>(&self) -> f64
//     where
//         S: SimdBasic<N>,
//         F: SimdBasic<f64> + SimdFloat<f64>;
//     unsafe fn _shannon_entropy_simd<S, F, const SC: usize, const FC: usize>(&self) -> f64
//     where
//         S: SimdBasic<N>,
//         F: SimdBasic<f64> + SimdFloat<f64>;
// }
//
// impl<N> MultisetSimdFn<N> for [N]
// where
//     N: SimdTypes + Copy + Zero + AsPrimitive<usize> + AsPrimitive<f64>,
// {
//     #[inline]
//     unsafe fn _intersection_simd<S, const C: usize>(&self, other: &Self, out: &mut Self)
//     where
//         S: SimdBasic<N>,
//     {
//         self.zip_map_chunks::<_, C>(&other, out, |a, b, out_slice| {
//             let simd_a = S::from_slice_unaligned_unchecked(a);
//             let simd_b = S::from_slice_unaligned_unchecked(b);
//             simd_a
//                 .min(simd_b)
//                 .write_to_slice_unaligned_unchecked(out_slice);
//         });
//     }
//
//     #[inline]
//     unsafe fn _union_simd<S, const C: usize>(&self, other: &Self, out: &mut Self)
//     where
//         S: SimdBasic<N>,
//     {
//         self.zip_map_chunks::<_, C>(&other, out, |a, b, out_slice| {
//             let simd_a = S::from_slice_unaligned_unchecked(a);
//             let simd_b = S::from_slice_unaligned_unchecked(b);
//             simd_a
//                 .max(simd_b)
//                 .write_to_slice_unaligned_unchecked(out_slice);
//         });
//     }
//
//     #[inline]
//     unsafe fn _count_non_zero_simd<S, const C: usize>(&self) -> usize
//     where
//         S: SimdBasic<N>,
//     {
//         self.fold_chunks::<_, _, C>(0, |acc, slice| {
//             let vec = S::from_slice_unaligned_unchecked(slice);
//             acc + vec.gt(S::splat(N::zero())).count_true()
//         })
//     }
//
//     #[inline]
//     unsafe fn _is_disjoint_simd<S, const C: usize>(&self, other: &Self) -> bool
//     where
//         S: SimdBasic<N>,
//     {
//         self.zip_all_chunks::<_, C>(&other, |a, b| {
//             let simd_a = S::from_slice_unaligned_unchecked(a);
//             let simd_b = S::from_slice_unaligned_unchecked(b);
//             simd_a.min(simd_b) == S::splat(N::zero())
//         })
//     }
//
//     #[inline]
//     unsafe fn _is_subset_simd<S, const C: usize>(&self, other: &Self) -> bool
//     where
//         S: SimdBasic<N>,
//     {
//         self.zip_all_chunks::<_, C>(&other, |a, b| {
//             let simd_a = S::from_slice_unaligned_unchecked(a);
//             let simd_b = S::from_slice_unaligned_unchecked(b);
//             simd_a.le(simd_b).all()
//         })
//     }
//
//     #[inline]
//     unsafe fn _is_superset_simd<S, const C: usize>(&self, other: &Self) -> bool
//     where
//         S: SimdBasic<N>,
//     {
//         self.zip_all_chunks::<_, C>(&other, |a, b| {
//             let simd_a = S::from_slice_unaligned_unchecked(a);
//             let simd_b = S::from_slice_unaligned_unchecked(b);
//             simd_a.ge(simd_b).all()
//         })
//     }
//
//     #[inline]
//     unsafe fn _is_any_lesser_simd<S, const C: usize>(&self, other: &Self) -> bool
//     where
//         S: SimdBasic<N>,
//     {
//         self.zip_any_chunks::<_, C>(&other, |a, b| {
//             let simd_a = S::from_slice_unaligned_unchecked(a);
//             let simd_b = S::from_slice_unaligned_unchecked(b);
//             simd_a.lt(simd_b).any()
//         })
//     }
//
//     #[inline]
//     unsafe fn _is_any_greater_simd<S, const C: usize>(&self, other: &Self) -> bool
//     where
//         S: SimdBasic<N>,
//     {
//         self.zip_any_chunks::<_, C>(&other, |a, b| {
//             let simd_a = S::from_slice_unaligned_unchecked(a);
//             let simd_b = S::from_slice_unaligned_unchecked(b);
//             simd_a.gt(simd_b).any()
//         })
//     }
//
//     #[inline]
//     unsafe fn _total_simd<S, const C: usize>(&self) -> usize
//     where
//         S: SimdBasic<N>,
//     {
//         if self.len() < C {
//             self.iter().map::<usize, _>(|e| (*e).as_()).sum()
//         } else {
//             let mut out = [N::zero(); C];
//             let sum_vec = self.fold_chunks::<_, _, C>(S::splat(N::zero()), |acc, a| {
//                 let simd_a = S::from_slice_unaligned_unchecked(a);
//                 acc + simd_a
//             });
//             sum_vec.write_to_slice_unaligned_unchecked(&mut out);
//             out.iter().map::<usize, _>(|e| (*e).as_()).sum()
//         }
//     }
//
//     #[inline]
//     unsafe fn _collision_entropy_simd<S, F, const SC: usize, const FC: usize>(&self) -> f64
//     where
//         S: SimdBasic<N>,
//         F: SimdBasic<f64> + SimdFloat<f64>,
//     {
//         let total: f64 = self._total_simd::<S, SC>() as f64;
//         -self
//             .fold_chunks::<_, _, FC>(F::splat(0.0), |acc, slice| {
//                 let mut f64_slice = MaybeUninit::<[f64; FC]>::uninit().assume_init();
//                 for i in 0..FC {
//                     *f64_slice.get_unchecked_mut(i) = (*slice.get_unchecked(i)).as_();
//                 }
//                 let data = F::from_slice_unaligned_unchecked(&f64_slice);
//                 let prob: F = data / total;
//                 acc + prob.powf(F::splat(2.0))
//             })
//             .sum()
//             .log2()
//     }
//
//     #[inline]
//     unsafe fn _shannon_entropy_simd<S, F, const SC: usize, const FC: usize>(&self) -> f64
//     where
//         S: SimdBasic<N>,
//         F: SimdBasic<f64> + SimdFloat<f64>,
//     {
//         let total: f64 = self._total_simd::<S, SC>() as f64;
//         -self
//             .fold_chunks::<_, _, FC>(F::splat(0.0), |acc, slice| {
//                 let mut f64_slice = MaybeUninit::<[f64; FC]>::uninit().assume_init();
//                 for i in 0..FC {
//                     *f64_slice.get_unchecked_mut(i) = (*slice.get_unchecked(i)).as_();
//                 }
//                 let data = F::from_slice_unaligned_unchecked(&f64_slice);
//                 let prob: F = data / total;
//                 let prob_log = prob * prob.ln();
//                 acc + prob_log.is_nan().select(F::splat(0.0), prob_log)
//             })
//             .sum()
//     }
// }
//
// impl<N: Counter, const SIZE: usize> Multiset<N, SIZE>
// where
//     [(); N::L128 * N::L256 * N::LF]: Sized,
// {
//     #[inline]
//     fn _intersection2_default(&self, other: &Self) -> Self {
//         self.zip_map(other, |s1, s2| s1.min(s2))
//     }
//
//     #[inline]
//     unsafe fn _array_intersection2_simd<S, const C: usize>(&self, other: &Self) -> Self
//     where
//         S: SimdBasic<N>,
//     {
//         let mut data = std::mem::MaybeUninit::<[N; SIZE]>::uninit().assume_init();
//         self.data._intersection_simd::<S, C>(&other.data, &mut data);
//         Multiset { data }
//     }
//
//     #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
//     #[target_feature(enable = "avx2,fma")]
//     #[inline]
//     unsafe fn _intersection2_avx2(&self, other: &Self) -> Self {
//         self._array_intersection2_simd::<N::SIMD256, { N::L256 }>(other)
//     }
//
//     #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
//     #[target_feature(enable = "avx")]
//     #[inline]
//     unsafe fn _intersection2_avx(&self, other: &Self) -> Self {
//         self._array_intersection2_simd::<N::SIMD256, { N::L256 }>(other)
//     }
//
//     #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
//     #[target_feature(enable = "sse4.2")]
//     #[inline]
//     unsafe fn _intersection2_sse42(&self, other: &Self) -> Self {
//         self._array_intersection2_simd::<N::SIMD128, { N::L128 }>(other)
//     }
//
//     #[inline]
//     pub fn intersection2(&self, other: &Self) -> Self {
//         unsafe {
//             if is_x86_feature_detected!("avx2") {
//                 self._intersection2_avx2(other)
//             } else if is_x86_feature_detected!("avx") {
//                 self._intersection2_avx(other)
//             } else if is_x86_feature_detected!("sse4.2") {
//                 self._intersection2_sse42(other)
//             } else {
//                 self._intersection2_default(other)
//             }
//         }
//     }
// }

#[allow(unused_braces)]
impl<N: Counter, const SIZE: usize> Multiset<N, SIZE>
where
    [(); N::L128 * N::L256 * N::LF]: Sized,
{
    simd_dispatch! {
        simd128 = N::SIMD128, simd256 = N::SIMD256, lanes128 = {N::L128}, lanes256 = {N::L256};
        pub fn intersection(&self, other: &Self) -> Self {
            self.zip_map(other, |s1, s2| s1.min(s2))
        }
    }

    simd_dispatch! {
        simd128 = N::SIMD128, simd256 = N::SIMD256, lanes128 = {N::L128}, lanes256 = {N::L256};
        pub fn union(&self, other: &Self) -> Self {
            self.zip_map(other, |s1, s2| s1.max(s2))
        }
    }

    simd_dispatch! {
        simd128 = N::SIMD128, simd256 = N::SIMD256, lanes128 = {N::L128}, lanes256 = {N::L256};
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
        simd128 = N::SIMD128, simd256 = N::SIMD256, lanes128 = {N::L128}, lanes256 = {N::L256};
        pub fn is_disjoint(&self, other: &Self) -> bool {
            self.iter()
                .zip(other.data.iter())
                .all(|(a, b)| a.min(b) == &N::zero())
        }
    }

    simd_dispatch! {
        simd128 = N::SIMD128, simd256 = N::SIMD256, lanes128 = {N::L128}, lanes256 = {N::L256};
        pub fn is_subset(&self, other: &Self) -> bool {
            self.iter().zip(other.data.iter()).all(|(a, b)| a <= b)
        }
    }

    simd_dispatch! {
        simd128 = N::SIMD128, simd256 = N::SIMD256, lanes128 = {N::L128}, lanes256 = {N::L256};
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
        simd128 = N::SIMD128, simd256 = N::SIMD256, lanes128 = {N::L128}, lanes256 = {N::L256};
        pub fn is_any_lesser(&self, other: &Self) -> bool {
            self.iter().zip(other.data.iter()).any(|(a, b)| a < b)
        }
    }

    simd_dispatch! {
        simd128 = N::SIMD128, simd256 = N::SIMD256, lanes128 = {N::L128}, lanes256 = {N::L256};
        pub fn is_any_greater(&self, other: &Self) -> bool {
            self.iter().zip(other.data.iter()).any(|(a, b)| a > b)
        }
    }

    simd_dispatch! {
        simd128 = N::SIMD128, simd256 = N::SIMD256, lanes128 = {N::L128}, lanes256 = {N::L256};
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
        simd128 = N::SIMDFloat, simd256 = N::SIMDFloat, lanes128 = {N::LF}, lanes256 = {N::LF};
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
        simd128 = N::SIMDFloat, simd256 = N::SIMDFloat, lanes128 = {N::LF}, lanes256 = {N::LF};
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

impl<N: Counter, const SIZE: usize> PartialOrd for Multiset<N, SIZE>
    where
        [(); N::L128 * N::L256 * N::LF]: Sized,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let mut order: Ordering = Ordering::Equal;
        for (e_self, e_other) in self.iter().zip(other.iter()) {
            match order {
                Ordering::Equal if e_self < e_other => order = Ordering::Less,
                Ordering::Equal if e_self > e_other => order = Ordering::Greater,
                Ordering::Less if e_self > e_other => return None,
                Ordering::Greater if e_self < e_other => return None,
                _ => (),
            }
        }
        Some(order)
    }

    #[inline]
    fn lt(&self, other: &Self) -> bool {
        self.is_proper_subset(other)
    }

    #[inline]
    fn le(&self, other: &Self) -> bool {
        self.is_subset(other)
    }

    #[inline]
    fn gt(&self, other: &Self) -> bool {
        self.is_proper_superset(other)
    }

    #[inline]
    fn ge(&self, other: &Self) -> bool {
        self.is_superset(other)
    }
}
