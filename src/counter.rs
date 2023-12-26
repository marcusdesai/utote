use core::fmt::Debug;
use core::hash::Hash;
#[cfg(feature = "simd")]
use core::simd::*;

mod sealed {
    pub trait Sealed {}

    macro_rules! impl_sealed {
        ($($t:ty),*) => {$(impl Sealed for $t {})*};
    }

    impl_sealed!(u8, u16, u32, u64, usize);
}

#[cfg(not(feature = "simd"))]
pub trait Counter: sealed::Sealed + Debug + Clone + Copy + Default + Eq + Ord + Hash {
    #[doc(hidden)]
    const ZERO: Self;

    #[doc(hidden)]
    const ONE: Self;

    #[doc(hidden)]
    fn saturating_add(self, rhs: Self) -> Self;

    #[doc(hidden)]
    fn saturating_sub(self, rhs: Self) -> Self;

    #[doc(hidden)]
    fn saturating_mul(self, rhs: Self) -> Self;

    #[doc(hidden)]
    fn saturating_div(self, rhs: Self) -> Self;

    #[doc(hidden)]
    fn saturating_pow(self, rhs: u32) -> Self;

    #[doc(hidden)]
    fn abs_diff(self, rhs: Self) -> Self;

    #[doc(hidden)]
    fn as_f64(self) -> f64;

    #[doc(hidden)]
    fn rem(self, rhs: Self) -> Self;

    #[doc(hidden)]
    fn rem_assign(&mut self, rhs: Self);
}

#[cfg(feature = "simd")]
pub trait Counter:
    sealed::Sealed + Debug + Clone + Copy + Default + Eq + Ord + Hash + SimdElement
{
    const ZERO: Self;

    const ONE: Self;

    const LANES: usize;

    type Mask;

    type Simd: SimdPartialOrd + SimdPartialEq + SimdUint;

    fn saturating_add(self, rhs: Self) -> Self;

    fn saturating_sub(self, rhs: Self) -> Self;

    fn saturating_mul(self, rhs: Self) -> Self;

    fn saturating_div(self, rhs: Self) -> Self;

    fn saturating_pow(self, rhs: u32) -> Self;

    fn abs_diff(self, rhs: Self) -> Self;

    fn as_f64(self) -> f64;

    fn rem(self, rhs: Self) -> Self;

    fn rem_assign(&mut self, rhs: Self);

    fn simd_saturating_add(s1: Self::Simd, s2: Self::Simd) -> Self::Simd;

    fn simd_lt(s1: Self::Simd, s2: Self::Simd) -> <Self as Counter>::Mask;

    fn simd_le(s1: Self::Simd, s2: Self::Simd) -> <Self as Counter>::Mask;

    fn simd_gt(s1: Self::Simd, s2: Self::Simd) -> <Self as Counter>::Mask;

    fn simd_ge(s1: Self::Simd, s2: Self::Simd) -> <Self as Counter>::Mask;

    fn simd_eq(s1: Self::Simd, s2: Self::Simd) -> <Self as Counter>::Mask;

    fn simd_slice(s: &[Self]) -> (&[Self], &[Self::Simd], &[Self]);

    fn simd_select(m: <Self as Counter>::Mask, s1: Self::Simd, s2: Self::Simd) -> Self::Simd;

    fn simd_all(m: <Self as Counter>::Mask) -> bool;

    fn simd_any(m: <Self as Counter>::Mask) -> bool;

    fn simd_as_array(s: &Self::Simd) -> &[Self];

    fn simd_zero() -> Self::Simd;
}

macro_rules! impl_counter {
    ($t:ty, mask = $mask:ty, lanes = $lanes:expr) => {
        impl Counter for $t {
            const ZERO: Self = 0;

            const ONE: Self = 1;

            #[cfg(feature = "simd")]
            const LANES: usize = $lanes;

            #[cfg(feature = "simd")]
            type Mask = Mask<$mask, $lanes>;

            #[cfg(feature = "simd")]
            type Simd = Simd<$t, $lanes>;

            #[inline]
            fn saturating_add(self, rhs: Self) -> Self {
                self.saturating_add(rhs)
            }

            #[inline]
            fn saturating_sub(self, rhs: Self) -> Self {
                self.saturating_sub(rhs)
            }

            #[inline]
            fn saturating_mul(self, rhs: Self) -> Self {
                self.saturating_mul(rhs)
            }

            #[inline]
            fn saturating_div(self, rhs: Self) -> Self {
                self.saturating_div(rhs)
            }

            #[inline]
            fn saturating_pow(self, rhs: u32) -> Self {
                self.saturating_pow(rhs)
            }

            #[inline]
            fn as_f64(self) -> f64 {
                self as f64
            }

            #[inline]
            fn abs_diff(self, rhs: Self) -> Self {
                self.abs_diff(rhs)
            }

            #[inline]
            fn rem(self, rhs: Self) -> Self {
                self % rhs
            }

            #[inline]
            fn rem_assign(&mut self, rhs: Self) {
                *self %= rhs
            }

            #[cfg(feature = "simd")]
            #[inline]
            fn simd_saturating_add(s1: Self::Simd, s2: Self::Simd) -> Self::Simd {
                s1.saturating_add(s2)
            }

            #[cfg(feature = "simd")]
            #[inline]
            fn simd_lt(s1: Self::Simd, s2: Self::Simd) -> <Self as Counter>::Mask {
                s1.simd_lt(s2)
            }

            #[cfg(feature = "simd")]
            #[inline]
            fn simd_le(s1: Self::Simd, s2: Self::Simd) -> <Self as Counter>::Mask {
                s1.simd_le(s2)
            }

            #[cfg(feature = "simd")]
            #[inline]
            fn simd_gt(s1: Self::Simd, s2: Self::Simd) -> <Self as Counter>::Mask {
                s1.simd_gt(s2)
            }

            #[cfg(feature = "simd")]
            #[inline]
            fn simd_ge(s1: Self::Simd, s2: Self::Simd) -> <Self as Counter>::Mask {
                s1.simd_ge(s2)
            }

            #[cfg(feature = "simd")]
            #[inline]
            fn simd_eq(s1: Self::Simd, s2: Self::Simd) -> <Self as Counter>::Mask {
                s1.simd_eq(s2)
            }

            #[cfg(feature = "simd")]
            #[inline]
            fn simd_slice(s: &[Self]) -> (&[Self], &[Self::Simd], &[Self]) {
                s.as_simd()
            }

            #[cfg(feature = "simd")]
            #[inline]
            fn simd_select(
                m: <Self as Counter>::Mask,
                s1: Self::Simd,
                s2: Self::Simd,
            ) -> Self::Simd {
                m.select(s1, s2)
            }

            #[cfg(feature = "simd")]
            #[inline]
            fn simd_all(m: <Self as Counter>::Mask) -> bool {
                m.all()
            }

            #[cfg(feature = "simd")]
            #[inline]
            fn simd_any(m: <Self as Counter>::Mask) -> bool {
                m.any()
            }

            #[cfg(feature = "simd")]
            #[inline]
            fn simd_as_array(s: &Self::Simd) -> &[Self] {
                s.as_array()
            }

            #[cfg(feature = "simd")]
            #[inline]
            fn simd_zero() -> Self::Simd {
                Simd::<$t, $lanes>::splat(Self::ZERO)
            }
        }
    };
}

impl_counter!(u8, mask = i8, lanes = 8);
impl_counter!(u16, mask = i16, lanes = 8);
impl_counter!(u32, mask = i32, lanes = 8);
impl_counter!(u64, mask = i64, lanes = 4);
impl_counter!(usize, mask = isize, lanes = 4);
