use core::fmt::Debug;
use core::hash::Hash;
#[cfg(feature = "simd")]
use core::simd::{LaneCount, Simd, SimdElement, SupportedLaneCount};

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

    fn saturating_add(self, rhs: Self) -> Self;

    fn saturating_sub(self, rhs: Self) -> Self;

    fn saturating_mul(self, rhs: Self) -> Self;

    fn saturating_div(self, rhs: Self) -> Self;

    fn saturating_pow(self, rhs: u32) -> Self;

    fn abs_diff(self, rhs: Self) -> Self;

    fn as_f64(self) -> f64;

    fn rem(self, rhs: Self) -> Self;

    fn rem_assign(&mut self, rhs: Self);

    fn simd_saturating_add<const LANES: usize>(
        s1: Simd<Self, LANES>,
        s2: Simd<Self, LANES>,
    ) -> Simd<Self, LANES>
    where
        LaneCount<LANES>: SupportedLaneCount;
}

macro_rules! impl_counter {
    ($($t:ty),*) => {$(
        impl Counter for $t {
            const ZERO: Self = 0;

            const ONE: Self = 1;

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
            fn simd_saturating_add<const LANES: usize>(
                s1: Simd<Self, LANES>,
                s2: Simd<Self, LANES>,
            ) -> Simd<Self, LANES>
            where
                LaneCount<LANES>: SupportedLaneCount
            {
                s1.saturating_add(s2)
            }
        }
    )*};
}

impl_counter!(u8, u16, u32, u64, usize);
