use core::iter::Sum;
use core::ops::{Rem, RemAssign};
#[cfg(feature = "simd")]
use core::simd::SimdElement;

mod sealed {
    pub trait Sealed {}

    macro_rules! impl_sealed {
        ($($t:ty),*) => {$(impl Sealed for $t {})*};
    }

    impl_sealed!(u8, u16, u32, u64, usize);
}

#[cfg(not(feature = "simd"))]
pub trait Counter:
    sealed::Sealed + Copy + Default + Ord + Sum + Rem<Output = Self> + RemAssign
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
}

#[cfg(feature = "simd")]
pub trait Counter:
    sealed::Sealed + Copy + Default + Ord + Sum + Rem<Output = Self> + RemAssign + SimdElement
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
        }
    )*};
}

impl_counter!(u8, u16, u32, u64, usize);
