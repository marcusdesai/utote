use packed_simd::*;
use rand::prelude::*;
use std::iter::FromIterator;
use typenum::U0;

use crate::multiset::Multiset;
use crate::small_num::SmallNumConsts;

macro_rules! multiset_simd {
    ($simd:ty, $scalar:ty, $simd_f:ty, $simd_m:ty) => {
        impl FromIterator<$scalar> for Multiset<$simd, U0> {
            #[inline]
            fn from_iter<T: IntoIterator<Item = $scalar>>(iter: T) -> Self {
                let mut data = <$simd>::ZERO;
                let mut it = iter.into_iter();

                for i in 0..<$simd>::lanes() {
                    if let Some(elem) = it.next() {
                        data = unsafe { data.replace_unchecked(i, elem) }
                    }
                }
                Multiset { data }
            }
        }

        impl Multiset<$simd, U0> {
            #[inline]
            pub const fn empty() -> Self {
                Multiset {
                    data: <$simd>::ZERO,
                }
            }

            #[inline]
            pub const fn repeat(elem: $scalar) -> Self {
                Multiset {
                    data: <$simd>::splat(elem),
                }
            }

            #[inline]
            pub fn from_slice(slice: &[$scalar]) -> Self {
                assert_eq!(slice.len(), Self::len());
                Multiset {
                    data: <$simd>::from_slice_unaligned(slice),
                }
            }

            #[inline]
            pub const fn len() -> usize {
                <$simd>::lanes()
            }

            #[inline]
            pub fn clear(&mut self) {
                self.data *= <$scalar>::ZERO
            }

            #[inline]
            pub fn contains(self, elem: usize) -> bool {
                elem < Self::len() && unsafe { self.data.extract_unchecked(elem) > <$scalar>::ZERO }
            }

            #[inline]
            pub unsafe fn contains_unchecked(self, elem: usize) -> bool {
                self.data.extract_unchecked(elem) > <$scalar>::ZERO
            }

            #[inline]
            pub fn intersection(&self, other: &Self) -> Self {
                Multiset {
                    data: self.data.min(other.data),
                }
            }

            #[inline]
            pub fn union(&self, other: &Self) -> Self {
                Multiset {
                    data: self.data.max(other.data),
                }
            }

            #[inline]
            pub fn count_zero(&self) -> u32 {
                self.data.eq(<$simd>::ZERO).bitmask().count_ones()
            }

            #[inline]
            pub fn count_non_zero(&self) -> u32 {
                self.data.gt(<$simd>::ZERO).bitmask().count_ones()
            }

            #[inline]
            pub fn is_empty(&self) -> bool {
                self.data == <$simd>::ZERO
            }

            #[inline]
            pub fn is_singleton(&self) -> bool {
                self.count_non_zero() == 1
            }

            #[inline]
            pub fn is_subset(&self, other: &Self) -> bool {
                self.data.le(other.data).all()
            }

            #[inline]
            pub fn is_superset(&self, other: &Self) -> bool {
                self.data.ge(other.data).all()
            }

            #[inline]
            pub fn is_any_lesser(&self, other: &Self) -> bool {
                self.data.lt(other.data).any()
            }

            #[inline]
            pub fn is_any_greater(&self, other: &Self) -> bool {
                self.data.gt(other.data).any()
            }

            /// May overflow & warning: horizontal
            #[inline]
            pub fn total(&self) -> $scalar {
                self.data.wrapping_sum()
            }

            /// Horizontal
            #[inline]
            pub fn argmax(&self) -> (usize, $scalar) {
                let mut the_max = unsafe { self.data.extract_unchecked(0) };
                let mut the_i = 0;

                for i in 1..Self::len() {
                    let val = unsafe { self.data.extract_unchecked(i) };
                    if val > the_max {
                        the_max = val;
                        the_i = i;
                    }
                }
                (the_i, the_max)
            }

            /// Horizontal
            #[inline]
            pub fn imax(&self) -> usize {
                self.argmax().0
            }

            /// Horizontal
            #[inline]
            pub fn max(&self) -> $scalar {
                self.data.max_element()
            }

            /// Horizontal
            #[inline]
            pub fn argmin(&self) -> (usize, $scalar) {
                let mut the_min = unsafe { self.data.extract_unchecked(0) };
                let mut the_i = 0;

                for i in 1..Self::len() {
                    let val = unsafe { self.data.extract_unchecked(i) };
                    if val < the_min {
                        the_min = val;
                        the_i = i;
                    }
                }
                (the_i, the_min)
            }

            /// Horizontal
            #[inline]
            pub fn imin(&self) -> usize {
                self.argmin().0
            }

            /// Horizontal
            #[inline]
            pub fn min(&self) -> $scalar {
                self.data.min_element()
            }

            #[inline]
            pub fn choose(&mut self, elem: usize) {
                let mask = <$simd_m>::splat(false).replace(elem, true);
                self.data = mask.select(self.data, <$simd>::ZERO)
            }

            /// Horizontal, really bad
            #[inline]
            pub fn choose_random(&mut self, rng: &mut StdRng) {
                let choice_value = rng.gen_range(<$scalar>::ZERO, self.total() + <$scalar>::ONE);
                let mut elem: usize = 0;
                let mut acc: $scalar = <$scalar>::ZERO;
                for i in 0..Self::len() {
                    acc += unsafe { self.data.extract_unchecked(i) };
                    if acc > choice_value {
                        elem = i;
                        break;
                    }
                }
                self.choose(elem)
            }

            /// partial horizontal
            #[inline]
            pub fn collision_entropy(&self) -> f64 {
                let total: f64 = From::from(self.total());
                let data = <$simd_f>::from(self.data);
                -(data / total).powf(<$simd_f>::splat(2.0)).sum().log2()
            }

            /// partial horizontal
            #[inline]
            pub fn shannon_entropy(&self) -> f64 {
                let total: f64 = From::from(self.total());
                let prob = <$simd_f>::from(self.data) / total;
                let data = prob * prob.ln();
                -data.is_nan().select(<$simd_f>::ZERO, data).sum()
            }
        }
    };
}

multiset_simd!(u8x2, u8, f64x2, m8x2);
multiset_simd!(u8x4, u8, f64x4, m8x4);
multiset_simd!(u8x8, u8, f64x8, m8x8);
multiset_simd!(u16x2, u16, f64x2, m16x2);
multiset_simd!(u16x4, u16, f64x4, m16x4);
multiset_simd!(u16x8, u16, f64x8, m16x8);
multiset_simd!(u32x2, u32, f64x2, m32x2);
multiset_simd!(u32x4, u32, f64x4, m32x4);
multiset_simd!(u32x8, u32, f64x8, m32x8);

#[cfg(test)]
mod tests {
    use super::*;
    tests_x4!(ms0u32x4, u32x4, typenum::U0);
    tests_x8!(ms0u16x8, u16x8, typenum::U0);
}
