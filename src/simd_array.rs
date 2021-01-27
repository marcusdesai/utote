use generic_array::ArrayLength;
use packed_simd::*;
use rand::prelude::*;
use std::iter::FromIterator;
use typenum::{UInt, Unsigned};

use crate::multiset::Multiset;
use crate::small_num::SmallNum;

macro_rules! multiset_simd_array {
    ($simd:ty, $scalar:ty, $simd_f:ty, $simd_m:ty) => {
        impl<U, B> FromIterator<$scalar> for Multiset<$simd, UInt<U, B>>
        where
            UInt<U, B>: ArrayLength<$simd>,
        {
            #[inline]
            fn from_iter<T: IntoIterator<Item = $scalar>>(iter: T) -> Self {
                let mut res = unsafe { Multiset::new_uninitialized() };
                let mut it = iter.into_iter();

                for i in 0..UInt::<U, B>::USIZE {
                    let mut elem_vec = <$simd>::ZERO;
                    for j in 0..<$simd>::lanes() {
                        if let Some(v) = it.next() {
                            elem_vec = elem_vec.replace(j, v)
                        }
                    }
                    unsafe { *res.data.get_unchecked_mut(i) = elem_vec }
                }

                res
            }
        }

        impl<U, B> Multiset<$simd, UInt<U, B>>
        where
            UInt<U, B>: ArrayLength<$simd>,
        {
            #[inline]
            pub fn empty() -> Self {
                Self::repeat(<$scalar>::ZERO)
            }

            #[inline]
            pub fn repeat(elem: $scalar) -> Self {
                let mut res = unsafe { Multiset::new_uninitialized() };
                for i in 0..UInt::<U, B>::USIZE {
                    unsafe { *res.data.get_unchecked_mut(i) = <$simd>::splat(elem) }
                }
                res
            }

            #[inline]
            pub fn from_slice(slice: &[$scalar]) -> Self {
                assert_eq!(slice.len(), Self::len());
                Self::from_iter(slice.iter().cloned())
            }

            #[inline]
            pub const fn len() -> usize {
                <$simd>::lanes() * UInt::<U, B>::USIZE
            }

            #[inline]
            pub fn clear(&mut self) {
                self.data.iter_mut().for_each(|e| *e *= <$scalar>::ZERO);
            }

            #[inline]
            pub fn contains(self, elem: usize) -> bool {
                if elem < Self::len() {
                    let array_index = elem / <$simd>::lanes();
                    let vector_index = elem % <$simd>::lanes();
                    unsafe {
                        self.data
                            .get_unchecked(array_index)
                            .extract_unchecked(vector_index)
                            > <$scalar>::ZERO
                    }
                } else {
                    false
                }
            }

            #[inline]
            pub unsafe fn contains_unchecked(self, elem: usize) -> bool {
                let array_index = elem / <$simd>::lanes();
                let vector_index = elem % <$simd>::lanes();
                self.data
                    .get_unchecked(array_index)
                    .extract_unchecked(vector_index)
                    > <$scalar>::ZERO
            }

            #[inline]
            pub fn intersection(&self, other: &Self) -> Self {
                self.zip_map(other, |s1, s2| s1.min(s2))
            }

            #[inline]
            pub fn union(&self, other: &Self) -> Self {
                self.zip_map(other, |s1, s2| s1.max(s2))
            }

            #[inline]
            pub fn count_zero(&self) -> u32 {
                self.fold(0, |acc, vec| {
                    acc + vec.eq(<$simd>::ZERO).bitmask().count_ones()
                })
            }

            #[inline]
            pub fn count_non_zero(&self) -> u32 {
                self.fold(0, |acc, vec| {
                    acc + vec.gt(<$simd>::ZERO).bitmask().count_ones()
                })
            }

            #[inline]
            pub fn is_empty(&self) -> bool {
                self.data.iter().all(|vec| vec == &<$simd>::ZERO)
            }

            #[inline]
            pub fn is_singleton(&self) -> bool {
                self.count_non_zero() == 1
            }

            #[inline]
            pub fn is_subset(&self, other: &Self) -> bool {
                self.data
                    .iter()
                    .zip(other.data.iter())
                    .all(|(s1, s2)| s1.le(*s2).all())
            }

            #[inline]
            pub fn is_superset(&self, other: &Self) -> bool {
                self.data
                    .iter()
                    .zip(other.data.iter())
                    .all(|(s1, s2)| s1.ge(*s2).all())
            }

            #[inline]
            pub fn is_any_lesser(&self, other: &Self) -> bool {
                self.data
                    .iter()
                    .zip(other.data.iter())
                    .any(|(s1, s2)| s1.lt(*s2).any())
            }

            #[inline]
            pub fn is_any_greater(&self, other: &Self) -> bool {
                self.data
                    .iter()
                    .zip(other.data.iter())
                    .any(|(s1, s2)| s1.gt(*s2).any())
            }

            /// May overflow && partial horizontal
            #[inline]
            pub fn total(&self) -> $scalar {
                self.fold(<$simd>::ZERO, |acc, vec| acc + vec)
                    .wrapping_sum()
            }

            /// Horizontal, really bad
            #[inline]
            pub fn argmax(&self) -> (usize, $scalar) {
                let mut the_max = unsafe { self.data.get_unchecked(0).extract_unchecked(0) };
                let mut the_i = 0;

                for arr_idx in 0..UInt::<U, B>::USIZE {
                    for i in 0..<$simd>::lanes() {
                        let val = unsafe { self.data.get_unchecked(arr_idx).extract_unchecked(i) };
                        if val > the_max {
                            the_max = val;
                            the_i = arr_idx * <$simd>::lanes() + i;
                        }
                    }
                }
                (the_i, the_max)
            }

            /// Horizontal, really bad
            #[inline]
            pub fn imax(&self) -> usize {
                self.argmax().0
            }

            /// Horizontal
            #[inline]
            pub fn max(&self) -> $scalar {
                self.fold(<$simd>::ZERO, |acc, vec| acc.max(vec))
                    .max_element()
            }

            /// Horizontal, really bad
            #[inline]
            pub fn argmin(&self) -> (usize, $scalar) {
                let mut the_min = unsafe { self.data.get_unchecked(0).extract_unchecked(0) };
                let mut the_i = 0;

                for arr_idx in 0..UInt::<U, B>::USIZE {
                    for i in 0..<$simd>::lanes() {
                        let val = unsafe { self.data.get_unchecked(arr_idx).extract_unchecked(i) };
                        if val < the_min {
                            the_min = val;
                            the_i = i;
                        }
                    }
                }
                (the_i, the_min)
            }

            /// Horizontal, really bad
            #[inline]
            pub fn imin(&self) -> usize {
                self.argmin().0
            }

            /// Horizontal
            #[inline]
            pub fn min(&self) -> $scalar {
                self.fold(<$simd>::MAX, |acc, vec| acc.min(vec))
                    .min_element()
            }

            #[inline]
            pub fn choose(&mut self, elem: usize) {
                let array_index = elem / <$simd>::lanes();
                let vector_index = elem % <$simd>::lanes();

                for i in 0..UInt::<U, B>::USIZE {
                    let data = unsafe { self.data.get_unchecked_mut(i) };
                    if i == array_index {
                        let mask = <$simd_m>::splat(false).replace(vector_index, true);
                        *data = mask.select(*data, <$simd>::ZERO)
                    } else {
                        *data *= <$scalar>::ZERO
                    }
                }
            }

            /// Horizontal, really bad
            #[inline]
            pub fn choose_random(&mut self, rng: &mut StdRng) {
                let choice_value = rng.gen_range(<$scalar>::ZERO, self.total() + <$scalar>::ONE);
                let mut vector_index: usize = 0;
                let mut acc: $scalar = <$scalar>::ZERO;
                let mut chosen: bool = false;
                for i in 0..UInt::<U, B>::USIZE {
                    let elem_vec = unsafe { self.data.get_unchecked_mut(i) };
                    if chosen {
                        *elem_vec *= <$scalar>::ZERO
                    } else {
                        'vec_loop: for j in 0..<$simd>::lanes() {
                            acc += unsafe { elem_vec.extract_unchecked(j) };
                            if acc > choice_value {
                                vector_index = j;
                                chosen = true;
                                break 'vec_loop;
                            }
                        }
                        if chosen {
                            let mask = <$simd_m>::splat(false).replace(vector_index, true);
                            *elem_vec = mask.select(*elem_vec, <$simd>::ZERO)
                        } else {
                            *elem_vec *= <$scalar>::ZERO
                        }
                    }
                }
            }

            /// partial horizontal
            #[inline]
            pub fn collision_entropy(&self) -> f64 {
                let total: f64 = From::from(self.total());
                -self
                    .fold(<$simd_f>::splat(0.0), |acc, vec| {
                        let data = <$simd_f>::from(vec);
                        acc + (data / total).powf(<$simd_f>::splat(2.0))
                    })
                    .sum()
                    .log2()
            }

            /// partial horizontal
            #[inline]
            pub fn shannon_entropy(&self) -> f64 {
                let total: f64 = From::from(self.total());
                -self
                    .fold(<$simd_f>::splat(0.0), |acc, vec| {
                        let prob = <$simd_f>::from(vec) / total;
                        let data = prob * prob.ln();
                        acc + data.is_nan().select(<$simd_f>::ZERO, data)
                    })
                    .sum()
            }
        }
    };
}

multiset_simd_array!(u32x4, u32, f64x4, m32x4);
multiset_simd_array!(u32x2, u32, f64x2, m32x2);
multiset_simd_array!(u8x8, u8, f64x8, m8x8);

#[cfg(test)]
mod tests {
    use super::*;
    tests_x4!(ms2u32x2, u32x2, typenum::U2);
    tests_x8!(ms1u8x8, u8x8, typenum::U1);
    tests_x8!(ms2u32x4, u32x4, typenum::U2);
}
