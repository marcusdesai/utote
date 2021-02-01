use generic_array::ArrayLength;
use paste::paste;
use rand::prelude::*;
use std::cmp::Ordering;
use std::iter::FromIterator;
use typenum::{UInt, Unsigned};

use crate::multiset::Multiset;
use crate::small_num::SmallNumConsts;

macro_rules! multiset_scalar_array {
    ($($scalar:ty),*) => {
        $(impl<U, B> FromIterator<$scalar> for Multiset<$scalar, UInt<U, B>>
            where
                UInt<U, B>: ArrayLength<$scalar>,
        {
            #[inline]
            fn from_iter<T: IntoIterator<Item=$scalar>>(iter: T) -> Self {
                let mut res = unsafe { Multiset::new_uninitialized() };
                let mut it = iter.into_iter();

                for i in 0..Self::len() {
                    let elem = match it.next() {
                        Some(v) => v,
                        None => <$scalar>::ZERO,
                    };
                    unsafe { *res.data.get_unchecked_mut(i) = elem }
                }
                res
            }
        }

        impl<U, B> PartialOrd for Multiset<$scalar, UInt<U, B>>
            where
                UInt<U, B>: ArrayLength<$scalar>,
        {
            partial_ord_body!();
        }

        impl<U, B> Multiset<$scalar, UInt<U, B>>
            where
                UInt<U, B>: ArrayLength<$scalar>,
        {

            /// Returns a Multiset of the given array size with all elements set to zero.
            #[inline]
            pub fn empty() -> Self {
                Self::repeat(<$scalar>::ZERO)
            }

            /// Returns a Multiset of the given array size with all elements set to `elem`.
            #[inline]
            pub fn repeat(elem: $scalar) -> Self {
                let mut res = unsafe { Multiset::new_uninitialized() };
                for i in 0..Self::len() {
                    unsafe { *res.data.get_unchecked_mut(i) = elem }
                }
                res
            }

            /// Returns a Multiset from a slice of the given array size.
            #[inline]
            pub fn from_slice(slice: &[$scalar]) -> Self
            {
                assert_eq!(slice.len(), Self::len());
                let mut res = unsafe { Multiset::new_uninitialized() };
                let mut iter = slice.iter();

                for i in 0..Self::len() {
                    unsafe {
                        *res.data.get_unchecked_mut(i) = *(iter.next().unwrap())
                    }
                }
                res
            }

            /// The number of elements in the multiset.
            #[inline]
            pub const fn len() -> usize { UInt::<U, B>::USIZE }

            /// Sets all element counts in the multiset to zero.
            #[inline]
            pub fn clear(&mut self) {
                self.data.iter_mut().for_each(|e| *e *= <$scalar>::ZERO);
            }

            /// Checks that a given element has at least one member in the multiset.
            #[inline]
            pub fn contains(self, elem: usize) -> bool {
                elem < Self::len() && unsafe { self.data.get_unchecked(elem) > &<$scalar>::ZERO }
            }

            /// Checks that a given element has at least one member in the multiset without bounds
            /// checks.
            ///
            /// # Safety
            /// Does not do bounds check on whether this element is an index in the underlying
            /// array.
            #[inline]
            pub unsafe fn contains_unchecked(self, elem: usize) -> bool {
                self.data.get_unchecked(elem) > &<$scalar>::ZERO
            }

            /// Returns a multiset which is the intersection of `self` and `other`.
            ///
            /// The Intersection of two multisets A & B is defined as the multiset C where
            /// C`[0]` == min(A`[0]`, B`[0]`).
            #[inline]
            pub fn intersection(&self, other: &Self) -> Self {
                self.zip_map(other, |s1, s2| s1.min(s2))
            }

            /// Returns a multiset which is the union of `self` and `other`.
            ///
            /// The union of two multisets A & B is defined as the multiset C where
            /// C`[0]` == max(A`[0]`, B`[0]`).
            #[inline]
            pub fn union(&self, other: &Self) -> Self {
                self.zip_map(other, |s1, s2| s1.max(s2))
            }

            /// Return the number of elements whose member count is zero.
            #[inline]
            pub fn count_zero(&self) -> $scalar {
                self.fold(Self::len() as $scalar, |acc, elem| acc - elem.min(1))
            }

            /// Return the number of elements whose member count is non-zero.
            #[inline]
            pub fn count_non_zero(&self) -> $scalar {
                self.fold(0, |acc, elem| acc + elem.min(1))
            }

            /// Check whether all elements have zero members.
            #[inline]
            pub fn is_empty(&self) -> bool {
                self.data.iter().all(|elem| elem == &<$scalar>::ZERO)
            }

            /// Check whether only one element has one or more members.
            #[inline]
            pub fn is_singleton(&self) -> bool {
                self.count_non_zero() == 1
            }

            /// Check whether `self` is a subset of `other`.
            ///
            /// Multisets A is a subset of B if A`[i]` <= B`[i]` for all `i` in A.
            #[inline]
            pub fn is_subset(&self, other: &Self) -> bool {
                self.data.iter().zip(other.data.iter()).all(|(a, b)| a <= b)
            }

            /// Check whether `self` is a superset of `other`.
            ///
            /// Multisets A is a superset of B if A`[i]` >= B`[i]` for all `i` in A.
            #[inline]
            pub fn is_superset(&self, other: &Self) -> bool {
                self.data.iter().zip(other.data.iter()).all(|(a, b)| a >= b)
            }

            /// Check whether `self` is a proper subset of `other`.
            ///
            /// Multisets A is a proper subset of B if A`[i]` <= B`[i]` for all `i` in A and there
            /// exists `j` such that A`[j]` < B`[j]`.
            #[inline]
            pub fn is_proper_subset(&self, other: &Self) -> bool {
                self.is_subset(other) && self.is_any_lesser(other)
            }

            /// Check whether `self` is a proper superset of `other`.
            ///
            /// Multisets A is a proper superset of B if A`[i]` >= B`[i]` for all `i` in A and
            /// there exists `j` such that A`[j]` > B`[j]`.
            #[inline]
            pub fn is_proper_superset(&self, other: &Self) -> bool {
                self.is_superset(other) && self.is_any_greater(other)
            }

            /// Check whether any element of `self` is less than an element of `other`.
            ///
            /// True if the exists some `i` such that A`[i]` < B`[i]`.
            #[inline]
            pub fn is_any_lesser(&self, other: &Self) -> bool {
                self.data.iter().zip(other.data.iter()).any(|(a, b)| a < b)
            }

            /// Check whether any element of `self` is greater than an element of `other`.
            ///
            /// True if the exists some `i` such that A`[i]` > B`[i]`.
            #[inline]
            pub fn is_any_greater(&self, other: &Self) -> bool {
                self.data.iter().zip(other.data.iter()).any(|(a, b)| a > b)
            }

            /// The total or cardinality of a multiset is the sum of all its elements member counts.
            ///
            /// Notes:
            /// - This may overflow.
            #[inline]
            pub fn total(&self) -> $scalar {
                self.data.iter().sum()
            }

            /// Returns a tuple containing the (element, corresponding largest member count) in the
            /// multiset.
            #[inline]
            pub fn argmax(&self) -> (usize, $scalar) {
                let mut the_max = unsafe { self.data.get_unchecked(0) };
                let mut the_i = 0;

                for i in 1..Self::len() {
                    let val = unsafe { self.data.get_unchecked(i) };
                    if val > the_max {
                        the_max = val;
                        the_i = i;
                    }
                }
                (the_i, *the_max)
            }

            /// Returns the element corresponding to the largest member count in the multiset.
            #[inline]
            pub fn imax(&self) -> usize {
                self.argmax().0
            }

            /// Returns the largest member count in the multiset.
            #[inline]
            pub fn max(&self) -> $scalar {
                let mut the_max = unsafe { self.data.get_unchecked(0) };

                for i in 1..Self::len() {
                    let val = unsafe { self.data.get_unchecked(i) };
                    if val > the_max {
                        the_max = val;
                    }
                }
                *the_max
            }

            /// Returns a tuple containing the (element, corresponding smallest member count) in the
            /// multiset.
            #[inline]
            pub fn argmin(&self) -> (usize, $scalar) {
                let mut the_min = unsafe { self.data.get_unchecked(0) };
                let mut the_i = 0;

                for i in 1..Self::len() {
                    let val = unsafe { self.data.get_unchecked(i) };
                    if val < the_min {
                        the_min = val;
                        the_i = i;
                    }
                }
                (the_i, *the_min)
            }

            /// Returns the element corresponding to the smallest member count in the multiset.
            #[inline]
            pub fn imin(&self) -> usize {
                self.argmin().0
            }

            /// Returns the smallest member count in the multiset.
            #[inline]
            pub fn min(&self) -> $scalar {
                let mut the_min = unsafe { self.data.get_unchecked(0) };

                for i in 1..Self::len() {
                    let val = unsafe { self.data.get_unchecked(i) };
                    if val < the_min {
                        the_min = val;
                    }
                }
                *the_min
            }

            /// Set all elements member counts, except for the given `elem`, to zero.
            #[inline]
            pub fn choose(&mut self, elem: usize) {
                for i in 0..Self::len() {
                    if i != elem {
                        unsafe { *self.data.get_unchecked_mut(i) = <$scalar>::ZERO }
                    }
                }
            }

            /// Set all elements member counts, except for a random one, to zero.
            #[inline]
            pub fn choose_random(&mut self, rng: &mut StdRng) {
                let choice_value = rng.gen_range(<$scalar>::ZERO, self.total() + <$scalar>::ONE);
                let mut acc = <$scalar>::ZERO;
                let mut chosen = false;
                self.data.iter_mut().for_each(|elem| {
                    if chosen {
                        *elem = <$scalar>::ZERO
                    } else {
                        acc += *elem;
                        if acc < choice_value {
                            *elem = <$scalar>::ZERO
                        } else {
                            chosen = true;
                        }
                    }
                })
            }

            /// Calculate the collision entropy of the multiset.
            #[inline]
            pub fn collision_entropy(&self) -> f64 {
                let total: f64 = From::from(self.total());
                -self.fold(0.0, |acc, frequency| {
                    acc + (f64::from(frequency) / total).powf(2.0)
                }).log2()
            }

            /// Calculate the shannon entropy of the multiset. Uses ln rather than log2.
            #[inline]
            pub fn shannon_entropy(&self) -> f64 {
                let total: f64 = From::from(self.total());
                -self.fold(0.0, |acc, frequency| {
                    if frequency > <$scalar>::ZERO {
                        let prob = f64::from(frequency) / total;
                        acc + prob * prob.ln()
                    } else {
                        acc
                    }
                })
            }
        })*
    }
}

multiset_scalar_array!(u8, u16, u32);

multiset_type!(u8, u16, u32);

#[cfg(test)]
mod tests {
    use super::*;
    tests_x4!(ms4u32, u32, typenum::U4);
    tests_x8!(ms8u16, u16, typenum::U8);
    tests_x4!(ms4u8, u8, typenum::U4);
}
