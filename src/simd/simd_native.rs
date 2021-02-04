use packed_simd::*;
use paste::paste;
use rand::prelude::*;
use std::cmp::Ordering;
use std::iter::FromIterator;
use typenum::U0;

use crate::multiset::{Multiset, MultisetIterator};
use crate::small_num::SmallNumConsts;

macro_rules! multiset_simd {
    ($alias:ty, $simd:ty, $scalar:ty, $simd_mask:ty) => {
        impl FromIterator<$scalar> for $alias {
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

        impl PartialOrd for $alias {
            partial_ord_body!();
        }

        impl IntoIterator for $alias {
            type Item = $scalar;
            type IntoIter = MultisetIterator<$simd, U0>;

            fn into_iter(self) -> Self::IntoIter {
                MultisetIterator {
                    multiset: self,
                    index: 0
                }
            }
        }

        impl Iterator for MultisetIterator<$simd, U0> {
            type Item = $scalar;

            fn next(&mut self) -> Option<Self::Item> {
                if self.index >= <$alias>::len() {
                    None
                } else {
                    let result = unsafe { self.multiset.data.extract_unchecked(self.index) };
                    self.index += 1;
                    Some(result)
                }
            }
        }

        impl ExactSizeIterator for MultisetIterator<$simd, U0> {
            fn len(&self) -> usize {
                <$alias>::len()
            }
        }

        impl $alias {
            /// Returns a Multiset of the given SIMD vector size with all elements set to zero.
            #[inline]
            pub const fn empty() -> Self {
                Multiset {
                    data: <$simd>::ZERO,
                }
            }

            /// Returns a Multiset of the given SIMD vector size with all elements set to `elem`.
            #[inline]
            pub const fn repeat(elem: $scalar) -> Self {
                Multiset {
                    data: <$simd>::splat(elem),
                }
            }

            /// Returns a Multiset from a slice of the given SIMD vector size.
            #[inline]
            pub fn from_slice(slice: &[$scalar]) -> Self {
                assert_eq!(slice.len(), Self::len());
                Multiset {
                    data: <$simd>::from_slice_unaligned(slice),
                }
            }

            /// The number of elements in the multiset.
            #[inline]
            pub const fn len() -> usize {
                <$simd>::lanes()
            }

            /// Sets all element counts in the multiset to zero.
            #[inline]
            pub fn clear(&mut self) {
                self.data *= <$scalar>::ZERO
            }

            /// Checks that a given element has at least one member in the multiset.
            #[inline]
            pub fn contains(self, elem: usize) -> bool {
                elem < Self::len() && unsafe { self.data.extract_unchecked(elem) > <$scalar>::ZERO }
            }

            /// Checks that a given element has at least one member in the multiset without bounds
            /// checks.
            ///
            /// # Safety
            /// Does not do bounds check on whether this element is an index in the underlying
            /// SIMD vector.
            #[inline]
            pub unsafe fn contains_unchecked(self, elem: usize) -> bool {
                self.data.extract_unchecked(elem) > <$scalar>::ZERO
            }

            /// Insert a given amount of an element into the multiset.
            #[inline]
            pub fn insert(&mut self, elem: usize, amount: $scalar) {
                if elem < Self::len() {
                    self.data = unsafe { self.data.replace_unchecked(elem, amount) };
                }
            }

            /// Insert a given amount of an element into the multiset without bounds checks.
            ///
            /// # Safety
            /// Does not do bounds check on whether this element is an index in the underlying
            /// SIMD vector.
            #[inline]
            pub unsafe fn insert_unchecked(&mut self, elem: usize, amount: $scalar) {
                self.data = self.data.replace_unchecked(elem, amount);
            }

            /// Set an element in the multiset to zero.
            #[inline]
            pub fn remove(&mut self, elem: usize) {
                if elem < Self::len() {
                    self.data = unsafe { self.data.replace_unchecked(elem, <$scalar>::ZERO) };
                }
            }

            /// Set an element in the multiset to zero without bounds checks.
            ///
            /// # Safety
            /// Does not do bounds check on whether this element is an index in the underlying
            /// SIMD vector.
            #[inline]
            pub unsafe fn remove_unchecked(&mut self, elem: usize) {
                self.data = self.data.replace_unchecked(elem, <$scalar>::ZERO);
            }

            /// Returns the amount of an element in the multiset.
            #[inline]
            pub fn get(self, elem: usize) -> Option<$scalar> {
                if elem < Self::len() {
                    unsafe { Some(self.data.extract_unchecked(elem)) }
                } else {
                    None
                }
            }

            /// Returns the amount of an element in the multiset without bounds checks.
            ///
            /// # Safety
            /// Does not do bounds check on whether this element is an index in the underlying
            /// SIMD vector.
            #[inline]
            pub unsafe fn get_unchecked(self, elem: usize) -> $scalar {
                self.data.extract_unchecked(elem)
            }

            /// Returns a multiset which is the intersection of `self` and `other`.
            ///
            /// The Intersection of two multisets A & B is defined as the multiset C where
            /// `C[0] == min(A[0], B[0]`).
            #[inline]
            pub fn intersection(&self, other: &Self) -> Self {
                Multiset {
                    data: self.data.min(other.data),
                }
            }

            /// Returns a multiset which is the union of `self` and `other`.
            ///
            /// The union of two multisets A & B is defined as the multiset C where
            /// `C[0] == max(A[0], B[0]`).
            #[inline]
            pub fn union(&self, other: &Self) -> Self {
                Multiset {
                    data: self.data.max(other.data),
                }
            }

            /// Return the number of elements whose member count is zero.
            #[inline]
            pub fn count_zero(&self) -> u32 {
                self.data.eq(<$simd>::ZERO).bitmask().count_ones()
            }

            /// Return the number of elements whose member count is non-zero.
            #[inline]
            pub fn count_non_zero(&self) -> u32 {
                self.data.gt(<$simd>::ZERO).bitmask().count_ones()
            }

            /// Check whether all elements have zero members.
            #[inline]
            pub fn is_empty(&self) -> bool {
                self.data == <$simd>::ZERO
            }

            /// Check whether only one element has one or more members.
            #[inline]
            pub fn is_singleton(&self) -> bool {
                self.count_non_zero() == 1
            }

            /// Check whether `self` is a subset of `other`.
            ///
            /// Multisets A is a subset of B if `A[i] <= B[i]` for all `i` in A.
            #[inline]
            pub fn is_subset(&self, other: &Self) -> bool {
                self.data.le(other.data).all()
            }

            /// Check whether `self` is a superset of `other`.
            ///
            /// Multisets A is a superset of B if `A[i] >= B[i]` for all `i` in A.
            #[inline]
            pub fn is_superset(&self, other: &Self) -> bool {
                self.data.ge(other.data).all()
            }

            /// Check whether `self` is a proper subset of `other`.
            ///
            /// Multisets A is a proper subset of B if `A[i] <= B[i]` for all `i` in A and there
            /// exists `j` such that `A[j] < B[j]`.
            #[inline]
            pub fn is_proper_subset(&self, other: &Self) -> bool {
                self.is_subset(other) && self.is_any_lesser(other)
            }

            /// Check whether `self` is a proper superset of `other`.
            ///
            /// Multisets A is a proper superset of B if `A[i] >= B[i]` for all `i` in A and
            /// there exists `j` such that `A[j] > B[j]`.
            #[inline]
            pub fn is_proper_superset(&self, other: &Self) -> bool {
                self.is_superset(other) && self.is_any_greater(other)
            }

            /// Check whether any element of `self` is less than an element of `other`.
            ///
            /// True if the exists some `i` such that `A[i] < B[i]`.
            #[inline]
            pub fn is_any_lesser(&self, other: &Self) -> bool {
                self.data.lt(other.data).any()
            }

            /// Check whether any element of `self` is greater than an element of `other`.
            ///
            /// True if the exists some `i` such that `A[i] > B[i]`.
            #[inline]
            pub fn is_any_greater(&self, other: &Self) -> bool {
                self.data.gt(other.data).any()
            }

            /// The total or cardinality of a multiset is the sum of all its elements member
            /// counts.
            ///
            /// Notes:
            /// - This may overflow.
            /// - The implementation uses a horizontal operation on SIMD vectors.
            #[inline]
            pub fn total(&self) -> $scalar {
                self.data.wrapping_sum()
            }

            /// Returns a tuple containing the (element, corresponding largest member count) in the
            /// multiset.
            ///
            /// Notes:
            /// - The implementation extracts values from the underlying SIMD vector.
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

            /// Returns the element corresponding to the largest member count in the multiset.
            ///
            /// Notes:
            /// - The implementation extracts values from the underlying SIMD vector.
            #[inline]
            pub fn imax(&self) -> usize {
                self.argmax().0
            }

            /// Returns the largest member count in the multiset.
            ///
            /// Notes:
            /// - The implementation uses a horizontal operation on the underlying SIMD vector.
            #[inline]
            pub fn max(&self) -> $scalar {
                self.data.max_element()
            }

            /// Returns a tuple containing the (element, corresponding smallest member count) in
            /// the multiset.
            ///
            /// Notes:
            /// - The implementation extracts values from the underlying SIMD vector.
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

            /// Returns the element corresponding to the smallest member count in the multiset.
            ///
            /// Notes:
            /// - The implementation extracts values from the underlying SIMD vector.
            #[inline]
            pub fn imin(&self) -> usize {
                self.argmin().0
            }

            /// Returns the smallest member count in the multiset.
            ///
            /// Notes:
            /// - The implementation uses a horizontal operation on the underlying SIMD vector.
            #[inline]
            pub fn min(&self) -> $scalar {
                self.data.min_element()
            }

            /// Set all elements member counts, except for the given `elem`, to zero.
            #[inline]
            pub fn choose(&mut self, elem: usize) {
                let mask = <$simd_mask>::splat(false).replace(elem, true);
                self.data = mask.select(self.data, <$simd>::ZERO)
            }

            /// Set all elements member counts, except for a random one, to zero.
            ///
            /// Notes:
            /// - The implementation extracts values from the underlying SIMD vector.
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
        }
    };
}

multiset_simd!(MS0u8x2, u8x2, u8, m8x2);
multiset_simd!(MS0u8x4, u8x4, u8, m8x4);
multiset_simd!(MS0u8x8, u8x8, u8, m8x8);
multiset_simd!(MS0u8x16, u8x16, u8, m8x16);
multiset_simd!(MS0u8x32, u8x32, u8, m8x32);
multiset_simd!(MS0u8x64, u8x64, u8, m8x64);
multiset_simd!(MS0u16x2, u16x2, u16, m16x2);
multiset_simd!(MS0u16x4, u16x4, u16, m16x4);
multiset_simd!(MS0u16x8, u16x8, u16, m16x8);
multiset_simd!(MS0u16x16, u16x16, u16, m16x16);
multiset_simd!(MS0u16x32, u16x32, u16, m16x32);
multiset_simd!(MS0u32x2, u32x2, u32, m32x2);
multiset_simd!(MS0u32x4, u32x4, u32, m32x4);
multiset_simd!(MS0u32x8, u32x8, u32, m32x8);
multiset_simd!(MS0u32x16, u32x16, u32, m32x16);
multiset_simd!(MS0u64x2, u64x2, u64, m64x2);
multiset_simd!(MS0u64x4, u64x4, u64, m64x4);
multiset_simd!(MS0u64x8, u64x8, u64, m64x8);

// Any alias where the simd type has an f64 equivalent lane-wise, can use this implementation.
macro_rules! multiset_simd_stats {
    ($alias:ty, $simd:ty, $scalar:ty, $simd_float:ty) => {
        impl $alias
            where
                f64: From<$scalar>,
                $simd_float: From<$simd>,
        {
            /// Calculate the collision entropy of the multiset.
            ///
            /// Notes:
            /// - The implementation uses a horizontal operation on SIMD vectors.
            #[inline]
            pub fn collision_entropy(&self) -> f64 {
                let total: f64 = From::from(self.total());
                let data = <$simd_float>::from(self.data);
                -(data / total).powf(<$simd_float>::splat(2.0)).sum().log2()
            }

            /// Calculate the shannon entropy of the multiset. Uses ln rather than log2.
            ///
            /// Notes:
            /// - The implementation uses a horizontal operation on SIMD vectors.
            #[inline]
            pub fn shannon_entropy(&self) -> f64 {
                let total: f64 = From::from(self.total());
                let prob = <$simd_float>::from(self.data) / total;
                let data = prob * prob.ln();
                -data.is_nan().select(<$simd_float>::ZERO, data).sum()
            }
        }
    }
}

multiset_simd_stats!(MS0u8x2, u8x2, u8, f64x2);
multiset_simd_stats!(MS0u8x4, u8x4, u8, f64x4);
multiset_simd_stats!(MS0u8x8, u8x8, u8, f64x8);
multiset_simd_stats!(MS0u16x2, u16x2, u16, f64x2);
multiset_simd_stats!(MS0u16x4, u16x4, u16, f64x4);
multiset_simd_stats!(MS0u16x8, u16x8, u16, f64x8);
multiset_simd_stats!(MS0u32x2, u32x2, u32, f64x2);
multiset_simd_stats!(MS0u32x4, u32x4, u32, f64x4);
multiset_simd_stats!(MS0u32x8, u32x8, u32, f64x8);

// Defines multiset aliases of the form: "MS0u32x4", with the type typenum::U0 built in. Each alias
// of this type uses the simd vector directly to store values in the multiset.
macro_rules! ms0_type {
    ($($elem_typ:ty),*) => {
        paste! { $(pub type [<MS0 $elem_typ>] = Multiset<$elem_typ, typenum::U0>; )* }
    }
}

ms0_type!(
    u8x2, u8x4, u8x8, u8x16, u8x32, u8x64, u16x2, u16x4, u16x8, u16x16, u16x32, u32x2, u32x4,
    u32x8, u32x16, u64x2, u64x4, u64x8
);

#[cfg(test)]
mod tests {
    use super::*;
    tests_x4!(ms0u32x4, u32x4, typenum::U0);
    tests_x8!(ms0u16x8, u16x8, typenum::U0);
}
