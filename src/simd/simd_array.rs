use generic_array::ArrayLength;
use packed_simd::*;
use paste::paste;
use rand::prelude::*;
use std::cmp::Ordering;
use std::iter::FromIterator;
use typenum::{UInt, Unsigned};

use crate::multiset::{Multiset, MultisetIterator};
use crate::small_num::SmallNumConsts;

macro_rules! multiset_simd_array {
    ($alias:ty, $simd:ty, $scalar:ty, $simd_mask:ty) => {
        impl<U, B> FromIterator<$scalar> for $alias
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

        impl<U, B> PartialOrd for $alias
            where
                UInt<U, B>: ArrayLength<$simd>,
        {
            partial_ord_body!();
        }

        impl<U, B> IntoIterator for $alias
            where
                UInt<U, B>: ArrayLength<$simd>,
        {
            type Item = $scalar;
            type IntoIter = MultisetIterator<$simd, UInt::<U, B>>;

            fn into_iter(self) -> Self::IntoIter {
                MultisetIterator {
                    multiset: self,
                    index: 0
                }
            }
        }

        impl<U, B> Iterator for MultisetIterator<$simd, UInt::<U, B>>
            where
                UInt<U, B>: ArrayLength<$simd>,
        {
            type Item = $scalar;

            fn next(&mut self) -> Option<Self::Item> {
                if self.index >= <$alias>::len() {
                    None
                } else {
                    let array_index = self.index / <$simd>::lanes();
                    let vector_index = self.index % <$simd>::lanes();
                    let result = unsafe {
                        self.multiset.data
                            .get_unchecked(array_index)
                            .extract_unchecked(vector_index)
                    };
                    self.index += 1;
                    Some(result)
                }
            }
        }

        impl<U, B> ExactSizeIterator for MultisetIterator<$simd, UInt::<U, B>>
            where
                UInt<U, B>: ArrayLength<$simd>,
        {
            fn len(&self) -> usize {
                <$alias>::len()
            }
        }

        impl<U, B> $alias
            where
                UInt<U, B>: ArrayLength<$simd>,
        {
            /// Returns a Multiset of the given array * SIMD vector size with all elements set to
            /// zero.
            #[inline]
            pub fn empty() -> Self {
                Self::repeat(<$scalar>::ZERO)
            }

            /// Returns a Multiset of the given array * SIMD vector size with all elements set to
            /// `elem`.
            #[inline]
            pub fn repeat(elem: $scalar) -> Self {
                let mut res = unsafe { Multiset::new_uninitialized() };
                for i in 0..UInt::<U, B>::USIZE {
                    unsafe { *res.data.get_unchecked_mut(i) = <$simd>::splat(elem) }
                }
                res
            }

            /// Returns a Multiset from a slice of the given array * SIMD vector size.
            #[inline]
            pub fn from_slice(slice: &[$scalar]) -> Self {
                assert_eq!(slice.len(), Self::len());
                Self::from_iter(slice.iter().cloned())
            }

            /// The number of elements in the multiset.
            #[inline]
            pub const fn len() -> usize {
                <$simd>::lanes() * UInt::<U, B>::USIZE
            }

            /// Sets all element counts in the multiset to zero.
            #[inline]
            pub fn clear(&mut self) {
                self.data.iter_mut().for_each(|e| *e *= <$scalar>::ZERO);
            }

            /// Checks that a given element has at least one member in the multiset.
            #[inline]
            pub fn contains(self, elem: usize) -> bool {
                elem < Self::len() && {
                    let array_index = elem / <$simd>::lanes();
                    let vector_index = elem % <$simd>::lanes();
                    unsafe {
                        self.data
                            .get_unchecked(array_index)
                            .extract_unchecked(vector_index)
                            > <$scalar>::ZERO
                    }
                }
            }

            /// Checks that a given element has at least one member in the multiset without bounds
            /// checks.
            ///
            /// # Safety
            /// Does not do bounds check on whether this element is an index in the underlying
            /// array.
            #[inline]
            pub unsafe fn contains_unchecked(self, elem: usize) -> bool {
                let array_index = elem / <$simd>::lanes();
                let vector_index = elem % <$simd>::lanes();
                self.data
                    .get_unchecked(array_index)
                    .extract_unchecked(vector_index)
                    > <$scalar>::ZERO
            }

            /// Insert a given amount of an element into the multiset.
            #[inline]
            pub fn insert(&mut self, elem: usize, amount: $scalar) {
                if elem < Self::len() {
                    let array_index = elem / <$simd>::lanes();
                    let vector_index = elem % <$simd>::lanes();
                    unsafe {
                        let vec = self.data.get_unchecked_mut(array_index);
                        *vec = vec.replace_unchecked(vector_index, amount)
                    }
                }
            }

            /// Insert a given amount of an element into the multiset without bounds checks.
            ///
            /// # Safety
            /// Does not do bounds check on whether this element is an index in the underlying
            /// array.
            #[inline]
            pub unsafe fn insert_unchecked(&mut self, elem: usize, amount: $scalar) {
                let array_index = elem / <$simd>::lanes();
                let vector_index = elem % <$simd>::lanes();
                let vec = self.data.get_unchecked_mut(array_index);
                *vec = vec.replace_unchecked(vector_index, amount)
            }

            /// Set an element in the multiset to zero.
            #[inline]
            pub fn remove(&mut self, elem: usize) {
                if elem < Self::len() {
                    let array_index = elem / <$simd>::lanes();
                    let vector_index = elem % <$simd>::lanes();
                    unsafe {
                        let vec = self.data.get_unchecked_mut(array_index);
                        *vec = vec.replace_unchecked(vector_index, <$scalar>::ZERO)
                    }
                }
            }

            /// Set an element in the multiset to zero without bounds checks.
            ///
            /// # Safety
            /// Does not do bounds check on whether this element is an index in the underlying
            /// array.
            #[inline]
            pub unsafe fn remove_unchecked(&mut self, elem: usize) {
                let array_index = elem / <$simd>::lanes();
                let vector_index = elem % <$simd>::lanes();
                let vec = self.data.get_unchecked_mut(array_index);
                *vec = vec.replace_unchecked(vector_index, <$scalar>::ZERO)
            }

            /// Returns the amount of an element in the multiset.
            #[inline]
            pub fn get(self, elem: usize) -> Option<$scalar> {
                let array_index = elem / <$simd>::lanes();
                let vector_index = elem % <$simd>::lanes();
                unsafe { self.data.get(array_index).map(|vec| vec.extract_unchecked(vector_index)) }
            }

            /// Returns the amount of an element in the multiset without bounds checks.
            ///
            /// # Safety
            /// Does not do bounds check on whether this element is an index in the underlying
            /// array.
            #[inline]
            pub unsafe fn get_unchecked(self, elem: usize) -> $scalar {
                let array_index = elem / <$simd>::lanes();
                let vector_index = elem % <$simd>::lanes();
                self.data.get_unchecked(array_index).extract_unchecked(vector_index)
            }

            /// Returns a multiset which is the intersection of `self` and `other`.
            ///
            /// The Intersection of two multisets A & B is defined as the multiset C where
            /// `C[0] == min(A[0], B[0]`).
            #[inline]
            pub fn intersection(&self, other: &Self) -> Self {
                self.zip_map(other, |s1, s2| s1.min(s2))
            }

            /// Returns a multiset which is the union of `self` and `other`.
            ///
            /// The union of two multisets A & B is defined as the multiset C where
            /// `C[0] == max(A[0], B[0]`).
            #[inline]
            pub fn union(&self, other: &Self) -> Self {
                self.zip_map(other, |s1, s2| s1.max(s2))
            }

            /// Return the number of elements whose member count is zero.
            #[inline]
            pub fn count_zero(&self) -> u32 {
                self.fold(0, |acc, vec| {
                    acc + vec.eq(<$simd>::ZERO).bitmask().count_ones()
                })
            }

            /// Return the number of elements whose member count is non-zero.
            #[inline]
            pub fn count_non_zero(&self) -> u32 {
                self.fold(0, |acc, vec| {
                    acc + vec.gt(<$simd>::ZERO).bitmask().count_ones()
                })
            }

            /// Check whether all elements have zero members.
            #[inline]
            pub fn is_empty(&self) -> bool {
                self.data.iter().all(|vec| vec == &<$simd>::ZERO)
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
                self.data
                    .iter()
                    .zip(other.data.iter())
                    .all(|(s1, s2)| s1.le(*s2).all())
            }

            /// Check whether `self` is a superset of `other`.
            ///
            /// Multisets A is a superset of B if `A[i] >= B[i]` for all `i` in A.
            #[inline]
            pub fn is_superset(&self, other: &Self) -> bool {
                self.data
                    .iter()
                    .zip(other.data.iter())
                    .all(|(s1, s2)| s1.ge(*s2).all())
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
                self.data
                    .iter()
                    .zip(other.data.iter())
                    .any(|(s1, s2)| s1.lt(*s2).any())
            }

            /// Check whether any element of `self` is greater than an element of `other`.
            ///
            /// True if the exists some `i` such that `A[i] > B[i]`.
            #[inline]
            pub fn is_any_greater(&self, other: &Self) -> bool {
                self.data
                    .iter()
                    .zip(other.data.iter())
                    .any(|(s1, s2)| s1.gt(*s2).any())
            }

            /// The total or cardinality of a multiset is the sum of all its elements member
            /// counts.
            ///
            /// Notes:
            /// - This may overflow.
            /// - The implementation uses a horizontal operation on SIMD vectors.
            #[inline]
            pub fn total(&self) -> $scalar {
                self.fold(<$simd>::ZERO, |acc, vec| acc + vec)
                    .wrapping_sum()
            }

            /// Returns a tuple containing the (element, corresponding largest member count) in the
            /// multiset.
            ///
            /// Notes:
            /// - The implementation extracts values from the underlying SIMD vectors.
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

            /// Returns the element corresponding to the largest member count in the multiset.
            ///
            /// Notes:
            /// - The implementation extracts values from the underlying SIMD vectors.
            #[inline]
            pub fn imax(&self) -> usize {
                self.argmax().0
            }

            /// Returns the largest member count in the multiset.
            ///
            /// Notes:
            /// - The implementation uses a horizontal operation on the underlying SIMD vectors.
            #[inline]
            pub fn max(&self) -> $scalar {
                self.fold(<$simd>::ZERO, |acc, vec| acc.max(vec))
                    .max_element()
            }

            /// Returns a tuple containing the (element, corresponding smallest member count) in
            /// the multiset.
            ///
            /// Notes:
            /// - The implementation extracts values from the underlying SIMD vectors.
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

            /// Returns the element corresponding to the smallest member count in the multiset.
            ///
            /// Notes:
            /// - The implementation extracts values from the underlying SIMD vectors.
            #[inline]
            pub fn imin(&self) -> usize {
                self.argmin().0
            }

            /// Returns the smallest member count in the multiset.
            ///
            /// Notes:
            /// - The implementation uses a horizontal operation on the underlying SIMD vectors.
            #[inline]
            pub fn min(&self) -> $scalar {
                self.fold(<$simd>::MAX, |acc, vec| acc.min(vec))
                    .min_element()
            }

            /// Set all elements member counts, except for the given `elem`, to zero.
            #[inline]
            pub fn choose(&mut self, elem: usize) {
                let array_index = elem / <$simd>::lanes();
                let vector_index = elem % <$simd>::lanes();

                for i in 0..UInt::<U, B>::USIZE {
                    let data = unsafe { self.data.get_unchecked_mut(i) };
                    if i == array_index {
                        let mask = <$simd_mask>::splat(false).replace(vector_index, true);
                        *data = mask.select(*data, <$simd>::ZERO)
                    } else {
                        *data *= <$scalar>::ZERO
                    }
                }
            }

            /// Set all elements member counts, except for a random one, to zero.
            ///
            /// Notes:
            /// - The implementation extracts values from the underlying SIMD vectors.
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
                            if acc >= choice_value {
                                vector_index = j;
                                chosen = true;
                                break 'vec_loop;
                            }
                        }
                        if chosen {
                            let mask = <$simd_mask>::splat(false).replace(vector_index, true);
                            *elem_vec = mask.select(*elem_vec, <$simd>::ZERO)
                        } else {
                            *elem_vec *= <$scalar>::ZERO
                        }
                    }
                }
            }
        }
    };
}

multiset_simd_array!(MSu8x2<UInt<U, B>>, u8x2, u8, m8x2);
multiset_simd_array!(MSu8x4<UInt<U, B>>, u8x4, u8, m8x4);
multiset_simd_array!(MSu8x8<UInt<U, B>>, u8x8, u8, m8x8);
multiset_simd_array!(MSu8x16<UInt<U, B>>, u8x16, u8, m8x16);
multiset_simd_array!(MSu8x32<UInt<U, B>>, u8x32, u8, m8x32);
multiset_simd_array!(MSu8x64<UInt<U, B>>, u8x64, u8, m8x64);
multiset_simd_array!(MSu16x2<UInt<U, B>>, u16x2, u16, m16x2);
multiset_simd_array!(MSu16x4<UInt<U, B>>, u16x4, u16, m16x4);
multiset_simd_array!(MSu16x8<UInt<U, B>>, u16x8, u16, m16x8);
multiset_simd_array!(MSu16x16<UInt<U, B>>, u16x16, u16, m16x16);
multiset_simd_array!(MSu16x32<UInt<U, B>>, u16x32, u16, m16x32);
multiset_simd_array!(MSu32x2<UInt<U, B>>, u32x2, u32, m32x2);
multiset_simd_array!(MSu32x4<UInt<U, B>>, u32x4, u32, m32x4);
multiset_simd_array!(MSu32x8<UInt<U, B>>, u32x8, u32, m32x8);
multiset_simd_array!(MSu32x16<UInt<U, B>>, u32x16, u32, m32x16);
multiset_simd_array!(MSu64x2<UInt<U, B>>, u64x2, u64, m8x2);
multiset_simd_array!(MSu64x4<UInt<U, B>>, u64x4, u64, m8x4);
multiset_simd_array!(MSu64x8<UInt<U, B>>, u64x8, u64, m8x8);

// Any alias where the simd type has an f64 equivalent lane-wise, can use this implementation.
macro_rules! multiset_simd_array_stats {
    ($alias:ty, $simd:ty, $scalar:ty, $simd_float:ty) => {
        impl<U, B> $alias
            where
                UInt<U, B>: ArrayLength<$simd>,
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
                -self
                    .fold(<$simd_float>::ZERO, |acc, vec| {
                        let data = <$simd_float>::from(vec);
                        acc + (data / total).powf(<$simd_float>::splat(2.0))
                    })
                    .sum()
                    .log2()
            }

            /// Calculate the shannon entropy of the multiset. Uses ln rather than log2.
            ///
            /// Notes:
            /// - The implementation uses a horizontal operation on SIMD vectors.
            #[inline]
            pub fn shannon_entropy(&self) -> f64 {
                let total: f64 = From::from(self.total());
                -self
                    .fold(<$simd_float>::ZERO, |acc, vec| {
                        let prob = <$simd_float>::from(vec) / total;
                        let data = prob * prob.ln();
                        acc + data.is_nan().select(<$simd_float>::ZERO, data)
                    })
                    .sum()
            }
        }
    }
}

multiset_simd_array_stats!(MSu8x2<UInt<U, B>>, u8x2, u8, f64x2);
multiset_simd_array_stats!(MSu8x4<UInt<U, B>>, u8x4, u8, f64x4);
multiset_simd_array_stats!(MSu8x8<UInt<U, B>>, u8x8, u8, f64x8);
multiset_simd_array_stats!(MSu16x2<UInt<U, B>>, u16x2, u16, f64x2);
multiset_simd_array_stats!(MSu16x4<UInt<U, B>>, u16x4, u16, f64x4);
multiset_simd_array_stats!(MSu16x8<UInt<U, B>>, u16x8, u16, f64x8);
multiset_simd_array_stats!(MSu32x2<UInt<U, B>>, u32x2, u32, f64x2);
multiset_simd_array_stats!(MSu32x4<UInt<U, B>>, u32x4, u32, f64x4);
multiset_simd_array_stats!(MSu32x8<UInt<U, B>>, u32x8, u32, f64x8);

multiset_type!(
    u8x2, u8x4, u8x8, u8x16, u8x32, u8x64, u16x2, u16x4, u16x8, u16x16, u16x32, u32x2, u32x4,
    u32x8, u32x16, u64x2, u64x4, u64x8
);

#[cfg(test)]
mod tests {
    use super::*;
    tests_x4!(ms2u32x2, u32x2, typenum::U2);
    tests_x8!(ms1u8x8, u8x8, typenum::U1);
    tests_x8!(ms2u32x4, u32x4, typenum::U2);
}
