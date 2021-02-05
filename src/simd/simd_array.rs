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
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu32x2;
            /// use typenum::U2;
            /// let multiset = MSu32x2::<U2>::empty();
            /// ```
            #[inline]
            pub fn empty() -> Self {
                Self::repeat(<$scalar>::ZERO)
            }

            /// Returns a Multiset of the given array * SIMD vector size with all elements set to
            /// `elem`.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu32x2;
            /// use typenum::U2;
            /// let multiset = MSu32x2::<U2>::repeat(5);
            /// ```
            #[inline]
            pub fn repeat(elem: $scalar) -> Self {
                let mut res = unsafe { Multiset::new_uninitialized() };
                for i in 0..UInt::<U, B>::USIZE {
                    unsafe { *res.data.get_unchecked_mut(i) = <$simd>::splat(elem) }
                }
                res
            }

            /// Returns a Multiset from a slice of the given array * SIMD vector size.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu32x2;
            /// use typenum::U2;
            /// let multiset = MSu32x2::<U2>::from_slice(&[1, 2, 3, 4]);
            /// ```
            #[inline]
            pub fn from_slice(slice: &[$scalar]) -> Self {
                assert_eq!(slice.len(), Self::len());
                Self::from_iter(slice.iter().cloned())
            }

            /// The number of elements in the multiset.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu32x2;
            /// use typenum::U2;
            /// assert_eq!(MSu32x2::<U2>::len(), 4);
            /// ```
            #[inline]
            pub const fn len() -> usize {
                <$simd>::lanes() * UInt::<U, B>::USIZE
            }

            /// Sets all element counts in the multiset to zero.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu32x2;
            /// use typenum::U2;
            /// let mut multiset = MSu32x2::<U2>::from_slice(&[1, 2, 3, 4]);
            /// multiset.clear();
            /// assert_eq!(multiset.is_empty(), true);
            /// ```
            #[inline]
            pub fn clear(&mut self) {
                self.data.iter_mut().for_each(|e| *e *= <$scalar>::ZERO);
            }

            /// Checks that a given element has at least one member in the multiset.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu32x2;
            /// use typenum::U2;
            /// let multiset = MSu32x2::<U2>::from_slice(&[1, 2, 0, 0]);
            /// assert_eq!(multiset.contains(1), true);
            /// assert_eq!(multiset.contains(3), false);
            /// assert_eq!(multiset.contains(5), false);
            /// ```
            ///
            /// ### Notes:
            /// - The implementation extracts values from the underlying SIMD vector.
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
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu32x2;
            /// use typenum::U2;
            /// let multiset = MSu32x2::<U2>::from_slice(&[1, 2, 0, 0]);
            /// assert_eq!(unsafe { multiset.contains_unchecked(1) }, true);
            /// assert_eq!(unsafe { multiset.contains_unchecked(3) }, false);
            /// // assert_eq!(unsafe { multiset.contains_unchecked(5) }, false);  NOT SAFE!!!
            /// ```
            ///
            /// ### Notes:
            /// - The implementation extracts values from the underlying SIMD vector.
            ///
            /// # Safety
            /// Does not run bounds check on whether this element is an index in the underlying
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

            /// Set the counter of an element in the multiset to `amount`.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu32x2;
            /// use typenum::U2;
            /// let mut multiset = MSu32x2::<U2>::from_slice(&[1, 2, 0, 0]);
            /// multiset.insert(2, 5);
            /// assert_eq!(multiset.get(2), Some(5));
            /// ```
            ///
            /// ### Notes:
            /// - The implementation replaces values from the underlying SIMD vector.
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

            /// Set the counter of an element in the multiset to `amount` without bounds checks.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu32x2;
            /// use typenum::U2;
            /// let mut multiset = MSu32x2::<U2>::from_slice(&[1, 2, 0, 0]);
            /// unsafe { multiset.insert_unchecked(2, 5) };
            /// assert_eq!(multiset.get(2), Some(5));
            /// // unsafe { multiset.insert_unchecked(5, 10) };  NOT SAFE!!!
            /// ```
            ///
            /// ### Notes:
            /// - The implementation replaces values from the underlying SIMD vector.
            ///
            /// # Safety
            /// Does not run bounds check on whether this element is an index in the underlying
            /// array.
            #[inline]
            pub unsafe fn insert_unchecked(&mut self, elem: usize, amount: $scalar) {
                let array_index = elem / <$simd>::lanes();
                let vector_index = elem % <$simd>::lanes();
                let vec = self.data.get_unchecked_mut(array_index);
                *vec = vec.replace_unchecked(vector_index, amount)
            }

            /// Set the counter of an element in the multiset to zero.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu32x2;
            /// use typenum::U2;
            /// let mut multiset = MSu32x2::<U2>::from_slice(&[1, 2, 0, 0]);
            /// multiset.remove(1);
            /// assert_eq!(multiset.get(1), Some(0));
            /// ```
            ///
            /// ### Notes:
            /// - The implementation replaces values from the underlying SIMD vector.
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

            /// Set the counter of an element in the multiset to zero without bounds checks.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu32x2;
            /// use typenum::U2;
            /// let mut multiset = MSu32x2::<U2>::from_slice(&[1, 2, 0, 0]);
            /// unsafe { multiset.remove_unchecked(1) };
            /// assert_eq!(multiset.get(1), Some(0));
            /// // unsafe { multiset.remove_unchecked(5) };  NOT SAFE!!!
            /// ```
            ///
            /// ### Notes:
            /// - The implementation replaces values from the underlying SIMD vector.
            ///
            /// # Safety
            /// Does not run bounds check on whether this element is an index in the underlying
            /// array.
            #[inline]
            pub unsafe fn remove_unchecked(&mut self, elem: usize) {
                let array_index = elem / <$simd>::lanes();
                let vector_index = elem % <$simd>::lanes();
                let vec = self.data.get_unchecked_mut(array_index);
                *vec = vec.replace_unchecked(vector_index, <$scalar>::ZERO)
            }

            /// Returns the amount of an element in the multiset.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu32x2;
            /// use typenum::U2;
            /// let multiset = MSu32x2::<U2>::from_slice(&[1, 2, 0, 0]);
            /// assert_eq!(multiset.get(1), Some(2));
            /// assert_eq!(multiset.get(3), Some(0));
            /// assert_eq!(multiset.get(5), None);
            /// ```
            ///
            /// ### Notes:
            /// - The implementation extracts values from the underlying SIMD vector.
            #[inline]
            pub fn get(self, elem: usize) -> Option<$scalar> {
                let array_index = elem / <$simd>::lanes();
                let vector_index = elem % <$simd>::lanes();
                unsafe { self.data.get(array_index).map(|vec| vec.extract_unchecked(vector_index)) }
            }

            /// Returns the amount of an element in the multiset without bounds checks.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu32x2;
            /// use typenum::U2;
            /// let multiset = MSu32x2::<U2>::from_slice(&[1, 2, 0, 0]);
            /// assert_eq!(unsafe { multiset.get_unchecked(1) }, 2);
            /// assert_eq!(unsafe { multiset.get_unchecked(3) }, 0);
            /// // unsafe { multiset.get_unchecked(5) };  NOT SAFE!!!
            /// ```
            ///
            /// ### Notes:
            /// - The implementation extracts values from the underlying SIMD vector.
            ///
            /// # Safety
            /// Does not run bounds check on whether this element is an index in the underlying
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
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu32x2;
            /// use typenum::U2;
            /// let a = MSu32x2::<U2>::from_slice(&[1, 2, 0, 0]);
            /// let b = MSu32x2::<U2>::from_slice(&[0, 2, 3, 0]);
            /// let c = MSu32x2::<U2>::from_slice(&[0, 2, 0, 0]);
            /// assert_eq!(a.intersection(&b), c);
            /// ```
            #[inline]
            pub fn intersection(&self, other: &Self) -> Self {
                self.zip_map(other, |s1, s2| s1.min(s2))
            }

            /// Returns a multiset which is the union of `self` and `other`.
            ///
            /// The union of two multisets A & B is defined as the multiset C where
            /// `C[0] == max(A[0], B[0]`).
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu32x2;
            /// use typenum::U2;
            /// let a = MSu32x2::<U2>::from_slice(&[1, 2, 0, 0]);
            /// let b = MSu32x2::<U2>::from_slice(&[0, 2, 3, 0]);
            /// let c = MSu32x2::<U2>::from_slice(&[1, 2, 3, 0]);
            /// assert_eq!(a.intersection(&b), c);
            /// ```
            #[inline]
            pub fn union(&self, other: &Self) -> Self {
                self.zip_map(other, |s1, s2| s1.max(s2))
            }

            /// Return the number of elements whose counter is zero.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu32x2;
            /// use typenum::U2;
            /// let multiset = MSu32x2::<U2>::from_slice(&[1, 0, 0, 0]);
            /// assert_eq!(multiset.count_zero(), 3);
            /// ```
            #[inline]
            pub fn count_zero(&self) -> u32 {
                self.fold(0, |acc, vec| {
                    acc + vec.eq(<$simd>::ZERO).bitmask().count_ones()
                })
            }

            /// Return the number of elements whose counter is non-zero.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu32x2;
            /// use typenum::U2;
            /// let multiset = MSu32x2::<U2>::from_slice(&[1, 0, 0, 0]);
            /// assert_eq!(multiset.count_non_zero(), 1);
            /// ```
            #[inline]
            pub fn count_non_zero(&self) -> u32 {
                self.fold(0, |acc, vec| {
                    acc + vec.gt(<$simd>::ZERO).bitmask().count_ones()
                })
            }

            /// Check whether all elements have a count of zero.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu32x2;
            /// use typenum::U2;
            /// let multiset = MSu32x2::<U2>::from_slice(&[0, 0, 0, 0]);
            /// assert_eq!(multiset.is_empty(), true);
            /// assert_eq!(MSu32x2::<U2>::empty().is_empty(), true);
            /// ```
            #[inline]
            pub fn is_empty(&self) -> bool {
                self.data.iter().all(|vec| vec == &<$simd>::ZERO)
            }

            /// Check whether only one element has a non-zero count.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu32x2;
            /// use typenum::U2;
            /// let multiset = MSu32x2::<U2>::from_slice(&[0, 5, 0, 0]);
            /// assert_eq!(multiset.is_singleton(), true);
            /// ```
            #[inline]
            pub fn is_singleton(&self) -> bool {
                self.count_non_zero() == 1
            }

            /// Returns `true` if `self` has no elements in common with `other`.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu32x2;
            /// use typenum::U2;
            /// let a = MSu32x2::<U2>::from_slice(&[1, 2, 0, 0]);
            /// let b = MSu32x2::<U2>::from_slice(&[0, 0, 3, 4]);
            /// assert_eq!(a.is_disjoint(&a), false);
            /// assert_eq!(a.is_disjoint(&b), true);
            /// ```
            #[inline]
            pub fn is_disjoint(&self, other: &Self) -> bool {
                self.data
                    .iter()
                    .zip(other.data.iter())
                    .fold(<$simd>::ZERO, |acc, (s1, s2)| acc + s1.min(*s2))
                    == <$simd>::ZERO
            }

            /// Check whether `self` is a subset of `other`.
            ///
            /// Multiset `A` is a subset of `B` if `A[i] <= B[i]` for all `i` in `A`.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu32x2;
            /// use typenum::U2;
            /// let a = MSu32x2::<U2>::from_slice(&[1, 2, 0, 0]);
            /// let b = MSu32x2::<U2>::from_slice(&[1, 3, 0, 0]);
            /// assert_eq!(a.is_subset(&a), true);
            /// assert_eq!(a.is_subset(&b), true);
            /// ```
            #[inline]
            pub fn is_subset(&self, other: &Self) -> bool {
                self.data
                    .iter()
                    .zip(other.data.iter())
                    .all(|(s1, s2)| s1.le(*s2).all())
            }

            /// Check whether `self` is a superset of `other`.
            ///
            /// Multiset `A` is a superset of `B` if `A[i] >= B[i]` for all `i` in `A`.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu32x2;
            /// use typenum::U2;
            /// let a = MSu32x2::<U2>::from_slice(&[1, 2, 0, 0]);
            /// let b = MSu32x2::<U2>::from_slice(&[1, 1, 0, 0]);
            /// assert_eq!(a.is_superset(&a), true);
            /// assert_eq!(a.is_superset(&b), true);
            /// ```
            #[inline]
            pub fn is_superset(&self, other: &Self) -> bool {
                self.data
                    .iter()
                    .zip(other.data.iter())
                    .all(|(s1, s2)| s1.ge(*s2).all())
            }

            /// Check whether `self` is a proper subset of `other`.
            ///
            /// Multiset `A` is a proper subset of `B` if `A[i] <= B[i]` for all `i` in `A` and
            /// there exists `j` such that `A[j] < B[j]`.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu32x2;
            /// use typenum::U2;
            /// let a = MSu32x2::<U2>::from_slice(&[1, 2, 0, 0]);
            /// let b = MSu32x2::<U2>::from_slice(&[1, 3, 0, 0]);
            /// assert_eq!(a.is_proper_subset(&a), false);
            /// assert_eq!(a.is_proper_subset(&b), true);
            /// ```
            #[inline]
            pub fn is_proper_subset(&self, other: &Self) -> bool {
                self.is_subset(other) && self.is_any_lesser(other)
            }

            /// Check whether `self` is a proper superset of `other`.
            ///
            /// Multiset `A` is a proper superset of `B` if `A[i] >= B[i]` for all `i` in `A` and
            /// there exists `j` such that `A[j] > B[j]`.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu32x2;
            /// use typenum::U2;
            /// let a = MSu32x2::<U2>::from_slice(&[1, 2, 0, 0]);
            /// let b = MSu32x2::<U2>::from_slice(&[1, 1, 0, 0]);
            /// assert_eq!(a.is_proper_superset(&a), false);
            /// assert_eq!(a.is_proper_superset(&b), true);
            /// ```
            #[inline]
            pub fn is_proper_superset(&self, other: &Self) -> bool {
                self.is_superset(other) && self.is_any_greater(other)
            }

            /// Check whether any element of `self` is less than an element of `other`.
            ///
            /// True if the exists some `i` such that `A[i] < B[i]`.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu32x2;
            /// use typenum::U2;
            /// let a = MSu32x2::<U2>::from_slice(&[1, 2, 4, 0]);
            /// let b = MSu32x2::<U2>::from_slice(&[1, 3, 0, 0]);
            /// assert_eq!(a.is_any_lesser(&a), false);
            /// assert_eq!(a.is_any_lesser(&b), true);
            /// ```
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
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu32x2;
            /// use typenum::U2;
            /// let a = MSu32x2::<U2>::from_slice(&[1, 2, 0, 0]);
            /// let b = MSu32x2::<U2>::from_slice(&[1, 1, 4, 0]);
            /// assert_eq!(a.is_any_greater(&a), false);
            /// assert_eq!(a.is_any_greater(&b), true);
            /// ```
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
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu32x2;
            /// use typenum::U2;
            /// let multiset = MSu32x2::<U2>::from_slice(&[1, 2, 3, 4]);
            /// assert_eq!(multiset.total(), 10);
            /// ```
            ///
            /// ### Notes:
            /// - This may overflow.
            /// - The implementation uses a horizontal operation on SIMD vectors.
            #[inline]
            pub fn total(&self) -> $scalar {
                self.fold(<$simd>::ZERO, |acc, vec| acc + vec)
                    .wrapping_sum()
            }

            /// Returns a tuple containing the (element, corresponding largest counter) in the
            /// multiset.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu32x2;
            /// use typenum::U2;
            /// let multiset = MSu32x2::<U2>::from_slice(&[2, 0, 5, 3]);
            /// assert_eq!(multiset.argmax(), (2, 5));
            /// ```
            ///
            /// ### Notes:
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

            /// Returns the element with the largest count in the multiset.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu32x2;
            /// use typenum::U2;
            /// let multiset = MSu32x2::<U2>::from_slice(&[2, 0, 5, 3]);
            /// assert_eq!(multiset.imax(), 2);
            /// ```
            ///
            /// ### Notes:
            /// - The implementation extracts values from the underlying SIMD vectors.
            #[inline]
            pub fn imax(&self) -> usize {
                self.argmax().0
            }

            /// Returns the largest counter in the multiset.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu32x2;
            /// use typenum::U2;
            /// let multiset = MSu32x2::<U2>::from_slice(&[2, 0, 5, 3]);
            /// assert_eq!(multiset.max(), 5);
            /// ```
            ///
            /// ### Notes:
            /// - The implementation uses a horizontal operation on the underlying SIMD vectors.
            #[inline]
            pub fn max(&self) -> $scalar {
                self.fold(<$simd>::ZERO, |acc, vec| acc.max(vec))
                    .max_element()
            }

            /// Returns a tuple containing the (element, corresponding smallest counter) in
            /// the multiset.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu32x2;
            /// use typenum::U2;
            /// let multiset = MSu32x2::<U2>::from_slice(&[2, 0, 5, 3]);
            /// assert_eq!(multiset.argmin(), (1, 0));
            /// ```
            ///
            /// ### Notes:
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

            /// Returns the element with the smallest count in the multiset.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu32x2;
            /// use typenum::U2;
            /// let multiset = MSu32x2::<U2>::from_slice(&[2, 0, 5, 3]);
            /// assert_eq!(multiset.imin(), 1);
            /// ```
            ///
            /// ### Notes:
            /// - The implementation extracts values from the underlying SIMD vectors.
            #[inline]
            pub fn imin(&self) -> usize {
                self.argmin().0
            }

            /// Returns the smallest counter in the multiset.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu32x2;
            /// use typenum::U2;
            /// let multiset = MSu32x2::<U2>::from_slice(&[2, 0, 5, 3]);
            /// assert_eq!(multiset.min(), 0);
            /// ```
            ///
            /// ### Notes:
            /// - The implementation uses a horizontal operation on the underlying SIMD vectors.
            #[inline]
            pub fn min(&self) -> $scalar {
                self.fold(<$simd>::MAX, |acc, vec| acc.min(vec))
                    .min_element()
            }

            /// Set all elements member counts, except for the given `elem`, to zero.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu32x2;
            /// use typenum::U2;
            /// let mut multiset = MSu32x2::<U2>::from_slice(&[2, 0, 5, 3]);
            /// multiset.choose(3);
            /// let result = MSu32x2::<U2>::from_slice(&[0, 0, 0, 3]);
            /// assert_eq!(multiset, result);
            /// ```
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

            /// Set all elements member counts, except for a random choice, to zero. The choice is
            /// weighted by the counts of the elements.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu32x2;
            /// use typenum::U2;
            /// use rand::prelude::*;
            /// let rng = &mut StdRng::seed_from_u64(thread_rng().next_u64());
            /// let mut multiset = MSu32x2::<U2>::from_slice(&[2, 0, 5, 3]);
            /// multiset.choose_random(rng);
            /// assert_eq!(multiset.is_singleton(), true);
            /// ```
            ///
            /// ### Notes:
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
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu32x2;
            /// use typenum::U2;
            /// let multiset = MSu32x2::<U2>::from_slice(&[2, 1, 1, 0]);
            /// let result = multiset.collision_entropy();
            /// // approximate: result == 1.415037499278844
            /// ```
            ///
            /// ### Notes:
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
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu32x2;
            /// use typenum::U2;
            /// let multiset = MSu32x2::<U2>::from_slice(&[2, 1, 1, 0]);
            /// let result = multiset.shannon_entropy();
            /// // approximate: result == 1.0397207708399179
            /// ```
            ///
            /// ### Notes:
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
