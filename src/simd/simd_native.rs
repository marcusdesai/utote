use packed_simd::*;
#[cfg(feature = "rand")]
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

        impl<'a> FromIterator<&'a $scalar> for $alias {
            #[inline]
            fn from_iter<T: IntoIterator<Item = &'a $scalar>>(iter: T) -> Self {
                let mut data = <$simd>::ZERO;
                let mut it = iter.into_iter();

                for i in 0..<$simd>::lanes() {
                    if let Some(elem) = it.next() {
                        data = unsafe { data.replace_unchecked(i, *elem) }
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
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MS0u16x4;
            /// let multiset = MS0u16x4::empty();
            /// ```
            #[inline]
            pub const fn empty() -> Self {
                Multiset {
                    data: <$simd>::ZERO,
                }
            }

            /// Returns a Multiset of the given SIMD vector size with all elements set to `elem`.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MS0u16x4;
            /// let multiset = MS0u16x4::repeat(5);
            /// ```
            #[inline]
            pub const fn repeat(elem: $scalar) -> Self {
                Multiset {
                    data: <$simd>::splat(elem),
                }
            }

            /// Returns a Multiset from a slice of the given SIMD vector size.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MS0u16x4;
            /// let multiset = MS0u16x4::from_slice(&[1, 2, 3, 4]);
            /// ```
            #[inline]
            pub fn from_slice(slice: &[$scalar]) -> Self {
                assert_eq!(slice.len(), Self::len());
                Multiset {
                    data: <$simd>::from_slice_unaligned(slice),
                }
            }

            /// The number of elements in the multiset.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MS0u16x4;
            /// assert_eq!(MS0u16x4::len(), 4);
            /// ```
            #[inline]
            pub const fn len() -> usize {
                <$simd>::lanes()
            }

            /// Sets all element counts in the multiset to zero.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MS0u16x4;
            /// let mut multiset = MS0u16x4::from_slice(&[1, 2, 3, 4]);
            /// multiset.clear();
            /// assert_eq!(multiset.is_empty(), true);
            /// ```
            #[inline]
            pub fn clear(&mut self) {
                self.data *= <$scalar>::ZERO
            }

            /// Checks that a given element has at least one member in the multiset.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MS0u16x4;
            /// let multiset = MS0u16x4::from_slice(&[1, 2, 0, 0]);
            /// assert_eq!(multiset.contains(1), true);
            /// assert_eq!(multiset.contains(3), false);
            /// assert_eq!(multiset.contains(5), false);
            /// ```
            ///
            /// ### Notes:
            /// - The implementation extracts values from the underlying SIMD vector.
            #[inline]
            pub fn contains(self, elem: usize) -> bool {
                elem < Self::len() && unsafe { self.data.extract_unchecked(elem) > <$scalar>::ZERO }
            }

            /// Checks that a given element has at least one member in the multiset without bounds
            /// checks.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MS0u16x4;
            /// let multiset = MS0u16x4::from_slice(&[1, 2, 0, 0]);
            /// assert_eq!(unsafe { multiset.contains_unchecked(1) }, true);
            /// assert_eq!(unsafe { multiset.contains_unchecked(3) }, false);
            /// // unsafe { multiset.contains_unchecked(5) };  NOT SAFE!!!
            /// ```
            ///
            /// ### Notes:
            /// - The implementation extracts values from the underlying SIMD vector.
            ///
            /// # Safety
            /// Does not run bounds check on whether this element is an index in the underlying
            /// SIMD vector.
            #[inline]
            pub unsafe fn contains_unchecked(self, elem: usize) -> bool {
                self.data.extract_unchecked(elem) > <$scalar>::ZERO
            }

            /// Set the counter of an element in the multiset to `amount`.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MS0u16x4;
            /// let mut multiset = MS0u16x4::from_slice(&[1, 2, 0, 0]);
            /// multiset.insert(2, 5);
            /// assert_eq!(multiset.get(2), Some(5));
            /// ```
            ///
            /// ### Notes:
            /// - The implementation replaces values from the underlying SIMD vector.
            #[inline]
            pub fn insert(&mut self, elem: usize, amount: $scalar) {
                if elem < Self::len() {
                    self.data = unsafe { self.data.replace_unchecked(elem, amount) };
                }
            }

            /// Set the counter of an element in the multiset to `amount` without bounds checks.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MS0u16x4;
            /// let mut multiset = MS0u16x4::from_slice(&[1, 2, 0, 0]);
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
            /// SIMD vector.
            #[inline]
            pub unsafe fn insert_unchecked(&mut self, elem: usize, amount: $scalar) {
                self.data = self.data.replace_unchecked(elem, amount);
            }

            /// Set the counter of an element in the multiset to zero.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MS0u16x4;
            /// let mut multiset = MS0u16x4::from_slice(&[1, 2, 0, 0]);
            /// multiset.remove(1);
            /// assert_eq!(multiset.get(1), Some(0));
            /// ```
            ///
            /// ### Notes:
            /// - The implementation replaces values from the underlying SIMD vector.
            #[inline]
            pub fn remove(&mut self, elem: usize) {
                if elem < Self::len() {
                    self.data = unsafe { self.data.replace_unchecked(elem, <$scalar>::ZERO) };
                }
            }

            /// Set the counter of an element in the multiset to zero without bounds checks.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MS0u16x4;
            /// let mut multiset = MS0u16x4::from_slice(&[1, 2, 0, 0]);
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
            /// SIMD vector.
            #[inline]
            pub unsafe fn remove_unchecked(&mut self, elem: usize) {
                self.data = self.data.replace_unchecked(elem, <$scalar>::ZERO);
            }

            /// Returns the amount of an element in the multiset.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MS0u16x4;
            /// let multiset = MS0u16x4::from_slice(&[1, 2, 0, 0]);
            /// assert_eq!(multiset.get(1), Some(2));
            /// assert_eq!(multiset.get(3), Some(0));
            /// assert_eq!(multiset.get(5), None);
            /// ```
            ///
            /// ### Notes:
            /// - The implementation extracts values from the underlying SIMD vector.
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
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MS0u16x4;
            /// let multiset = MS0u16x4::from_slice(&[1, 2, 0, 0]);
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
            /// SIMD vector.
            #[inline]
            pub unsafe fn get_unchecked(self, elem: usize) -> $scalar {
                self.data.extract_unchecked(elem)
            }

            /// Returns a multiset which is the intersection of `self` and `other`.
            ///
            /// The Intersection of two multisets A & B is defined as the multiset C where
            /// `C[0] == min(A[0], B[0]`).
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MS0u16x4;
            /// let a = MS0u16x4::from_slice(&[1, 2, 0, 0]);
            /// let b = MS0u16x4::from_slice(&[0, 2, 3, 0]);
            /// let c = MS0u16x4::from_slice(&[0, 2, 0, 0]);
            /// assert_eq!(a.intersection(&b), c);
            /// ```
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
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MS0u16x4;
            /// let a = MS0u16x4::from_slice(&[1, 2, 0, 0]);
            /// let b = MS0u16x4::from_slice(&[0, 2, 3, 0]);
            /// let c = MS0u16x4::from_slice(&[1, 2, 3, 0]);
            /// assert_eq!(a.union(&b), c);
            /// ```
            #[inline]
            pub fn union(&self, other: &Self) -> Self {
                Multiset {
                    data: self.data.max(other.data),
                }
            }

            /// Return the number of elements whose counter is zero.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MS0u16x4;
            /// let multiset = MS0u16x4::from_slice(&[1, 0, 0, 0]);
            /// assert_eq!(multiset.count_zero(), 3);
            /// ```
            #[inline]
            pub fn count_zero(&self) -> u32 {
                self.data.eq(<$simd>::ZERO).bitmask().count_ones()
            }

            /// Return the number of elements whose counter is non-zero.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MS0u16x4;
            /// let multiset = MS0u16x4::from_slice(&[1, 0, 0, 0]);
            /// assert_eq!(multiset.count_non_zero(), 1);
            /// ```
            #[inline]
            pub fn count_non_zero(&self) -> u32 {
                self.data.gt(<$simd>::ZERO).bitmask().count_ones()
            }

            /// Check whether all elements have a count of zero.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MS0u16x4;
            /// let multiset = MS0u16x4::from_slice(&[0, 0, 0, 0]);
            /// assert_eq!(multiset.is_empty(), true);
            /// assert_eq!(MS0u16x4::empty().is_empty(), true);
            /// ```
            #[inline]
            pub fn is_empty(&self) -> bool {
                self.data == <$simd>::ZERO
            }

            /// Check whether only one element has a non-zero count.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MS0u16x4;
            /// let multiset = MS0u16x4::from_slice(&[0, 5, 0, 0]);
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
            /// use utote::MS0u16x4;
            /// let a = MS0u16x4::from_slice(&[1, 2, 0, 0]);
            /// let b = MS0u16x4::from_slice(&[0, 0, 3, 4]);
            /// assert_eq!(a.is_disjoint(&a), false);
            /// assert_eq!(a.is_disjoint(&b), true);
            /// ```
            #[inline]
            pub fn is_disjoint(&self, other: &Self) -> bool {
                self.data.min(other.data) == <$simd>::ZERO
            }

            /// Check whether `self` is a subset of `other`.
            ///
            /// Multiset `A` is a subset of `B` if `A[i] <= B[i]` for all `i` in `A`.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MS0u16x4;
            /// let a = MS0u16x4::from_slice(&[1, 2, 0, 0]);
            /// let b = MS0u16x4::from_slice(&[1, 3, 0, 0]);
            /// assert_eq!(a.is_subset(&a), true);
            /// assert_eq!(a.is_subset(&b), true);
            /// ```
            #[inline]
            pub fn is_subset(&self, other: &Self) -> bool {
                self.data.le(other.data).all()
            }

            /// Check whether `self` is a superset of `other`.
            ///
            /// Multiset `A` is a superset of `B` if `A[i] >= B[i]` for all `i` in `A`.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MS0u16x4;
            /// let a = MS0u16x4::from_slice(&[1, 2, 0, 0]);
            /// let b = MS0u16x4::from_slice(&[1, 1, 0, 0]);
            /// assert_eq!(a.is_superset(&a), true);
            /// assert_eq!(a.is_superset(&b), true);
            /// ```
            #[inline]
            pub fn is_superset(&self, other: &Self) -> bool {
                self.data.ge(other.data).all()
            }

            /// Check whether `self` is a proper subset of `other`.
            ///
            /// Multiset `A` is a proper subset of `B` if `A[i] <= B[i]` for all `i` in `A` and
            /// there exists `j` such that `A[j] < B[j]`.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MS0u16x4;
            /// let a = MS0u16x4::from_slice(&[1, 2, 0, 0]);
            /// let b = MS0u16x4::from_slice(&[1, 3, 0, 0]);
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
            /// use utote::MS0u16x4;
            /// let a = MS0u16x4::from_slice(&[1, 2, 0, 0]);
            /// let b = MS0u16x4::from_slice(&[1, 1, 0, 0]);
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
            /// use utote::MS0u16x4;
            /// let a = MS0u16x4::from_slice(&[1, 2, 4, 0]);
            /// let b = MS0u16x4::from_slice(&[1, 3, 0, 0]);
            /// assert_eq!(a.is_any_lesser(&a), false);
            /// assert_eq!(a.is_any_lesser(&b), true);
            /// ```
            #[inline]
            pub fn is_any_lesser(&self, other: &Self) -> bool {
                self.data.lt(other.data).any()
            }

            /// Check whether any element of `self` is greater than an element of `other`.
            ///
            /// True if the exists some `i` such that `A[i] > B[i]`.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MS0u16x4;
            /// let a = MS0u16x4::from_slice(&[1, 2, 0, 0]);
            /// let b = MS0u16x4::from_slice(&[1, 1, 4, 0]);
            /// assert_eq!(a.is_any_greater(&a), false);
            /// assert_eq!(a.is_any_greater(&b), true);
            /// ```
            #[inline]
            pub fn is_any_greater(&self, other: &Self) -> bool {
                self.data.gt(other.data).any()
            }

            /// The total or cardinality of a multiset is the sum of all its elements member
            /// counts.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MS0u16x4;
            /// let multiset = MS0u16x4::from_slice(&[1, 2, 3, 4]);
            /// assert_eq!(multiset.total(), 10);
            /// ```
            ///
            /// ### Notes:
            /// - This may overflow.
            /// - The implementation uses a horizontal operation on SIMD vectors.
            #[inline]
            pub fn total(&self) -> $scalar {
                self.data.wrapping_sum()
            }

            /// Returns a tuple containing the (element, corresponding largest counter) in the
            /// multiset.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MS0u16x4;
            /// let multiset = MS0u16x4::from_slice(&[2, 0, 5, 3]);
            /// assert_eq!(multiset.argmax(), (2, 5));
            /// ```
            ///
            /// ### Notes:
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

            /// Returns the element with the largest count in the multiset.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MS0u16x4;
            /// let multiset = MS0u16x4::from_slice(&[2, 0, 5, 3]);
            /// assert_eq!(multiset.imax(), 2);
            /// ```
            ///
            /// ### Notes:
            /// - The implementation extracts values from the underlying SIMD vector.
            #[inline]
            pub fn imax(&self) -> usize {
                self.argmax().0
            }

            /// Returns the largest counter in the multiset.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MS0u16x4;
            /// let multiset = MS0u16x4::from_slice(&[2, 0, 5, 3]);
            /// assert_eq!(multiset.max(), 5);
            /// ```
            ///
            /// ### Notes:
            /// - The implementation uses a horizontal operation on the underlying SIMD vector.
            #[inline]
            pub fn max(&self) -> $scalar {
                self.data.max_element()
            }

            /// Returns a tuple containing the (element, corresponding smallest counter) in the
            /// multiset.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MS0u16x4;
            /// let multiset = MS0u16x4::from_slice(&[2, 0, 5, 3]);
            /// assert_eq!(multiset.argmin(), (1, 0));
            /// ```
            ///
            /// ### Notes:
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

            /// Returns the element with the smallest count in the multiset.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MS0u16x4;
            /// let multiset = MS0u16x4::from_slice(&[2, 0, 5, 3]);
            /// assert_eq!(multiset.imin(), 1);
            /// ```
            ///
            /// ### Notes:
            /// - The implementation extracts values from the underlying SIMD vector.
            #[inline]
            pub fn imin(&self) -> usize {
                self.argmin().0
            }

            /// Returns the smallest counter in the multiset.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MS0u16x4;
            /// let multiset = MS0u16x4::from_slice(&[2, 0, 5, 3]);
            /// assert_eq!(multiset.min(), 0);
            /// ```
            ///
            /// ### Notes:
            /// - The implementation uses a horizontal operation on the underlying SIMD vector.
            #[inline]
            pub fn min(&self) -> $scalar {
                self.data.min_element()
            }

            /// Set all element counts, except for the given `elem`, to zero.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MS0u16x4;
            /// let mut multiset = MS0u16x4::from_slice(&[2, 0, 5, 3]);
            /// multiset.choose(2);
            /// let result = MS0u16x4::from_slice(&[0, 0, 5, 0]);
            /// assert_eq!(multiset, result);
            /// ```
            #[inline]
            pub fn choose(&mut self, elem: usize) {
                let mask = <$simd_mask>::splat(false).replace(elem, true);
                self.data = mask.select(self.data, <$simd>::ZERO)
            }

            /// Set all element counts, except for a random choice, to zero. The choice is
            /// weighted by the counts of the elements.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MS0u16x4;
            /// use rand::prelude::*;
            /// let rng = &mut SmallRng::seed_from_u64(thread_rng().next_u64());
            /// let mut multiset = MS0u16x4::from_slice(&[2, 0, 5, 3]);
            /// multiset.choose_random(rng);
            /// assert_eq!(multiset.is_singleton(), true);
            /// ```
            ///
            /// ### Notes:
            /// - The implementation extracts values from the underlying SIMD vector.
            #[cfg(feature = "rand")]
            #[inline]
            pub fn choose_random<T: RngCore>(&mut self, rng: &mut T) {
                let choice_value = rng.gen_range(<$scalar>::ONE..=self.total());
                let mut elem: usize = 0;
                let mut acc: $scalar = <$scalar>::ZERO;
                for i in 0..Self::len() {
                    acc += unsafe { self.data.extract_unchecked(i) };
                    if acc >= choice_value {
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
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MS0u16x4;
            /// let multiset = MS0u16x4::from_slice(&[2, 1, 1, 0]);
            /// let result = multiset.collision_entropy();
            /// // approximate: result == 1.415037499278844
            /// ```
            ///
            /// ### Notes:
            /// - The implementation uses a horizontal operation on SIMD vectors.
            #[inline]
            pub fn collision_entropy(&self) -> f64 {
                let total: f64 = From::from(self.total());
                let data = <$simd_float>::from(self.data);
                -(data / total).powf(<$simd_float>::splat(2.0)).sum().log2()
            }

            /// Calculate the shannon entropy of the multiset. Uses ln rather than log2.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MS0u16x4;
            /// let multiset = MS0u16x4::from_slice(&[2, 1, 1, 0]);
            /// let result = multiset.shannon_entropy();
            /// // approximate: result == 1.0397207708399179
            /// ```
            ///
            /// ### Notes:
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

pub type MS0u8x2 = Multiset<u8x2, typenum::UTerm>;
pub type MS0u8x4 = Multiset<u8x4, typenum::UTerm>;
pub type MS0u8x8 = Multiset<u8x8, typenum::UTerm>;
pub type MS0u8x16 = Multiset<u8x16, typenum::UTerm>;
pub type MS0u8x32 = Multiset<u8x32, typenum::UTerm>;
pub type MS0u8x64 = Multiset<u8x64, typenum::UTerm>;

pub type MS0u16x2 = Multiset<u16x2, typenum::UTerm>;
pub type MS0u16x4 = Multiset<u16x4, typenum::UTerm>;
pub type MS0u16x8 = Multiset<u16x8, typenum::UTerm>;
pub type MS0u16x16 = Multiset<u16x16, typenum::UTerm>;
pub type MS0u16x32 = Multiset<u16x32, typenum::UTerm>;

pub type MS0u32x2 = Multiset<u32x2, typenum::UTerm>;
pub type MS0u32x4 = Multiset<u32x4, typenum::UTerm>;
pub type MS0u32x8 = Multiset<u32x8, typenum::UTerm>;
pub type MS0u32x16 = Multiset<u32x16, typenum::UTerm>;

pub type MS0u64x2 = Multiset<u64x2, typenum::UTerm>;
pub type MS0u64x4 = Multiset<u64x4, typenum::UTerm>;
pub type MS0u64x8 = Multiset<u64x8, typenum::UTerm>;

#[cfg(test)]
mod tests {
    use super::*;
    tests_x4!(ms0u32x4, u32x4, typenum::U0);
    tests_x8!(ms0u16x8, u16x8, typenum::U0);
}
