#[cfg(feature = "rand")]
use rand::prelude::*;
use std::cmp::Ordering;
use std::iter::FromIterator;

use crate::multiset::{Multiset, MultisetIterator};
use crate::small_num::SmallNumConsts;

macro_rules! multiset_scalar_array {
    ($($alias:ty, $scalar:ty),*) => {
        $(impl<const SIZE: usize> FromIterator<$scalar> for $alias {
            #[inline]
            fn from_iter<T: IntoIterator<Item=$scalar>>(iter: T) -> Self {
                let mut res: Self = unsafe { Multiset::new_uninitialized() };
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

        impl<'a, const SIZE: usize> FromIterator<&'a $scalar> for $alias {
            #[inline]
            fn from_iter<T: IntoIterator<Item=&'a $scalar>>(iter: T) -> Self {
                let mut res: Self = unsafe { Multiset::new_uninitialized() };
                let mut it = iter.into_iter();

                for i in 0..Self::len() {
                    let elem = match it.next() {
                        Some(v) => *v,
                        None => <$scalar>::ZERO,
                    };
                    unsafe { *res.data.get_unchecked_mut(i) = elem }
                }
                res
            }
        }

        impl<const SIZE: usize> PartialOrd for $alias {
            partial_ord_body!();
        }

        impl<const SIZE: usize> IntoIterator for $alias {
            type Item = $scalar;
            type IntoIter = MultisetIterator<$scalar, SIZE>;

            fn into_iter(self) -> Self::IntoIter {
                MultisetIterator {
                    multiset: self,
                    index: 0,
                }
            }
        }

        impl<const SIZE: usize> Iterator for MultisetIterator<$scalar, SIZE> {
            type Item = $scalar;

            fn next(&mut self) -> Option<Self::Item> {
                if self.index >= <$alias>::len() {
                    None
                } else {
                    let result = unsafe { *self.multiset.data.get_unchecked(self.index) };
                    self.index += 1;
                    Some(result)
                }
            }
        }

        impl<const SIZE: usize> ExactSizeIterator for MultisetIterator<$scalar, SIZE> {
            fn len(&self) -> usize {
                <$alias>::len()
            }
        }

        impl<const SIZE: usize> $alias {
            pub const SIZE: usize = SIZE;

            /// Returns a Multiset of the given array size with all elements set to zero.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu8;
            /// let multiset = MSu8::<4>::empty();
            /// ```
            #[inline]
            pub const fn empty() -> Self {
                Self::repeat(<$scalar>::ZERO)
            }

            /// Returns a Multiset of the given array size with all elements set to `elem`.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu8;
            /// let multiset = MSu8::<4>::repeat(5);
            /// ```
            #[inline]
            pub const fn repeat(elem: $scalar) -> Self {
                Multiset { data: [elem; SIZE] }
            }

            /// Returns a Multiset from a slice of the given array size.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu8;
            /// let multiset = MSu8::<4>::from_slice(&[1, 2, 3, 4]);
            /// ```
            #[inline]
            pub fn from_slice(slice: &[$scalar]) -> Self {
                slice.iter().collect()
            }

            /// The number of elements in the multiset.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu8;
            /// assert_eq!(MSu8::<4>::len(), 4);
            /// ```
            #[inline]
            pub const fn len() -> usize { SIZE }

            /// Sets all element counts in the multiset to zero.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu8;
            /// let mut multiset = MSu8::<4>::from_slice(&[1, 2, 3, 4]);
            /// multiset.clear();
            /// assert_eq!(multiset.is_empty(), true);
            /// ```
            #[inline]
            pub fn clear(&mut self) {
                self.data.iter_mut().for_each(|e| *e = <$scalar>::ZERO);
            }

            /// Checks that a given element has at least one member in the multiset.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu8;
            /// let multiset = MSu8::<4>::from_slice(&[1, 2, 0, 0]);
            /// assert_eq!(multiset.contains(1), true);
            /// assert_eq!(multiset.contains(3), false);
            /// assert_eq!(multiset.contains(5), false);
            /// ```
            #[inline]
            pub fn contains(&self, elem: usize) -> bool {
                elem < Self::len() && unsafe { self.data.get_unchecked(elem) > &<$scalar>::ZERO }
            }

            /// Checks that a given element has at least one member in the multiset without bounds
            /// checks.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu8;
            /// let multiset = MSu8::<4>::from_slice(&[1, 2, 0, 0]);
            /// assert_eq!(unsafe { multiset.contains_unchecked(1) }, true);
            /// assert_eq!(unsafe { multiset.contains_unchecked(3) }, false);
            /// // unsafe { multiset.contains_unchecked(5) };  NOT SAFE!!!
            /// ```
            ///
            /// # Safety
            /// Does not run bounds check on whether this element is an index in the underlying
            /// array.
            #[inline]
            pub unsafe fn contains_unchecked(&self, elem: usize) -> bool {
                self.data.get_unchecked(elem) > &<$scalar>::ZERO
            }

            /// Set the counter of an element in the multiset to `amount`.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu8;
            /// let mut multiset = MSu8::<4>::from_slice(&[1, 2, 0, 0]);
            /// multiset.insert(2, 5);
            /// assert_eq!(multiset.get(2), Some(&5));
            /// ```
            #[inline]
            pub fn insert(&mut self, elem: usize, amount: $scalar) {
                if elem < Self::len() {
                    unsafe { *self.data.get_unchecked_mut(elem) = amount };
                }
            }

            /// Set the counter of an element in the multiset to `amount` without bounds checks.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu8;
            /// let mut multiset = MSu8::<4>::from_slice(&[1, 2, 0, 0]);
            /// unsafe { multiset.insert_unchecked(2, 5) };
            /// assert_eq!(multiset.get(2), Some(&5));
            /// // unsafe { multiset.insert_unchecked(5, 10) };  NOT SAFE!!!
            /// ```
            ///
            /// # Safety
            /// Does not run bounds check on whether this element is an index in the underlying
            /// array.
            #[inline]
            pub unsafe fn insert_unchecked(&mut self, elem: usize, amount: $scalar) {
                *self.data.get_unchecked_mut(elem) = amount
            }

            /// Set the counter of an element in the multiset to zero.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu8;
            /// let mut multiset = MSu8::<4>::from_slice(&[1, 2, 0, 0]);
            /// multiset.remove(1);
            /// assert_eq!(multiset.get(1), Some(&0));
            /// ```
            #[inline]
            pub fn remove(&mut self, elem: usize) {
                if elem < Self::len() {
                    unsafe { *self.data.get_unchecked_mut(elem) = <$scalar>::ZERO };
                }
            }

            /// Set the counter of an element in the multiset to zero without bounds checks.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu8;
            /// let mut multiset = MSu8::<4>::from_slice(&[1, 2, 0, 0]);
            /// unsafe { multiset.remove_unchecked(1) };
            /// assert_eq!(multiset.get(1), Some(&0));
            /// // unsafe { multiset.remove_unchecked(5) };  NOT SAFE!!!
            /// ```
            ///
            /// # Safety
            /// Does not run bounds check on whether this element is an index in the underlying
            /// array.
            #[inline]
            pub unsafe fn remove_unchecked(&mut self, elem: usize) {
                *self.data.get_unchecked_mut(elem) = <$scalar>::ZERO
            }

            /// Returns the amount of an element in the multiset.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu8;
            /// let multiset = MSu8::<4>::from_slice(&[1, 2, 0, 0]);
            /// assert_eq!(multiset.get(1), Some(&2));
            /// assert_eq!(multiset.get(3), Some(&0));
            /// assert_eq!(multiset.get(5), None);
            /// ```
            #[inline]
            pub fn get(&self, elem: usize) -> Option<&$scalar> {
                self.data.get(elem)
            }

            /// Returns the amount of an element in the multiset without bounds checks.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu8;
            /// let multiset = MSu8::<4>::from_slice(&[1, 2, 0, 0]);
            /// assert_eq!(unsafe { multiset.get_unchecked(1) }, 2);
            /// assert_eq!(unsafe { multiset.get_unchecked(3) }, 0);
            /// // unsafe { multiset.get_unchecked(5) };  NOT SAFE!!!
            /// ```
            ///
            /// # Safety
            /// Does not run bounds check on whether this element is an index in the underlying
            /// array.
            #[inline]
            pub unsafe fn get_unchecked(&self, elem: usize) -> $scalar {
                *self.data.get_unchecked(elem)
            }

            /// Returns a multiset which is the intersection of `self` and `other`.
            ///
            /// The Intersection of two multisets A & B is defined as the multiset C where
            /// `C[0] == min(A[0], B[0]`).
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu8;
            /// let a = MSu8::<4>::from_slice(&[1, 2, 0, 0]);
            /// let b = MSu8::<4>::from_slice(&[0, 2, 3, 0]);
            /// let c = MSu8::<4>::from_slice(&[0, 2, 0, 0]);
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
            /// use utote::MSu8;
            /// let a = MSu8::<4>::from_slice(&[1, 2, 0, 0]);
            /// let b = MSu8::<4>::from_slice(&[0, 2, 3, 0]);
            /// let c = MSu8::<4>::from_slice(&[1, 2, 3, 0]);
            /// assert_eq!(a.union(&b), c);
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
            /// use utote::MSu8;
            /// let multiset = MSu8::<4>::from_slice(&[1, 0, 0, 0]);
            /// assert_eq!(multiset.count_zero(), 3);
            /// ```
            #[inline]
            pub fn count_zero(&self) -> $scalar {
                self.fold(Self::len() as $scalar, |acc, elem| acc - elem.min(1))
            }

            /// Return the number of elements whose counter is non-zero.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu8;
            /// let multiset = MSu8::<4>::from_slice(&[1, 0, 0, 0]);
            /// assert_eq!(multiset.count_non_zero(), 1);
            /// ```
            #[inline]
            pub fn count_non_zero(&self) -> $scalar {
                self.fold(0, |acc, elem| acc + elem.min(1))
            }

            /// Check whether all elements have a count of zero.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu8;
            /// let multiset = MSu8::<4>::from_slice(&[0, 0, 0, 0]);
            /// assert_eq!(multiset.is_empty(), true);
            /// assert_eq!(MSu8::<4>::empty().is_empty(), true);
            /// ```
            #[inline]
            pub fn is_empty(&self) -> bool {
                self.data.iter().all(|elem| elem == &<$scalar>::ZERO)
            }

            /// Check whether only one element has a non-zero count.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu8;
            /// let multiset = MSu8::<4>::from_slice(&[0, 5, 0, 0]);
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
            /// use utote::MSu8;
            /// let a = MSu8::<4>::from_slice(&[1, 2, 0, 0]);
            /// let b = MSu8::<4>::from_slice(&[0, 0, 3, 4]);
            /// assert_eq!(a.is_disjoint(&a), false);
            /// assert_eq!(a.is_disjoint(&b), true);
            /// ```
            #[inline]
            pub fn is_disjoint(&self, other: &Self) -> bool {
                self.data
                    .iter()
                    .zip(other.data.iter())
                    .fold(<$scalar>::ZERO, |acc, (a, b)| acc + a.min(b))
                    == <$scalar>::ZERO
            }

            /// Check whether `self` is a subset of `other`.
            ///
            /// Multiset `A` is a subset of `B` if `A[i] <= B[i]` for all `i` in `A`.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu8;
            /// let a = MSu8::<4>::from_slice(&[1, 2, 0, 0]);
            /// let b = MSu8::<4>::from_slice(&[1, 3, 0, 0]);
            /// assert_eq!(a.is_subset(&a), true);
            /// assert_eq!(a.is_subset(&b), true);
            /// ```
            #[inline]
            pub fn is_subset(&self, other: &Self) -> bool {
                self.data.iter().zip(other.data.iter()).all(|(a, b)| a <= b)
            }

            /// Check whether `self` is a superset of `other`.
            ///
            /// Multiset `A` is a superset of `B` if `A[i] >= B[i]` for all `i` in `A`.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu8;
            /// let a = MSu8::<4>::from_slice(&[1, 2, 0, 0]);
            /// let b = MSu8::<4>::from_slice(&[1, 1, 0, 0]);
            /// assert_eq!(a.is_superset(&a), true);
            /// assert_eq!(a.is_superset(&b), true);
            /// ```
            #[inline]
            pub fn is_superset(&self, other: &Self) -> bool {
                self.data.iter().zip(other.data.iter()).all(|(a, b)| a >= b)
            }

            /// Check whether `self` is a proper subset of `other`.
            ///
            /// Multiset `A` is a proper subset of `B` if `A[i] <= B[i]` for all `i` in `A` and
            /// there exists `j` such that `A[j] < B[j]`.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu8;
            /// let a = MSu8::<4>::from_slice(&[1, 2, 0, 0]);
            /// let b = MSu8::<4>::from_slice(&[1, 3, 0, 0]);
            /// assert_eq!(a.is_proper_subset(&a), false);
            /// assert_eq!(a.is_proper_subset(&b), true);
            /// ```
            #[inline]
            pub fn is_proper_subset(&self, other: &Self) -> bool {
                self != other && self.is_subset(other)
            }

            /// Check whether `self` is a proper superset of `other`.
            ///
            /// Multiset `A` is a proper superset of `B` if `A[i] >= B[i]` for all `i` in `A` and
            /// there exists `j` such that `A[j] > B[j]`.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu8;
            /// let a = MSu8::<4>::from_slice(&[1, 2, 0, 0]);
            /// let b = MSu8::<4>::from_slice(&[1, 1, 0, 0]);
            /// assert_eq!(a.is_proper_superset(&a), false);
            /// assert_eq!(a.is_proper_superset(&b), true);
            /// ```
            #[inline]
            pub fn is_proper_superset(&self, other: &Self) -> bool {
                self != other && self.is_superset(other)
            }

            /// Check whether any element of `self` is less than an element of `other`.
            ///
            /// True if the exists some `i` such that `A[i] < B[i]`.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu8;
            /// let a = MSu8::<4>::from_slice(&[1, 2, 4, 0]);
            /// let b = MSu8::<4>::from_slice(&[1, 3, 0, 0]);
            /// assert_eq!(a.is_any_lesser(&a), false);
            /// assert_eq!(a.is_any_lesser(&b), true);
            /// ```
            #[inline]
            pub fn is_any_lesser(&self, other: &Self) -> bool {
                self.data.iter().zip(other.data.iter()).any(|(a, b)| a < b)
            }

            /// Check whether any element of `self` is greater than an element of `other`.
            ///
            /// True if the exists some `i` such that `A[i] > B[i]`.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu8;
            /// let a = MSu8::<4>::from_slice(&[1, 2, 0, 0]);
            /// let b = MSu8::<4>::from_slice(&[1, 1, 4, 0]);
            /// assert_eq!(a.is_any_greater(&a), false);
            /// assert_eq!(a.is_any_greater(&b), true);
            /// ```
            #[inline]
            pub fn is_any_greater(&self, other: &Self) -> bool {
                self.data.iter().zip(other.data.iter()).any(|(a, b)| a > b)
            }

            /// The total or cardinality of a multiset is the sum of all its elements member
            /// counts.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu8;
            /// let multiset = MSu8::<4>::from_slice(&[1, 2, 3, 4]);
            /// assert_eq!(multiset.total(), 10);
            /// ```
            ///
            /// ### Notes:
            /// - This may overflow.
            #[inline]
            pub fn total(&self) -> $scalar {
                self.data.iter().sum()
            }

            /// Returns a tuple containing the (element, corresponding largest counter) in the
            /// multiset.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu8;
            /// let multiset = MSu8::<4>::from_slice(&[2, 0, 5, 3]);
            /// assert_eq!(multiset.argmax(), (2, 5));
            /// ```
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

            /// Returns the element with the largest count in the multiset.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu8;
            /// let multiset = MSu8::<4>::from_slice(&[2, 0, 5, 3]);
            /// assert_eq!(multiset.imax(), 2);
            /// ```
            #[inline]
            pub fn imax(&self) -> usize {
                self.argmax().0
            }

            /// Returns the largest counter in the multiset.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu8;
            /// let multiset = MSu8::<4>::from_slice(&[2, 0, 5, 3]);
            /// assert_eq!(multiset.max(), 5);
            /// ```
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

            /// Returns a tuple containing the (element, corresponding smallest counter) in
            /// the multiset.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu8;
            /// let multiset = MSu8::<4>::from_slice(&[2, 0, 5, 3]);
            /// assert_eq!(multiset.argmin(), (1, 0));
            /// ```
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

            /// Returns the element with the smallest count in the multiset.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu8;
            /// let multiset = MSu8::<4>::from_slice(&[2, 0, 5, 3]);
            /// assert_eq!(multiset.imin(), 1);
            /// ```
            #[inline]
            pub fn imin(&self) -> usize {
                self.argmin().0
            }

            /// Returns the smallest counter in the multiset.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu8;
            /// let multiset = MSu8::<4>::from_slice(&[2, 0, 5, 3]);
            /// assert_eq!(multiset.min(), 0);
            /// ```
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

            /// Set all element counts, except for the given `elem`, to zero.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu8;
            /// let mut multiset = MSu8::<4>::from_slice(&[2, 0, 5, 3]);
            /// multiset.choose(2);
            /// let result = MSu8::<4>::from_slice(&[0, 0, 5, 0]);
            /// assert_eq!(multiset, result);
            /// ```
            #[inline]
            pub fn choose(&mut self, elem: usize) {
                self.data.iter_mut().enumerate().for_each(|(i, e)| {
                    if i != elem { *e = <$scalar>::ZERO }
                })
            }

            /// Set all element counts, except for a random choice, to zero. The choice is
            /// weighted by the counts of the elements.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu8;
            /// use rand::prelude::*;
            /// let rng = &mut SmallRng::seed_from_u64(thread_rng().next_u64());
            /// let mut multiset = MSu8::<4>::from_slice(&[2, 0, 5, 3]);
            /// multiset.choose_random(rng);
            /// assert_eq!(multiset.is_singleton(), true);
            /// ```
            #[cfg(feature = "rand")]
            #[inline]
            pub fn choose_random<T: RngCore>(&mut self, rng: &mut T) {
                let total = self.total();
                if total == 0 {
                    return
                }
                let choice_value = rng.gen_range(<$scalar>::ONE..=total);
                let mut acc = <$scalar>::ZERO;
                let mut chosen = false;
                self.data.iter_mut().for_each(|elem| {
                    if chosen {
                        *elem = <$scalar>::ZERO
                    } else {
                        acc += *elem;
                        if acc >= choice_value {
                            chosen = true;
                        } else {
                            *elem = <$scalar>::ZERO
                        }
                    }
                })
            }
        })*
    }
}

multiset_scalar_array!(
    MSu8<SIZE>,
    u8,
    MSu16<SIZE>,
    u16,
    MSu32<SIZE>,
    u32,
    MSu64<SIZE>,
    u64
);

// Any alias where the scalar type can lossless cast to f64 can use this implementation.
macro_rules! multiset_scalar_array_stats {
    ($($alias:ty, $scalar:ty),*) => {
        $(impl<const SIZE: usize> $alias
            where
                f64: From<$scalar>,
        {
            /// Calculate the collision entropy of the multiset.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu8;
            /// let multiset = MSu8::<4>::from_slice(&[2, 1, 1, 0]);
            /// let result = multiset.collision_entropy();
            /// // approximate: result == 1.415037499278844
            /// ```
            #[inline]
            pub fn collision_entropy(&self) -> f64 {
                let total: f64 = From::from(self.total());
                -self.fold(0.0, |acc, frequency| {
                    acc + (f64::from(frequency) / total).powf(2.0)
                }).log2()
            }

            /// Calculate the shannon entropy of the multiset. Uses ln rather than log2.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use utote::MSu8;
            /// let multiset = MSu8::<4>::from_slice(&[2, 1, 1, 0]);
            /// let result = multiset.shannon_entropy();
            /// // approximate: result == 1.0397207708399179
            /// ```
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

multiset_scalar_array_stats!(MSu8<SIZE>, u8, MSu16<SIZE>, u16, MSu32<SIZE>, u32);

pub type MSu8<const SIZE: usize> = Multiset<u8, SIZE>;
pub type MSu16<const SIZE: usize> = Multiset<u16, SIZE>;
pub type MSu32<const SIZE: usize> = Multiset<u32, SIZE>;
pub type MSu64<const SIZE: usize> = Multiset<u64, SIZE>;

#[cfg(test)]
mod tests {
    use super::*;
    tests_x4!(ms4u32, Multiset<u32, 4>, u32);
    tests_x8!(ms8u16, Multiset<u16, 8>, u16);
    tests_x4!(ms4u8, Multiset<u8, 4>, u8);
}
