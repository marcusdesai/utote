use num_traits::{AsPrimitive, Zero};
use rand::prelude::*;
use std::cmp::Ordering;
use std::fmt::{Debug, Formatter, Result};
use std::hash::{Hash, Hasher};
use std::iter::FromIterator;
use std::mem;
use std::ops::AddAssign;
use std::slice::{Iter, IterMut};

pub struct Multiset2<N, const SIZE: usize> {
    data: [N; SIZE],
}

impl<N: Hash, const SIZE: usize> Hash for Multiset2<N, SIZE> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data.hash(state)
    }
}

impl<N: Debug, const SIZE: usize> Debug for Multiset2<N, SIZE> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("Multiset")
            .field("data", &self.data)
            .finish()
    }
}

impl<N: Copy, const SIZE: usize> Multiset2<N, SIZE> {
    #[inline]
    pub(crate) unsafe fn new_uninitialized() -> Self {
        Multiset2 {
            data: mem::MaybeUninit::<[N; SIZE]>::uninit().assume_init(),
        }
    }

    #[inline]
    pub(crate) fn fold<Acc, F>(&self, init: Acc, mut f: F) -> Acc
    where
        F: FnMut(Acc, N) -> Acc,
    {
        let mut res = init;
        for e in self.data.iter() {
            res = f(res, *e)
        }
        res
    }

    #[inline]
    pub(crate) fn zip_map<N2, N3, F>(
        &self,
        other: &Multiset2<N2, SIZE>,
        mut f: F,
    ) -> Multiset2<N3, SIZE>
    where
        N2: Copy,
        N3: Copy,
        F: FnMut(N, N2) -> N3,
    {
        let mut res = unsafe { Multiset2::new_uninitialized() };
        res.data
            .iter_mut()
            .zip(self.data.iter().zip(other.data.iter()))
            .for_each(|(r, (a, b))| *r = f(*a, *b));
        res
    }
}

impl<N: Clone, const SIZE: usize> Clone for Multiset2<N, SIZE> {
    fn clone(&self) -> Self {
        Multiset2 {
            data: self.data.clone(),
        }
    }
}

impl<N: Copy, const SIZE: usize> Copy for Multiset2<N, SIZE> {}

impl<N, const SIZE: usize> AddAssign for Multiset2<N, SIZE>
where
    N: AddAssign + Copy,
{
    fn add_assign(&mut self, rhs: Self) {
        // Safety: `i` will always be in range of `rhs.data`.
        self.data
            .iter_mut()
            .enumerate()
            .for_each(|(i, e)| unsafe { *e += *rhs.data.get_unchecked(i) })
    }
}

impl<N: PartialEq, const SIZE: usize> PartialEq for Multiset2<N, SIZE> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<N: PartialEq, const SIZE: usize> Eq for Multiset2<N, SIZE> {}

impl<N: Copy, const SIZE: usize> FromIterator<N> for Multiset2<N, SIZE>
where
    N: Zero,
{
    #[inline]
    fn from_iter<T: IntoIterator<Item = N>>(iter: T) -> Self {
        let mut res: Self = unsafe { Multiset2::new_uninitialized() };
        let it = iter.into_iter().chain(std::iter::repeat(N::zero()));
        res.data.iter_mut().zip(it).for_each(|(r, e)| *r = e);
        res
    }
}

impl<'a, N: 'a + Copy, const SIZE: usize> FromIterator<&'a N> for Multiset2<N, SIZE>
where
    N: Zero,
{
    #[inline]
    fn from_iter<T: IntoIterator<Item = &'a N>>(iter: T) -> Self {
        iter.into_iter().copied().collect()
    }
}

impl<N: PartialOrd + PartialEq, const SIZE: usize> PartialOrd for Multiset2<N, SIZE> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let mut order: Ordering = Ordering::Equal;
        for (e_self, e_other) in self.data.iter().zip(other.data.iter()) {
            match order {
                Ordering::Equal => {
                    if e_self < e_other {
                        order = Ordering::Less
                    } else if e_self > e_other {
                        order = Ordering::Greater
                    }
                }
                Ordering::Less => {
                    if e_self > e_other {
                        return None;
                    }
                }
                Ordering::Greater => {
                    if e_self < e_other {
                        return None;
                    }
                }
            }
        }
        Some(order)
    }
}

impl<'a, N, const SIZE: usize> IntoIterator for &'a Multiset2<N, SIZE> {
    type Item = &'a N;
    type IntoIter = Iter<'a, N>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<N: Copy + Default, const SIZE: usize> Default for Multiset2<N, SIZE> {
    fn default() -> Self {
        Multiset2 {
            data: [N::default(); SIZE],
        }
    }
}

impl<const SIZE: usize> Multiset2<u16, SIZE> {
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
        Multiset2 { data: [0; SIZE] }
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
    pub const fn repeat(elem: u16) -> Self {
        Multiset2 { data: [elem; SIZE] }
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
    pub fn from_slice(slice: &[u16]) -> Self {
        slice.iter().collect()
    }

    /// blah
    #[inline]
    pub const fn from_array(data: [u16; SIZE]) -> Self {
        Multiset2 { data }
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
    pub const fn len() -> usize {
        SIZE
    }

    /// blah
    #[inline]
    pub fn iter(&self) -> Iter<'_, u16> {
        self.data.iter()
    }

    /// blah
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, u16> {
        self.data.iter_mut()
    }

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
        self.data = [0; SIZE]
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
        // Safety: trivially guaranteed by bounds check on `elem` param.
        elem < SIZE && unsafe { self.data.get_unchecked(elem) > &0 }
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
        self.data.get_unchecked(elem) > &0
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
    pub fn insert(&mut self, elem: usize, amount: u16) {
        if elem < SIZE {
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
    pub unsafe fn insert_unchecked(&mut self, elem: usize, amount: u16) {
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
        if elem < SIZE {
            unsafe { *self.data.get_unchecked_mut(elem) = 0 };
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
        *self.data.get_unchecked_mut(elem) = 0
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
    pub fn get(&self, elem: usize) -> Option<&u16> {
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
    pub unsafe fn get_unchecked(&self, elem: usize) -> u16 {
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
    pub fn count_zero(&self) -> usize {
        self.fold(SIZE, |acc, elem| acc - elem.min(1) as usize)
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
    pub fn count_non_zero(&self) -> usize {
        self.fold(0, |acc, elem| acc + elem.min(1) as usize)
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
        self.data.iter().all(|elem| elem == &0)
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
            .all(|(a, b)| a.min(b) == &0)
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
    /// True if there exists some `i` such that `A[i] < B[i]`.
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
    /// True if there exists some `i` such that `A[i] > B[i]`.
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
    pub fn total(&self) -> usize {
        self.data.iter().map(|e| *e as usize).sum()
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
    pub fn argmax(&self) -> (usize, u16) {
        // iter cannot be empty, so it's fine to unwrap
        let (index, max) = self
            .data
            .iter()
            .enumerate()
            .max_by_key(|(_, e)| *e)
            .unwrap();
        (index, *max)
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
    pub fn max(&self) -> u16 {
        // iter cannot be empty, so it's fine to unwrap
        *self.data.iter().max().unwrap()
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
    pub fn argmin(&self) -> (usize, u16) {
        // iter cannot be empty, so it's fine to unwrap
        let (index, min) = self
            .data
            .iter()
            .enumerate()
            .min_by_key(|(_, e)| *e)
            .unwrap();
        (index, *min)
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
    pub fn min(&self) -> u16 {
        // iter cannot be empty, so it's fine to unwrap
        *self.data.iter().min().unwrap()
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
        let mut res = [0; SIZE];
        if elem < SIZE {
            unsafe { *res.get_unchecked_mut(elem) = *self.data.get_unchecked(elem) };
        }
        self.data = res
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
            return;
        }
        let choice_value = rng.gen_range(1..=total);
        let mut res = [0; SIZE];
        let mut acc = 0;
        for (i, elem) in self.data.iter().enumerate() {
            acc += *elem as usize;
            if acc >= choice_value {
                // Safety: `i` cannot be outside of `res`.
                unsafe { *res.get_unchecked_mut(i) = *elem }
                break;
            }
        }
        self.data = res
    }
}

impl<const SIZE: usize> Multiset2<u16, SIZE> {
    // todo: mention use of .as_() in changelog, important 1.45 potential break
    /// Calculate the collision entropy of the multiset.
    ///
    /// safety:
    /// **In Rust versions before 1.45.0**, some uses of the `as` operator were not entirely safe.
    /// In particular, it was undefined behavior if
    /// a truncated floating point value could not fit in the target integer
    /// type ([#10184](https://github.com/rust-lang/rust/issues/10184));
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
        let total: f64 = self.total().as_(); // todo: note use of .as_()
        -self
            .fold(0.0, |acc, frequency| {
                let freq_f64: f64 = frequency.as_();
                acc + (freq_f64 / total).powf(2.0)
            })
            .log2()
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
        let total: f64 = self.total().as_();
        -self.fold(0.0, |acc, frequency| {
            if frequency > 0 {
                let freq_f64: f64 = frequency.as_();
                let prob = freq_f64 / total;
                acc + prob * prob.ln()
            } else {
                acc
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    tests_x4!(multiset2u8_4, Multiset2<u16, 4>, u16);
    tests_x8!(multiset2u8_8, Multiset2<u16, 8>, u16);
}
