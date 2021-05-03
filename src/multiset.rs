#[cfg(feature = "packed_simd")]
use crate::simd::SimdTypes;
use num_traits::{AsPrimitive, One, Unsigned, Zero};
#[cfg(all(not(feature = "packed_simd"), feature = "rand"))]
use rand::{Rng, RngCore};
use std::cmp::Ordering;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::iter::FromIterator;
use std::mem;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign};
use std::slice::{Iter, IterMut};

mod sealed {
    pub trait Sealed {}

    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
    impl Sealed for usize {}
}

// Just arithmetic impls for unsigned ints.
#[doc(hidden)]
pub trait CounterArithmetic:
    sealed::Sealed
    + Sized
    + Add
    + AddAssign
    + Div
    + DivAssign
    + Mul
    + MulAssign
    + Rem
    + RemAssign
    + Sub
    + SubAssign
{
    // empty
}

impl CounterArithmetic for u8 {}
impl CounterArithmetic for u16 {}
impl CounterArithmetic for u32 {}
impl CounterArithmetic for u64 {}
impl CounterArithmetic for usize {}

// Collects various properties into one trait for more concise implementations. All of these
// properties are implemented by unsigned integers.
#[doc(hidden)]
pub trait CounterBasic:
    sealed::Sealed
    + Clone
    + Copy
    + Debug
    + Default
    + Hash
    + One
    + Ord
    + PartialEq
    + PartialOrd
    + Unsigned
    + Zero
    + AsPrimitive<usize>
    + AsPrimitive<f64>
{
    // empty
}

impl CounterBasic for u8 {}
impl CounterBasic for u16 {}
impl CounterBasic for u32 {}
impl CounterBasic for u64 {}
impl CounterBasic for usize {}

#[cfg(not(feature = "packed_simd"))]
/// Docs for counter type
pub trait Counter: CounterArithmetic + CounterBasic {
    // empty
}

#[cfg(feature = "packed_simd")]
#[doc(hidden)]
pub trait Counter: sealed::Sealed + CounterArithmetic + CounterBasic + SimdTypes {
    // empty
}

impl Counter for u8 {}
impl Counter for u16 {}
impl Counter for u32 {}
impl Counter for u64 {}
impl Counter for usize {}

/// Multiset! yay
#[derive(Debug)]
pub struct Multiset<N: Counter, const SIZE: usize> {
    pub(crate) data: [N; SIZE],
}

impl<N: Counter, const SIZE: usize> Hash for Multiset<N, SIZE> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data.hash(state)
    }
}

impl<N: Counter, const SIZE: usize> Clone for Multiset<N, SIZE> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<N: Counter, const SIZE: usize> Copy for Multiset<N, SIZE> {}

impl<N: Counter, const SIZE: usize> PartialEq for Multiset<N, SIZE> {
    // Array compare will always use memcmp because N will be a unit, so no need for SIMD
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<N: Counter, const SIZE: usize> Eq for Multiset<N, SIZE> {}

impl<N: Counter, const SIZE: usize> FromIterator<N> for Multiset<N, SIZE> {
    #[inline]
    fn from_iter<T: IntoIterator<Item = N>>(iter: T) -> Self {
        let mut res: Self = unsafe { Multiset::new_uninitialized() };
        let it = iter.into_iter().chain(std::iter::repeat(N::zero()));
        res.iter_mut().zip(it).for_each(|(r, e)| *r = e);
        res
    }
}

impl<'a, N: 'a + Counter, const SIZE: usize> FromIterator<&'a N> for Multiset<N, SIZE> {
    #[inline]
    fn from_iter<T: IntoIterator<Item = &'a N>>(iter: T) -> Self {
        iter.into_iter().copied().collect()
    }
}

impl<N: Counter, const SIZE: usize> IntoIterator for Multiset<N, SIZE> {
    type Item = N;
    type IntoIter = std::array::IntoIter<N, SIZE>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        std::array::IntoIter::new(self.data)
    }
}

impl<'a, N: Counter, const SIZE: usize> IntoIterator for &'a Multiset<N, SIZE> {
    type Item = &'a N;
    type IntoIter = Iter<'a, N>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<N: Counter, const SIZE: usize> From<[N; SIZE]> for Multiset<N, SIZE> {
    fn from(data: [N; SIZE]) -> Self {
        Multiset { data }
    }
}

impl<N: Counter, const SIZE: usize> From<&[N; SIZE]> for Multiset<N, SIZE> {
    fn from(data: &[N; SIZE]) -> Self {
        Multiset { data: *data }
    }
}

impl<N: Counter, const SIZE: usize> From<Multiset<N, SIZE>> for [N; SIZE] {
    fn from(set: Multiset<N, SIZE>) -> Self {
        set.data
    }
}

impl<N: Counter, const SIZE: usize> From<&[N]> for Multiset<N, SIZE> {
    fn from(slice: &[N]) -> Self {
        slice.iter().collect()
    }
}

impl<'a, N: Counter, const SIZE: usize> From<&'a Multiset<N, SIZE>> for &'a [N] {
    fn from(set: &'a Multiset<N, SIZE>) -> Self {
        &set.data
    }
}

impl<N: Counter, const SIZE: usize> PartialOrd for Multiset<N, SIZE> {
    #[allow(clippy::comparison_chain)]
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

impl<N: Counter, const SIZE: usize> Default for Multiset<N, SIZE> {
    #[inline]
    fn default() -> Self {
        Multiset {
            data: [N::default(); SIZE],
        }
    }
}

// Ops implementations
// todo: use SIMD for ops?

impl<N: Counter, const SIZE: usize> Add for Multiset<N, SIZE> {
    type Output = Multiset<N, SIZE>;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        self.into_iter()
            .zip(rhs.into_iter())
            .map(|(a, b)| a + b)
            .collect()
    }
}

impl<N: Counter, const SIZE: usize> AddAssign for Multiset<N, SIZE> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.iter_mut()
            .zip(rhs.into_iter())
            .for_each(|(l, r)| *l += r);
    }
}

impl<N: Counter, const SIZE: usize> Div for Multiset<N, SIZE> {
    type Output = Multiset<N, SIZE>;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        self.into_iter()
            .zip(rhs.into_iter())
            .map(|(a, b)| a / b)
            .collect()
    }
}

impl<N: Counter, const SIZE: usize> DivAssign for Multiset<N, SIZE> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        self.iter_mut()
            .zip(rhs.into_iter())
            .for_each(|(l, r)| *l /= r);
    }
}

impl<N: Counter, const SIZE: usize> Mul for Multiset<N, SIZE> {
    type Output = Multiset<N, SIZE>;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self.into_iter()
            .zip(rhs.into_iter())
            .map(|(a, b)| a * b)
            .collect()
    }
}

impl<N: Counter, const SIZE: usize> MulAssign for Multiset<N, SIZE> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.iter_mut()
            .zip(rhs.into_iter())
            .for_each(|(l, r)| *l *= r);
    }
}

impl<N: Counter, const SIZE: usize> Rem for Multiset<N, SIZE> {
    type Output = Multiset<N, SIZE>;

    #[inline]
    fn rem(self, rhs: Self) -> Self::Output {
        self.into_iter()
            .zip(rhs.into_iter())
            .map(|(a, b)| a % b)
            .collect()
    }
}

impl<N: Counter, const SIZE: usize> RemAssign for Multiset<N, SIZE> {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        self.iter_mut()
            .zip(rhs.into_iter())
            .for_each(|(l, r)| *l %= r);
    }
}

impl<N: Counter, const SIZE: usize> Sub for Multiset<N, SIZE> {
    type Output = Multiset<N, SIZE>;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        self.into_iter()
            .zip(rhs.into_iter())
            .map(|(a, b)| a - b)
            .collect()
    }
}

impl<N: Counter, const SIZE: usize> SubAssign for Multiset<N, SIZE> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.iter_mut()
            .zip(rhs.into_iter())
            .for_each(|(l, r)| *l -= r);
    }
}

impl<N: Counter, const SIZE: usize> Multiset<N, SIZE> {
    /// The number of elements in the multiset.
    pub const SIZE: usize = SIZE;

    /// The number of elements in the multiset.
    #[inline]
    pub fn len() -> usize {
        SIZE
    }

    #[inline]
    pub(crate) unsafe fn new_uninitialized() -> Self {
        Multiset {
            data: mem::MaybeUninit::<[N; SIZE]>::uninit().assume_init(),
        }
    }

    #[inline]
    pub(crate) fn zip_map<N2, N3, F>(
        &self,
        other: &Multiset<N2, SIZE>,
        mut f: F,
    ) -> Multiset<N3, SIZE>
    where
        N2: Counter,
        N3: Counter,
        F: FnMut(N, N2) -> N3,
    {
        let mut res = unsafe { Multiset::new_uninitialized() };
        res.iter_mut()
            .zip(self.iter().zip(other.iter()))
            .for_each(|(r, (a, b))| *r = f(*a, *b));
        res
    }

    /// Returns a Multiset of the given array size with all elements set to zero.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use utote::Multiset;
    /// let multiset = Multiset::<u8, 4>::empty();
    /// ```
    #[inline]
    pub fn empty() -> Self {
        Multiset {
            data: [N::zero(); SIZE],
        }
    }

    /// Returns a Multiset of the given array size with all elements set to `elem`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use utote::Multiset;
    /// let multiset = Multiset::<u8, 4>::repeat(5);
    /// ```
    #[inline]
    pub fn repeat(elem: N) -> Self {
        Multiset { data: [elem; SIZE] }
    }

    /// blah
    #[inline]
    pub fn iter(&self) -> Iter<'_, N> {
        self.data.iter()
    }

    /// blah
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, N> {
        self.data.iter_mut()
    }

    /// Sets all element counts in the multiset to zero.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use utote::Multiset;
    /// let mut multiset = Multiset::<u8, 4>::from_slice(&[1, 2, 3, 4]);
    /// multiset.clear();
    /// assert_eq!(multiset.is_empty(), true);
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        self.data = [N::zero(); SIZE]
    }

    /// Checks that a given element has at least one member in the multiset.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use utote::Multiset;
    /// let multiset = Multiset::<u8, 4>::from_slice(&[1, 2, 0, 0]);
    /// assert_eq!(multiset.contains(1), true);
    /// assert_eq!(multiset.contains(3), false);
    /// assert_eq!(multiset.contains(5), false);
    /// ```
    #[inline]
    pub fn contains(&self, elem: usize) -> bool {
        // Safety: trivially guaranteed by bounds check on `elem` param.
        elem < SIZE && unsafe { self.data.get_unchecked(elem) > &N::zero() }
    }

    /// Checks that a given element has at least one member in the multiset without bounds
    /// checks.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use utote::Multiset;
    /// let multiset = Multiset::<u8, 4>::from_slice(&[1, 2, 0, 0]);
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
        self.data.get_unchecked(elem) > &N::zero()
    }

    /// Set the counter of an element in the multiset to `amount`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use utote::Multiset;
    /// let mut multiset = Multiset::<u8, 4>::from_slice(&[1, 2, 0, 0]);
    /// multiset.insert(2, 5);
    /// assert_eq!(multiset.get(2), Some(&5));
    /// ```
    #[inline]
    pub fn insert(&mut self, elem: usize, amount: N) {
        if elem < SIZE {
            unsafe { *self.data.get_unchecked_mut(elem) = amount };
        }
    }

    /// Set the counter of an element in the multiset to `amount` without bounds checks.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use utote::Multiset;
    /// let mut multiset = Multiset::<u8, 4>::from_slice(&[1, 2, 0, 0]);
    /// unsafe { multiset.insert_unchecked(2, 5) };
    /// assert_eq!(multiset.get(2), Some(&5));
    /// // unsafe { multiset.insert_unchecked(5, 10) };  NOT SAFE!!!
    /// ```
    ///
    /// # Safety
    /// Does not run bounds check on whether this element is an index in the underlying
    /// array.
    #[inline]
    pub unsafe fn insert_unchecked(&mut self, elem: usize, amount: N) {
        *self.data.get_unchecked_mut(elem) = amount
    }

    /// Set the counter of an element in the multiset to zero.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use utote::Multiset;
    /// let mut multiset = Multiset::<u8, 4>::from_slice(&[1, 2, 0, 0]);
    /// multiset.remove(1);
    /// assert_eq!(multiset.get(1), Some(&0));
    /// ```
    #[inline]
    pub fn remove(&mut self, elem: usize) {
        if elem < SIZE {
            unsafe { *self.data.get_unchecked_mut(elem) = N::zero() };
        }
    }

    /// Set the counter of an element in the multiset to zero without bounds checks.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use utote::Multiset;
    /// let mut multiset = Multiset::<u8, 4>::from_slice(&[1, 2, 0, 0]);
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
        *self.data.get_unchecked_mut(elem) = N::zero()
    }

    /// Returns the amount of an element in the multiset.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use utote::Multiset;
    /// let multiset = Multiset::<u8, 4>::from_slice(&[1, 2, 0, 0]);
    /// assert_eq!(multiset.get(1), Some(&2));
    /// assert_eq!(multiset.get(3), Some(&0));
    /// assert_eq!(multiset.get(5), None);
    /// ```
    #[inline]
    pub fn get(&self, elem: usize) -> Option<&N> {
        self.data.get(elem)
    }

    /// Returns the amount of an element in the multiset without bounds checks.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use utote::Multiset;
    /// let multiset = Multiset::<u8, 4>::from_slice(&[1, 2, 0, 0]);
    /// assert_eq!(unsafe { multiset.get_unchecked(1) }, 2);
    /// assert_eq!(unsafe { multiset.get_unchecked(3) }, 0);
    /// // unsafe { multiset.get_unchecked(5) };  NOT SAFE!!!
    /// ```
    ///
    /// # Safety
    /// Does not run bounds check on whether this element is an index in the underlying
    /// array.
    #[inline]
    pub unsafe fn get_unchecked(&self, elem: usize) -> N {
        *self.data.get_unchecked(elem)
    }

    /// Returns a multiset which is the intersection of `self` and `other`.
    ///
    /// The Intersection of two multisets A & B is defined as the multiset C where
    /// `C[0] == min(A[0], B[0])`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use utote::Multiset;
    /// let a = Multiset::<u8, 4>::from_slice(&[1, 2, 0, 0]);
    /// let b = Multiset::<u8, 4>::from_slice(&[0, 2, 3, 0]);
    /// let c = Multiset::<u8, 4>::from_slice(&[0, 2, 0, 0]);
    /// assert_eq!(a.intersection(&b), c);
    /// ```
    #[cfg(not(feature = "packed_simd"))]
    #[inline]
    pub fn intersection(&self, other: &Self) -> Self {
        self.zip_map(other, |s1, s2| s1.min(s2))
    }

    /// Returns a multiset which is the union of `self` and `other`.
    ///
    /// The union of two multisets A & B is defined as the multiset C where
    /// `C[0] == max(A[0], B[0])`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use utote::Multiset;
    /// let a = Multiset::<u8, 4>::from_slice(&[1, 2, 0, 0]);
    /// let b = Multiset::<u8, 4>::from_slice(&[0, 2, 3, 0]);
    /// let c = Multiset::<u8, 4>::from_slice(&[1, 2, 3, 0]);
    /// assert_eq!(a.union(&b), c);
    /// ```
    #[cfg(not(feature = "packed_simd"))]
    #[inline]
    pub fn union(&self, other: &Self) -> Self {
        self.zip_map(other, |s1, s2| s1.max(s2))
    }

    /// Return the number of elements whose counter is non-zero.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use utote::Multiset;
    /// let multiset = Multiset::<u8, 4>::from_slice(&[1, 0, 0, 0]);
    /// assert_eq!(multiset.count_non_zero(), 1);
    /// ```
    #[cfg(not(feature = "packed_simd"))]
    #[inline]
    pub fn count_non_zero(&self) -> usize {
        self.iter().fold(0, |acc, &elem| {
            acc + <N as AsPrimitive<usize>>::as_(elem.min(N::one()))
        })
    }

    /// Return the number of elements whose counter is zero.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use utote::Multiset;
    /// let multiset = Multiset::<u8, 4>::from_slice(&[1, 0, 0, 0]);
    /// assert_eq!(multiset.count_zero(), 3);
    /// ```
    #[cfg(not(feature = "packed_simd"))]
    #[inline]
    pub fn count_zero(&self) -> usize {
        SIZE - self.count_non_zero()
    }

    /// Check whether only one element has a non-zero count.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use utote::Multiset;
    /// let multiset = Multiset::<u8, 4>::from_slice(&[0, 5, 0, 0]);
    /// assert_eq!(multiset.is_singleton(), true);
    /// ```
    #[cfg(not(feature = "packed_simd"))]
    #[inline]
    pub fn is_singleton(&self) -> bool {
        self.count_non_zero() == 1
    }

    /// Returns `true` if `self` has no elements in common with `other`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use utote::Multiset;
    /// let a = Multiset::<u8, 4>::from_slice(&[1, 2, 0, 0]);
    /// let b = Multiset::<u8, 4>::from_slice(&[0, 0, 3, 4]);
    /// assert_eq!(a.is_disjoint(&a), false);
    /// assert_eq!(a.is_disjoint(&b), true);
    /// ```
    #[cfg(not(feature = "packed_simd"))]
    #[inline]
    pub fn is_disjoint(&self, other: &Self) -> bool {
        self.iter()
            .zip(other.iter())
            .all(|(a, b)| a.min(b) == &N::zero())
    }

    /// Check whether `self` is a subset of `other`.
    ///
    /// Multiset `A` is a subset of `B` if `A[i] <= B[i]` for all `i` in `A`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use utote::Multiset;
    /// let a = Multiset::<u8, 4>::from_slice(&[1, 2, 0, 0]);
    /// let b = Multiset::<u8, 4>::from_slice(&[1, 3, 0, 0]);
    /// assert_eq!(a.is_subset(&a), true);
    /// assert_eq!(a.is_subset(&b), true);
    /// ```
    #[cfg(not(feature = "packed_simd"))]
    #[inline]
    pub fn is_subset(&self, other: &Self) -> bool {
        self.iter().zip(other.iter()).all(|(a, b)| a <= b)
    }

    /// Check whether `self` is a superset of `other`.
    ///
    /// Multiset `A` is a superset of `B` if `A[i] >= B[i]` for all `i` in `A`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use utote::Multiset;
    /// let a = Multiset::<u8, 4>::from_slice(&[1, 2, 0, 0]);
    /// let b = Multiset::<u8, 4>::from_slice(&[1, 1, 0, 0]);
    /// assert_eq!(a.is_superset(&a), true);
    /// assert_eq!(a.is_superset(&b), true);
    /// ```
    #[cfg(not(feature = "packed_simd"))]
    #[inline]
    pub fn is_superset(&self, other: &Self) -> bool {
        self.iter().zip(other.iter()).all(|(a, b)| a >= b)
    }

    /// Check whether `self` is a proper subset of `other`.
    ///
    /// Multiset `A` is a proper subset of `B` if `A[i] <= B[i]` for all `i` in `A` and
    /// there exists `j` such that `A[j] < B[j]`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use utote::Multiset;
    /// let a = Multiset::<u8, 4>::from_slice(&[1, 2, 0, 0]);
    /// let b = Multiset::<u8, 4>::from_slice(&[1, 3, 0, 0]);
    /// assert_eq!(a.is_proper_subset(&a), false);
    /// assert_eq!(a.is_proper_subset(&b), true);
    /// ```
    #[cfg(not(feature = "packed_simd"))]
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
    /// use utote::Multiset;
    /// let a = Multiset::<u8, 4>::from_slice(&[1, 2, 0, 0]);
    /// let b = Multiset::<u8, 4>::from_slice(&[1, 1, 0, 0]);
    /// assert_eq!(a.is_proper_superset(&a), false);
    /// assert_eq!(a.is_proper_superset(&b), true);
    /// ```
    #[cfg(not(feature = "packed_simd"))]
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
    /// use utote::Multiset;
    /// let a = Multiset::<u8, 4>::from_slice(&[1, 2, 4, 0]);
    /// let b = Multiset::<u8, 4>::from_slice(&[1, 3, 0, 0]);
    /// assert_eq!(a.is_any_lesser(&a), false);
    /// assert_eq!(a.is_any_lesser(&b), true);
    /// ```
    #[cfg(not(feature = "packed_simd"))]
    #[inline]
    pub fn is_any_lesser(&self, other: &Self) -> bool {
        self.iter().zip(other.iter()).any(|(a, b)| a < b)
    }

    /// Check whether any element of `self` is greater than an element of `other`.
    ///
    /// True if there exists some `i` such that `A[i] > B[i]`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use utote::Multiset;
    /// let a = Multiset::<u8, 4>::from_slice(&[1, 2, 0, 0]);
    /// let b = Multiset::<u8, 4>::from_slice(&[1, 1, 4, 0]);
    /// assert_eq!(a.is_any_greater(&a), false);
    /// assert_eq!(a.is_any_greater(&b), true);
    /// ```
    #[cfg(not(feature = "packed_simd"))]
    #[inline]
    pub fn is_any_greater(&self, other: &Self) -> bool {
        self.iter().zip(other.iter()).any(|(a, b)| a > b)
    }

    /// Check whether all elements have a count of zero.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use utote::Multiset;
    /// let multiset = Multiset::<u8, 4>::from_slice(&[0, 0, 0, 0]);
    /// assert_eq!(multiset.is_empty(), true);
    /// assert_eq!(Multiset::<u8, 4>::empty().is_empty(), true);
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data == [N::zero(); SIZE]
    }

    /// The total or cardinality of a multiset is the sum of all its elements member
    /// counts.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use utote::Multiset;
    /// let multiset = Multiset::<u8, 4>::from_slice(&[1, 2, 3, 4]);
    /// assert_eq!(multiset.total(), 10);
    /// ```
    ///
    /// ### Notes:
    /// - This may overflow.
    #[cfg(not(feature = "packed_simd"))]
    #[inline]
    pub fn total(&self) -> usize {
        self.iter()
            .map(|e| <N as AsPrimitive<usize>>::as_(*e))
            .sum()
    }

    /// Returns a tuple containing the (element, corresponding largest counter) in the
    /// multiset.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use utote::Multiset;
    /// let multiset = Multiset::<u8, 4>::from_slice(&[2, 0, 5, 3]);
    /// assert_eq!(multiset.argmax(), (2, 5));
    /// ```
    #[inline]
    pub fn argmax(&self) -> (usize, N) {
        // iter cannot be empty, so it's fine to unwrap
        let (index, max) = self.iter().enumerate().max_by_key(|(_, e)| *e).unwrap();
        (index, *max)
    }

    /// Returns the element with the largest count in the multiset.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use utote::Multiset;
    /// let multiset = Multiset::<u8, 4>::from_slice(&[2, 0, 5, 3]);
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
    /// use utote::Multiset;
    /// let multiset = Multiset::<u8, 4>::from_slice(&[2, 0, 5, 3]);
    /// assert_eq!(multiset.max(), 5);
    /// ```
    #[inline]
    pub fn max(&self) -> N {
        // iter cannot be empty, so it's fine to unwrap
        *self.iter().max().unwrap()
    }

    /// Returns a tuple containing the (element, corresponding smallest counter) in
    /// the multiset.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use utote::Multiset;
    /// let multiset = Multiset::<u8, 4>::from_slice(&[2, 0, 5, 3]);
    /// assert_eq!(multiset.argmin(), (1, 0));
    /// ```
    #[inline]
    pub fn argmin(&self) -> (usize, N) {
        // iter cannot be empty, so it's fine to unwrap
        let (index, min) = self.iter().enumerate().min_by_key(|(_, e)| *e).unwrap();
        (index, *min)
    }

    /// Returns the element with the smallest count in the multiset.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use utote::Multiset;
    /// let multiset = Multiset::<u8, 4>::from_slice(&[2, 0, 5, 3]);
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
    /// use utote::Multiset;
    /// let multiset = Multiset::<u8, 4>::from_slice(&[2, 0, 5, 3]);
    /// assert_eq!(multiset.min(), 0);
    /// ```
    #[inline]
    pub fn min(&self) -> N {
        // iter cannot be empty, so it's fine to unwrap
        *self.iter().min().unwrap()
    }

    /// Set all element counts, except for the given `elem`, to zero.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use utote::Multiset;
    /// let mut multiset = Multiset::<u8, 4>::from_slice(&[2, 0, 5, 3]);
    /// multiset.choose(2);
    /// let result = Multiset::<u8, 4>::from_slice(&[0, 0, 5, 0]);
    /// assert_eq!(multiset, result);
    /// ```
    #[inline]
    pub fn choose(&mut self, elem: usize) {
        let mut res = [N::zero(); SIZE];
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
    /// use utote::Multiset;
    /// use rand::prelude::*;
    /// let rng = &mut SmallRng::seed_from_u64(thread_rng().next_u64());
    /// let mut multiset = Multiset::<u8, 4>::from_slice(&[2, 0, 5, 3]);
    /// multiset.choose_random(rng);
    /// assert_eq!(multiset.is_singleton(), true);
    /// ```
    #[cfg(feature = "rand")]
    #[cfg(not(feature = "packed_simd"))]
    #[inline]
    pub fn choose_random<T: RngCore>(&mut self, rng: &mut T) {
        let total = self.total();
        if total == 0 {
            return;
        }
        let choice_value = rng.gen_range(1..=total);
        let mut res = [N::zero(); SIZE];
        let mut acc = 0;
        for (i, elem) in self.iter().enumerate() {
            acc += <N as AsPrimitive<usize>>::as_(*elem);
            if acc >= choice_value {
                // Safety: `i` cannot be outside of `res`.
                unsafe { *res.get_unchecked_mut(i) = *elem }
                break;
            }
        }
        self.data = res
    }

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
    /// use utote::Multiset;
    /// let multiset = Multiset::<u8, 4>::from_slice(&[2, 1, 1, 0]);
    /// let result = multiset.collision_entropy();
    /// // approximate: result == 1.415037499278844
    /// ```
    #[cfg(not(feature = "packed_simd"))]
    #[inline]
    pub fn collision_entropy(&self) -> f64 {
        let total: f64 = self.total().as_(); // todo: note use of .as_()
        -self
            .into_iter()
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
    /// use utote::Multiset;
    /// let multiset = Multiset::<u8, 4>::from_slice(&[2, 1, 1, 0]);
    /// let result = multiset.shannon_entropy();
    /// // approximate: result == 1.0397207708399179
    /// ```
    #[cfg(not(feature = "packed_simd"))]
    #[inline]
    pub fn shannon_entropy(&self) -> f64 {
        let total: f64 = self.total().as_();
        -self.into_iter().fold(0.0, |acc, frequency| {
            if frequency > &N::zero() {
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
    use approx::assert_relative_eq;
    #[cfg(feature = "rand")]
    use rand::prelude::*;

    #[test]
    fn test_zip_map() {
        let set1: Multiset<u8, 4> = Multiset::from_slice(&[1, 5, 2, 8]);
        let set2: Multiset<u8, 4> = Multiset::from_slice(&[1, 5, 2, 8]);
        let result = set1.zip_map(&set2, |e1, e2| e1 + e2);
        let expected = Multiset::from_slice(&[2, 10, 4, 16]);
        assert_eq!(result, expected)
    }

    #[test]
    fn test_len() {
        assert_eq!(Multiset::<u16, 9>::len(), 9)
    }

    #[test]
    fn test_empty() {
        let result = Multiset::<u32, 7>::empty();
        let expected = Multiset::from([0u32; 7]);
        assert_eq!(result, expected)
    }

    #[test]
    fn test_repeat() {
        let result = Multiset::<u8, 4>::repeat(3);
        let expected = Multiset::from_array([3u8; 4]);
        assert_eq!(result, expected)
    }

    #[test]
    fn from_array() {
        let set: Multiset<u16, 3> = Multiset::from([5, 4, 3]);
        assert_eq!(set.get(1), Some(&4))
    }

    #[test]
    fn from_slice() {
        let set: Multiset<u16, 3> = Multiset::from(&[5, 4, 3]);
        assert_eq!(set.get(1), Some(&4))
    }

    #[test]
    fn to_array() {
        let set: Multiset<u16, 3> = Multiset::from(&[5, 4, 3]);
        let arr: [u16; 3] = set.into();
        assert_eq!(arr, [5, 4, 3])
    }

    #[test]
    fn test_clear() {
        let mut set = Multiset::<u8, 4>::repeat(3);
        set.clear();
        let expected = Multiset::<u8, 4>::empty();
        assert_eq!(set, expected)
    }

    #[test]
    fn test_contains() {
        let set = Multiset::<u8, 5>::from([1, 0, 1, 0, 1]);
        [0, 2, 4].iter().for_each(|elem| {
            assert!(set.contains(*elem));
        });

        [1, 3].iter().for_each(|elem| {
            assert!(!set.contains(*elem));
        });

        [5, 10].iter().for_each(|elem| {
            assert!(!set.contains(*elem));
        });
    }

    #[test]
    fn test_contains_unchecked() {
        let set = Multiset::<u8, 5>::from([1, 0, 1, 0, 1]);
        unsafe {
            [0, 2, 4].iter().for_each(|elem| {
                assert!(set.contains_unchecked(*elem));
            });

            [1, 3].iter().for_each(|elem| {
                assert!(!set.contains_unchecked(*elem));
            });
        }
    }

    #[test]
    fn test_insert() {
        let mut set = Multiset::<u8, 4>::from([1, 1, 1, 1]);
        assert_eq!(set.get(2), Some(&1));
        set.insert(2, 5);
        assert_eq!(set.get(2), Some(&5))
    }

    #[test]
    fn test_insert_unchecked() {
        let mut set = Multiset::<u8, 4>::from([1, 1, 1, 1]);
        assert_eq!(set.get(2), Some(&1));
        unsafe { set.insert_unchecked(2, 5) };
        assert_eq!(set.get(2), Some(&5));
    }

    #[test]
    fn test_remove() {
        let mut set = Multiset::<u64, 3>::from([2, 3, 4]);
        assert_eq!(set.get(1), Some(&3));
        set.remove(1);
        assert_eq!(set.get(1), Some(&0))
    }

    #[test]
    fn test_remove_unchecked() {
        let mut set = Multiset::<u64, 3>::from([2, 3, 4]);
        assert_eq!(set.get(1), Some(&3));
        unsafe { set.remove_unchecked(1) };
        assert_eq!(set.get(1), Some(&0))
    }

    #[test]
    fn test_get() {
        let set = Multiset::<usize, 4>::from([6, 7, 8, 9]);
        assert_eq!(set.get(1), Some(&7));
        assert_eq!(set.get(9), None)
    }

    #[test]
    fn test_get_unchecked() {
        let set = Multiset::<usize, 4>::from([6, 7, 8, 9]);
        unsafe { assert_eq!(set.get_unchecked(1), 7) }
    }

    #[test]
    fn test_intersection() {
        let a = Multiset::<u8, 4>::from([1, 2, 5, 6]);
        let b = Multiset::from([0, 1, 8, 9]);
        let c = Multiset::from([0, 1, 5, 6]);
        assert_eq!(c, a.intersection(&b))
    }

    #[test]
    fn test_union() {
        let a = Multiset::<u8, 4>::from([1, 2, 5, 6]);
        let b = Multiset::from([0, 1, 8, 9]);
        let c = Multiset::from([1, 2, 8, 9]);
        assert_eq!(c, a.union(&b))
    }

    #[test]
    fn test_count_zero() {
        let set = Multiset::<u16, 7>::from([0, 1, 3, 0, 8, 0, 0]);
        assert_eq!(set.count_zero(), 4)
    }

    #[test]
    fn test_count_non_zero() {
        let set = Multiset::<u16, 7>::from([0, 1, 3, 0, 8, 0, 0]);
        assert_eq!(set.count_non_zero(), 3)
    }

    #[test]
    fn test_is_empty() {
        let a = Multiset::from([1u8; 3]);
        let b = Multiset::<u8, 3>::empty();
        assert!(!a.is_empty());
        assert!(b.is_empty());
    }

    #[test]
    fn test_is_singleton() {
        let a = Multiset::<u16, 4>::from([0, 0, 8, 0]);
        assert!(a.is_singleton());

        let b = Multiset::<u16, 4>::from([0, 2, 8, 0]);
        assert!(!b.is_singleton());

        let c = Multiset::<u16, 4>::empty();
        assert!(!c.is_singleton());
    }

    #[test]
    fn test_is_disjoint() {
        let a = Multiset::<u8, 4>::from([1, 1, 0, 0]);
        let b = Multiset::from([0, 0, 1, 1]);
        let c = Multiset::from([0, 1, 1, 0]);

        assert!(a.is_disjoint(&b));
        assert!(!a.is_disjoint(&c));
    }

    #[test]
    fn test_is_subset() {
        let a = Multiset::<u8, 3>::from([1, 1, 1]);
        let b = Multiset::from([2, 2, 2]);
        assert!(a.is_subset(&b));
        assert!(!b.is_subset(&a));

        let c = Multiset::from([0, 1, 2]);
        assert!(!a.is_subset(&c));
        assert!(!c.is_subset(&a));

        let d = Multiset::from([1, 1, 1]);
        assert!(a.is_subset(&d));
        assert!(d.is_subset(&a));
    }

    #[test]
    fn test_is_superset() {
        let a = Multiset::<u8, 3>::from([1, 1, 1]);
        let b = Multiset::from([2, 2, 2]);
        assert!(!a.is_superset(&b));
        assert!(b.is_superset(&a));

        let c = Multiset::from([0, 1, 2]);
        assert!(!a.is_superset(&c));
        assert!(!c.is_superset(&a));

        let d = Multiset::from([1, 1, 1]);
        assert!(a.is_superset(&d));
        assert!(d.is_superset(&a));
    }

    #[test]
    fn test_is_proper_subset() {
        let a = Multiset::from([1u8; 3]);
        let b = Multiset::from([2u8; 3]);
        assert!(a.is_proper_subset(&b));
        assert!(!b.is_proper_subset(&a));

        let c = Multiset::from([0u8, 1, 2]);
        assert!(!a.is_proper_subset(&c));
        assert!(!c.is_proper_subset(&a));

        let d = Multiset::from([1u8; 3]);
        assert!(!a.is_proper_subset(&d));
        assert!(!d.is_proper_subset(&a));
    }

    #[test]
    fn test_is_proper_superset() {
        let a = Multiset::from([1u8; 3]);
        let b = Multiset::from([2u8; 3]);
        assert!(!a.is_proper_superset(&b));
        assert!(b.is_proper_superset(&a));

        let c = Multiset::from([0u8, 1, 2]);
        assert!(!a.is_proper_superset(&c));
        assert!(!c.is_proper_superset(&a));

        let d = Multiset::from([1u8; 3]);
        assert!(!a.is_proper_superset(&d));
        assert!(!d.is_proper_superset(&a));
    }

    #[test]
    fn test_is_any_lesser() {
        let a = Multiset::from([1u8; 3]);
        let b = Multiset::from([2u8; 3]);
        assert!(a.is_any_lesser(&b));
        assert!(!b.is_any_lesser(&a));

        let c = Multiset::from([0u8, 1, 2]);
        assert!(a.is_any_lesser(&c));
        assert!(c.is_any_lesser(&a));

        let d = Multiset::from([1u8; 3]);
        assert!(!a.is_any_lesser(&d));
        assert!(!d.is_any_lesser(&a));
    }

    #[test]
    fn test_is_any_greater() {
        let a = Multiset::from([1u8; 3]);
        let b = Multiset::from([2u8; 3]);
        assert!(!a.is_any_greater(&b));
        assert!(b.is_any_greater(&a));

        let c = Multiset::from([0u8, 1, 2]);
        assert!(a.is_any_greater(&c));
        assert!(c.is_any_greater(&a));

        let d = Multiset::from([1u8; 3]);
        assert!(!a.is_any_greater(&d));
        assert!(!d.is_any_greater(&a));
    }

    #[test]
    fn test_total() {
        let set = Multiset::from([1u8, 2, 3, 4]);
        assert_eq!(set.total(), 10)
    }

    #[test]
    fn test_argmax() {
        let set = Multiset::from([1u8, 0, 3, 1]);
        let expected = (2, 3);
        assert_eq!(set.argmax(), expected)
    }

    #[test]
    fn test_imax() {
        let set = Multiset::from([1u8, 0, 3, 1]);
        let expected = 2;
        assert_eq!(set.imax(), expected)
    }

    #[test]
    fn test_max() {
        let set = Multiset::from([1u8, 0, 3, 1]);
        let expected = 3;
        assert_eq!(set.max(), expected)
    }

    #[test]
    fn test_argmin() {
        let set = Multiset::from([1u8, 0, 3, 1]);
        let expected = (1, 0);
        assert_eq!(set.argmin(), expected)
    }

    #[test]
    fn test_imin() {
        let set = Multiset::from([1u8, 0, 3, 1]);
        let expected = 1;
        assert_eq!(set.imin(), expected)
    }

    #[test]
    fn test_min() {
        let set = Multiset::from([1u8, 0, 3, 1]);
        let expected = 0;
        assert_eq!(set.min(), expected)
    }

    #[test]
    fn test_choose() {
        let mut set = Multiset::from([1u8, 2, 3, 4, 5]);
        let expected = Multiset::from([0u8, 0, 0, 4, 0]);
        set.choose(3);
        assert_eq!(set, expected)
    }

    #[cfg(feature = "rand")]
    #[test]
    fn test_choose_random() {
        let mut result1 = Multiset::from([1u8, 2, 3, 4, 5]);
        let test_rng1 = &mut SmallRng::seed_from_u64(thread_rng().next_u64());
        result1.choose_random(test_rng1);
        assert!(result1.is_singleton() && result1.is_subset(&Multiset::from([1u8, 2, 3, 4, 5])));

        let mut result2 = Multiset::from([1u8, 2, 3, 4, 5]);
        let test_rng2 = &mut SmallRng::seed_from_u64(thread_rng().next_u64());
        result2.choose_random(test_rng2);
        assert!(result2.is_singleton() && result2.is_subset(&Multiset::from([1u8, 2, 3, 4, 5])));
    }

    #[cfg(feature = "rand")]
    #[test]
    fn test_choose_random_empty() {
        let mut result = Multiset::<u32, 5>::empty();
        let test_rng = &mut SmallRng::seed_from_u64(thread_rng().next_u64());
        result.choose_random(test_rng);
        let expected = Multiset::empty();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_collision_entropy() {
        let simple: Multiset<u8, 4> = Multiset::from([200, 0, 0, 0]);
        assert_eq!(simple.collision_entropy(), 0.0);

        let set: Multiset<u8, 4> = Multiset::from([2, 1, 1, 0]);
        assert_relative_eq!(
            set.collision_entropy(),
            1.415037499278844,
            epsilon = f64::EPSILON
        );
    }

    #[test]
    fn test_shannon_entropy1() {
        let a: Multiset<u8, 4> = Multiset::from([200, 0, 0, 0]);
        assert_eq!(a.shannon_entropy(), 0.0);

        let b: Multiset<u8, 4> = Multiset::from([2, 1, 1, 0]);
        assert_relative_eq!(
            b.shannon_entropy(),
            1.0397207708399179,
            epsilon = f64::EPSILON
        );
    }
}
