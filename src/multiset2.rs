#[cfg(feature = "packed_simd")]
use crate::simd_impl::SimdTypes;
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

#[doc(hidden)]
// Just arithmetic impls for unsigned ints.
pub trait CounterArithmetic:
    Sized + Add + AddAssign + Div + DivAssign + Mul + MulAssign + Rem + RemAssign + Sub + SubAssign
{
    // empty
}

impl CounterArithmetic for u8 {}
impl CounterArithmetic for u16 {}
impl CounterArithmetic for u32 {}
impl CounterArithmetic for u64 {}
impl CounterArithmetic for usize {}

#[doc(hidden)]
// Collects various properties into one trait for more concise implementations. All of these
// properties are implemented by unsigned integers.
pub trait CounterBasic:
    Clone
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
pub trait Counter: CounterArithmetic + CounterBasic + SimdTypes {
    // empty
}

impl Counter for u8 {}
impl Counter for u16 {}
impl Counter for u32 {}
impl Counter for u64 {}
impl Counter for usize {}

/// Multiset! yay
#[derive(Debug)]
pub struct Multiset2<N: Counter, const SIZE: usize> {
    pub(crate) data: [N; SIZE],
}

impl<N: Counter, const SIZE: usize> Hash for Multiset2<N, SIZE> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data.hash(state)
    }
}

impl<N: Counter, const SIZE: usize> Clone for Multiset2<N, SIZE> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<N: Counter, const SIZE: usize> Copy for Multiset2<N, SIZE> {}

impl<N: Counter, const SIZE: usize> PartialEq for Multiset2<N, SIZE> {
    // Array compare will always use memcmp because N will be a unit, so no need for SIMD
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<N: Counter, const SIZE: usize> Eq for Multiset2<N, SIZE> {}

impl<N: Counter, const SIZE: usize> FromIterator<N> for Multiset2<N, SIZE> {
    #[inline]
    fn from_iter<T: IntoIterator<Item = N>>(iter: T) -> Self {
        let mut res: Self = unsafe { Multiset2::new_uninitialized() };
        let it = iter.into_iter().chain(std::iter::repeat(N::zero()));
        res.data.iter_mut().zip(it).for_each(|(r, e)| *r = e);
        res
    }
}

impl<'a, N: 'a + Counter, const SIZE: usize> FromIterator<&'a N> for Multiset2<N, SIZE> {
    #[inline]
    fn from_iter<T: IntoIterator<Item = &'a N>>(iter: T) -> Self {
        iter.into_iter().copied().collect()
    }
}

impl<N: Counter, const SIZE: usize> IntoIterator for Multiset2<N, SIZE> {
    type Item = N;
    type IntoIter = std::array::IntoIter<N, SIZE>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        std::array::IntoIter::new(self.data)
    }
}

impl<'a, N: Counter, const SIZE: usize> IntoIterator for &'a Multiset2<N, SIZE> {
    type Item = &'a N;
    type IntoIter = Iter<'a, N>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<N: Counter, const SIZE: usize> PartialOrd for Multiset2<N, SIZE> {
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

impl<N: Counter, const SIZE: usize> Default for Multiset2<N, SIZE> {
    #[inline]
    fn default() -> Self {
        Multiset2 {
            data: [N::default(); SIZE],
        }
    }
}

// Ops implementations
// todo: use SIMD for ops?

impl<N: Counter, const SIZE: usize> Add for Multiset2<N, SIZE> {
    type Output = Multiset2<N, SIZE>;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        self.into_iter()
            .zip(rhs.into_iter())
            .map(|(a, b)| a + b)
            .collect()
    }
}

impl<N: Counter, const SIZE: usize> AddAssign for Multiset2<N, SIZE> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.iter_mut()
            .zip(rhs.into_iter())
            .for_each(|(l, r)| *l += r);
    }
}

impl<N: Counter, const SIZE: usize> Div for Multiset2<N, SIZE> {
    type Output = Multiset2<N, SIZE>;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        self.into_iter()
            .zip(rhs.into_iter())
            .map(|(a, b)| a / b)
            .collect()
    }
}

impl<N: Counter, const SIZE: usize> DivAssign for Multiset2<N, SIZE> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        self.iter_mut()
            .zip(rhs.into_iter())
            .for_each(|(l, r)| *l /= r);
    }
}

impl<N: Counter, const SIZE: usize> Mul for Multiset2<N, SIZE> {
    type Output = Multiset2<N, SIZE>;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self.into_iter()
            .zip(rhs.into_iter())
            .map(|(a, b)| a * b)
            .collect()
    }
}

impl<N: Counter, const SIZE: usize> MulAssign for Multiset2<N, SIZE> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.iter_mut()
            .zip(rhs.into_iter())
            .for_each(|(l, r)| *l *= r);
    }
}

impl<N: Counter, const SIZE: usize> Rem for Multiset2<N, SIZE> {
    type Output = Multiset2<N, SIZE>;

    #[inline]
    fn rem(self, rhs: Self) -> Self::Output {
        self.into_iter()
            .zip(rhs.into_iter())
            .map(|(a, b)| a % b)
            .collect()
    }
}

impl<N: Counter, const SIZE: usize> RemAssign for Multiset2<N, SIZE> {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        self.iter_mut()
            .zip(rhs.into_iter())
            .for_each(|(l, r)| *l %= r);
    }
}

impl<N: Counter, const SIZE: usize> Sub for Multiset2<N, SIZE> {
    type Output = Multiset2<N, SIZE>;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        self.into_iter()
            .zip(rhs.into_iter())
            .map(|(a, b)| a - b)
            .collect()
    }
}

impl<N: Counter, const SIZE: usize> SubAssign for Multiset2<N, SIZE> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.iter_mut()
            .zip(rhs.into_iter())
            .for_each(|(l, r)| *l -= r);
    }
}

// Array specific

impl<N: Counter, const SIZE: usize> Multiset2<N, SIZE> {
    /// The number of elements in the multiset.
    pub const SIZE: usize = SIZE;

    /// The number of elements in the multiset.
    #[inline]
    pub fn len() -> usize {
        SIZE
    }

    #[inline]
    pub(crate) unsafe fn new_uninitialized() -> Self {
        Multiset2 {
            data: mem::MaybeUninit::<[N; SIZE]>::uninit().assume_init(),
        }
    }

    #[inline]
    pub(crate) fn zip_map<N2, N3, F>(
        &self,
        other: &Multiset2<N2, SIZE>,
        mut f: F,
    ) -> Multiset2<N3, SIZE>
    where
        N2: Counter,
        N3: Counter,
        F: FnMut(N, N2) -> N3,
    {
        let mut res = unsafe { Multiset2::new_uninitialized() };
        res.data
            .iter_mut()
            .zip(self.data.iter().zip(other.data.iter()))
            .for_each(|(r, (a, b))| *r = f(*a, *b));
        res
    }

    /// Returns a Multiset of the given array size with all elements set to zero.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use utote::MSu8;
    /// let multiset = MSu8::<4>::empty();
    /// ```
    #[inline]
    pub fn empty() -> Self {
        Multiset2 {
            data: [N::zero(); SIZE],
        }
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
    pub fn repeat(elem: N) -> Self {
        Multiset2 { data: [elem; SIZE] }
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

    /// Returns a Multiset from a slice of the given array size.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use utote::MSu8;
    /// let multiset = MSu8::<4>::from_slice(&[1, 2, 3, 4]);
    /// ```
    #[inline]
    pub fn from_slice(slice: &[N]) -> Self {
        slice.iter().collect()
    }

    /// blah
    #[inline]
    pub fn from_array(data: [N; SIZE]) -> Self {
        Multiset2 { data }
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
        self.data = [N::zero(); SIZE]
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
        elem < SIZE && unsafe { self.data.get_unchecked(elem) > &N::zero() }
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
        self.data.get_unchecked(elem) > &N::zero()
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
    pub unsafe fn insert_unchecked(&mut self, elem: usize, amount: N) {
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
            unsafe { *self.data.get_unchecked_mut(elem) = N::zero() };
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
        *self.data.get_unchecked_mut(elem) = N::zero()
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
    pub fn get(&self, elem: usize) -> Option<&N> {
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
    /// use utote::MSu8;
    /// let a = MSu8::<4>::from_slice(&[1, 2, 0, 0]);
    /// let b = MSu8::<4>::from_slice(&[0, 2, 3, 0]);
    /// let c = MSu8::<4>::from_slice(&[0, 2, 0, 0]);
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
    /// use utote::MSu8;
    /// let a = MSu8::<4>::from_slice(&[1, 2, 0, 0]);
    /// let b = MSu8::<4>::from_slice(&[0, 2, 3, 0]);
    /// let c = MSu8::<4>::from_slice(&[1, 2, 3, 0]);
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
    /// use utote::MSu8;
    /// let multiset = MSu8::<4>::from_slice(&[1, 0, 0, 0]);
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
    /// use utote::MSu8;
    /// let multiset = MSu8::<4>::from_slice(&[1, 0, 0, 0]);
    /// assert_eq!(multiset.count_zero(), 3);
    /// ```
    #[cfg(not(feature = "packed_simd"))]
    #[inline]
    pub fn count_zero(&self) -> usize
    where
        N: AsPrimitive<usize>,
    {
        SIZE - self.count_non_zero()
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
    #[cfg(not(feature = "packed_simd"))]
    #[inline]
    pub fn is_singleton(&self) -> bool
    where
        N: AsPrimitive<usize>,
    {
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
    /// use utote::MSu8;
    /// let a = MSu8::<4>::from_slice(&[1, 2, 0, 0]);
    /// let b = MSu8::<4>::from_slice(&[1, 3, 0, 0]);
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
    /// use utote::MSu8;
    /// let a = MSu8::<4>::from_slice(&[1, 2, 0, 0]);
    /// let b = MSu8::<4>::from_slice(&[1, 1, 0, 0]);
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
    /// use utote::MSu8;
    /// let a = MSu8::<4>::from_slice(&[1, 2, 0, 0]);
    /// let b = MSu8::<4>::from_slice(&[1, 3, 0, 0]);
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
    /// use utote::MSu8;
    /// let a = MSu8::<4>::from_slice(&[1, 2, 0, 0]);
    /// let b = MSu8::<4>::from_slice(&[1, 1, 0, 0]);
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
    /// use utote::MSu8;
    /// let a = MSu8::<4>::from_slice(&[1, 2, 4, 0]);
    /// let b = MSu8::<4>::from_slice(&[1, 3, 0, 0]);
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
    /// use utote::MSu8;
    /// let a = MSu8::<4>::from_slice(&[1, 2, 0, 0]);
    /// let b = MSu8::<4>::from_slice(&[1, 1, 4, 0]);
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
    /// use utote::MSu8;
    /// let multiset = MSu8::<4>::from_slice(&[0, 0, 0, 0]);
    /// assert_eq!(multiset.is_empty(), true);
    /// assert_eq!(MSu8::<4>::empty().is_empty(), true);
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
    /// use utote::MSu8;
    /// let multiset = MSu8::<4>::from_slice(&[1, 2, 3, 4]);
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
    /// use utote::MSu8;
    /// let multiset = MSu8::<4>::from_slice(&[2, 0, 5, 3]);
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
    /// use utote::MSu8;
    /// let multiset = MSu8::<4>::from_slice(&[2, 0, 5, 3]);
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
    pub fn min(&self) -> N {
        // iter cannot be empty, so it's fine to unwrap
        *self.iter().min().unwrap()
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
    /// use utote::MSu8;
    /// use rand::prelude::*;
    /// let rng = &mut SmallRng::seed_from_u64(thread_rng().next_u64());
    /// let mut multiset = MSu8::<4>::from_slice(&[2, 0, 5, 3]);
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
    /// use utote::MSu8;
    /// let multiset = MSu8::<4>::from_slice(&[2, 1, 1, 0]);
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
    /// use utote::MSu8;
    /// let multiset = MSu8::<4>::from_slice(&[2, 1, 1, 0]);
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
    use rand::prelude::*;

    tests_x4!(multiset2u16_4, Multiset2<u16, 4>, u16);
    // tests_x8!(multiset2u16_8, Multiset2<u16, 8>, u16);
}
