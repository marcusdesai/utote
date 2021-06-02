#[cfg(feature = "simd")]
use crate::simd::SimdTypes;
use num_traits::{AsPrimitive, One, Unsigned, Zero};
#[cfg(all(not(feature = "simd"), feature = "rand"))]
use rand::{Rng, RngCore};
#[cfg(not(feature = "simd"))]
use std::cmp::Ordering;
use std::fmt::{Debug, Display, Formatter, Result};
use std::hash::{Hash, Hasher};
use std::iter::FromIterator;
use std::mem;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign,
};
use std::slice::{Iter, IterMut, SliceIndex};

mod sealed {
    pub trait Sealed {}

    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
    impl Sealed for usize {}
}

// Just arithmetic impls for ints.
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

// Collects various properties into one trait for more concise implementations.
// All of these properties are implemented by unsigned integers.
#[doc(hidden)]
pub trait CounterBasic:
    sealed::Sealed
    + Clone
    + Copy
    + Debug
    + Default
    + Display
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

#[cfg(not(feature = "simd"))]
/// The Counter trait simplifies the use of Multiset with generics. This trait
/// is sealed and not implementable outside of this crate.
pub trait Counter: sealed::Sealed + CounterArithmetic + CounterBasic {
    // empty
}

#[cfg(feature = "simd")]
#[doc(hidden)]
pub trait Counter: sealed::Sealed + CounterArithmetic + CounterBasic + SimdTypes {
    // empty
}

impl Counter for u8 {}
impl Counter for u16 {}
impl Counter for u32 {}
impl Counter for u64 {}
impl Counter for usize {}

/// A stack allocated multiset of unsigned integers.
///
/// Each index in the `Multiset` array corresponds to an element in the
/// multiset. If the value at an index is 0 then there are no instances of the
/// corresponding uint in the multiset. Because of this correspondence the
/// order of the counts must remain consistent for any `Multiset` to behave
/// consistently.
///
/// # Examples
///
/// ```
/// use utote::Multiset;
///
/// // A multiset of 5 elements, which can be counted up to u8::MAX
/// let mut multiset = Multiset::from([0u8, 3, 4, 0, 5]);
/// assert_eq!(multiset.total(), 12);
///
/// let equivalent_multiset = Multiset::<u8, 5>::from([0, 3, 4, 0, 5]);
/// assert_eq!(multiset, equivalent_multiset);
///
/// multiset.insert(2, 6);
/// assert_eq!(multiset, Multiset::from([0, 3, 6, 0, 5]));
///
/// for elem in multiset.iter() {
///     println!("{}", elem);
/// }
///
/// assert_eq!(multiset.contains(0), false);
/// assert_eq!(multiset.contains(1), true);
/// ```
///
/// Some common set-like operations:
///
/// ```
/// use utote::Multiset;
///
/// let ms_sub: Multiset<u32, 3> = Multiset::from([0, 1, 1]);
/// let ms_super = Multiset::from([1, 1, 2]);
///
/// assert_eq!(ms_sub.is_subset(&ms_super), true);
///
/// assert_eq!(ms_sub.union(&ms_super), Multiset::from([1, 1, 2]));
///
/// assert_eq!(ms_super.is_proper_superset(&ms_sub), true);
///
/// // Any multiset where all counters are zero is equivalent to
/// // the empty multiset.
/// let empty: Multiset<u64, 2> = Multiset::from([0, 0]);
/// assert_eq!(empty, Multiset::empty());
/// ```
///
/// # Indexing
///
/// The `Multiset` type allows accessing values by index in the same manner
/// that `Vec` does.
///
/// ```
/// use utote::Multiset;
/// let multiset = Multiset::from([1u16, 2, 3]);
/// println!("{}", multiset[1]); // it will display '2'
/// ```
///
/// As with [`slice`], [`get`] and [`get_mut`] are provided (along with
/// unchecked versions).
///
/// [`slice`]: std::slice
/// [`get`]: Multiset::get
/// [`get_mut`]: Multiset::get_mut
///
/// # Using Generically
///
/// The `Counter` trait is provided to simplify using `Multiset` generically.
/// `Counter` is implemented for all unsigned integers.
///
/// ```no_run
/// use utote::{Counter, Multiset};
///
/// fn generic_ms<T: Counter, const SIZE: usize>(ms: Multiset<T, SIZE>) {
///     println!("{}", ms.is_empty());
/// }
/// ```
pub struct Multiset<N: Counter, const SIZE: usize> {
    pub(crate) data: [N; SIZE],
}

////////////////////////////////////////////////////////////////////////////////
// Common trait implementations for Multiset
////////////////////////////////////////////////////////////////////////////////

impl<N: Counter, const SIZE: usize> Debug for Multiset<N, SIZE> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("Multiset")
            .field("data", &self.data)
            .finish()
    }
}

impl<N: Counter, const SIZE: usize> Hash for Multiset<N, SIZE> {
    #[inline]
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
    // Array compare will always use memcmp because N will be a unit, so no
    // need for SIMD
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<N: Counter, const SIZE: usize> Eq for Multiset<N, SIZE> {}

impl<N: Counter, I: SliceIndex<[N]>, const SIZE: usize> Index<I> for Multiset<N, SIZE> {
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        Index::index(&self.data, index)
    }
}

impl<N: Counter, I: SliceIndex<[N]>, const SIZE: usize> IndexMut<I> for Multiset<N, SIZE> {
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        IndexMut::index_mut(&mut self.data, index)
    }
}

impl<N: Counter, const SIZE: usize> FromIterator<N> for Multiset<N, SIZE> {
    #[inline]
    fn from_iter<T: IntoIterator<Item = N>>(iter: T) -> Self {
        // Safety: iter chained with repeated zeroes guarantees that we have
        // enough elements to fully initialise the Multiset.
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

impl<'a, N: Counter, const SIZE: usize> IntoIterator for &'a mut Multiset<N, SIZE> {
    type Item = &'a mut N;
    type IntoIter = IterMut<'a, N>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<N: Counter, const SIZE: usize> From<[N; SIZE]> for Multiset<N, SIZE> {
    #[inline]
    fn from(data: [N; SIZE]) -> Self {
        Multiset { data }
    }
}

impl<N: Counter, const SIZE: usize> From<&[N; SIZE]> for Multiset<N, SIZE> {
    #[inline]
    fn from(data: &[N; SIZE]) -> Self {
        Multiset { data: *data }
    }
}

impl<N: Counter, const SIZE: usize> From<Multiset<N, SIZE>> for [N; SIZE] {
    #[inline]
    fn from(set: Multiset<N, SIZE>) -> Self {
        set.data
    }
}

impl<N: Counter, const SIZE: usize> From<&[N]> for Multiset<N, SIZE> {
    #[inline]
    fn from(slice: &[N]) -> Self {
        slice.iter().collect()
    }
}

impl<'a, N: Counter, const SIZE: usize> From<&'a Multiset<N, SIZE>> for &'a [N] {
    #[inline]
    fn from(set: &'a Multiset<N, SIZE>) -> Self {
        &set.data
    }
}

impl<'a, N: Counter, const SIZE: usize> From<&'a mut Multiset<N, SIZE>> for &'a mut [N] {
    fn from(set: &'a mut Multiset<N, SIZE>) -> Self {
        &mut set.data
    }
}

/// Partial order based on proper sub/super sets
#[cfg(not(feature = "simd"))]
impl<N: Counter, const SIZE: usize> PartialOrd for Multiset<N, SIZE> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let mut order: Ordering = Ordering::Equal;
        for (e_self, e_other) in self.iter().zip(other.iter()) {
            match order {
                Ordering::Equal if e_self < e_other => order = Ordering::Less,
                Ordering::Equal if e_self > e_other => order = Ordering::Greater,
                Ordering::Less if e_self > e_other => return None,
                Ordering::Greater if e_self < e_other => return None,
                _ => (),
            }
        }
        Some(order)
    }

    #[inline]
    fn lt(&self, other: &Self) -> bool {
        self.is_proper_subset(other)
    }

    #[inline]
    fn le(&self, other: &Self) -> bool {
        self.is_subset(other)
    }

    #[inline]
    fn gt(&self, other: &Self) -> bool {
        self.is_proper_superset(other)
    }

    #[inline]
    fn ge(&self, other: &Self) -> bool {
        self.is_superset(other)
    }
}

// todo: implement Ord using Dershowitz–Manna ordering?
//  See https://en.wikipedia.org/wiki/Dershowitz%E2%80%93Manna_ordering
// relies on ordering of the integers, but the int order becomes in essence a
// lexicographic order when we use the ints as aliases of other data. So if
// there is no partial-order to that data being aliased then the D-M ordering
// should not be used.

impl<N: Counter, const SIZE: usize> Default for Multiset<N, SIZE> {
    #[inline]
    fn default() -> Self {
        Multiset {
            data: [N::default(); SIZE],
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Common ops implementations for Multiset
////////////////////////////////////////////////////////////////////////////////

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

impl<N: Counter, const SIZE: usize> Add<N> for Multiset<N, SIZE> {
    type Output = Multiset<N, SIZE>;

    #[inline]
    fn add(self, rhs: N) -> Self::Output {
        self.into_iter().map(|a| a + rhs).collect()
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

impl<N: Counter, const SIZE: usize> AddAssign<N> for Multiset<N, SIZE> {
    #[inline]
    fn add_assign(&mut self, rhs: N) {
        self.iter_mut().for_each(|l| *l += rhs);
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

impl<N: Counter, const SIZE: usize> Div<N> for Multiset<N, SIZE> {
    type Output = Multiset<N, SIZE>;

    #[inline]
    fn div(self, rhs: N) -> Self::Output {
        self.into_iter().map(|a| a / rhs).collect()
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

impl<N: Counter, const SIZE: usize> DivAssign<N> for Multiset<N, SIZE> {
    #[inline]
    fn div_assign(&mut self, rhs: N) {
        self.iter_mut().for_each(|l| *l /= rhs);
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

impl<N: Counter, const SIZE: usize> Mul<N> for Multiset<N, SIZE> {
    type Output = Multiset<N, SIZE>;

    #[inline]
    fn mul(self, rhs: N) -> Self::Output {
        self.into_iter().map(|a| a * rhs).collect()
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

impl<N: Counter, const SIZE: usize> MulAssign<N> for Multiset<N, SIZE> {
    #[inline]
    fn mul_assign(&mut self, rhs: N) {
        self.iter_mut().for_each(|l| *l *= rhs);
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

impl<N: Counter, const SIZE: usize> Rem<N> for Multiset<N, SIZE> {
    type Output = Multiset<N, SIZE>;

    #[inline]
    fn rem(self, rhs: N) -> Self::Output {
        self.into_iter().map(|a| a % rhs).collect()
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

impl<N: Counter, const SIZE: usize> RemAssign<N> for Multiset<N, SIZE> {
    #[inline]
    fn rem_assign(&mut self, rhs: N) {
        self.iter_mut().for_each(|l| *l %= rhs);
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

impl<N: Counter, const SIZE: usize> Sub<N> for Multiset<N, SIZE> {
    type Output = Multiset<N, SIZE>;

    #[inline]
    fn sub(self, rhs: N) -> Self::Output {
        self.into_iter().map(|a| a - rhs).collect()
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

impl<N: Counter, const SIZE: usize> SubAssign<N> for Multiset<N, SIZE> {
    #[inline]
    fn sub_assign(&mut self, rhs: N) {
        self.iter_mut().for_each(|l| *l -= rhs);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Inherent methods
////////////////////////////////////////////////////////////////////////////////

impl<N: Counter, const SIZE: usize> Multiset<N, SIZE> {
    /// The number of elements in the multiset.
    pub const SIZE: usize = SIZE;

    /// Returns the number of elements in the multiset.
    #[inline]
    pub fn len() -> usize {
        SIZE
    }

    /// Constructs a new Multiset.
    ///
    /// This method is equivalent to calling [`Multiset::from`] with an array
    /// of the correct type.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let multiset = Multiset::new([1u8, 2, 3, 4]);
    /// let multiset_from = Multiset::from([1, 2, 3, 4]);
    /// assert_eq!(multiset, multiset_from);
    /// ```
    #[inline]
    pub fn new(data: [N; SIZE]) -> Multiset<N, SIZE> {
        Multiset { data }
    }

    // Safety: The Multiset created by this function must only be used once the
    // Multiset.data array is guaranteed to be fully initialised.
    #[inline]
    unsafe fn new_uninitialized() -> Self {
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
        // Safety: both self and other *must* be the same length as res,
        // therefore every element of res is guaranteed to be initialised upon
        // completing the iter_mut.
        let mut res = unsafe { Multiset::new_uninitialized() };
        res.iter_mut()
            .zip(self.iter().zip(other.iter()))
            .for_each(|(r, (a, b))| *r = f(*a, *b));
        res
    }

    /// Returns a Multiset of the given array size with all element counts set
    /// to zero.
    ///
    /// [`Multiset::default`] will also construct an empty set.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let empty_multiset = Multiset::<u8, 4>::empty();
    /// assert_eq!(empty_multiset, Multiset::from([0, 0, 0, 0]));
    /// assert_eq!(empty_multiset, Multiset::default());
    /// ```
    #[inline]
    pub fn empty() -> Self {
        Multiset {
            data: [N::zero(); SIZE],
        }
    }

    /// Returns a Multiset of the given array size with all elements set to
    /// `count`.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let multiset = Multiset::<u8, 4>::repeat(5);
    /// assert_eq!(multiset, Multiset::from([5, 5, 5, 5]))
    /// ```
    #[inline]
    pub fn repeat(count: N) -> Self {
        Multiset {
            data: [count; SIZE],
        }
    }

    /// Constructs a Multiset from an iterator of elements in the multiset,
    /// incrementing the count of each element as it occurs in the iterator.
    ///
    /// # Panics
    /// If any item in the iterator is out of bounds of the Multiset.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let multiset = Multiset::<u8, 4>::from_elements(&[1, 1, 0, 2, 2, 2]);
    /// assert_eq!(multiset, Multiset::from([1, 2, 3, 0]))
    /// ```
    #[inline]
    pub fn from_elements<'a, I>(elements: I) -> Self
    where
        I: IntoIterator<Item = &'a usize>,
    {
        elements.into_iter().fold(Multiset::empty(), |mut acc, &e| {
            if e >= SIZE {
                panic!("element: {} not in Multiset (element > SIZE)", e)
            }
            // Safety: Above condition ensures `e` is not out of bounds.
            unsafe { *acc.get_unchecked_mut(e) += N::one() };
            acc
        })
    }

    /// Return an [Iter](`std::slice::Iter`) of the element counts in the
    /// Multiset.
    #[inline]
    pub fn iter(&self) -> Iter<'_, N> {
        self.data.iter()
    }

    /// Return an [IterMut](`std::slice::IterMut`) of the element counts in the
    /// Multiset.
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, N> {
        self.data.iter_mut()
    }

    /// Sets all element counts in the multiset to zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let mut multiset = Multiset::<u8, 4>::from([1, 2, 3, 4]);
    /// multiset.clear();
    /// assert_eq!(multiset.is_empty(), true);
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        self.data = [N::zero(); SIZE]
    }

    /// Returns `true` if `elem` has count > 0 in the multiset.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let multiset = Multiset::from([1u8, 2, 0, 0]);
    /// assert_eq!(multiset.contains(1), true);
    /// assert_eq!(multiset.contains(3), false);
    /// assert_eq!(multiset.contains(5), false);
    /// ```
    #[inline]
    pub fn contains(&self, elem: usize) -> bool {
        // Safety: Guaranteed by bounds check on `elem`.
        elem < SIZE && unsafe { self.get_unchecked(elem) > &N::zero() }
    }

    /// Returns `true` if `elem` has count > 0 in the multiset, without doing
    /// bounds checking.
    ///
    /// For a safe alternative see [`contains`].
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is
    /// *[undefined behavior]* even if the resulting boolean is not used.
    ///
    /// [`contains`]: Multiset::contains
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let multiset = Multiset::from([1u8, 2, 0, 0]);
    ///
    /// unsafe {
    ///     assert_eq!(multiset.contains_unchecked(1), true);
    ///     assert_eq!(multiset.contains_unchecked(3), false);
    /// }
    /// ```
    #[inline]
    pub unsafe fn contains_unchecked(&self, elem: usize) -> bool {
        self.get_unchecked(elem) > &N::zero()
    }

    /// Set the count of `elem` in the multiset to `amount`.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let mut multiset = Multiset::from([1u8, 2, 0, 0]);
    /// multiset.insert(2, 5);
    /// assert_eq!(multiset.get(2), Some(&5));
    /// ```
    #[inline]
    pub fn insert(&mut self, elem: usize, amount: N) {
        if elem < SIZE {
            // Safety: Guaranteed by bounds check on `elem`.
            unsafe { *self.get_unchecked_mut(elem) = amount };
        }
    }

    /// Set the count of `elem` in the multiset to `amount`, without doing
    /// bounds checking.
    ///
    /// For a safe alternative see [`insert`].
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is
    /// *[undefined behavior]*.
    ///
    /// [`insert`]: Multiset::insert
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let mut multiset = Multiset::from([1u8, 2, 0, 0]);
    ///
    /// unsafe {
    ///     multiset.insert_unchecked(2, 5);
    ///    assert_eq!(multiset.get(2), Some(&5));
    /// }
    /// ```
    #[inline]
    pub unsafe fn insert_unchecked(&mut self, elem: usize, amount: N) {
        *self.get_unchecked_mut(elem) = amount
    }

    /// Set the count of `elem` in the multiset to zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let mut multiset = Multiset::from([1u8, 2, 0, 0]);
    /// multiset.remove(1);
    /// assert_eq!(multiset.get(1), Some(&0));
    /// ```
    #[inline]
    pub fn remove(&mut self, elem: usize) {
        if elem < SIZE {
            // Safety: Guaranteed by bounds check on `elem`.
            unsafe { *self.get_unchecked_mut(elem) = N::zero() };
        }
    }

    /// Set the count of `elem` in the multiset to zero, without doing bounds
    /// checking.
    ///
    /// For a safe alternative see [`remove`].
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is
    /// *[undefined behavior]*.
    ///
    /// [`remove`]: Multiset::remove
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let mut multiset = Multiset::from([1u8, 2, 0, 0]);
    /// unsafe {
    ///     multiset.remove_unchecked(1);
    ///     assert_eq!(multiset.get(1), Some(&0));
    /// }
    /// ```
    #[inline]
    pub unsafe fn remove_unchecked(&mut self, elem: usize) {
        *self.get_unchecked_mut(elem) = N::zero()
    }

    /// Returns a reference to a count or subslice of counts depending on the
    /// type of index.
    ///
    /// - If given a position, returns a reference to the count at that
    ///   position or `None` if out of bounds.
    /// - If given a range, returns the subslice corresponding to that range,
    ///   or `None` if out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let multiset = Multiset::from([1u8, 2, 0, 0]);
    /// assert_eq!(multiset.get(1), Some(&2));
    /// assert_eq!(multiset.get(2..), Some(&[0, 0][..]));
    /// assert_eq!(multiset.get(5), None);
    /// assert_eq!(multiset.get(0..5), None);
    /// ```
    #[inline]
    pub fn get<I>(&self, index: I) -> Option<&I::Output>
    where
        I: SliceIndex<[N]>,
    {
        self.data.get(index)
    }

    /// Returns a mutable reference to a count or subslice of counts depending
    /// on the type of index (see [`get`]) or `None` if the index is out of
    /// bounds.
    ///
    /// [`get`]: Multiset::get
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let mut multiset = Multiset::from([1u8, 2, 4]);
    ///
    /// if let Some(elem) = multiset.get_mut(1) {
    ///     *elem = 42;
    /// }
    /// assert_eq!(multiset, Multiset::from([1u8, 42, 4]));
    /// ```
    #[inline]
    pub fn get_mut<I>(&mut self, index: I) -> Option<&mut I::Output>
    where
        I: SliceIndex<[N]>,
    {
        self.data.get_mut(index)
    }

    /// Returns a reference to a count or subslice of counts, without doing
    /// bounds checking.
    ///
    /// For a safe alternative see [`get`].
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is
    /// *[undefined behavior]* even if the resulting reference is not used.
    ///
    /// [`get`]: Multiset::get
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let multiset = Multiset::from([1u32, 2, 0, 0]);
    ///
    /// unsafe {
    ///     assert_eq!(multiset.get_unchecked(1), &2);
    /// }
    /// ```
    #[inline]
    pub unsafe fn get_unchecked<I>(&self, index: I) -> &I::Output
    where
        I: SliceIndex<[N]>,
    {
        self.data.get_unchecked(index)
    }

    /// Returns a mutable reference to a count or subslice of counts, without
    /// doing bounds checking.
    ///
    /// For a safe alternative see [`get_mut`].
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is
    /// *[undefined behavior]* even if the resulting reference is not used.
    ///
    /// [`get_mut`]: Multiset::get_mut
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let mut multiset = Multiset::from([1u8, 2, 4]);
    ///
    /// unsafe {
    ///     let elem = multiset.get_unchecked_mut(1);
    ///     *elem = 13;
    /// }
    /// assert_eq!(multiset, Multiset::from([1u8, 13, 4]));
    /// ```
    #[inline]
    pub unsafe fn get_unchecked_mut<I>(&mut self, index: I) -> &mut I::Output
    where
        I: SliceIndex<[N]>,
    {
        self.data.get_unchecked_mut(index)
    }

    /// Returns a multiset which is the intersection of `self` and `other`.
    ///
    /// The Intersection of two multisets is the counts of elements which
    /// occurs in both. `A` intersection `B` is the multiset `C` where
    /// `C[i] == min(A[i], B[i])` for all `i` in `C`.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let a = Multiset::from([1u8, 2, 0, 0]);
    /// let b = Multiset::from([0, 1, 3, 0]);
    /// let c = Multiset::from([0, 1, 0, 0]);
    /// assert_eq!(a.intersection(&b), c);
    /// ```
    #[cfg(not(feature = "simd"))]
    #[inline]
    pub fn intersection(&self, other: &Self) -> Self {
        self.zip_map(other, |e1, e2| e1.min(e2))
    }

    /// Returns a multiset which is the union of `self` and `other`.
    ///
    /// The union of two multisets is the counts of elements which occur in
    /// either, without repeats. `A` union `B` is the multiset `C` where
    /// `C[i] == max(A[i], B[i])` for all `i` in `C`.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let a = Multiset::from([1u8, 2, 0, 0]);
    /// let b = Multiset::from([0, 1, 3, 0]);
    /// let c = Multiset::from([1, 2, 3, 0]);
    /// assert_eq!(a.union(&b), c);
    /// ```
    #[cfg(not(feature = "simd"))]
    #[inline]
    pub fn union(&self, other: &Self) -> Self {
        self.zip_map(other, |e1, e2| e1.max(e2))
    }

    /// Returns a multiset which is the difference of `self` and `other`.
    ///
    /// The difference of this multiset and another is the count of elements in
    /// this and those that occur in this and other without repeats. `A`
    /// difference `B` is the multiset `C` where `C[i] == min(A[i], B[i])` if
    /// `A[i] > 0` and `B[i] > 0`, otherwise `A[i]` for all `i` in `C`.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let a = Multiset::from([1u8, 2, 0, 0]);
    /// let b = Multiset::from([0, 1, 3, 0]);
    /// let c = Multiset::from([1, 1, 0, 0]);
    /// assert_eq!(a.difference(&b), c);
    /// ```
    #[inline]
    pub fn difference(&self, other: &Self) -> Self {
        self.zip_map(other, |e1, e2| {
            if e1 > N::zero() && e2 > N::zero() {
                e1.min(e2)
            } else {
                e1
            }
        })
    }

    /// Returns a multiset which is the symmetric_difference of `self` and
    /// `other`.
    ///
    /// The symmetric_difference of two multisets is the count of elements that
    /// occur in either, and the count, without repeats that occurs in both.
    /// `A` symmetric_difference `B` is the multiset `C` where
    /// `C[i] == min(A[i], B[i])` if `A[i] > 0` and `B[i] > 0`, otherwise
    /// `max(A[i], B[i])` for all `i` in `C`.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let a = Multiset::from([1u8, 2, 0, 0]);
    /// let b = Multiset::from([0, 1, 3, 0]);
    /// let c = Multiset::from([1, 1, 3, 0]);
    /// assert_eq!(a.symmetric_difference(&b), c);
    /// ```
    #[inline]
    pub fn symmetric_difference(&self, other: &Self) -> Self {
        self.zip_map(other, |e1, e2| {
            if e1 > N::zero() && e2 > N::zero() {
                e1.min(e2)
            } else {
                e1.max(e2)
            }
        })
    }

    /// Returns the number of elements whose count is non-zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let multiset = Multiset::from([1u8, 0, 0, 0]);
    /// assert_eq!(multiset.count_non_zero(), 1);
    /// ```
    #[cfg(not(feature = "simd"))]
    #[inline]
    pub fn count_non_zero(&self) -> usize {
        self.iter().fold(0, |acc, &elem| {
            acc + <N as AsPrimitive<usize>>::as_(elem.min(N::one()))
        })
    }

    /// Returns the number of elements whose count is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let multiset = Multiset::from([1u8, 0, 0, 0]);
    /// assert_eq!(multiset.count_zero(), 3);
    /// ```
    #[cfg(not(feature = "simd"))]
    #[inline]
    pub fn count_zero(&self) -> usize {
        SIZE - self.count_non_zero()
    }

    /// Returns `true` if only one element in the multiset has a non-zero
    /// count.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let multiset = Multiset::from([0u8, 5, 0, 0]);
    /// assert_eq!(multiset.is_singleton(), true);
    /// ```
    #[cfg(not(feature = "simd"))]
    #[inline]
    pub fn is_singleton(&self) -> bool {
        self.count_non_zero() == 1
    }

    /// Returns `true` if `self` is disjoint to `other`.
    ///
    /// Multiset `A` is disjoint to `B` if `A` has no elements in common with
    /// `B`. `A` is disjoint `B` if `min(A[i], B[i]) == 0` for all `i` in
    /// `A`.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let a = Multiset::from([1u8, 2, 0, 0]);
    /// assert_eq!(a.is_disjoint(&a), false);
    ///
    /// let b = Multiset::from([0, 0, 3, 4]);
    /// assert_eq!(a.is_disjoint(&b), true);
    ///
    /// let c = Multiset::from([0, 1, 1, 0]);
    /// assert_eq!(a.is_disjoint(&c), false);
    /// ```
    #[cfg(not(feature = "simd"))]
    #[inline]
    pub fn is_disjoint(&self, other: &Self) -> bool {
        self.iter()
            .zip(other.iter())
            .all(|(a, b)| a.min(b) == &N::zero())
    }

    /// Returns `true` if `self` is a subset of `other`.
    ///
    /// Multiset `A` is a subset of `B` if all the element counts in `A` are
    /// less than or equal to the counts in `B`. `A` is subset `B` if
    /// `A[i] <= B[i]` for all `i` in `A`.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let a = Multiset::from([1u64, 2, 0, 0]);
    /// assert_eq!(a.is_subset(&a), true);
    ///
    /// let b = Multiset::from([1, 3, 0, 0]);
    /// assert_eq!(a.is_subset(&b), true);
    ///
    /// assert_eq!(a.is_subset(&Multiset::empty()), false);
    /// ```
    #[cfg(not(feature = "simd"))]
    #[inline]
    pub fn is_subset(&self, other: &Self) -> bool {
        self.iter().zip(other.iter()).all(|(a, b)| a <= b)
    }

    /// Returns `true` if `self` is a superset of `other`.
    ///
    /// Multiset `A` is a subset of `B` if all the element counts in `A` are
    /// greater than or equal to the counts in `B`. `A` is superset `B` if
    /// `A[i] >= B[i]` for all `i` in `A`.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let a = Multiset::from([1u8, 2, 0, 0]);
    /// assert_eq!(a.is_superset(&a), true);
    ///
    /// let b = Multiset::from([1, 1, 0, 0]);
    /// assert_eq!(a.is_superset(&b), true);
    ///
    /// assert_eq!(a.is_superset(&Multiset::empty()), true);
    /// ```
    #[cfg(not(feature = "simd"))]
    #[inline]
    pub fn is_superset(&self, other: &Self) -> bool {
        self.iter().zip(other.iter()).all(|(a, b)| a >= b)
    }

    /// Returns `true` if `self` is a proper subset of `other`.
    ///
    /// Multiset `A` is a proper subset of `B` if `A` is a subset of `B` and at
    /// least one element count of `A` is less than that element count in `B`.
    /// `A` is proper subset `B` if `A[i] <= B[i]` for all `i` in `A` and there
    /// exists `j` such that `A[j] < B[j]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let a = Multiset::from([1u8, 2, 0, 0]);
    /// assert_eq!(a.is_proper_subset(&a), false);
    ///
    /// let b = Multiset::from([1, 3, 0, 0]);
    /// assert_eq!(a.is_proper_subset(&b), true);
    /// ```
    #[cfg(not(feature = "simd"))]
    #[inline]
    pub fn is_proper_subset(&self, other: &Self) -> bool {
        self != other && self.is_subset(other)
    }

    /// Returns `true` if `self` is a proper superset of `other`.
    ///
    /// Multiset `A` is a proper superset of `B` if `A` is a superset of `B`
    /// and at least one element count of `A` is greater than that element
    /// count in `B`. `A` is proper superset `B` if `A[i] >= B[i]` for all `i`
    /// in `A` and there exists `j` such that `A[j] > B[j]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let a = Multiset::from([1u8, 2, 0, 0]);
    /// assert_eq!(a.is_proper_superset(&a), false);
    ///
    /// let b = Multiset::from([1, 1, 0, 0]);
    /// assert_eq!(a.is_proper_superset(&b), true);
    /// ```
    #[cfg(not(feature = "simd"))]
    #[inline]
    pub fn is_proper_superset(&self, other: &Self) -> bool {
        self != other && self.is_superset(other)
    }

    /// Returns `true` if any element count in `self` is less than that element
    /// count in `other`.
    ///
    /// `A` is any lesser `B` if there exists some `i` such that `A[i] < B[i]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let a = Multiset::from([1u8, 2, 4, 0]);
    /// assert_eq!(a.is_any_lesser(&a), false);
    ///
    /// let b = Multiset::from([1, 3, 0, 0]);
    /// assert_eq!(a.is_any_lesser(&b), true);
    /// ```
    #[cfg(not(feature = "simd"))]
    #[inline]
    pub fn is_any_lesser(&self, other: &Self) -> bool {
        self.iter().zip(other.iter()).any(|(a, b)| a < b)
    }

    /// Returns `true` if any element count in `self` is greater than that
    /// element count in `other`.
    ///
    /// `A` is any greater `B` if there exists some `i` such that
    /// `A[i] > B[i]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let a = Multiset::from([1u8, 2, 0, 0]);
    /// assert_eq!(a.is_any_greater(&a), false);
    ///
    /// let b = Multiset::from([1, 1, 4, 0]);
    /// assert_eq!(a.is_any_greater(&b), true);
    /// ```
    #[cfg(not(feature = "simd"))]
    #[inline]
    pub fn is_any_greater(&self, other: &Self) -> bool {
        self.iter().zip(other.iter()).any(|(a, b)| a > b)
    }

    /// Returns `true` if all elements have a count of zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let multiset = Multiset::from([0u8, 0, 0, 0]);
    /// assert_eq!(multiset.is_empty(), true);
    /// assert_eq!(Multiset::<u8, 4>::empty().is_empty(), true);
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data == [N::zero(); SIZE]
    }

    /// The total or cardinality of a multiset is the sum of all element
    /// counts.
    ///
    /// This function converts counts to `usize` to try and avoid overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let multiset = Multiset::from([1u8, 2, 3, 4]);
    /// assert_eq!(multiset.total(), 10);
    /// ```
    #[cfg(not(feature = "simd"))]
    #[inline]
    pub fn total(&self) -> usize {
        self.iter()
            .map(|e| <N as AsPrimitive<usize>>::as_(*e))
            .sum()
    }

    /// Returns a tuple containing the element and a reference to the largest
    /// count in the multiset.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let multiset = Multiset::from([2u16, 0, 5, 3]);
    /// assert_eq!(multiset.elem_count_max(), (2, &5));
    /// ```
    #[inline]
    pub fn elem_count_max(&self) -> (usize, &N) {
        // iter cannot be empty, so it's fine to unwrap
        self.iter()
            .enumerate()
            .max_by_key(|(_, count)| *count)
            .unwrap()
    }

    /// Returns the element with the largest count in the multiset.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let multiset = Multiset::from([2u8, 0, 5, 3]);
    /// assert_eq!(multiset.elem_max(), 2);
    /// ```
    #[inline]
    pub fn elem_max(&self) -> usize {
        self.elem_count_max().0
    }

    /// Returns a reference to the largest count in the multiset.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let multiset = Multiset::from([2u8, 0, 5, 3]);
    /// assert_eq!(multiset.count_max(), &5);
    /// ```
    #[inline]
    pub fn count_max(&self) -> &N {
        // iter cannot be empty, so it's fine to unwrap
        self.iter().max().unwrap()
    }

    /// Returns a tuple containing the element and a reference to the smallest
    /// count in the multiset.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let multiset = Multiset::from([2u8, 0, 5, 3]);
    /// assert_eq!(multiset.elem_count_min(), (1, &0));
    /// ```
    #[inline]
    pub fn elem_count_min(&self) -> (usize, &N) {
        // iter cannot be empty, so it's fine to unwrap
        self.iter()
            .enumerate()
            .min_by_key(|(_, count)| *count)
            .unwrap()
    }

    /// Returns the element with the smallest count in the multiset.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let multiset = Multiset::from([2u8, 0, 5, 3]);
    /// assert_eq!(multiset.elem_min(), 1);
    /// ```
    #[inline]
    pub fn elem_min(&self) -> usize {
        self.elem_count_min().0
    }

    /// Returns a reference to the smallest count in the multiset.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let multiset = Multiset::from([2u8, 0, 5, 3]);
    /// assert_eq!(multiset.count_min(), &0);
    /// ```
    #[inline]
    pub fn count_min(&self) -> &N {
        // iter cannot be empty, so it's fine to unwrap
        self.iter().min().unwrap()
    }

    /// Set all element counts, except for the given `elem`, to zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let mut multiset = Multiset::from([2u8, 0, 5, 3]);
    /// multiset.choose(2);
    /// let result = Multiset::from([0, 0, 5, 0]);
    /// assert_eq!(multiset, result);
    /// ```
    #[inline]
    pub fn choose(&mut self, elem: usize) {
        let mut res = [N::zero(); SIZE];
        if elem < SIZE {
            // Safety: Guaranteed by bounds check on `elem`.
            unsafe { *res.get_unchecked_mut(elem) = *self.get_unchecked(elem) };
        }
        self.data = res
    }

    /// Set all element counts, except for a random choice, to zero.
    ///
    /// The choice is weighted by the counts of the elements, and unless the
    /// multiset is empty an element with non-zero count will always be chosen.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    /// use rand::prelude::*;
    ///
    /// let rng = &mut StdRng::seed_from_u64(thread_rng().next_u64());
    /// let mut multiset = Multiset::from([2u8, 0, 5, 3]);
    /// multiset.choose_random(rng);
    /// assert_eq!(multiset.is_singleton(), true);
    /// ```
    #[cfg(feature = "rand")]
    #[cfg(not(feature = "simd"))]
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

    /// Calculate the collision entropy of the multiset.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let multiset = Multiset::from([2u8, 1, 1, 0]);
    /// let result = multiset.collision_entropy();
    /// // approximate: result == 1.415037499278844
    /// ```
    ///
    /// # Warning
    /// Should not be used if [`Multiset::total`] or any counter in the
    /// multiset cannot be converted to `f64`. The conversions are handled by
    /// [`AsPrimitive<f64>`].
    ///
    /// [`AsPrimitive<f64>`]: num_traits::AsPrimitive
    #[cfg(not(feature = "simd"))]
    #[inline]
    pub fn collision_entropy(&self) -> f64 {
        let total: f64 = self.total().as_();
        -self
            .into_iter()
            .fold(0.0, |acc, count| {
                let freq_f64: f64 = count.as_();
                acc + (freq_f64 / total).powf(2.0)
            })
            .log2()
    }

    /// Calculate the shannon entropy of the multiset. Uses ln rather than log2.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let multiset = Multiset::from([2u8, 1, 1, 0]);
    /// let result = multiset.shannon_entropy();
    /// // approximate: result == 1.0397207708399179
    /// ```
    ///
    /// # Warning
    /// Should not be used if [`Multiset::total`] or any counter in the
    /// multiset cannot be converted to `f64`. The conversions are handled by
    /// [`AsPrimitive<f64>`].
    ///
    /// [`AsPrimitive<f64>`]: num_traits::AsPrimitive
    #[cfg(not(feature = "simd"))]
    #[inline]
    pub fn shannon_entropy(&self) -> f64 {
        let total: f64 = self.total().as_();
        -self.into_iter().fold(0.0, |acc, count| {
            if count > &N::zero() {
                let freq_f64: f64 = count.as_();
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
    use std::panic;

    fn catch_unwind_silent<F: FnOnce() -> R + panic::UnwindSafe, R>(
        f: F,
    ) -> std::thread::Result<R> {
        let prev_hook = panic::take_hook();
        panic::set_hook(Box::new(|_| {}));
        let result = panic::catch_unwind(f);
        panic::set_hook(prev_hook);
        result
    }

    #[test]
    fn test_index() {
        let set = Multiset::from([1u8, 2, 3, 4]);
        assert_eq!(set.index(1), &2);
        assert_eq!(set.index(2..), &[3, 4]);
    }

    #[test]
    fn test_index_mut() {
        let mut set = Multiset::from([1u8, 2, 3, 4]);
        *set.index_mut(1) = 3;
        assert_eq!(set, Multiset::from([1u8, 3, 3, 4]));
        set.index_mut(2..).iter_mut().for_each(|e| *e = 5);
        assert_eq!(set, Multiset::from([1u8, 3, 5, 5]))
    }

    #[test]
    fn test_add() {
        let set = Multiset::from([2u8, 2, 2, 2]);
        assert_eq!(set + set, Multiset::from([4u8, 4, 4, 4]));
        assert_eq!(set + 3, Multiset::from([5u8, 5, 5, 5]));

        let mut set_assign = Multiset::from([3u8, 3, 3, 3]);
        set_assign += set;
        assert_eq!(set_assign, Multiset::from([5u8, 5, 5, 5]));
        set_assign += 2;
        assert_eq!(set_assign, Multiset::from([7u8, 7, 7, 7]));
    }

    #[test]
    fn test_div() {
        let set = Multiset::from([10u8, 10, 10, 10]);
        assert_eq!(set / set, Multiset::from([1u8, 1, 1, 1]));
        assert_eq!(set / 2, Multiset::from([5u8, 5, 5, 5]));

        let mut set_assign = Multiset::from([20u8, 20, 20, 20]);
        set_assign /= set;
        assert_eq!(set_assign, Multiset::from([2u8, 2, 2, 2]));
        set_assign /= 2;
        assert_eq!(set_assign, Multiset::from([1u8, 1, 1, 1]));
    }

    #[test]
    fn test_mul() {
        let set = Multiset::from([2u8, 2, 2, 2]);
        assert_eq!(set * set, Multiset::from([4u8, 4, 4, 4]));
        assert_eq!(set * 3, Multiset::from([6u8, 6, 6, 6]));

        let mut set_assign = Multiset::from([3u8, 3, 3, 3]);
        set_assign *= set;
        assert_eq!(set_assign, Multiset::from([6u8, 6, 6, 6]));
        set_assign *= 2;
        assert_eq!(set_assign, Multiset::from([12u8, 12, 12, 12]));
    }

    #[test]
    fn test_rem() {
        let set = Multiset::from([10u8, 10, 10, 10]);
        assert_eq!(set % set, Multiset::from([0u8, 0, 0, 0]));
        assert_eq!(set % 3, Multiset::from([1u8, 1, 1, 1]));

        let mut set_assign = Multiset::from([3u8, 3, 3, 3]);
        set_assign %= set;
        assert_eq!(set_assign, Multiset::from([3u8, 3, 3, 3]));
        set_assign %= 2;
        assert_eq!(set_assign, Multiset::from([1u8, 1, 1, 1]));
    }

    #[test]
    fn test_sub() {
        let set = Multiset::from([5u8, 5, 5, 5]);
        assert_eq!(set - set, Multiset::from([0u8, 0, 0, 0]));
        assert_eq!(set - 3, Multiset::from([2u8, 2, 2, 2]));

        let mut set_assign = Multiset::from([8u8, 8, 8, 8]);
        set_assign -= set;
        assert_eq!(set_assign, Multiset::from([3u8, 3, 3, 3]));
        set_assign -= 2;
        assert_eq!(set_assign, Multiset::from([1u8, 1, 1, 1]));
    }

    #[test]
    fn test_zip_map() {
        let set1: Multiset<u8, 4> = Multiset::from([1, 5, 2, 8]);
        let set2: Multiset<u8, 4> = Multiset::from([1, 5, 2, 8]);
        let result = set1.zip_map(&set2, |e1, e2| e1 + e2);
        let expected = Multiset::from([2, 10, 4, 16]);
        assert_eq!(result, expected)
    }

    #[test]
    fn test_len() {
        assert_eq!(Multiset::<u16, 9>::len(), 9)
    }

    #[test]
    fn test_new() {
        assert_eq!(Multiset::new([1u8, 3]), Multiset::from([1u8, 3]))
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
        let expected = Multiset::from([3u8; 4]);
        assert_eq!(result, expected)
    }

    #[test]
    fn test_from_elements() {
        let into_it = &[0, 1, 1, 2, 2, 2];
        let result = Multiset::<u8, 4>::from_elements(into_it);
        assert_eq!(result, Multiset::from([1, 2, 3, 0]))
    }

    #[test]
    fn test_from_elements_panic() {
        let into_it = &[9]; // contains a value larger than the multiset size
        let res = catch_unwind_silent(|| Multiset::<u8, 4>::from_elements(into_it));
        assert!(res.is_err())
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
        let set: Multiset<u16, 3> = Multiset::from([5, 4, 3]);
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
        unsafe { assert_eq!(set.get_unchecked(1), &7) }
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
    fn test_difference() {
        let a = Multiset::<u8, 4>::from([0, 2, 5, 6]);
        let b = Multiset::from([1, 1, 8, 0]);
        let c = Multiset::from([0, 1, 5, 6]);
        assert_eq!(c, a.difference(&b))
    }

    #[test]
    fn test_symmetric_difference() {
        let a = Multiset::<u8, 4>::from([0, 2, 5, 6]);
        let b = Multiset::from([1, 1, 8, 0]);
        let c = Multiset::from([1, 1, 5, 6]);
        assert_eq!(c, a.symmetric_difference(&b))
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
        let expected = (2, &3);
        assert_eq!(set.elem_count_max(), expected)
    }

    #[test]
    fn test_imax() {
        let set = Multiset::from([1u8, 0, 3, 1]);
        let expected = 2;
        assert_eq!(set.elem_max(), expected)
    }

    #[test]
    fn test_max() {
        let set = Multiset::from([1u8, 0, 3, 1]);
        let expected = &3;
        assert_eq!(set.count_max(), expected)
    }

    #[test]
    fn test_argmin() {
        let set = Multiset::from([1u8, 0, 3, 1]);
        let expected = (1, &0);
        assert_eq!(set.elem_count_min(), expected)
    }

    #[test]
    fn test_imin() {
        let set = Multiset::from([1u8, 0, 3, 1]);
        let expected = 1;
        assert_eq!(set.elem_min(), expected)
    }

    #[test]
    fn test_min() {
        let set = Multiset::from([1u8, 0, 3, 1]);
        let expected = &0;
        assert_eq!(set.count_min(), expected)
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
        let test_rng1 = &mut StdRng::seed_from_u64(thread_rng().next_u64());
        result1.choose_random(test_rng1);
        assert!(result1.is_singleton() && result1.is_subset(&Multiset::from([1u8, 2, 3, 4, 5])));

        let mut result2 = Multiset::from([1u8, 2, 3, 4, 5]);
        let test_rng2 = &mut StdRng::seed_from_u64(thread_rng().next_u64());
        result2.choose_random(test_rng2);
        assert!(result2.is_singleton() && result2.is_subset(&Multiset::from([1u8, 2, 3, 4, 5])));
    }

    #[cfg(feature = "rand")]
    #[test]
    fn test_choose_random_empty() {
        let mut result = Multiset::<u32, 5>::empty();
        let test_rng = &mut StdRng::seed_from_u64(thread_rng().next_u64());
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
    fn test_shannon_entropy() {
        let a: Multiset<u8, 4> = Multiset::from([200, 0, 0, 0]);
        assert_eq!(a.shannon_entropy(), 0.0);

        let b: Multiset<u8, 4> = Multiset::from([2, 1, 1, 0]);
        assert_relative_eq!(
            b.shannon_entropy(),
            1.0397207708399179,
            epsilon = f64::EPSILON
        );
    }

    #[test]
    fn test_generic() {
        #[cfg(feature = "simd")]
        fn generic_total<T: Counter, const SIZE: usize>(ms: Multiset<T, SIZE>) -> usize
        where
            [(); T::L128 * T::L256 * T::LF]: Sized,
        {
            ms.total()
        }

        #[cfg(not(feature = "simd"))]
        fn generic_total<T: Counter, const SIZE: usize>(ms: Multiset<T, SIZE>) -> usize {
            ms.total()
        }

        assert_eq!(generic_total(Multiset::from([1u8, 2, 3])), 6);

        #[cfg(feature = "simd")]
        fn generic_union<T: Counter, const SIZE: usize>(
            ms1: Multiset<T, SIZE>,
            ms2: Multiset<T, SIZE>,
        ) -> Multiset<T, SIZE>
        where
            [(); T::L128 * T::L256 * T::LF]: Sized,
        {
            ms1.union(&ms2)
        }

        #[cfg(not(feature = "simd"))]
        fn generic_union<T: Counter, const SIZE: usize>(
            ms1: Multiset<T, SIZE>,
            ms2: Multiset<T, SIZE>,
        ) -> Multiset<T, SIZE> {
            ms1.union(&ms2)
        }

        let a = Multiset::from([0u8, 0]);
        let b = Multiset::from([1u8, 1]);
        assert_eq!(generic_union(a, b), Multiset::from([1, 1]));
    }
}
