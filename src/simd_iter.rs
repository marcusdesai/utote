use crate::counter::Counter;
use core::array;
use core::cmp::Ordering;
use core::fmt::Debug;
use core::iter::{FromIterator, IntoIterator};
use core::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign,
};
#[cfg(feature = "simd")]
use core::simd::*;
use core::slice::{Iter, IterMut, SliceIndex};
#[cfg(feature = "rand")]
use rand::{Rng, RngCore};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Multiset<T: Counter, const SIZE: usize>([T; SIZE]);

////////////////////////////////////////////////////////////////////////////////
// Inherent methods
////////////////////////////////////////////////////////////////////////////////

impl<T: Counter, const SIZE: usize> Multiset<T, SIZE> {
    /// Length of the multiset.
    pub const SIZE: usize = SIZE;

    /// Get the length of the multiset.
    pub const fn len(&self) -> usize {
        SIZE
    }

    /// Construct an empty multiset.
    pub const fn new() -> Self {
        Multiset([T::ZERO; SIZE])
    }

    /// Construct a multiset setting all counts to the given value.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let multiset = Multiset::<u8, 4>::repeat(5);
    /// assert_eq!(multiset, Multiset::from([5, 5, 5, 5]))
    /// ```
    pub const fn repeat(count: T) -> Self {
        Multiset([count; SIZE])
    }

    /// Convert an array to a multiset.
    pub const fn from_array(array: [T; SIZE]) -> Self {
        Multiset(array)
    }

    /// Returns a reference to an array of the multiset counts.
    pub const fn as_array(&self) -> &[T; SIZE] {
        &self.0
    }

    /// Returns a mutable reference to an array of the multiset counts.
    pub fn as_mut_array(&mut self) -> &mut [T; SIZE] {
        &mut self.0
    }

    /// Convert a multiset to an array.
    pub const fn to_array(self) -> [T; SIZE] {
        self.0
    }

    /// Construct a multiset from a slice. If the slice's `len` is less than
    /// the length of the multiset the remaining counts will be set to 0. If
    /// the slice's `len` is greater than that of the multiset then the excess
    /// counts will be ignored.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let slice = &[5u32, 4, 3, 2, 1];
    /// let ms = Multiset::from([5u32, 4, 3, 2, 1]);
    /// assert_eq!(ms, Multiset::from_slice(slice));
    ///
    /// let short_slice = &[5u32, 4, 3];
    /// let full_ms = Multiset::from([5u32, 4, 3, 0, 0]);
    /// assert_eq!(full_ms, Multiset::from_slice(short_slice));
    ///
    /// let long_slice = &[5u32, 4, 3, 2, 1, 0];
    /// assert_eq!(ms, Multiset::from_slice(long_slice))
    /// ```
    pub fn from_slice(slice: &[T]) -> Self {
        slice.iter().copied().collect()
    }

    /// Constructs a Multiset from an iterator of elements in the multiset,
    /// incrementing the count of each element as it occurs in the iterator.
    /// Elements greater than the length of the multiset are ignored.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let multiset = Multiset::<u8, 4>::from_elements([1, 1, 0, 2, 2, 2, 5]);
    /// assert_eq!(multiset, Multiset::from([1, 2, 3, 0]))
    /// ```
    pub fn from_elements<I: IntoIterator<Item = usize>>(elements: I) -> Self {
        let mut out = [T::ZERO; SIZE];
        elements
            .into_iter()
            .filter(|e| e < &SIZE)
            .for_each(|e| out[e] = out[e].saturating_add(T::ONE));
        Multiset(out)
    }

    /// Constructs a Multiset from an iterator of elements in the multiset,
    /// incrementing the count of each element as it occurs in the iterator,
    /// without bounds checking.
    ///
    /// For a safe alternative see [`from_elements`].
    ///
    /// # Safety
    ///
    /// Calling this method with an iterator containing elements which are
    /// out-of-bounds indexes is *[undefined behavior]* even if the resulting
    /// Multiset is not used.
    ///
    /// [`from_elements`]: Multiset::from_elements
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let multiset = unsafe { Multiset::<u8, 4>::from_elements_unchecked([1, 1, 0, 2, 2, 2]) };
    /// assert_eq!(multiset, Multiset::from([1, 2, 3, 0]))
    /// ```
    pub unsafe fn from_elements_unchecked<I: IntoIterator<Item = usize>>(elements: I) -> Self {
        let mut out = [T::ZERO; SIZE];
        elements
            .into_iter()
            .for_each(|e| {
                *out.get_unchecked_mut(e) = out.get_unchecked(e).saturating_add(T::ONE)
            });
        Multiset(out)
    }

    /// Return an [Iter](`std::slice::Iter`) of the element counts in the
    /// Multiset.
    pub fn iter(&self) -> Iter<T> {
        self.0.iter()
    }

    /// Return a [IterMut](`std::slice::IterMut`) of the element counts in the
    /// Multiset.
    pub fn iter_mut(&mut self) -> IterMut<T> {
        self.0.iter_mut()
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
    pub fn contains(&self, elem: usize) -> bool {
        elem < SIZE && self[elem] > T::ZERO
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
    pub unsafe fn contains_unchecked(&self, elem: usize) -> bool {
        self.get_unchecked(elem) > &T::ZERO
    }

    /// Set the count of `elem` in the multiset to `amount`. If `elem` is
    /// out-of-bounds the insert is a silent noop.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let mut multiset = Multiset::from([1u8, 2, 0, 0]);
    /// multiset.insert(2, 5);
    /// assert_eq!(multiset.get(2), Some(&5));
    ///
    /// let mut ms100 = Multiset::<u64, 8>::repeat(100);
    /// multiset.insert(10, 200);
    /// assert_eq!(ms100, Multiset::repeat(100))
    /// ```
    pub fn insert(&mut self, elem: usize, amount: T) {
        if elem < SIZE {
            self[elem] = amount
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
    /// unsafe { multiset.insert_unchecked(2, 5) };
    /// assert_eq!(multiset.get(2), Some(&5));
    /// ```
    pub unsafe fn insert_unchecked(&mut self, elem: usize, amount: T) {
        *self.get_unchecked_mut(elem) = amount
    }

    /// Set the count of `elem` in the multiset to zero. If `elem` is
    /// out-of-bounds the remove is a silent noop.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let mut multiset = Multiset::from([1u8, 2, 0, 0]);
    /// multiset.remove(1);
    /// assert_eq!(multiset.get(1), Some(&0));
    ///
    /// let mut ms100 = Multiset::<u64, 8>::repeat(100);
    /// multiset.remove(10);
    /// assert_eq!(ms100, Multiset::repeat(100))
    /// ```
    pub fn remove(&mut self, elem: usize) {
        if elem < SIZE {
            self[elem] = T::ZERO
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
    /// unsafe { multiset.remove_unchecked(1) };
    /// assert_eq!(multiset.get(1), Some(&0));
    /// ```
    pub unsafe fn remove_unchecked(&mut self, elem: usize) {
        *self.get_unchecked_mut(elem) = T::ZERO
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
    pub fn get<I>(&self, index: I) -> Option<&I::Output>
    where
        I: SliceIndex<[T]>,
    {
        self.0.get(index)
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
    /// unsafe {
    ///     assert_eq!(multiset.get_unchecked(1), &2);
    /// }
    /// ```
    pub unsafe fn get_unchecked<I>(&self, index: I) -> &I::Output
    where
        I: SliceIndex<[T]>,
    {
        self.0.get_unchecked(index)
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
    /// if let Some(elem) = multiset.get_mut(1) {
    ///     *elem = 42;
    /// }
    /// assert_eq!(multiset, Multiset::from([1u8, 42, 4]));
    /// ```
    pub fn get_mut<I>(&mut self, index: I) -> Option<&mut I::Output>
    where
        I: SliceIndex<[T]>,
    {
        self.0.get_mut(index)
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
    /// unsafe { *multiset.get_unchecked_mut(1) = 13 };
    /// assert_eq!(multiset, Multiset::from([1u8, 13, 4]));
    /// ```
    pub unsafe fn get_unchecked_mut<I>(&mut self, index: I) -> &mut I::Output
    where
        I: SliceIndex<[T]>,
    {
        self.0.get_unchecked_mut(index)
    }

    /// Increment the count of `elem` in the multiset by `amount`, saturating
    /// at the limit of the counter type. If `elem` is out-of-bounds the
    /// increment is a silent noop.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let mut multiset = Multiset::from([1u8, 2, 0, 0]);
    /// multiset.increment(1, 5);
    /// assert_eq!(multiset.get(1), Some(&7));
    ///
    /// let mut empty = Multiset::<u64, 8>::new();
    /// multiset.increment(10, 25);
    /// assert!(empty.is_empty())
    /// ```
    pub fn increment(&mut self, elem: usize, amount: T) {
        if elem < SIZE {
            self.0[elem] = self.0[elem].saturating_add(amount)
        }
    }

    /// Increment the count of `elem` in the multiset by `amount`, without doing
    /// bounds checking, saturating at the limit of the counter type.
    ///
    /// For a safe alternative see [`increment`].
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds `elem` is
    /// *[undefined behavior]*.
    ///
    /// [`increment`]: Multiset::increment
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let mut multiset = Multiset::from([1u8, 2, 0, 0]);
    /// unsafe { multiset.increment_unchecked(1, 5) };
    /// assert_eq!(multiset.get(1), Some(&7));
    /// ```
    pub unsafe fn increment_unchecked(&mut self, elem: usize, amount: T) {
        *self.get_unchecked_mut(elem) = self.get_unchecked(elem).saturating_add(amount)
    }

    /// Decrement the count of `elem` in the multiset by `amount`, saturating
    /// at the limit of the counter type. If `elem` is out-of-bounds the
    /// decrement is a silent noop.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let mut multiset = Multiset::from([1u8, 2, 0, 0]);
    /// multiset.decrement(1, 5);
    /// assert_eq!(multiset.get(1), Some(&0));
    ///
    /// let mut empty = Multiset::<u64, 8>::new();
    /// multiset.decrement(10, 25);
    /// assert!(empty.is_empty())
    /// ```
    pub fn decrement(&mut self, elem: usize, amount: T) {
        if elem < SIZE {
            self.0[elem] = self.0[elem].saturating_sub(amount)
        }
    }

    /// Decrement the count of `elem` in the multiset by `amount`, without doing
    /// bounds checking, saturating at the limit of the counter type.
    ///
    /// For a safe alternative see [`decrement`].
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds `elem` is
    /// *[undefined behavior]*.
    ///
    /// [`decrement`]: Multiset::decrement
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let mut multiset = Multiset::from([1u8, 2, 0, 0]);
    /// unsafe { multiset.decrement_unchecked(1, 5) };
    /// assert_eq!(multiset.get(1), Some(&0));
    /// ```
    pub unsafe fn decrement_unchecked(&mut self, elem: usize, amount: T) {
        *self.get_unchecked_mut(elem) = self.get_unchecked(elem).saturating_sub(amount)
    }

    /// Creates a multiset with all counts raised to the power of the given
    /// exponent. This operation saturates at numeric bounds instead of
    /// overflowing.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let multiset = Multiset::from([100u8, 5]);
    /// assert_eq!(multiset.pow(2), Multiset::from([255, 25]))
    /// ```
    pub fn pow(self, exp: u32) -> Self {
        self.iter().map(|a| a.saturating_pow(exp)).collect()
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
    pub fn count_non_zero(&self) -> usize {
        self.iter().filter(|&c| *c > T::ZERO).count()
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
    pub fn count_zero(&self) -> usize {
        SIZE - self.count_non_zero()
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
    /// assert!(multiset.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.0 = [T::ZERO; SIZE];
    }

    pub fn intersection(&self, other: &Self) -> Self {
        self.iter()
            .zip(other.iter())
            .map(|(a, b)| *a.min(b))
            .collect()
    }

    pub fn intersection_mut(&mut self, other: &Self) {
        self.iter_mut()
            .zip(other.iter())
            .for_each(|(a, b)| *a = (*a).min(*b))
    }

    pub fn union(&self, other: &Self) -> Self {
        self.iter()
            .zip(other.iter())
            .map(|(a, b)| *a.max(b))
            .collect()
    }

    pub fn union_mut(&mut self, other: &Self) {
        self.iter_mut()
            .zip(other.iter())
            .for_each(|(a, b)| *a = (*a).max(*b))
    }

    pub fn difference(&self, other: &Self) -> Self {
        self.iter()
            .zip(other.iter())
            .map(|(a, b)| a.saturating_sub(*b))
            .collect()
    }

    pub fn difference_mut(&mut self, other: &Self) {
        self.iter_mut()
            .zip(other.iter())
            .for_each(|(a, b)| *a = a.saturating_sub(*b))
    }

    pub fn complement(&self, other: &Self) -> Self {
        self.iter()
            .zip(other.iter())
            .map(|(a, b)| b.saturating_sub(*a))
            .collect()
    }

    pub fn complement_mut(&mut self, other: &Self) {
        self.iter_mut()
            .zip(other.iter())
            .for_each(|(a, b)| *a = b.saturating_sub(*a))
    }

    pub fn sum(&self, other: &Self) -> Self {
        self.iter()
            .zip(other.iter())
            .map(|(a, b)| a.saturating_add(*b))
            .collect()
    }

    pub fn sum_mut(&mut self, other: &Self) {
        self.iter_mut()
            .zip(other.iter())
            .for_each(|(a, b)| *a = a.saturating_add(*b))
    }

    pub fn symmetric_difference(&self, other: &Self) -> Self {
        self.iter()
            .zip(other.iter())
            .map(|(a, b)| a.abs_diff(*b))
            .collect()
    }

    pub fn symmetric_difference_mut(&mut self, other: &Self) {
        self.iter_mut()
            .zip(other.iter())
            .for_each(|(a, b)| *a = a.abs_diff(*b))
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
    /// assert!(multiset.is_singleton());
    /// ```
    pub fn is_singleton(&self) -> bool {
        self.count_non_zero() == 1
    }

    /// Returns `true` if all elements have a count of zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let multiset = Multiset::from([0u8, 0, 0, 0]);
    /// assert!(multiset.is_empty());
    /// assert!(Multiset::<u8, 50>::new().is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.0 == [T::ZERO; SIZE]
    }

    #[cfg(not(feature = "simd"))]
    pub fn is_disjoint(&self, other: &Self) -> bool {
        self.iter()
            .zip(other.iter())
            .all(|(a, b)| (*a).min(*b) == T::ZERO)
    }

    #[cfg(feature = "simd")]
    pub fn is_disjoint(&self, other: &Self) -> bool {
        const LANES: usize = 8;
        let mut disjoint = self
            .0
            .array_chunks::<LANES>()
            .zip(other.0.array_chunks())
            .all(|(a1, a2)| {
                let s1 = Simd::from_array(*a1);
                let s2 = Simd::from_array(*a2);
                s1.lanes_lt(s2)
                    .select(s1, s2)
                    .lanes_eq(Simd::splat(T::ZERO))
                    .all()
            });
        if SIZE % LANES != 0 && disjoint {
            let mut temp = [T::ZERO; LANES];
            let rem1 = &self.0[(SIZE - (SIZE % LANES))..];
            let rem2 = &other.0[(SIZE - (SIZE % LANES))..];
            for (elem, (a, b)) in temp.iter_mut().zip(rem1.iter().zip(rem2.iter())) {
                *elem = (*a).min(*b)
            }
            disjoint &= Simd::from_array(temp).lanes_eq(Simd::splat(T::ZERO)).all()
        }
        disjoint
    }

    #[cfg(not(feature = "simd"))]
    pub fn is_subset(&self, other: &Self) -> bool {
        self.iter().zip(other.iter()).all(|(a, b)| a <= b)
    }

    #[cfg(feature = "simd")]
    pub fn is_subset(&self, other: &Self) -> bool {
        const LANES: usize = 8;
        let mut subset = self
            .0
            .array_chunks::<LANES>()
            .zip(other.0.array_chunks())
            .all(|(a1, a2)| {
                let s1 = Simd::from_array(*a1);
                let s2 = Simd::from_array(*a2);
                s1.lanes_le(s2).all()
            });
        if SIZE % LANES != 0 && subset {
            let rem1 = &self.0[(SIZE - (SIZE % LANES))..];
            let rem2 = &other.0[(SIZE - (SIZE % LANES))..];
            for (a, b) in rem1.iter().zip(rem2.iter()) {
                subset &= a <= b
            }
        }
        subset
    }

    #[cfg(not(feature = "simd"))]
    pub fn is_superset(&self, other: &Self) -> bool {
        self.iter().zip(other.iter()).all(|(a, b)| a >= b)
    }

    #[cfg(feature = "simd")]
    pub fn is_superset(&self, other: &Self) -> bool {
        const LANES: usize = 8;
        let mut superset = self
            .0
            .array_chunks::<LANES>()
            .zip(other.0.array_chunks())
            .all(|(a1, a2)| {
                let s1 = Simd::from_array(*a1);
                let s2 = Simd::from_array(*a2);
                s1.lanes_ge(s2).all()
            });
        if SIZE % LANES != 0 && superset {
            let rem1 = &self.0[(SIZE - (SIZE % LANES))..];
            let rem2 = &other.0[(SIZE - (SIZE % LANES))..];
            for (a, b) in rem1.iter().zip(rem2.iter()) {
                superset &= a >= b
            }
        }
        superset
    }

    #[cfg(not(feature = "simd"))]
    pub fn is_proper_subset(&self, other: &Self) -> bool {
        self.iter().zip(other.iter()).all(|(a, b)| a < b)
    }

    #[cfg(feature = "simd")]
    pub fn is_proper_subset(&self, other: &Self) -> bool {
        const LANES: usize = 8;
        let mut proper_subset = self
            .0
            .array_chunks::<LANES>()
            .zip(other.0.array_chunks())
            .all(|(a1, a2)| {
                let s1 = Simd::from_array(*a1);
                let s2 = Simd::from_array(*a2);
                s1.lanes_lt(s2).all()
            });
        if SIZE % LANES != 0 && proper_subset {
            let rem1 = &self.0[(SIZE - (SIZE % LANES))..];
            let rem2 = &other.0[(SIZE - (SIZE % LANES))..];
            for (a, b) in rem1.iter().zip(rem2.iter()) {
                proper_subset &= a < b
            }
        }
        proper_subset
    }

    #[cfg(not(feature = "simd"))]
    pub fn is_proper_superset(&self, other: &Self) -> bool {
        self.iter().zip(other.iter()).all(|(a, b)| a > b)
    }

    #[cfg(feature = "simd")]
    pub fn is_proper_superset(&self, other: &Self) -> bool {
        const LANES: usize = 8;
        let mut proper_superset = self
            .0
            .array_chunks::<LANES>()
            .zip(other.0.array_chunks())
            .all(|(a1, a2)| {
                let s1 = Simd::from_array(*a1);
                let s2 = Simd::from_array(*a2);
                s1.lanes_gt(s2).all()
            });
        if SIZE % LANES != 0 && proper_superset {
            let rem1 = &self.0[(SIZE - (SIZE % LANES))..];
            let rem2 = &other.0[(SIZE - (SIZE % LANES))..];
            for (a, b) in rem1.iter().zip(rem2.iter()) {
                proper_superset &= a > b
            }
        }
        proper_superset
    }

    /// The sum of all element counts, also known as the cardinality of the
    /// multiset. This operation saturates at numeric bounds instead of
    /// overflowing.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let multiset = Multiset::from([1u8, 2, 3, 4]);
    /// assert_eq!(multiset.total(), 10);
    ///
    /// let saturating = Multiset::from([100u8, 100, 100]);
    /// assert_eq!(saturating.total(), 255);
    /// ```
    #[cfg(not(feature = "simd"))]
    pub fn total(&self) -> T {
        self.iter().fold(T::ZERO, |acc, e| acc.saturating_add(*e))
    }

    #[cfg(feature = "simd")]
    pub fn total(&self) -> T {
        const LANES: usize = 8;
        let mut t: Simd<T, LANES> = Simd::splat(T::ZERO);
        for a in self.0.array_chunks() {
            let s = Simd::from_array(*a);
            t = T::simd_saturating_add(t, s)
        }
        if SIZE % LANES != 0 {
            let rem = &self.0[(SIZE - (SIZE % LANES))..];
            let s = simd_from_slice_or_zero(rem);
            t = T::simd_saturating_add(t, s)
        }
        t.to_array()
            .iter()
            .fold(T::ZERO, |acc, e| acc.saturating_add(*e))
    }

    /// The sum of all element counts as a given counter type. This operation
    /// saturates at numeric bounds instead of overflowing.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let saturating = Multiset::from([100u8, 100, 100]);
    /// assert_eq!(saturating.total_as::<u32>(), 300);
    /// ```
    #[cfg(not(feature = "simd"))]
    pub fn total_as<U: Counter>(&self) -> U
    where
        U: From<T>,
    {
        self.iter()
            .fold(U::ZERO, |acc, e| acc.saturating_add((*e).into()))
    }

    #[cfg(feature = "simd")]
    pub fn total_as<U: Counter>(&self) -> U
    where
        U: From<T>,
    {
        const LANES: usize = 8;
        let mut t: Simd<U, LANES> = Simd::splat(U::ZERO);
        for a in self.0.array_chunks() {
            let s = Simd::from_array(*a).cast();
            t = U::simd_saturating_add(t, s)
        }
        if SIZE % LANES != 0 {
            let rem = &self.0[(SIZE - (SIZE % LANES))..];
            let s = simd_from_slice_or_zero(rem).cast();
            t = U::simd_saturating_add(t, s)
        }
        t.to_array()
            .iter()
            .fold(U::ZERO, |acc, e| acc.saturating_add(*e))
    }

    /// Returns the element and a reference to it's corresponding count that is
    /// the maximum value in the multiset.
    ///
    /// If several counts are equally maximum, the last pair is returned. If
    /// the size of the multiset is zero, [`None`] is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let multiset = Multiset::from([2u16, 0, 5, 3]);
    /// assert_eq!(multiset.elem_count_max(), Some((2, &5)));
    ///
    /// let multi_max = Multiset::from([0u8, 2, 6, 6, 6]);
    /// assert_eq!(multi_max.elem_count_max(), Some((4, &6)));
    ///
    /// let zero_size = Multiset::<u8, 0>::new();
    /// assert_eq!(zero_size.elem_count_max(), None)
    /// ```
    pub fn elem_count_max(&self) -> Option<(usize, &T)> {
        self.iter().enumerate().max_by_key(|(_, count)| *count)
    }

    /// Returns the element whose corresponding count is the maximum value in
    /// the multiset.
    ///
    /// If several counts are equally maximum, the last element is returned. If
    /// the size of the multiset is zero, [`None`] is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let multiset = Multiset::from([2u16, 0, 5, 3]);
    /// assert_eq!(multiset.elem_max(), Some(2));
    ///
    /// let multi_max = Multiset::from([0u8, 2, 6, 6, 6]);
    /// assert_eq!(multi_max.elem_max(), Some(4));
    ///
    /// let zero_size = Multiset::<u8, 0>::new();
    /// assert_eq!(zero_size.elem_max(), None)
    /// ```
    pub fn elem_max(&self) -> Option<usize> {
        self.elem_count_max().map(|t| t.0)
    }

    /// Returns the count that is the maximum value in the multiset.
    ///
    /// If several counts are equally maximum, the last is returned. If
    /// the size of the multiset is zero, [`None`] is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let multiset = Multiset::from([2u16, 0, 5, 3]);
    /// assert_eq!(multiset.count_max(), Some(5));
    ///
    /// let multi_max = Multiset::from([0u8, 2, 6, 6, 6]);
    /// assert_eq!(multi_max.count_max(), Some(6));
    ///
    /// let zero_size = Multiset::<u8, 0>::new();
    /// assert_eq!(zero_size.count_max(), None)
    /// ```
    pub fn count_max(&self) -> Option<T> {
        self.iter().copied().max()
    }

    /// Returns the element and a reference to it's corresponding count that is
    /// the minimum value in the multiset.
    ///
    /// If several counts are equally minimum, the first pair is returned. If
    /// the size of the multiset is zero, [`None`] is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let multiset = Multiset::from([2u16, 1, 5, 3]);
    /// assert_eq!(multiset.elem_count_min(), Some((1, &1)));
    ///
    /// let multi_min = Multiset::from([1u8, 2, 0, 0, 0]);
    /// assert_eq!(multi_min.elem_count_min(), Some((2, &0)));
    ///
    /// let zero_size = Multiset::<u8, 0>::new();
    /// assert_eq!(zero_size.elem_count_min(), None)
    /// ```
    pub fn elem_count_min(&self) -> Option<(usize, &T)> {
        self.iter().enumerate().min_by_key(|(_, count)| *count)
    }

    /// Returns the element whose corresponding count is the minimum value in
    /// the multiset.
    ///
    /// If several counts are equally minimum, the first element is returned.
    /// If the size of the multiset is zero, [`None`] is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let multiset = Multiset::from([2u16, 1, 5, 3]);
    /// assert_eq!(multiset.elem_min(), Some(1));
    ///
    /// let multi_min = Multiset::from([1u8, 2, 0, 0, 0]);
    /// assert_eq!(multi_min.elem_min(), Some(2));
    ///
    /// let zero_size = Multiset::<u8, 0>::new();
    /// assert_eq!(zero_size.elem_min(), None)
    /// ```
    pub fn elem_min(&self) -> Option<usize> {
        self.elem_count_min().map(|t| t.0)
    }

    /// Returns the count that is the minimum value in the multiset.
    ///
    /// If several counts are equally minimum, the first is returned. If
    /// the size of the multiset is zero, [`None`] is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let multiset = Multiset::from([2u16, 0, 5, 3]);
    /// assert_eq!(multiset.count_min(), Some(0));
    ///
    /// let multi_min = Multiset::from([1u8, 2, 0, 0, 0]);
    /// assert_eq!(multi_min.count_min(), Some(0));
    ///
    /// let zero_size = Multiset::<u8, 0>::new();
    /// assert_eq!(zero_size.count_min(), None)
    /// ```
    pub fn count_min(&self) -> Option<T> {
        self.iter().copied().min()
    }

    /// Set all element counts, except for the given element, to zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use utote::Multiset;
    ///
    /// let mut multiset = Multiset::from([2u8, 0, 5, 3]);
    /// multiset.choose(2);
    /// assert_eq!(multiset, Multiset::from([0, 0, 5, 0]));
    /// ```
    pub fn choose(&mut self, elem: usize) {
        let mut res = [T::ZERO; SIZE];
        if elem < SIZE {
            res[elem] = self.0[elem];
        }
        self.0 = res
    }

    /// Set all element counts, except for a random choice, to zero.
    ///
    /// The choice is weighted by the counts of the elements, unless the
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
    /// assert!(multiset.is_singleton());
    /// ```
    #[cfg(feature = "rand")]
    pub fn choose_random<R: RngCore>(&mut self, rng: &mut R)
    where
        u64: From<T>,
    {
        if SIZE == 1 {
            return;
        }
        let total: u64 = self.total_as();
        if total == 0 {
            return;
        }
        let choice_value = rng.gen_range(1..=total);
        let mut res = [T::ZERO; SIZE];
        let mut acc = 0;
        for i in 0..SIZE {
            let val: u64 = self.0[i].into();
            acc += val;
            if acc >= choice_value {
                res[i] = self.0[i];
                break;
            }
        }
        self.0 = res
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
    ///
    /// let zero_entropy = Multiset::from([20u64, 0, 0, 0]);
    /// assert_eq!(zero_entropy.collision_entropy(), 0.0)
    /// ```
    #[cfg(not(feature = "simd"))]
    pub fn collision_entropy(&self) -> f64
    where
        u64: From<T>,
    {
        let total = self.total_as::<u64>() as f64;
        let mut s: f64 = 0.0;
        for count in self.iter() {
            s += ((*count).as_f64() / total).powf(2.0)
        }
        -s.log2()
    }

    #[cfg(feature = "simd")]
    pub fn collision_entropy(&self) -> f64
    where
        u64: From<T>,
    {
        const LANES: usize = 4;
        let total: u64 = self.total_as();
        let total_vec: Simd<f64, LANES> = Simd::splat(total).cast();
        let mut out = Simd::splat(0.0);
        for a in self.0.array_chunks() {
            let fs = Simd::from_array(*a).cast();
            let prob = fs / total_vec;
            out += prob * prob;
        }
        if SIZE % LANES != 0 {
            let rem = &self.0[(SIZE - (SIZE % LANES))..];
            let fs = simd_from_slice_or_zero(rem).cast();
            let prob = fs / total_vec;
            out += prob * prob;
        }
        -(out.reduce_sum().log2())
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
    ///
    /// let zero_entropy = Multiset::from([20u64, 0, 0, 0]);
    /// assert_eq!(zero_entropy.shannon_entropy(), 0.0)
    /// ```
    pub fn shannon_entropy(&self) -> f64
    where
        u64: From<T>,
    {
        let total = self.total_as::<u64>() as f64;
        let mut acc = 0.0;
        for c in self.iter().filter(|&c| *c > T::ZERO) {
            let prob = c.as_f64() / total;
            acc += prob * prob.ln()
        }
        -acc
    }
}

////////////////////////////////////////////////////////////////////////////////
// Common traits
////////////////////////////////////////////////////////////////////////////////

impl<T: Counter, const SIZE: usize> Default for Multiset<T, SIZE> {
    fn default() -> Self {
        Multiset([T::default(); SIZE])
    }
}

impl<T: Counter, I: SliceIndex<[T]>, const SIZE: usize> Index<I> for Multiset<T, SIZE> {
    type Output = I::Output;

    fn index(&self, index: I) -> &Self::Output {
        self.0.index(index)
    }
}

impl<T: Counter, I: SliceIndex<[T]>, const SIZE: usize> IndexMut<I> for Multiset<T, SIZE> {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

impl<T: Counter, const SIZE: usize> FromIterator<T> for Multiset<T, SIZE> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut out = [T::ZERO; SIZE];
        for (elem, val) in out.iter_mut().zip(iter.into_iter()) {
            *elem = val
        }
        Multiset(out)
    }
}

impl<T: Counter, const SIZE: usize> IntoIterator for Multiset<T, SIZE> {
    type Item = T;
    type IntoIter = array::IntoIter<T, SIZE>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIterator::into_iter(self.0)
    }
}

impl<'a, T: Counter, const SIZE: usize> IntoIterator for &'a Multiset<T, SIZE> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T: Counter, const SIZE: usize> IntoIterator for &'a mut Multiset<T, SIZE> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

#[cfg(feature = "simd")]
#[inline]
fn simd_ord_check<T: Counter, const LANES: usize>(
    order: Ordering,
    s1: Simd<T, LANES>,
    s2: Simd<T, LANES>,
) -> Option<Ordering>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    match order {
        Ordering::Equal if s1.lanes_eq(s2).all() => Some(order),
        Ordering::Equal if s1.lanes_le(s2).all() => Some(Ordering::Less),
        Ordering::Equal if s1.lanes_ge(s2).all() => Some(Ordering::Greater),
        Ordering::Equal => None,
        Ordering::Less if s1.lanes_gt(s2).any() => None,
        Ordering::Greater if s1.lanes_lt(s2).any() => None,
        _ => Some(order),
    }
}

impl<T: Counter, const SIZE: usize> PartialOrd for Multiset<T, SIZE> {
    #[cfg(not(feature = "simd"))]
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

    #[cfg(feature = "simd")]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        const LANES: usize = 8;
        let mut order: Ordering = Ordering::Equal;
        for (a1, a2) in self.0.array_chunks::<LANES>().zip(other.0.array_chunks()) {
            let s1 = Simd::from_array(*a1);
            let s2 = Simd::from_array(*a2);
            match simd_ord_check(order, s1, s2) {
                Some(o) => order = o,
                None => return None,
            }
        }
        if SIZE % LANES != 0 {
            let rem1 = &self.0[(SIZE - (SIZE % LANES))..];
            let rem2 = &other.0[(SIZE - (SIZE % LANES))..];
            let s1 = simd_from_slice_or_zero::<_, LANES>(rem1);
            let s2 = simd_from_slice_or_zero(rem2);
            match simd_ord_check(order, s1, s2) {
                Some(o) => order = o,
                None => return None,
            }
        }
        Some(order)
    }

    fn lt(&self, other: &Self) -> bool {
        self.is_proper_subset(other)
    }

    fn le(&self, other: &Self) -> bool {
        self.is_subset(other)
    }

    fn gt(&self, other: &Self) -> bool {
        self.is_proper_superset(other)
    }

    fn ge(&self, other: &Self) -> bool {
        self.is_superset(other)
    }
}

////////////////////////////////////////////////////////////////////////////////
// Common conversions
////////////////////////////////////////////////////////////////////////////////

impl<T: Counter, const SIZE: usize> AsRef<Multiset<T, SIZE>> for Multiset<T, SIZE> {
    fn as_ref(&self) -> &Multiset<T, SIZE> {
        self
    }
}

impl<T: Counter, const SIZE: usize> AsMut<Multiset<T, SIZE>> for Multiset<T, SIZE> {
    fn as_mut(&mut self) -> &mut Multiset<T, SIZE> {
        self
    }
}

impl<T: Counter, const SIZE: usize> AsRef<[T; SIZE]> for Multiset<T, SIZE> {
    fn as_ref(&self) -> &[T; SIZE] {
        &self.0
    }
}

impl<T: Counter, const SIZE: usize> AsMut<[T; SIZE]> for Multiset<T, SIZE> {
    fn as_mut(&mut self) -> &mut [T; SIZE] {
        &mut self.0
    }
}

impl<T: Counter, const SIZE: usize> AsRef<[T]> for Multiset<T, SIZE> {
    fn as_ref(&self) -> &[T] {
        &self.0
    }
}

impl<T: Counter, const SIZE: usize> AsMut<[T]> for Multiset<T, SIZE> {
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.0
    }
}

impl<T: Counter, const SIZE: usize> From<[T; SIZE]> for Multiset<T, SIZE> {
    fn from(arr: [T; SIZE]) -> Self {
        Multiset(arr)
    }
}

impl<T: Counter, const SIZE: usize> From<Multiset<T, SIZE>> for [T; SIZE] {
    fn from(ms: Multiset<T, SIZE>) -> Self {
        ms.0
    }
}

////////////////////////////////////////////////////////////////////////////////
// Common ops
////////////////////////////////////////////////////////////////////////////////

impl<T: Counter, const SIZE: usize> Add for Multiset<T, SIZE> {
    type Output = Multiset<T, SIZE>;

    fn add(self, rhs: Self) -> Self::Output {
        self.sum(&rhs)
    }
}

impl<T: Counter, const SIZE: usize> Add<T> for Multiset<T, SIZE> {
    type Output = Multiset<T, SIZE>;

    fn add(self, rhs: T) -> Self::Output {
        self.iter().map(|a| a.saturating_add(rhs)).collect()
    }
}

impl<T: Counter, const SIZE: usize> AddAssign for Multiset<T, SIZE> {
    fn add_assign(&mut self, rhs: Self) {
        self.sum_mut(&rhs)
    }
}

impl<T: Counter, const SIZE: usize> AddAssign<T> for Multiset<T, SIZE> {
    fn add_assign(&mut self, rhs: T) {
        self.iter_mut().for_each(|a| *a = a.saturating_add(rhs))
    }
}

impl<T: Counter, const SIZE: usize> Sub for Multiset<T, SIZE> {
    type Output = Multiset<T, SIZE>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.difference(&rhs)
    }
}

impl<T: Counter, const SIZE: usize> Sub<T> for Multiset<T, SIZE> {
    type Output = Multiset<T, SIZE>;

    fn sub(self, rhs: T) -> Self::Output {
        self.iter().map(|a| a.saturating_sub(rhs)).collect()
    }
}

impl<T: Counter, const SIZE: usize> SubAssign for Multiset<T, SIZE> {
    fn sub_assign(&mut self, rhs: Self) {
        self.difference_mut(&rhs)
    }
}

impl<T: Counter, const SIZE: usize> SubAssign<T> for Multiset<T, SIZE> {
    fn sub_assign(&mut self, rhs: T) {
        self.iter_mut().for_each(|a| *a = a.saturating_sub(rhs))
    }
}

impl<T: Counter, const SIZE: usize> Mul for Multiset<T, SIZE> {
    type Output = Multiset<T, SIZE>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.iter()
            .zip(rhs.iter())
            .map(|(a, b)| a.saturating_mul(*b))
            .collect()
    }
}

impl<T: Counter, const SIZE: usize> Mul<T> for Multiset<T, SIZE> {
    type Output = Multiset<T, SIZE>;

    fn mul(self, rhs: T) -> Self::Output {
        self.iter().map(|a| a.saturating_mul(rhs)).collect()
    }
}

impl<T: Counter, const SIZE: usize> MulAssign for Multiset<T, SIZE> {
    fn mul_assign(&mut self, rhs: Self) {
        self.iter_mut()
            .zip(rhs.iter())
            .for_each(|(a, b)| *a = a.saturating_mul(*b))
    }
}

impl<T: Counter, const SIZE: usize> MulAssign<T> for Multiset<T, SIZE> {
    fn mul_assign(&mut self, rhs: T) {
        self.iter_mut().for_each(|a| *a = a.saturating_mul(rhs))
    }
}

impl<T: Counter, const SIZE: usize> Div for Multiset<T, SIZE> {
    type Output = Multiset<T, SIZE>;

    fn div(self, rhs: Self) -> Self::Output {
        self.iter()
            .zip(rhs.iter())
            .map(|(a, b)| a.saturating_div(*b))
            .collect()
    }
}

impl<T: Counter, const SIZE: usize> Div<T> for Multiset<T, SIZE> {
    type Output = Multiset<T, SIZE>;

    fn div(self, rhs: T) -> Self::Output {
        self.iter().map(|a| a.saturating_div(rhs)).collect()
    }
}

impl<T: Counter, const SIZE: usize> DivAssign for Multiset<T, SIZE> {
    fn div_assign(&mut self, rhs: Self) {
        self.iter_mut()
            .zip(rhs.iter())
            .for_each(|(a, b)| *a = a.saturating_div(*b))
    }
}

impl<T: Counter, const SIZE: usize> DivAssign<T> for Multiset<T, SIZE> {
    fn div_assign(&mut self, rhs: T) {
        self.iter_mut().for_each(|a| *a = a.saturating_div(rhs))
    }
}

impl<T: Counter, const SIZE: usize> Rem for Multiset<T, SIZE> {
    type Output = Multiset<T, SIZE>;

    fn rem(self, rhs: Self) -> Self::Output {
        self.iter().zip(rhs.iter()).map(|(a, b)| *a % *b).collect()
    }
}

impl<T: Counter, const SIZE: usize> Rem<T> for Multiset<T, SIZE> {
    type Output = Multiset<T, SIZE>;

    fn rem(self, rhs: T) -> Self::Output {
        self.iter().map(|val| *val % rhs).collect()
    }
}

impl<T: Counter, const SIZE: usize> RemAssign for Multiset<T, SIZE> {
    fn rem_assign(&mut self, rhs: Self) {
        self.iter_mut().zip(rhs.iter()).for_each(|(a, b)| *a %= *b)
    }
}

impl<T: Counter, const SIZE: usize> RemAssign<T> for Multiset<T, SIZE> {
    fn rem_assign(&mut self, rhs: T) {
        self.iter_mut().for_each(|a| *a %= rhs)
    }
}

////////////////////////////////////////////////////////////////////////////////
// Utils
////////////////////////////////////////////////////////////////////////////////

#[cfg(feature = "simd")]
fn simd_from_slice_or_zero<T: Counter, const LANES: usize>(slice: &[T]) -> Simd<T, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let mut temp = [T::ZERO; LANES];
    for (elem, val) in temp.iter_mut().zip(slice.iter()) {
        *elem = *val
    }
    Simd::from_array(temp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // todo: tests where SIZE = 0

    #[test]
    fn test_index() {
        let set = Multiset::from([1u8, 2, 3, 4]);
        assert_eq!(set.index(1), &2);
        assert_eq!(set.index(2..), &[3, 4]);
    }

    #[test]
    fn test_intersection() {
        let a = Multiset::<u32, 10>::from_array([1, 6, 3, 8, 5, 2, 7, 4, 10, 20]);
        let b = Multiset::<u32, 10>::from_array([5, 2, 7, 4, 1, 6, 3, 8, 20, 10]);
        let expected = Multiset::<u32, 10>::from_array([1, 2, 3, 4, 1, 2, 3, 4, 10, 10]);
        assert_eq!(a.intersection(&b), expected)
    }

    #[test]
    fn test_union() {
        let a = Multiset::<u32, 10>::from_array([1, 6, 3, 8, 5, 2, 7, 4, 10, 20]);
        let b = Multiset::<u32, 10>::from_array([5, 2, 7, 4, 1, 6, 3, 8, 20, 10]);
        let expected = Multiset::<u32, 10>::from_array([5, 6, 7, 8, 5, 6, 7, 8, 20, 20]);
        assert_eq!(a.union(&b), expected)
    }

    #[test]
    fn test_total() {
        let a = Multiset::<u32, 4>::from_array([200, 0, 0, 0]);
        assert_eq!(a.total(), 200);

        let b = Multiset::<u16, 4>::from_array([10, 0, 20, 0]);
        assert_eq!(b.total(), 30);

        let c = Multiset::<u8, 4>::from_array([255, 0, 100, 100]);
        assert_eq!(c.total(), 255);
    }

    #[test]
    fn test_wide_total() {
        let a = Multiset::<u32, 4>::from_array([200, 0, 0, 0]);
        assert_eq!(a.total_as::<u64>(), 200);

        let b = Multiset::<u16, 4>::from_array([10, 0, 20, 0]);
        assert_eq!(b.total_as::<u32>(), 30);

        let c = Multiset::<u8, 4>::from_array([255, 0, 100, 100]);
        assert_eq!(c.total_as::<u16>(), 455);
    }

    #[test]
    fn test_collision_entropy() {
        let simple = Multiset::<u32, 4>::from_array([200, 0, 0, 0]);
        assert_eq!(simple.collision_entropy(), 0.0);

        let set = Multiset::<u32, 4>::from_array([2, 1, 1, 0]);
        assert_relative_eq!(
            set.collision_entropy(),
            1.415037499278844,
            epsilon = f64::EPSILON
        );
    }
}
