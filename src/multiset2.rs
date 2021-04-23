use num_traits::{AsPrimitive, Zero};
use paste::paste;
use rand::prelude::*;
use std::cmp::Ordering;
use std::fmt::{Debug, Formatter, Result};
use std::hash::{Hash, Hasher};
use std::iter::FromIterator;
use std::mem;
use std::ops::AddAssign;
use std::slice::{Iter, IterMut};

use crate::chunks::ChunkUtils;
use crate::small_num::SmallNumConsts;

#[repr(transparent)]
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

impl<N, const SIZE: usize> Multiset2<N, SIZE> {
    pub const SIZE: usize = SIZE;

    /// blah
    #[inline]
    pub const fn from_array(data: [N; SIZE]) -> Self {
        Multiset2 { data }
    }

    // todo: deprecate len
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

    /// Returns a Multiset of the given array size with all elements set to zero.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use utote::MSu8;
    /// let multiset = MSu8::<4>::empty();
    /// ```
    #[inline]
    pub fn empty() -> Self
    where
        N: Zero,
    {
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
    pub fn from_slice(slice: &[N]) -> Self
    where
        N: Zero,
    {
        slice.iter().collect()
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
    pub fn clear(&mut self)
    where
        N: Zero,
    {
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
    pub fn contains(&self, elem: usize) -> bool
    where
        N: Zero + PartialOrd,
    {
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
    pub unsafe fn contains_unchecked(&self, elem: usize) -> bool
    where
        N: Zero + PartialOrd,
    {
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
    pub fn remove(&mut self, elem: usize)
    where
        N: Zero,
    {
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
    pub unsafe fn remove_unchecked(&mut self, elem: usize)
    where
        N: Zero,
    {
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
    pub fn is_empty(&self) -> bool
    where
        N: PartialEq + Zero,
    {
        self.data == [N::zero(); SIZE]
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
    pub fn argmax(&self) -> (usize, N)
    where
        N: Ord,
    {
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
    pub fn imax(&self) -> usize
    where
        N: Ord,
    {
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
    pub fn max(&self) -> N
    where
        N: Ord,
    {
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
    pub fn argmin(&self) -> (usize, N)
    where
        N: Ord,
    {
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
    pub fn imin(&self) -> usize
    where
        N: Ord,
    {
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
    pub fn min(&self) -> N
    where
        N: Ord,
    {
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
    pub fn choose(&mut self, elem: usize)
    where
        N: Zero,
    {
        let mut res = [N::zero(); SIZE];
        if elem < SIZE {
            unsafe { *res.get_unchecked_mut(elem) = *self.data.get_unchecked(elem) };
        }
        self.data = res
    }
}

impl<N: Copy, const SIZE: usize> Clone for Multiset2<N, SIZE> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<N: Copy, const SIZE: usize> Copy for Multiset2<N, SIZE> {}

impl<N, const SIZE: usize> AddAssign for Multiset2<N, SIZE>
where
    N: AddAssign + Copy,
{
    // todo: use SIMD
    fn add_assign(&mut self, rhs: Self) {
        self.data
            .iter_mut()
            .zip(rhs.data.iter())
            .for_each(|(l, r)| *l += *r);
    }
}

impl<N: PartialEq, const SIZE: usize> PartialEq for Multiset2<N, SIZE> {
    // Array compare will always use memcmp because N will be a unit, so no need for SIMD
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

use lazy_static::lazy_static;
use packed_simd::*;

enum CPUFeature {
    AVX2,
    AVX,
    SSE42,
    DEF,
}

impl CPUFeature {
    #[inline]
    fn val(&self) -> &CPUFeature {
        &self
    }
}

lazy_static! {
    static ref CPU_FEATURE: CPUFeature = {
        if is_x86_feature_detected!("avx2") {
            CPUFeature::AVX2
        } else if is_x86_feature_detected!("avx") {
            CPUFeature::AVX
        } else if is_x86_feature_detected!("sse4.2") {
            CPUFeature::SSE42
        } else {
            CPUFeature::DEF
        }
    };
}

macro_rules! simd_variants {
    ($name:ty, $fn_macro:ident, $lanes128:expr, $lanes256:expr, $simd128:ty, $simd256:ty
    $(, $simd_f128:ty, $simd_f256:ty, $simd_m128:ty, $simd_m256:ty)*) => {
        paste! {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            #[target_feature(enable = "avx2,fma")]
            $fn_macro! { [<_ $name _avx2>], $simd256, $lanes256 $(, $simd_f256, $simd_m256)* }

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            #[target_feature(enable = "avx")]
            $fn_macro! { [<_ $name _avx>], $simd256, $lanes256 $(, $simd_f256, $simd_m256)* }

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            #[target_feature(enable = "sse4.2")]
            $fn_macro! { [<_ $name _sse42>], $simd128, $lanes128 $(, $simd_f128, $simd_m128)* }
        }
    };
}

macro_rules! simd_dispatch {
    ($(#[$outer:meta])*
    pub fn $name:ident (&self $(, $arg:ident: $typ:ty)*) -> $ret:ty; default = $def:ident) => {
        paste! {
            $(#[$outer])*
            #[inline]
            pub fn $name(&self, $($arg: $typ),*) -> $ret {
                unsafe {
                    match CPU_FEATURE.val() {
                        CPUFeature::AVX2 => self.[<_ $name _avx2>]($($arg),*),
                        CPUFeature::AVX => self.[<_ $name _avx>]($($arg),*),
                        CPUFeature::SSE42 => self.[<_ $name _sse42>]($($arg),*),
                        CPUFeature::DEF => self.$def($($arg),*),
                    }
                }
            }
        }
    };
}

macro_rules! intersection_simd {
    ($name:ident, $simd:ty, $lanes:expr) => {
        #[doc(hidden)]
        #[inline]
        unsafe fn $name(&self, other: &Self) -> Self {
            let data = self
                .data
                .zip_map_chunks::<_, $lanes>(&other.data, |a, b, out| {
                    let simd_a = <$simd>::from_slice_unaligned_unchecked(a);
                    let simd_b = <$simd>::from_slice_unaligned_unchecked(b);
                    simd_a.min(simd_b).write_to_slice_unaligned_unchecked(out);
                });
            Multiset2 { data }
        }
    };
}

macro_rules! union_simd {
    ($name:ident, $simd:ty, $lanes:expr) => {
        #[doc(hidden)]
        #[inline]
        unsafe fn $name(&self, other: &Self) -> Self {
            let data = self
                .data
                .zip_map_chunks::<_, $lanes>(&other.data, |a, b, out| {
                    let simd_a = <$simd>::from_slice_unaligned_unchecked(a);
                    let simd_b = <$simd>::from_slice_unaligned_unchecked(b);
                    simd_a.max(simd_b).write_to_slice_unaligned_unchecked(out);
                });
            Multiset2 { data }
        }
    };
}

macro_rules! count_zero_simd {
    ($name:ident, $simd:ty, $lanes:expr) => {
        #[doc(hidden)]
        #[inline]
        unsafe fn $name(&self) -> usize {
            SIZE - self.data.fold_chunks::<_, _, $lanes>(0, |acc, slice| {
                let vec = <$simd>::from_slice_unaligned_unchecked(slice);
                acc + vec.gt(<$simd>::ZERO).bitmask().count_ones() as usize
            })
        }
    };
}

macro_rules! count_non_zero_simd {
    ($name:ident, $simd:ty, $lanes:expr) => {
        #[doc(hidden)]
        #[inline]
        unsafe fn $name(&self) -> usize {
            self.data.fold_chunks::<_, _, $lanes>(0, |acc, slice| {
                let vec = <$simd>::from_slice_unaligned_unchecked(slice);
                acc + vec.gt(<$simd>::ZERO).bitmask().count_ones() as usize
            })
        }
    };
}

// SIMD implementations will go in this impl block for now
impl<const SIZE: usize> Multiset2<u16, SIZE> {
    #[doc(hidden)]
    #[inline]
    fn _intersection_default(&self, other: &Self) -> Self {
        self.zip_map(other, |s1, s2| s1.min(s2))
    }

    simd_variants!(intersection, intersection_simd, 8, 16, u16x8, u16x16);
    simd_dispatch! {
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
        pub fn intersection(&self, other: &Self) -> Self;
        default = _intersection_default
    }

    #[doc(hidden)]
    #[inline]
    fn _union_default(&self, other: &Self) -> Self {
        self.zip_map(other, |s1, s2| s1.max(s2))
    }

    simd_variants!(union, union_simd, 8, 16, u16x8, u16x16);
    simd_dispatch! {
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
        pub fn union(&self, other: &Self) -> Self;
        default = _union_default
    }

    #[doc(hidden)]
    #[inline]
    fn _count_zero_default(&self) -> usize {
        self.fold(SIZE, |acc, elem| acc - elem.min(1) as usize)
    }

    simd_variants!(count_zero, count_zero_simd, 8, 16, u16x8, u16x16);
    simd_dispatch! {
        /// Return the number of elements whose counter is zero.
        ///
        /// # Examples
        ///
        /// ```no_run
        /// use utote::MSu8;
        /// let multiset = MSu8::<4>::from_slice(&[1, 0, 0, 0]);
        /// assert_eq!(multiset.count_zero(), 3);
        /// ```
        pub fn count_zero(&self) -> usize;
        default = _count_zero_default
    }

    #[doc(hidden)]
    #[inline]
    pub fn _count_non_zero_default(&self) -> usize {
        self.fold(0, |acc, elem| acc + elem.min(1) as usize)
    }

    simd_variants!(count_non_zero, count_non_zero_simd, 8, 16, u16x8, u16x16);
    simd_dispatch! {
        /// Return the number of elements whose counter is non-zero.
        ///
        /// # Examples
        ///
        /// ```no_run
        /// use utote::MSu8;
        /// let multiset = MSu8::<4>::from_slice(&[1, 0, 0, 0]);
        /// assert_eq!(multiset.count_non_zero(), 1);
        /// ```
        pub fn count_non_zero(&self) -> usize;
        default = _count_non_zero_default
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

// trait SIMDType<N> {
//     type SIMD128: SIMDFunc<N> + Add<N>;
//     type SIMD256: SIMDFunc<N> + Add<N>;
// }
//
// impl SIMDType<u16> for u16 {
//     type SIMD128 = u16x8;
//     type SIMD256 = u16x16;
// }
//
// trait SIMDFunc<N> {
//     unsafe fn from_slice_unaligned_unchecked(t: &[N]) -> Self;
// }
//
// impl SIMDFunc<u16> for u16x8 {
//     unsafe fn from_slice_unaligned_unchecked(t: &[u16]) -> Self {
//         u16x8::from_slice_unaligned_unchecked(t)
//     }
// }
//
// impl SIMDFunc<u16> for u16x16 {
//     unsafe fn from_slice_unaligned_unchecked(t: &[u16]) -> Self {
//         u16x16::from_slice_unaligned_unchecked(t)
//     }
// }
//
// impl<N: SIMDType<N>, const SIZE: usize> Multiset2<N, SIZE> {
//     pub fn _d(t: &[N]) {
//         unsafe { N::SIMD128::from_slice_unaligned_unchecked(t) };
//     }
// }
