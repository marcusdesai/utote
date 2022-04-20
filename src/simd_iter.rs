use num_traits::{AsPrimitive, SaturatingAdd, SaturatingSub, Unsigned};
use rand::{Rng, RngCore};
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{Index, IndexMut};
use std::simd::*;
use std::slice::{Iter, IterMut, SliceIndex};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Multiset<T, const SIZE: usize>([T; SIZE])
where
    T: Unsigned;

impl<T: Unsigned, I: SliceIndex<[T]>, const SIZE: usize> Index<I> for Multiset<T, SIZE> {
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        Index::index(&self.0, index)
    }
}

impl<T: Unsigned, I: SliceIndex<[T]>, const SIZE: usize> IndexMut<I> for Multiset<T, SIZE> {
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        IndexMut::index_mut(&mut self.0, index)
    }
}

impl<T: Unsigned, const SIZE: usize> Multiset<T, SIZE> {
    pub const SIZE: usize = SIZE;

    pub const fn len(&self) -> usize {
        SIZE
    }

    pub fn new() -> Self
    where
        T: Copy,
    {
        Multiset([T::zero(); SIZE])
    }

    pub const fn from_array(data: [T; SIZE]) -> Self {
        Multiset(data)
    }

    pub const fn as_array(&self) -> &[T; SIZE] {
        &self.0
    }

    pub fn as_mut_array(&mut self) -> &mut [T; SIZE] {
        &mut self.0
    }

    pub const fn to_array(self) -> [T; SIZE]
    where
        T: Copy,
    {
        self.0
    }

    #[inline]
    pub fn iter(&self) -> Iter<T> {
        self.0.iter()
    }

    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<T> {
        self.0.iter_mut()
    }

    #[inline]
    pub fn count_non_zero(&self) -> usize
    where
        T: Ord,
    {
        self.iter().filter(|&c| *c > T::zero()).count()
    }

    #[inline]
    pub fn count_zero(&self) -> usize
    where
        T: Ord,
    {
        SIZE - self.count_non_zero()
    }

    #[inline]
    pub fn zeroed(&mut self) {
        self.0.iter_mut().for_each(|m| *m = T::zero());
    }

    #[inline]
    pub fn intersection(&self, other: &Self) -> Self
    where
        T: Copy + Ord,
    {
        let mut out: [T; SIZE] = [T::zero(); SIZE];
        for (elem, (a, b)) in out.iter_mut().zip(self.iter().zip(other.iter())) {
            *elem = *a.min(b)
        }
        Multiset::from_array(out)
    }

    #[inline]
    pub fn intersection_mut(&mut self, other: &Self)
    where
        T: Copy + Ord,
    {
        for (a, b) in self.iter_mut().zip(other.iter()) {
            *a = (*a).min(*b)
        }
    }

    #[inline]
    pub fn union(&self, other: &Self) -> Self
    where
        T: Copy + Ord,
    {
        let mut out: [T; SIZE] = [T::zero(); SIZE];
        for (elem, (a, b)) in out.iter_mut().zip(self.iter().zip(other.iter())) {
            *elem = *a.max(b)
        }
        Multiset::from_array(out)
    }

    #[inline]
    pub fn union_mut(&mut self, other: &Self)
    where
        T: Copy + Ord,
    {
        for (a, b) in self.iter_mut().zip(other.iter()) {
            *a = (*a).max(*b)
        }
    }

    #[inline]
    pub fn difference(&self, other: &Self) -> Self
    where
        T: Copy + SaturatingSub,
    {
        let mut out = [T::zero(); SIZE];
        for (elem, (a, b)) in out.iter_mut().zip(self.iter().zip(other.iter())) {
            *elem = a.saturating_sub(b)
        }
        Multiset::from_array(out)
    }

    #[inline]
    pub fn difference_mut(&mut self, other: &Self)
    where
        T: SaturatingSub,
    {
        for (a, b) in self.iter_mut().zip(other.iter()) {
            *a = a.saturating_sub(b)
        }
    }

    #[inline]
    pub fn complement(&self, other: &Self) -> Self
    where
        T: Copy + SaturatingSub,
    {
        let mut out = [T::zero(); SIZE];
        for (elem, (a, b)) in out.iter_mut().zip(self.iter().zip(other.iter())) {
            *elem = b.saturating_sub(a)
        }
        Multiset::from_array(out)
    }

    #[inline]
    pub fn complement_mut(&mut self, other: &Self)
    where
        T: SaturatingSub,
    {
        for (a, b) in self.iter_mut().zip(other.iter()) {
            *a = b.saturating_sub(a)
        }
    }

    #[inline]
    pub fn sum(&self, other: &Self) -> Self
        where
            T: Copy + SaturatingAdd,
    {
        let mut out = [T::zero(); SIZE];
        for (elem, (a, b)) in out.iter_mut().zip(self.iter().zip(other.iter())) {
            *elem = a.saturating_add(b)
        }
        Multiset::from_array(out)
    }

    #[inline]
    pub fn sum_mut(&mut self, other: &Self)
        where
            T: SaturatingAdd,
    {
        for (a, b) in self.iter_mut().zip(other.iter()) {
            *a = a.saturating_add(b)
        }
    }

    #[inline]
    pub fn symmetric_difference(&self, other: &Self) -> Self
    where
        T: Copy + Ord,
    {
        let mut out = [T::zero(); SIZE];
        for (elem, (a, b)) in out.iter_mut().zip(self.iter().zip(other.iter())) {
            *elem = if a > b { *a - *b } else { *b - *a }
        }
        Multiset::from_array(out)
    }

    #[inline]
    pub fn symmetric_difference_mut(&mut self, other: &Self)
    where
        T: Copy + Ord,
    {
        for (a, b) in self.iter_mut().zip(other.iter()) {
            *a = if *a > *b { *a - *b } else { *b - *a }
        }
    }

    #[inline]
    pub fn is_singleton(&self) -> bool
    where
        T: Ord,
    {
        self.count_non_zero() == 1
    }

    #[inline]
    pub fn is_empty(&self) -> bool
    where
        T: Copy,
    {
        self.0 == [T::zero(); SIZE]
    }

    #[inline]
    pub fn is_disjoint_scalar(&self, other: &Self) -> bool
    where
        T: Copy + Ord,
    {
        self.iter()
            .zip(other.iter())
            .all(|(a, b)| (*a).min(*b) == T::zero())
    }

    #[inline]
    pub fn is_disjoint(&self, other: &Self) -> bool
    where
        T: Ord + SimdElement,
    {
        const LANES: usize = 8;
        let mut disjoint = self
            .0
            .array_chunks()
            .zip(other.0.array_chunks())
            .all(|(c1, c2)| {
                let s1: Simd<T, LANES> = Simd::from_array(*c1);
                let s2 = Simd::from_array(*c2);
                s1.lanes_lt(s2).select(s1, s2) == Simd::splat(T::zero())
            });
        if disjoint && SIZE % LANES != 0 {
            let mut temp = [T::zero(); LANES];
            let rem1 = &self.0[(SIZE - (SIZE % LANES))..];
            let rem2 = &other.0[(SIZE - (SIZE % LANES))..];
            for (elem, (a, b)) in temp.iter_mut().zip(rem1.iter().zip(rem2.iter())) {
                *elem = (*a).min(*b)
            }
            disjoint &= Simd::from_array(temp) == Simd::splat(T::zero())
        }
        disjoint
    }

    #[inline]
    pub fn is_subset_scalar(&self, other: &Self) -> bool
    where
        T: Ord,
    {
        self.iter().zip(other.iter()).all(|(a, b)| a <= b)
    }

    #[inline]
    pub fn is_subset(&self, other: &Self) -> bool
    where
        T: Ord + SimdElement,
    {
        const LANES: usize = 8;
        let mut subset = self
            .0
            .array_chunks()
            .zip(other.0.array_chunks())
            .all(|(c1, c2)| {
                let s1: Simd<T, LANES> = Simd::from_array(*c1);
                let s2 = Simd::from_array(*c2);
                s1.lanes_le(s2).all()
            });
        if subset && SIZE % LANES != 0 {
            let rem1 = &self.0[(SIZE - (SIZE % LANES))..];
            let rem2 = &other.0[(SIZE - (SIZE % LANES))..];
            for (a, b) in rem1.iter().zip(rem2.iter()) {
                subset &= a <= b
            }
        }
        subset
    }

    #[inline]
    pub fn is_superset_scalar(&self, other: &Self) -> bool
    where
        T: Ord,
    {
        self.iter().zip(other.iter()).all(|(a, b)| a >= b)
    }

    #[inline]
    pub fn is_superset(&self, other: &Self) -> bool
    where
        T: Ord + SimdElement,
    {
        const LANES: usize = 8;
        let mut subset = self
            .0
            .array_chunks()
            .zip(other.0.array_chunks())
            .all(|(c1, c2)| {
                let s1: Simd<T, LANES> = Simd::from_array(*c1);
                let s2 = Simd::from_array(*c2);
                s1.lanes_ge(s2).all()
            });
        if subset && SIZE % LANES != 0 {
            let rem1 = &self.0[(SIZE - (SIZE % LANES))..];
            let rem2 = &other.0[(SIZE - (SIZE % LANES))..];
            for (a, b) in rem1.iter().zip(rem2.iter()) {
                subset &= a >= b
            }
        }
        subset
    }

    #[inline]
    pub fn is_proper_subset_scalar(&self, other: &Self) -> bool
    where
        T: Ord,
    {
        self.iter().zip(other.iter()).all(|(a, b)| a < b)
    }

    #[inline]
    pub fn is_proper_subset(&self, other: &Self) -> bool
    where
        T: Ord + SimdElement,
    {
        const LANES: usize = 8;
        let mut subset = self
            .0
            .array_chunks()
            .zip(other.0.array_chunks())
            .all(|(c1, c2)| {
                let s1: Simd<T, LANES> = Simd::from_array(*c1);
                let s2 = Simd::from_array(*c2);
                s1.lanes_lt(s2).all()
            });
        if subset && SIZE % LANES != 0 {
            let rem1 = &self.0[(SIZE - (SIZE % LANES))..];
            let rem2 = &other.0[(SIZE - (SIZE % LANES))..];
            for (a, b) in rem1.iter().zip(rem2.iter()) {
                subset &= a < b
            }
        }
        subset
    }

    #[inline]
    pub fn is_proper_superset_scalar(&self, other: &Self) -> bool
    where
        T: Ord,
    {
        self.iter().zip(other.iter()).all(|(a, b)| a > b)
    }

    #[inline]
    pub fn is_proper_superset(&self, other: &Self) -> bool
    where
        T: Ord + SimdElement,
    {
        const LANES: usize = 8;
        let mut subset = self
            .0
            .array_chunks()
            .zip(other.0.array_chunks())
            .all(|(c1, c2)| {
                let s1: Simd<T, LANES> = Simd::from_array(*c1);
                let s2 = Simd::from_array(*c2);
                s1.lanes_gt(s2).all()
            });
        if subset && SIZE % LANES != 0 {
            let rem1 = &self.0[(SIZE - (SIZE % LANES))..];
            let rem2 = &other.0[(SIZE - (SIZE % LANES))..];
            for (a, b) in rem1.iter().zip(rem2.iter()) {
                subset &= a > b
            }
        }
        subset
    }

    #[deprecated(since = "0.5.0", note = "equivalent to negation of superset")]
    #[inline]
    pub fn is_any_lesser(&self, other: &Self) -> bool
    where
        T: Ord + SimdElement,
    {
        !self.is_superset(other)
    }

    #[deprecated(since = "0.5.0", note = "equivalent to negation of subset")]
    #[inline]
    pub fn is_any_greater(&self, other: &Self) -> bool
    where
        T: Ord + SimdElement,
    {
        !self.is_subset(other)
    }

    #[inline]
    pub fn total(&self) -> T
    where
        T: Copy + Sum,
    {
        self.iter().copied().sum()
    }

    #[inline]
    pub fn wide_total<U: From<T> + Sum>(&self) -> U
    where
        T: Copy,
    {
        self.iter().map(|&c| c.into()).sum()
    }

    #[inline]
    pub fn elem_count_max(&self) -> Option<(usize, &T)>
    where
        T: Ord,
    {
        self.iter().enumerate().max_by_key(|(_, count)| *count)
    }

    #[inline]
    pub fn elem_max(&self) -> Option<usize>
    where
        T: Ord,
    {
        self.elem_count_max().map(|t| t.0)
    }

    #[inline]
    pub fn count_max(&self) -> Option<T>
    where
        T: Copy + Ord,
    {
        self.iter().copied().max()
    }

    #[inline]
    pub fn elem_count_min(&self) -> Option<(usize, &T)>
    where
        T: Ord,
    {
        self.iter().enumerate().min_by_key(|(_, count)| *count)
    }

    #[inline]
    pub fn elem_min(&self) -> Option<usize>
    where
        T: Ord,
    {
        self.elem_count_min().map(|t| t.0)
    }

    #[inline]
    pub fn count_min(&self) -> Option<T>
    where
        T: Copy + Ord,
    {
        self.iter().copied().min()
    }

    #[inline]
    pub fn choose(&mut self, elem: usize)
    where
        T: Copy,
    {
        let mut res = [T::zero(); SIZE];
        if elem < SIZE {
            res[elem] = self.0[elem];
        }
        self.0 = res
    }

    #[inline]
    pub fn choose_random<R: RngCore>(&mut self, rng: &mut R)
    where
        u64: From<T>,
        T: Copy,
    {
        if SIZE == 1 {
            return;
        }
        let total: u64 = self.wide_total();
        if total == 0 {
            return;
        }
        let choice_value = rng.gen_range(1..=total);
        let mut res = [T::zero(); SIZE];
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

    #[inline]
    pub fn collision_entropy_scalar(&self) -> f64
    where
        T: AsPrimitive<f64> + Sum,
    {
        let total = self.total().as_();
        let mut s = 0.0;
        for count in self.iter() {
            s += ((*count).as_() / total).powf(2.0)
        }
        -(s.log2())
    }

    #[inline]
    pub fn collision_entropy(&self) -> f64
    where
        T: Sum + SimdElement,
    {
        const LANES: usize = 4;
        let total_vec: Simd<f64, LANES> = Simd::splat(self.total()).cast();
        let mut out = Simd::splat(0.0);
        for a in self.0.array_chunks() {
            let fs = Simd::from_array(*a).cast();
            let prob = fs / total_vec;
            out += prob * prob;
        }
        if SIZE % LANES != 0 {
            let mut temp = [T::zero(); LANES];
            let rem_slice = &self.0[(SIZE - (SIZE % LANES))..];
            for (elem, count) in temp.iter_mut().zip(rem_slice.iter()) {
                *elem = *count
            }
            let fs = Simd::from_array(temp).cast();
            let prob = fs / total_vec;
            out += prob * prob;
        }
        -(out.reduce_sum().log2())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // todo: tests where SIZE = 0

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
    fn test_collision_entropy() {
        let simple = Multiset::<u32, 4>::from_array([200, 0, 0, 0]);
        assert_eq!(simple.collision_entropy(), 0.0);
        assert_eq!(simple.collision_entropy_scalar(), 0.0);

        let set = Multiset::<u32, 4>::from_array([2, 1, 1, 0]);
        assert_relative_eq!(
            set.collision_entropy(),
            1.415037499278844,
            epsilon = f64::EPSILON
        );
        assert_relative_eq!(
            set.collision_entropy_scalar(),
            1.415037499278844,
            epsilon = f64::EPSILON
        );
    }
}
