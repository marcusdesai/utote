use crate::counter::Counter;
use rand::{Rng, RngCore};
use std::cmp::Ordering;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign};
#[cfg(feature = "simd")]
use std::simd::*;
use std::slice::{Iter, IterMut, SliceIndex};
use num_traits::Pow;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(transparent)]
pub struct Multiset<T: Counter, const SIZE: usize>([T; SIZE]);

////////////////////////////////////////////////////////////////////////////////
// Inherent methods
////////////////////////////////////////////////////////////////////////////////

impl<T: Counter, const SIZE: usize> Multiset<T, SIZE> {
    pub const SIZE: usize = SIZE;

    pub const fn len(&self) -> usize {
        SIZE
    }

    pub const fn new() -> Self {
        Multiset([T::ZERO; SIZE])
    }

    #[deprecated(since = "0.7.0", note = "equivalent to new")]
    pub fn empty() -> Self {
        Self::new()
    }

    pub const fn repeat(count: T) -> Self {
        Multiset([count; SIZE])
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

    pub const fn to_array(self) -> [T; SIZE] {
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
    pub fn count_non_zero(&self) -> usize {
        self.iter().filter(|&c| *c > T::ZERO).count()
    }

    #[inline]
    pub fn count_zero(&self) -> usize {
        SIZE - self.count_non_zero()
    }

    #[inline]
    pub fn zeroed(&mut self) {
        self.0.iter_mut().for_each(|m| *m = T::ZERO);
    }

    #[inline]
    pub fn intersection(&self, other: &Self) -> Self {
        let mut out: [T; SIZE] = [T::ZERO; SIZE];
        for (elem, (a, b)) in out.iter_mut().zip(self.iter().zip(other.iter())) {
            *elem = *a.min(b)
        }
        Multiset::from_array(out)
    }

    #[inline]
    pub fn intersection_mut(&mut self, other: &Self) {
        for (a, b) in self.iter_mut().zip(other.iter()) {
            *a = (*a).min(*b)
        }
    }

    #[inline]
    pub fn union(&self, other: &Self) -> Self {
        let mut out: [T; SIZE] = [T::ZERO; SIZE];
        for (elem, (a, b)) in out.iter_mut().zip(self.iter().zip(other.iter())) {
            *elem = *a.max(b)
        }
        Multiset::from_array(out)
    }

    #[inline]
    pub fn union_mut(&mut self, other: &Self) {
        for (a, b) in self.iter_mut().zip(other.iter()) {
            *a = (*a).max(*b)
        }
    }

    #[inline]
    pub fn difference(&self, other: &Self) -> Self {
        let mut out = [T::ZERO; SIZE];
        for (elem, (a, b)) in out.iter_mut().zip(self.iter().zip(other.iter())) {
            *elem = a.saturating_sub(*b)
        }
        Multiset::from_array(out)
    }

    #[inline]
    pub fn difference_mut(&mut self, other: &Self) {
        for (a, b) in self.iter_mut().zip(other.iter()) {
            *a = a.saturating_sub(*b)
        }
    }

    #[inline]
    pub fn complement(&self, other: &Self) -> Self {
        let mut out = [T::ZERO; SIZE];
        for (elem, (a, b)) in out.iter_mut().zip(self.iter().zip(other.iter())) {
            *elem = b.saturating_sub(*a)
        }
        Multiset::from_array(out)
    }

    #[inline]
    pub fn complement_mut(&mut self, other: &Self) {
        for (a, b) in self.iter_mut().zip(other.iter()) {
            *a = b.saturating_sub(*a)
        }
    }

    #[inline]
    pub fn sum(&self, other: &Self) -> Self {
        let mut out = [T::ZERO; SIZE];
        for (elem, (a, b)) in out.iter_mut().zip(self.iter().zip(other.iter())) {
            *elem = a.saturating_add(*b)
        }
        Multiset::from_array(out)
    }

    #[inline]
    pub fn sum_mut(&mut self, other: &Self) {
        for (a, b) in self.iter_mut().zip(other.iter()) {
            *a = a.saturating_add(*b)
        }
    }

    #[inline]
    pub fn symmetric_difference(&self, other: &Self) -> Self {
        let mut out = [T::ZERO; SIZE];
        for (elem, (a, b)) in out.iter_mut().zip(self.iter().zip(other.iter())) {
            *elem = a.abs_diff(*b)
        }
        Multiset::from_array(out)
    }

    #[inline]
    pub fn symmetric_difference_mut(&mut self, other: &Self) {
        for (a, b) in self.iter_mut().zip(other.iter()) {
            *a = a.abs_diff(*b)
        }
    }

    #[inline]
    pub fn is_singleton(&self) -> bool {
        self.count_non_zero() == 1
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0 == [T::ZERO; SIZE]
    }

    #[cfg(not(feature = "simd"))]
    #[inline]
    pub fn is_disjoint(&self, other: &Self) -> bool {
        self.iter()
            .zip(other.iter())
            .all(|(a, b)| (*a).min(*b) == T::ZERO)
    }

    #[cfg(feature = "simd")]
    #[inline]
    pub fn is_disjoint(&self, other: &Self) -> bool {
        const LANES: usize = 8;
        let mut disjoint = self
            .0
            .array_chunks::<LANES>()
            .zip(other.0.array_chunks())
            .all(|(c1, c2)| {
                let s1 = Simd::from_array(*c1);
                let s2 = Simd::from_array(*c2);
                s1.lanes_lt(s2)
                    .select(s1, s2)
                    .lanes_eq(Simd::splat(T::ZERO))
                    .all()
            });
        if disjoint && SIZE % LANES != 0 {
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
    #[inline]
    pub fn is_subset(&self, other: &Self) -> bool {
        self.iter().zip(other.iter()).all(|(a, b)| a <= b)
    }

    #[cfg(feature = "simd")]
    #[inline]
    pub fn is_subset(&self, other: &Self) -> bool {
        const LANES: usize = 8;
        let mut subset = self
            .0
            .array_chunks::<LANES>()
            .zip(other.0.array_chunks())
            .all(|(c1, c2)| {
                let s1 = Simd::from_array(*c1);
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

    #[cfg(not(feature = "simd"))]
    #[inline]
    pub fn is_superset(&self, other: &Self) -> bool {
        self.iter().zip(other.iter()).all(|(a, b)| a >= b)
    }

    #[cfg(feature = "simd")]
    #[inline]
    pub fn is_superset(&self, other: &Self) -> bool {
        const LANES: usize = 8;
        let mut subset = self
            .0
            .array_chunks::<LANES>()
            .zip(other.0.array_chunks())
            .all(|(c1, c2)| {
                let s1 = Simd::from_array(*c1);
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

    #[cfg(not(feature = "simd"))]
    #[inline]
    pub fn is_proper_subset(&self, other: &Self) -> bool {
        self.iter().zip(other.iter()).all(|(a, b)| a < b)
    }

    #[cfg(feature = "simd")]
    #[inline]
    pub fn is_proper_subset(&self, other: &Self) -> bool {
        const LANES: usize = 8;
        let mut subset = self
            .0
            .array_chunks::<LANES>()
            .zip(other.0.array_chunks())
            .all(|(c1, c2)| {
                let s1 = Simd::from_array(*c1);
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

    #[cfg(not(feature = "simd"))]
    #[inline]
    pub fn is_proper_superset(&self, other: &Self) -> bool {
        self.iter().zip(other.iter()).all(|(a, b)| a > b)
    }

    #[cfg(feature = "simd")]
    #[inline]
    pub fn is_proper_superset(&self, other: &Self) -> bool {
        const LANES: usize = 8;
        let mut subset = self
            .0
            .array_chunks::<LANES>()
            .zip(other.0.array_chunks())
            .all(|(c1, c2)| {
                let s1 = Simd::from_array(*c1);
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

    #[deprecated(since = "0.7.0", note = "equivalent to negation of superset")]
    #[inline]
    pub fn is_any_lesser(&self, other: &Self) -> bool {
        !self.is_superset(other)
    }

    #[deprecated(since = "0.7.0", note = "equivalent to negation of subset")]
    #[inline]
    pub fn is_any_greater(&self, other: &Self) -> bool {
        !self.is_subset(other)
    }

    #[inline]
    pub fn total(&self) -> T {
        self.iter().copied().sum()
    }

    // #[inline(never)]
    pub fn wide_total<U: From<T> + Sum>(&self) -> U {
        self.iter().map(|&c| c.into()).sum()
    }

    #[inline]
    pub fn elem_count_max(&self) -> Option<(usize, &T)> {
        self.iter().enumerate().max_by_key(|(_, count)| *count)
    }

    #[inline]
    pub fn elem_max(&self) -> Option<usize> {
        self.elem_count_max().map(|t| t.0)
    }

    #[inline]
    pub fn count_max(&self) -> Option<T> {
        self.iter().copied().max()
    }

    #[inline]
    pub fn elem_count_min(&self) -> Option<(usize, &T)> {
        self.iter().enumerate().min_by_key(|(_, count)| *count)
    }

    #[inline]
    pub fn elem_min(&self) -> Option<usize> {
        self.elem_count_min().map(|t| t.0)
    }

    #[inline]
    pub fn count_min(&self) -> Option<T> {
        self.iter().copied().min()
    }

    #[inline]
    pub fn choose(&mut self, elem: usize) {
        let mut res = [T::ZERO; SIZE];
        if elem < SIZE {
            res[elem] = self.0[elem];
        }
        self.0 = res
    }

    #[inline]
    pub fn choose_random<R: RngCore>(&mut self, rng: &mut R)
    where
        u64: From<T>,
    {
        if SIZE == 1 {
            return;
        }
        let total: u64 = self.wide_total();
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

    #[cfg(not(feature = "simd"))]
    #[inline]
    pub fn collision_entropy(&self) -> f64
    where
        u64: From<T>,
    {
        let total = self.wide_total::<u64>() as f64;
        let mut s: f64 = 0.0;
        for count in self.iter() {
            s += ((*count).as_f64() / total).powf(2.0)
        }
        -s.log2()
    }

    #[cfg(feature = "simd")]
    #[inline]
    pub fn collision_entropy(&self) -> f64
    where
        u64: From<T>,
    {
        const LANES: usize = 4;
        let total: u64 = self.wide_total();
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

    pub fn shannon_entropy(&self) -> f64
    where
        u64: From<T>,
    {
        let total = self.wide_total::<u64>() as f64;
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

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        Index::index(&self.0, index)
    }
}

impl<T: Counter, I: SliceIndex<[T]>, const SIZE: usize> IndexMut<I> for Multiset<T, SIZE> {
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        IndexMut::index_mut(&mut self.0, index)
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
        for (e_self, e_other) in self.0.iter().zip(other.0.iter()) {
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
        let mut out = [T::ZERO; SIZE];
        for (elem, val) in out.iter_mut().zip(self.0.iter()) {
            *elem = val.saturating_add(rhs)
        }
        Multiset::from_array(out)
    }
}

impl<T: Counter, const SIZE: usize> AddAssign for Multiset<T, SIZE> {
    fn add_assign(&mut self, rhs: Self) {
        self.sum_mut(&rhs)
    }
}

impl<T: Counter, const SIZE: usize> AddAssign<T> for Multiset<T, SIZE> {
    fn add_assign(&mut self, rhs: T) {
        for val in self.0.iter_mut() {
            *val = val.saturating_add(rhs)
        }
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
        let mut out = [T::ZERO; SIZE];
        for (elem, val) in out.iter_mut().zip(self.0.iter()) {
            *elem = val.saturating_sub(rhs)
        }
        Multiset::from_array(out)
    }
}

impl<T: Counter, const SIZE: usize> SubAssign for Multiset<T, SIZE> {
    fn sub_assign(&mut self, rhs: Self) {
        self.difference_mut(&rhs)
    }
}

impl<T: Counter, const SIZE: usize> SubAssign<T> for Multiset<T, SIZE> {
    fn sub_assign(&mut self, rhs: T) {
        for val in self.0.iter_mut() {
            *val = val.saturating_sub(rhs)
        }
    }
}

impl<T: Counter, const SIZE: usize> Mul for Multiset<T, SIZE> {
    type Output = Multiset<T, SIZE>;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut out = [T::ZERO; SIZE];
        for (elem, (val1, val2)) in out.iter_mut().zip(self.0.iter().zip(rhs.0.iter())) {
            *elem = val1.saturating_mul(*val2)
        }
        Multiset::from_array(out)
    }
}

impl<T: Counter, const SIZE: usize> Mul<T> for Multiset<T, SIZE> {
    type Output = Multiset<T, SIZE>;

    fn mul(self, rhs: T) -> Self::Output {
        let mut out = [T::ZERO; SIZE];
        for (elem, val) in out.iter_mut().zip(self.0.iter()) {
            *elem = val.saturating_mul(rhs)
        }
        Multiset::from_array(out)
    }
}

impl<T: Counter, const SIZE: usize> MulAssign for Multiset<T, SIZE> {
    fn mul_assign(&mut self, rhs: Self) {
        for (val1, val2) in self.0.iter_mut().zip(rhs.0.iter()) {
            *val1 = val1.saturating_mul(*val2)
        }
    }
}

impl<T: Counter, const SIZE: usize> MulAssign<T> for Multiset<T, SIZE> {
    fn mul_assign(&mut self, rhs: T) {
        for val in self.0.iter_mut() {
            *val = val.saturating_mul(rhs)
        }
    }
}

impl<T: Counter, const SIZE: usize> Div for Multiset<T, SIZE> {
    type Output = Multiset<T, SIZE>;

    fn div(self, rhs: Self) -> Self::Output {
        let mut out = [T::ZERO; SIZE];
        for (elem, (val1, val2)) in out.iter_mut().zip(self.0.iter().zip(rhs.0.iter())) {
            *elem = val1.saturating_div(*val2)
        }
        Multiset::from_array(out)
    }
}

impl<T: Counter, const SIZE: usize> Div<T> for Multiset<T, SIZE> {
    type Output = Multiset<T, SIZE>;

    fn div(self, rhs: T) -> Self::Output {
        let mut out = [T::ZERO; SIZE];
        for (elem, val) in out.iter_mut().zip(self.0.iter()) {
            *elem = val.saturating_div(rhs)
        }
        Multiset::from_array(out)
    }
}

impl<T: Counter, const SIZE: usize> DivAssign for Multiset<T, SIZE> {
    fn div_assign(&mut self, rhs: Self) {
        for (val1, val2) in self.0.iter_mut().zip(rhs.0.iter()) {
            *val1 = val1.saturating_div(*val2)
        }
    }
}

impl<T: Counter, const SIZE: usize> DivAssign<T> for Multiset<T, SIZE> {
    fn div_assign(&mut self, rhs: T) {
        for val in self.0.iter_mut() {
            *val = val.saturating_div(rhs)
        }
    }
}

impl<T: Counter, const SIZE: usize> Rem for Multiset<T, SIZE> {
    type Output = Multiset<T, SIZE>;

    fn rem(self, rhs: Self) -> Self::Output {
        let mut out = [T::ZERO; SIZE];
        for (elem, (val1, val2)) in out.iter_mut().zip(self.0.iter().zip(rhs.0.iter())) {
            *elem = *val1 % *val2
        }
        Multiset::from_array(out)
    }
}

impl<T: Counter, const SIZE: usize> Rem<T> for Multiset<T, SIZE> {
    type Output = Multiset<T, SIZE>;

    fn rem(self, rhs: T) -> Self::Output {
        let mut out = [T::ZERO; SIZE];
        for (elem, val) in out.iter_mut().zip(self.0.iter()) {
            *elem = *val % rhs
        }
        Multiset::from_array(out)
    }
}

impl<T: Counter, const SIZE: usize> RemAssign for Multiset<T, SIZE> {
    fn rem_assign(&mut self, rhs: Self) {
        for (val1, val2) in self.0.iter_mut().zip(rhs.0.iter()) {
            *val1 %= *val2
        }
    }
}

impl<T: Counter, const SIZE: usize> RemAssign<T> for Multiset<T, SIZE> {
    fn rem_assign(&mut self, rhs: T) {
        for val in self.0.iter_mut() {
            *val %= rhs
        }
    }
}

impl<T: Counter, const SIZE: usize> Pow<u32> for Multiset<T, SIZE> {
    type Output = Multiset<T, SIZE>;

    fn pow(self, rhs: u32) -> Self::Output {
        let mut out = [T::ZERO; SIZE];
        for (elem, val) in out.iter_mut().zip(self.0.iter()) {
            *elem = val.saturating_pow(rhs)
        }
        Multiset::from_array(out)
    }
}

////////////////////////////////////////////////////////////////////////////////
// Utils
////////////////////////////////////////////////////////////////////////////////

#[cfg(feature = "simd")]
#[inline]
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

        let set = Multiset::<u32, 4>::from_array([2, 1, 1, 0]);
        assert_relative_eq!(
            set.collision_entropy(),
            1.415037499278844,
            epsilon = f64::EPSILON
        );
    }
}
