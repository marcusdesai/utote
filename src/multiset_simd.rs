use packed_simd::{u32x2, f64x2};
use typenum::U0;

use crate::{Multiset, SmallZero};


impl Multiset<u32x2, U0> {
    #[inline]
    pub const fn empty() -> Self {
        Multiset { data: u32x2::ZERO }
    }

    #[inline]
    pub const fn repeat(elem: u32) -> Self {
        Multiset { data: u32x2::splat(elem) }
    }

    pub fn from_iter<I>(iter: I) -> Self
        where
            I: IntoIterator<Item=u32>,
    {
        let mut data = u32x2::ZERO;
        let mut it = iter.into_iter();

        for i in 0..u32x2::lanes() {
            if let Some(elem) = it.next() {
                data = unsafe { data.replace_unchecked(i, elem) }
            }
        }
        Multiset { data }
    }

    pub fn from_slice(slice: &[u32]) -> Self
    {
        assert_eq!(slice.len(), u32x2::lanes());
        Self::from_iter(slice.iter().cloned())
    }

    #[inline]
    pub const fn len() -> usize { u32x2::lanes() }

    #[inline]
    pub fn contains(self, index: usize) -> bool {
        index < Self::len() && unsafe { self.data.extract_unchecked(index) > u32::ZERO }
    }

    #[inline]
    pub unsafe fn contains_unchecked(self, index: usize) -> bool {
        self.data.extract_unchecked(index) > u32::ZERO
    }

    #[inline]
    pub fn intersection(&self, other: &Self) -> Self {
        Multiset { data: self.data.min(other.data) }
    }

    #[inline]
    pub fn union(&self, other: &Self) -> Self {
        Multiset { data: self.data.max(other.data) }
    }

    #[inline]
    fn count_zero(&self) -> u32 {
        self.data.eq(u32x2::ZERO).bitmask().count_ones()
    }

    #[inline]
    fn count_non_zero(&self) -> u32 {
        self.data.gt(u32x2::ZERO).bitmask().count_ones()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data == u32x2::ZERO
    }

    #[inline]
    pub fn is_singleton(&self) -> bool {
        self.count_non_zero() == 1
    }

    #[inline]
    pub fn is_subset(&self, other: &Self) -> bool {
        self.data.le(other.data).all()
    }

    #[inline]
    pub fn is_superset(&self, other: &Self) -> bool {
        self.data.ge(other.data).all()
    }

    #[inline]
    pub fn is_any_lesser(&self, other: &Self) -> bool {
        self.data.lt(other.data).any()
    }

    #[inline]
    pub fn is_any_greater(&self, other: &Self) -> bool {
        self.data.gt(other.data).any()
    }

    /// May overflow & warning: horizontal
    #[inline]
    pub fn total(&self) -> u32 {
        self.data.wrapping_sum()
    }

    /// partial horizontal
    #[inline]
    pub fn collision_entropy(&self) -> f64 {
        let total: f64 = From::from(self.total());
        let data = f64x2::from(self.data);
        -(data / total).powf(f64x2::splat(2.0)).sum().log2()
    }

    /// partial horizontal
    #[inline]
    pub fn shannon_entropy(&self) -> f64 {
        let total: f64 = From::from(self.total());
        let prob = f64x2::from(self.data) / total;
        let data = prob * prob.ln();
        -data.is_nan().select(data, f64x2::ZERO).sum()
    }
}