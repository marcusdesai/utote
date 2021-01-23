use generic_array::ArrayLength;
use packed_simd::{u32x2, f64x2};
use typenum::{UInt, Unsigned};

use crate::{Multiset, SmallZero};


impl<U, B> Multiset<u32x2, UInt<U, B>>
    where
        UInt<U, B>: ArrayLength<u32x2>
{
    #[inline]
    pub fn empty() -> Self {
        Self::repeat(u32::ZERO)
    }

    #[inline]
    pub fn repeat(elem: u32) -> Self {
        let mut res = unsafe { Multiset::new_uninitialized() };
        for i in 0..UInt::<U, B>::USIZE {
            unsafe { *res.data.get_unchecked_mut(i) = u32x2::splat(elem) }
        }
        res
    }

    pub fn from_iter<I>(iter: I) -> Self
        where
            I: IntoIterator<Item=u32>,
    {
        let mut res = unsafe { Multiset::new_uninitialized() };
        let mut it = iter.into_iter();

        for i in 0.. UInt::<U, B>::USIZE {
            let mut elem_vec = u32x2::ZERO;
            for j in 0..u32x2::lanes() {
                if let Some(v) = it.next() {
                    elem_vec = elem_vec.replace(j, v)
                }
            }
            unsafe { *res.data.get_unchecked_mut(i) = elem_vec }
        }

        res
    }

    pub fn from_slice(slice: &[u32]) -> Self
    {
        assert_eq!(slice.len(), Self::len());
        Self::from_iter(slice.iter().cloned())
    }

    #[inline]
    pub const fn len() -> usize { u32x2::lanes() * UInt::<U, B>::USIZE }

    #[inline]
    pub fn contains(self, index: usize) -> bool {
        if index < Self::len() {
            let array_index = index / u32x2::lanes();
            let vector_index = index % u32x2::lanes();
            unsafe { self.data.get_unchecked(array_index).extract_unchecked(vector_index) > u32::ZERO }
        } else {
            false
        }
    }

    #[inline]
    pub unsafe fn contains_unchecked(self, index: usize) -> bool {
        let array_index = index / u32x2::lanes();
        let vector_index = index % u32x2::lanes();
        self.data.get_unchecked(array_index).extract_unchecked(vector_index) > u32::ZERO
    }

    #[inline]
    pub fn intersection(&self, other: &Self) -> Self {
        self.zip_map(other, |s1, s2| s1.min(s2))
    }

    #[inline]
    pub fn union(&self, other: &Self) -> Self {
        self.zip_map(other, |s1, s2| s1.max(s2))
    }

    #[inline]
    fn count_zero(&self) -> u32 {
        self.fold(0, |acc, vec| acc + vec.eq(u32x2::ZERO).bitmask().count_ones())
    }

    #[inline]
    fn count_non_zero(&self) -> u32 {
        self.fold(0, |acc, vec| acc + vec.gt(u32x2::ZERO).bitmask().count_ones())
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.iter().all(|vec| vec == &u32x2::ZERO)
    }

    #[inline]
    pub fn is_singleton(&self) -> bool {
        self.count_non_zero() == 1
    }

    #[inline]
    pub fn is_subset(&self, other: &Self) -> bool {
        self.data.iter().zip(other.data.iter()).all(|(s1, s2)| s1.le(*s2).all())
    }

    #[inline]
    pub fn is_superset(&self, other: &Self) -> bool {
        self.data.iter().zip(other.data.iter()).all(|(s1, s2)| s1.ge(*s2).all())
    }

    #[inline]
    pub fn is_any_lesser(&self, other: &Self) -> bool {
        self.data.iter().zip(other.data.iter()).any(|(s1, s2)| s1.lt(*s2).any())
    }

    #[inline]
    pub fn is_any_greater(&self, other: &Self) -> bool {
        self.data.iter().zip(other.data.iter()).any(|(s1, s2)| s1.gt(*s2).any())
    }

    /// May overflow && partial horizontal
    #[inline]
    pub fn total(&self) -> u32 {
        self.fold(u32x2::ZERO, |acc, vec| acc + vec).wrapping_sum()
    }

    /// partial horizontal
    #[inline]
    pub fn collision_entropy(&self) -> f64 {
        let total: f64 = From::from(self.total());
        -self.fold(f64x2::splat(0.0), |acc, vec| {
            let data = f64x2::from(vec);
            acc + (data / total).powf(f64x2::splat(2.0))
        }).sum().log2()
    }

    /// partial horizontal
    #[inline]
    pub fn shannon_entropy(&self) -> f64 {
        let total: f64 = From::from(self.total());
        -self.fold(f64x2::splat(0.0), |acc, vec| {
            let prob = f64x2::from(vec) / total;
            let data = prob * prob.ln();
            acc + data.is_nan().select(data, f64x2::ZERO)
        }).sum()
    }
}
