use generic_array::ArrayLength;
use typenum::{UInt, Unsigned};

use crate::{Multiset, SmallZero};


impl<U, B> Multiset<u32, UInt<U, B>>
    where
        UInt<U, B>: ArrayLength<u32>,
{
    #[inline]
    pub fn empty() -> Self {
        Self::repeat(u32::ZERO)
    }

    #[inline]
    pub fn repeat(elem: u32) -> Self {
        let mut res = unsafe { Multiset::new_uninitialized() };
        for i in 0..UInt::<U, B>::USIZE {
            unsafe { *res.data.get_unchecked_mut(i) = elem }
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
            let elem = match it.next() {
                Some(v) => v,
                None => u32::ZERO,
            };
            unsafe { *res.data.get_unchecked_mut(i) = elem }
        }
        res
    }

    pub fn from_slice(slice: &[u32]) -> Self
    {
        assert_eq!(slice.len(), Self::len());
        let mut res = unsafe { Multiset::new_uninitialized() };
        let mut iter = slice.iter();

        for i in 0..UInt::<U, B>::USIZE {
            unsafe {
                *res.data.get_unchecked_mut(i) = iter.next().unwrap().clone()
            }
        }
        res
    }

    #[inline]
    pub const fn len() -> usize { UInt::<U, B>::USIZE }

    #[inline]
    pub fn contains(self, elem: usize) -> bool {
        elem < Self::len() && unsafe { self.data.get_unchecked(elem) > &u32::ZERO }
    }

    #[inline]
    pub unsafe fn contains_unchecked(self, elem: usize) -> bool {
        self.data.get_unchecked(elem) > &u32::ZERO
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
        self.fold(Self::len() as u32, |acc, elem| acc - elem.min(1))
    }

    #[inline]
    fn count_non_zero(&self) -> u32 {
        self.fold(0, |acc, elem| acc + elem.min(1))
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.iter().all(|elem| elem == &u32::ZERO)
    }

    #[inline]
    pub fn is_singleton(&self) -> bool {
        self.count_non_zero() == 1
    }

    #[inline]
    pub fn is_subset(&self, other: &Self) -> bool {
        self.data.iter().zip(other.data.iter()).all(|(a, b)| a <= b)
    }

    #[inline]
    pub fn is_superset(&self, other: &Self) -> bool {
        self.data.iter().zip(other.data.iter()).all(|(a, b)| a >= b)
    }

    #[inline]
    pub fn is_any_lesser(&self, other: &Self) -> bool {
        self.data.iter().zip(other.data.iter()).any(|(a, b)| a < b)
    }

    #[inline]
    pub fn is_any_greater(&self, other: &Self) -> bool {
        self.data.iter().zip(other.data.iter()).any(|(a, b)| a > b)
    }

    /// May overflow
    #[inline]
    pub fn total(&self) -> u32 {
        self.data.iter().sum()
    }

    #[inline]
    pub fn collision_entropy(&self) -> f64 {
        let total: f64 = From::from(self.total());
        -self.fold(0.0, |acc, frequency| {
            acc + (f64::from(frequency) / total).powf(2.0)
        }).log2()
    }

    #[inline]
    pub fn shannon_entropy(&self) -> f64 {
        let total: f64 = From::from(self.total());
        -self.fold(0.0, |acc, frequency| {
            if frequency > u32::ZERO {
                let prob = f64::from(frequency) / total;
                acc + prob * prob.log2()
            } else {
                acc
            }
        })
    }
}
