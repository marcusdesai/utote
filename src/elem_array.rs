use generic_array::ArrayLength;
use rand::prelude::*;
use typenum::{UInt, Unsigned};

use crate::{Multiset, SmallNum};


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
        for i in 0..Self::len() {
            unsafe { *res.data.get_unchecked_mut(i) = elem }
        }
        res
    }

    #[inline]
    pub fn from_iter<I>(iter: I) -> Self
        where
            I: IntoIterator<Item=u32>,
    {
        let mut res = unsafe { Multiset::new_uninitialized() };
        let mut it = iter.into_iter();

        for i in 0..Self::len() {
            let elem = match it.next() {
                Some(v) => v,
                None => u32::ZERO,
            };
            unsafe { *res.data.get_unchecked_mut(i) = elem }
        }
        res
    }

    #[inline]
    pub fn from_slice(slice: &[u32]) -> Self
    {
        assert_eq!(slice.len(), Self::len());
        let mut res = unsafe { Multiset::new_uninitialized() };
        let mut iter = slice.iter();

        for i in 0..Self::len() {
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
    pub fn argmax(&self) -> (usize, u32) {
        let mut the_max = unsafe { self.data.get_unchecked(0) };
        let mut the_i = 0;

        for i in 1..Self::len() {
            let val = unsafe { self.data.get_unchecked(i) };
            if val > the_max {
                the_max = val;
                the_i = i;
            }
        }
        (the_i, *the_max)
    }

    #[inline]
    pub fn imax(&self) -> usize {
        self.argmax().0
    }

    #[inline]
    pub fn max(&self) -> u32 {
        let mut the_max = unsafe { self.data.get_unchecked(0) };

        for i in 1..Self::len() {
            let val = unsafe { self.data.get_unchecked(i) };
            if val > the_max {
                the_max = val;
            }
        }
        *the_max
    }

    #[inline]
    pub fn argmin(&self) -> (usize, u32) {
        let mut the_min = unsafe { self.data.get_unchecked(0) };
        let mut the_i = 0;

        for i in 1..Self::len() {
            let val = unsafe { self.data.get_unchecked(i) };
            if val < the_min {
                the_min = val;
                the_i = i;
            }
        }
        (the_i, *the_min)
    }

    #[inline]
    pub fn imin(&self) -> usize {
        self.argmin().0
    }

    #[inline]
    pub fn min(&self) -> u32 {
        let mut the_min = unsafe { self.data.get_unchecked(0) };

        for i in 1..Self::len() {
            let val = unsafe { self.data.get_unchecked(i) };
            if val < the_min {
                the_min = val;
            }
        }
        *the_min
    }

    #[inline]
    pub fn choose(&mut self, elem: usize) {
        for i in 0..Self::len() {
            if i != elem {
                unsafe { *self.data.get_unchecked_mut(i) = u32::ZERO }
            }
        }
    }

    #[inline]
    pub fn choose_random(&mut self, rng: &mut StdRng) {
        let choice = rng.gen_range(u32::ZERO, self.total() + u32::ONE);
        let mut acc = u32::ZERO;
        let mut chosen = false;
        self.data.iter_mut().for_each(|elem| {
            if chosen {
                *elem = u32::ZERO
            } else {
                acc += *elem;
                if acc < choice {
                    *elem = u32::ZERO
                } else {
                    chosen = true;
                }
            }
        })
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


#[cfg(test)]
mod tests {
    use super::*;
    type MS4u8 = Multiset<u32, typenum::U4>;

    #[test]
    fn test_repeat() {
        let result = MS4u8::repeat(3);
        let expected = MS4u8::from_iter(vec![3; 4].into_iter());
        assert_eq!(result, expected)
    }

    #[test]
    fn test_zeroes() {
        let result = MS4u8::empty();
        let expected = MS4u8::from_iter(vec![0; 4].into_iter());
        assert_eq!(result, expected)
    }

    #[test]
    fn test_contains() {
        let set = MS4u8::from_slice(&[1, 0, 1, 0]);
        assert!(set.contains(2));
        assert!(!set.contains(1));
        assert!(!set.contains(4))
    }

    #[test]
    fn test_union() {
        let a = MS4u8::from_slice(&[2, 0, 4, 0]);
        let b = MS4u8::from_slice(&[0, 0, 3, 1]);
        let c = MS4u8::from_slice(&[2, 0, 4, 1]);
        assert_eq!(c, a.union(&b))
    }

    #[test]
    fn test_intersection() {
        let a = MS4u8::from_slice(&[2, 0, 4, 0]);
        let b = MS4u8::from_slice(&[0, 0, 3, 1]);
        let c = MS4u8::from_slice(&[0, 0, 3, 0]);
        assert_eq!(c, a.intersection(&b))
    }

    #[test]
    fn test_is_subset() {
        let a = MS4u8::from_slice(&[2, 0, 4, 0]);
        let b = MS4u8::from_slice(&[2, 0, 4, 1]);
        assert!(a.is_subset(&b));
        assert!(!b.is_subset(&a));

        let c = MS4u8::from_slice(&[1, 3, 4, 5]);
        assert!(!a.is_subset(&c));
        assert!(!c.is_subset(&a));
    }

    #[test]
    fn test_is_superset() {
        let a = MS4u8::from_slice(&[2, 0, 4, 0]);
        let b = MS4u8::from_slice(&[2, 0, 4, 1]);
        assert!(!a.is_superset(&b));
        assert!(b.is_superset(&a));

        let c = MS4u8::from_slice(&[1, 3, 4, 5]);
        assert!(!a.is_superset(&c));
        assert!(!c.is_superset(&a));
    }

    #[test]
    fn test_is_singleton() {
        let a = MS4u8::from_slice(&[1, 0, 0, 0]);
        assert!(a.is_singleton());

        let b = MS4u8::from_slice(&[0, 0, 0, 5]);
        assert!(b.is_singleton());

        let c = MS4u8::from_slice(&[1, 0, 0, 5]);
        assert!(!c.is_singleton());

        let d = MS4u8::from_slice(&[0, 0, 0, 0]);
        assert!(!d.is_singleton());
    }

    #[test]
    fn test_is_empty() {
        let a = MS4u8::from_slice(&[2, 0, 4, 0]);
        let b = MS4u8::from_slice(&[0, 0, 0, 0]);
        assert!(!a.is_empty());
        assert!(b.is_empty());
    }

    #[test]
    fn test_shannon_entropy1() {
        let a = MS4u8::from_slice(&[200, 0, 0, 0]);
        let b = MS4u8::from_slice(&[2, 1, 1, 0]);
        assert_eq!(a.shannon_entropy(), 0.0);
        assert_eq!(b.shannon_entropy(), 1.5);
    }

    #[test]
    fn test_shannon_entropy2() {
        let a = MS4u8::from_slice(&[4, 6, 1, 6]);
        let entropy = a.shannon_entropy();
        let lt = 1.79219;
        let gt = 1.79220;
        assert!(lt < entropy && entropy < gt);

        let b = MS4u8::from_slice(&[4, 6, 0, 6]);
        let entropy = b.shannon_entropy();
        let lt = 1.56127;
        let gt = 1.56128;
        assert!(lt < entropy && entropy < gt);
    }

    #[test]
    fn test_collision_entropy() {
        let a = MS4u8::from_slice(&[200, 0, 0, 0]);
        assert_eq!(a.collision_entropy(), 0.0);

        let entropy = MS4u8::from_slice(&[2, 1, 1, 0]).collision_entropy();
        let lt = 1.41502;
        let gt = 1.41504;
        assert!(lt < entropy && entropy < gt);
    }

    #[test]
    fn test_choose() {
        let mut set = MS4u8::from_slice(&[2, 1, 3, 4]);
        let expected = MS4u8::from_slice(&[0, 0, 3, 0]);
        set.choose(2);
        assert_eq!(set, expected)
    }

    #[test]
    fn test_choose_random() {
        let mut result1 = MS4u8::from_slice(&[2, 1, 3, 4]);
        let expected1 = MS4u8::from_slice(&[0, 0, 3, 0]);
        let test_rng1 = &mut StdRng::seed_from_u64(5);
        result1.choose_random(test_rng1);
        assert_eq!(result1, expected1);

        let mut result2 = MS4u8::from_slice(&[2, 1, 3, 4]);
        let expected2 = MS4u8::from_slice(&[2, 0, 0, 0]);
        let test_rng2 = &mut StdRng::seed_from_u64(10);
        result2.choose_random(test_rng2);
        assert_eq!(result2, expected2);
    }

    #[test]
    fn test_choose_random_empty() {
        let mut result = MS4u8::from_slice(&[0, 0, 0, 0]);
        let expected = MS4u8::from_slice(&[0, 0, 0, 0]);
        let test_rng = &mut StdRng::seed_from_u64(1);
        result.choose_random(test_rng);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_count_zero() {
        let set = MS4u8::from_slice(&[0, 0, 3, 0]);
        assert_eq!(set.count_zero(), 3)
    }

    #[test]
    fn test_count_non_zero() {
        let set = MS4u8::from_slice(&[0, 2, 3, 0]);
        assert_eq!(set.count_non_zero(), 2)
    }

    #[test]
    fn test_map() {
        let set = MS4u8::from_slice(&[1, 5, 2, 8]);
        let result = set.map(|e| e * 2);
        let expected = MS4u8::from_slice(&[2, 10, 4, 16]);
        assert_eq!(result, expected)
    }

    #[test]
    fn test_argmax() {
        let set = MS4u8::from_slice(&[1, 5, 2, 8]);
        let expected = (3, 8);
        assert_eq!(set.argmax(), expected)
    }

    #[test]
    fn test_imax() {
        let set = MS4u8::from_slice(&[1, 5, 2, 8]);
        let expected = 3;
        assert_eq!(set.imax(), expected)
    }

    #[test]
    fn test_max() {
        let set = MS4u8::from_slice(&[1, 5, 2, 8]);
        let expected = 8;
        assert_eq!(set.max(), expected)
    }

    #[test]
    fn test_argmin() {
        let set = MS4u8::from_slice(&[1, 5, 2, 8]);
        let expected = (0, 1);
        assert_eq!(set.argmin(), expected)
    }

    #[test]
    fn test_imin() {
        let set = MS4u8::from_slice(&[1, 5, 2, 8]);
        let expected = 0;
        assert_eq!(set.imin(), expected)
    }

    #[test]
    fn test_min() {
        let set = MS4u8::from_slice(&[1, 5, 2, 8]);
        let expected = 1;
        assert_eq!(set.min(), expected)
    }
}
