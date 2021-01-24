use packed_simd::{u32x2, f64x2};
use typenum::U0;

use crate::{Multiset, SmallNum};


impl Multiset<u32x2, U0> {
    #[inline]
    pub const fn empty() -> Self {
        Multiset { data: u32x2::ZERO }
    }

    #[inline]
    pub const fn repeat(elem: u32) -> Self {
        Multiset { data: u32x2::splat(elem) }
    }

    #[inline]
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

    #[inline]
    pub fn from_slice(slice: &[u32]) -> Self
    {
        assert_eq!(slice.len(), Self::len());
        Multiset { data: u32x2::from_slice_unaligned(slice) }
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

    /// Horizontal
    #[inline]
    pub fn argmax(&self) -> (usize, u32) {
        let mut the_max = unsafe { self.data.extract_unchecked(0) };
        let mut the_i = 0;

        for i in 1..Self::len() {
            let val = unsafe { self.data.extract_unchecked(i) };
            if val > the_max {
                the_max = val;
                the_i = i;
            }
        }
        (the_i, the_max)
    }

    /// Horizontal
    #[inline]
    pub fn imax(&self) -> usize {
        self.argmax().0
    }

    /// Horizontal
    #[inline]
    pub fn max(&self) -> u32 {
        self.data.max_element()
    }

    /// Horizontal
    #[inline]
    pub fn argmin(&self) -> (usize, u32) {
        let mut the_min = unsafe { self.data.extract_unchecked(0) };
        let mut the_i = 0;

        for i in 1..Self::len() {
            let val = unsafe { self.data.extract_unchecked(i) };
            if val < the_min {
                the_min = val;
                the_i = i;
            }
        }
        (the_i, the_min)
    }

    /// Horizontal
    #[inline]
    pub fn imin(&self) -> usize {
        self.argmin().0
    }

    /// Horizontal
    #[inline]
    pub fn min(&self) -> u32 {
        self.data.min_element()
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


#[cfg(test)]
mod tests {
    use super::*;

    type MS2u32 = Multiset<u32x2, U0>;

    #[test]
    fn test_repeat() {
        let result = MS2u32::repeat(3);
        let expected = MS2u32::from_iter(vec![3; 2].into_iter());
        assert_eq!(result, expected)
    }

    #[test]
    fn test_zeroes() {
        let result = MS2u32::empty();
        let expected = MS2u32::from_iter(vec![0; 2].into_iter());
        assert_eq!(result, expected)
    }

    #[test]
    fn test_contains() {
        let set = MS2u32::from_slice(&[1, 0]);
        assert!(set.contains(0));
        assert!(!set.contains(1));
        assert!(!set.contains(4))
    }

    #[test]
    fn test_union() {
        let a = MS2u32::from_slice(&[2, 1]);
        let b = MS2u32::from_slice(&[0, 4]);
        let c = MS2u32::from_slice(&[2, 4]);
        assert_eq!(c, a.union(&b))
    }

    #[test]
    fn test_intersection() {
        let a = MS2u32::from_slice(&[2, 1]);
        let b = MS2u32::from_slice(&[0, 4]);
        let c = MS2u32::from_slice(&[0, 1]);
        assert_eq!(c, a.intersection(&b))
    }

    #[test]
    fn test_is_subset() {
        let a = MS2u32::from_slice(&[2, 0]);
        let b = MS2u32::from_slice(&[2, 1]);
        assert!(a.is_subset(&b));
        assert!(!b.is_subset(&a));

        let c = MS2u32::from_slice(&[1, 3]);
        assert!(!a.is_subset(&c));
        assert!(!c.is_subset(&a));
    }

    #[test]
    fn test_is_superset() {
        let a = MS2u32::from_slice(&[2, 0]);
        let b = MS2u32::from_slice(&[2, 1]);
        assert!(!a.is_superset(&b));
        assert!(b.is_superset(&a));

        let c = MS2u32::from_slice(&[1, 3]);
        assert!(!a.is_superset(&c));
        assert!(!c.is_superset(&a));
    }

    #[test]
    fn test_is_singleton() {
        let a = MS2u32::from_slice(&[1, 0]);
        assert!(a.is_singleton());

        let b = MS2u32::from_slice(&[0, 5]);
        assert!(b.is_singleton());

        let c = MS2u32::from_slice(&[1, 5]);
        assert!(!c.is_singleton());

        let d = MS2u32::from_slice(&[0, 0]);
        assert!(!d.is_singleton());
    }

    #[test]
    fn test_is_empty() {
        let a = MS2u32::from_slice(&[2, 0]);
        let b = MS2u32::from_slice(&[0, 0]);
        assert!(!a.is_empty());
        assert!(b.is_empty());
        assert!(MS2u32::empty().is_empty())
    }

    // #[test]
    // fn test_shannon_entropy1() {
    //     let a = MS2u32::from_slice(&[200, 0, 0, 0]);
    //     let b = MS2u32::from_slice(&[2, 1, 1, 0]);
    //     assert_eq!(a.shannon_entropy(), 0.0);
    //     assert_eq!(b.shannon_entropy(), 1.5);
    // }
    //
    // #[test]
    // fn test_shannon_entropy2() {
    //     let a = MS2u32::from_slice(&[4, 6, 1, 6]);
    //     let entropy = a.shannon_entropy();
    //     let lt = 1.79219;
    //     let gt = 1.79220;
    //     assert!(lt < entropy && entropy < gt);
    //
    //     let b = MS2u32::from_slice(&[4, 6, 0, 6]);
    //     let entropy = b.shannon_entropy();
    //     let lt = 1.56127;
    //     let gt = 1.56128;
    //     assert!(lt < entropy && entropy < gt);
    // }
    //
    // #[test]
    // fn test_collision_entropy() {
    //     let a = MS2u32::from_slice(&[200, 0, 0, 0]);
    //     assert_eq!(a.collision_entropy(), 0.0);
    //
    //     let entropy = MS2u32::from_slice(&[2, 1, 1, 0]).collision_entropy();
    //     let lt = 1.41502;
    //     let gt = 1.41504;
    //     assert!(lt < entropy && entropy < gt);
    // }
    //
    // #[test]
    // fn test_choose() {
    //     let mut set = MS2u32::from_slice(&[2, 1, 3, 4]);
    //     let expected = MS2u32::from_slice(&[0, 0, 3, 0]);
    //     set.choose(2);
    //     assert_eq!(set, expected)
    // }
    //
    // #[test]
    // fn test_choose_random() {
    //     let mut result1 = MS2u32::from_slice(&[2, 1, 3, 4]);
    //     let expected1 = MS2u32::from_slice(&[0, 0, 3, 0]);
    //     let test_rng1 = &mut StdRng::seed_from_u64(5);
    //     result1.choose_random(test_rng1);
    //     assert_eq!(result1, expected1);
    //
    //     let mut result2 = MS2u32::from_slice(&[2, 1, 3, 4]);
    //     let expected2 = MS2u32::from_slice(&[2, 0, 0, 0]);
    //     let test_rng2 = &mut StdRng::seed_from_u64(10);
    //     result2.choose_random(test_rng2);
    //     assert_eq!(result2, expected2);
    // }
    //
    // #[test]
    // fn test_choose_random_empty() {
    //     let mut result = MS2u32::from_slice(&[0, 0, 0, 0]);
    //     let expected = MS2u32::from_slice(&[0, 0, 0, 0]);
    //     let test_rng = &mut StdRng::seed_from_u64(1);
    //     result.choose_random(test_rng);
    //     assert_eq!(result, expected);
    // }

    #[test]
    fn test_count_zero() {
        let set = MS2u32::from_slice(&[0, 0]);
        assert_eq!(set.count_zero(), 2)
    }

    #[test]
    fn test_count_non_zero() {
        let set = MS2u32::from_slice(&[0, 2]);
        assert_eq!(set.count_non_zero(), 1)
    }

    // #[test]
    // fn test_map() {
    //     let set = MS2u32::from_slice(&[1, 5, 2, 8]);
    //     let result = set.map(|e| e * 2);
    //     let expected = MS2u32::from_slice(&[2, 10, 4, 16]);
    //     assert_eq!(result, expected)
    // }

    #[test]
    fn test_argmax() {
        let set = MS2u32::from_slice(&[1, 5]);
        let expected = (1, 5);
        assert_eq!(set.argmax(), expected)
    }

    #[test]
    fn test_imax() {
        let set = MS2u32::from_slice(&[1, 5]);
        let expected = 1;
        assert_eq!(set.imax(), expected)
    }

    #[test]
    fn test_max() {
        let set = MS2u32::from_slice(&[1, 5]);
        let expected = 5;
        assert_eq!(set.max(), expected)
    }

    #[test]
    fn test_argmin() {
        let set = MS2u32::from_slice(&[1, 5]);
        let expected = (0, 1);
        assert_eq!(set.argmin(), expected)
    }

    #[test]
    fn test_imin() {
        let set = MS2u32::from_slice(&[1, 5]);
        let expected = 0;
        assert_eq!(set.imin(), expected)
    }

    #[test]
    fn test_min() {
        let set = MS2u32::from_slice(&[1, 5]);
        let expected = 1;
        assert_eq!(set.min(), expected)
    }
}
