use packed_simd::{u32x4, f64x4, m32x4};
use rand::prelude::*;
use std::iter::FromIterator;
use typenum::U0;

use crate::{Multiset, SmallNum};


macro_rules! multiset_simd {
    ($simd:ty, $scalar:ty, $simd_f:ty, $simd_m:ty) => {
        impl FromIterator<$scalar> for Multiset<$simd, U0> {
            #[inline]
            fn from_iter<T: IntoIterator<Item=$scalar>>(iter: T) -> Self {
                let mut data = <$simd>::ZERO;
                let mut it = iter.into_iter();

                for i in 0..<$simd>::lanes() {
                    if let Some(elem) = it.next() {
                        data = unsafe { data.replace_unchecked(i, elem) }
                    }
                }
                Multiset { data }
            }
        }

        impl Multiset<$simd, U0> {
            #[inline]
            pub const fn empty() -> Self {
                Multiset { data: <$simd>::ZERO }
            }

            #[inline]
            pub const fn repeat(elem: $scalar) -> Self {
                Multiset { data: <$simd>::splat(elem) }
            }

            #[inline]
            pub fn from_slice(slice: &[$scalar]) -> Self
            {
                assert_eq!(slice.len(), Self::len());
                Multiset { data: <$simd>::from_slice_unaligned(slice) }
            }

            #[inline]
            pub const fn len() -> usize { <$simd>::lanes() }

            #[inline]
            pub fn clear(&mut self) {
                self.data *= <$scalar>::ZERO
            }

            #[inline]
            pub fn contains(self, elem: usize) -> bool {
                elem < Self::len() && unsafe { self.data.extract_unchecked(elem) > <$scalar>::ZERO }
            }

            #[inline]
            pub unsafe fn contains_unchecked(self, elem: usize) -> bool {
                self.data.extract_unchecked(elem) > <$scalar>::ZERO
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
            pub fn count_zero(&self) -> $scalar {
                self.data.eq(<$simd>::ZERO).bitmask().count_ones()
            }

            #[inline]
            pub fn count_non_zero(&self) -> $scalar {
                self.data.gt(<$simd>::ZERO).bitmask().count_ones()
            }

            #[inline]
            pub fn is_empty(&self) -> bool {
                self.data == <$simd>::ZERO
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
            pub fn total(&self) -> $scalar {
                self.data.wrapping_sum()
            }

            /// Horizontal
            #[inline]
            pub fn argmax(&self) -> (usize, $scalar) {
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
            pub fn max(&self) -> $scalar {
                self.data.max_element()
            }

            /// Horizontal
            #[inline]
            pub fn argmin(&self) -> (usize, $scalar) {
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
            pub fn min(&self) -> $scalar {
                self.data.min_element()
            }

            #[inline]
            pub fn choose(&mut self, elem: usize) {
                let mask = <$simd_m>::splat(false).replace(elem, true);
                self.data = mask.select(self.data, <$simd>::ZERO)
            }

            /// Horizontal, really bad
            #[inline]
            pub fn choose_random(&mut self, rng: &mut StdRng) {
                let choice_value = rng.gen_range(<$scalar>::ZERO, self.total() + <$scalar>::ONE);
                let mut elem: usize = 0;
                let mut acc: $scalar = <$scalar>::ZERO;
                for i in 0..Self::len() {
                    acc += unsafe { self.data.extract_unchecked(i) };
                    if acc > choice_value {
                        elem = i;
                        break
                    }
                }
                self.choose(elem)
            }

            /// partial horizontal
            #[inline]
            pub fn collision_entropy(&self) -> f64 {
                let total: f64 = From::from(self.total());
                let data = <$simd_f>::from(self.data);
                -(data / total).powf(<$simd_f>::splat(2.0)).sum().log2()
            }

            /// partial horizontal
            #[inline]
            pub fn shannon_entropy(&self) -> f64 {
                let total: f64 = From::from(self.total());
                let prob = <$simd_f>::from(self.data) / total;
                let data = prob * prob.ln();
                -data.is_nan().select(<$simd_f>::ZERO, data).sum()
            }
        }
    }
}


multiset_simd!(u32x4, u32, f64x4, m32x4);


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    type MS0u32x4 = Multiset<u32x4, U0>;

    #[test]
    fn test_empty() {
        let result = MS0u32x4::empty();
        let expected = MS0u32x4::from_iter(vec![0; 4].into_iter());
        assert_eq!(result, expected)
    }

    #[test]
    fn test_repeat() {
        let result = MS0u32x4::repeat(3);
        let expected = MS0u32x4::from_iter(vec![3; 4].into_iter());
        assert_eq!(result, expected)
    }

    #[test]
    fn test_len() {
        assert_eq!(MS0u32x4::len(), 4)
    }

    #[test]
    fn test_clear() {
        let mut set = MS0u32x4::repeat(3);
        set.clear();
        let expected = MS0u32x4::empty();
        assert_eq!(set, expected)
    }

    #[test]
    fn test_contains() {
        let set = MS0u32x4::from_slice(&[1, 0, 1, 0]);
        assert!(set.contains(0));
        assert!(!set.contains(1));
        assert!(set.contains(2));
        assert!(!set.contains(3));
        assert!(!set.contains(4))
    }

    #[test]
    fn test_contains_unchecked() {
        let set = MS0u32x4::from_slice(&[1, 0, 1, 0]);
        unsafe {
            assert!(set.contains_unchecked(0));
            assert!(!set.contains_unchecked(1));
            assert!(set.contains_unchecked(2));
            assert!(!set.contains_unchecked(3));
        }
    }

    #[test]
    fn test_intersection() {
        let a = MS0u32x4::from_slice(&[2, 0, 4, 0]);
        let b = MS0u32x4::from_slice(&[0, 0, 3, 1]);
        let c = MS0u32x4::from_slice(&[0, 0, 3, 0]);
        assert_eq!(c, a.intersection(&b))
    }

    #[test]
    fn test_union() {
        let a = MS0u32x4::from_slice(&[2, 0, 4, 0]);
        let b = MS0u32x4::from_slice(&[0, 0, 3, 1]);
        let c = MS0u32x4::from_slice(&[2, 0, 4, 1]);
        assert_eq!(c, a.union(&b))
    }

    #[test]
    fn test_count_zero() {
        let set = MS0u32x4::from_slice(&[0, 0, 3, 0]);
        assert_eq!(set.count_zero(), 3)
    }

    #[test]
    fn test_count_non_zero() {
        let set = MS0u32x4::from_slice(&[0, 2, 3, 0]);
        assert_eq!(set.count_non_zero(), 2)
    }

    #[test]
    fn test_is_empty() {
        let a = MS0u32x4::from_slice(&[2, 0, 4, 0]);
        let b = MS0u32x4::from_slice(&[0, 0, 0, 0]);
        assert!(!a.is_empty());
        assert!(b.is_empty());
    }

    #[test]
    fn test_is_singleton() {
        let a = MS0u32x4::from_slice(&[1, 0, 0, 0]);
        assert!(a.is_singleton());

        let b = MS0u32x4::from_slice(&[0, 0, 0, 5]);
        assert!(b.is_singleton());

        let c = MS0u32x4::from_slice(&[1, 0, 0, 5]);
        assert!(!c.is_singleton());

        let d = MS0u32x4::from_slice(&[0, 0, 0, 0]);
        assert!(!d.is_singleton());
    }

    #[test]
    fn test_is_subset() {
        let a = MS0u32x4::from_slice(&[2, 0, 4, 0]);
        let b = MS0u32x4::from_slice(&[2, 0, 4, 1]);
        assert!(a.is_subset(&b));
        assert!(!b.is_subset(&a));

        let c = MS0u32x4::from_slice(&[1, 3, 4, 5]);
        assert!(!a.is_subset(&c));
        assert!(!c.is_subset(&a));
    }

    #[test]
    fn test_is_superset() {
        let a = MS0u32x4::from_slice(&[2, 0, 4, 0]);
        let b = MS0u32x4::from_slice(&[2, 0, 4, 1]);
        assert!(!a.is_superset(&b));
        assert!(b.is_superset(&a));

        let c = MS0u32x4::from_slice(&[1, 3, 4, 5]);
        assert!(!a.is_superset(&c));
        assert!(!c.is_superset(&a));
    }

    #[test]
    fn test_is_any_lesser() {
        let a = MS0u32x4::from_slice(&[2, 0, 4, 0]);
        let b = MS0u32x4::from_slice(&[2, 0, 4, 1]);
        assert!(a.is_any_lesser(&b));
        assert!(!b.is_any_lesser(&a));

        let c = MS0u32x4::from_slice(&[1, 3, 4, 5]);
        assert!(a.is_any_lesser(&c));
        assert!(c.is_any_lesser(&a));
    }

    #[test]
    fn test_is_any_greater() {
        let a = MS0u32x4::from_slice(&[2, 0, 4, 0]);
        let b = MS0u32x4::from_slice(&[2, 0, 4, 1]);
        assert!(!a.is_any_greater(&b));
        assert!(b.is_any_greater(&a));

        let c = MS0u32x4::from_slice(&[1, 3, 4, 5]);
        assert!(a.is_any_greater(&c));
        assert!(c.is_any_greater(&a));
    }

    #[test]
    fn test_total() {
        let set = MS0u32x4::from_slice(&[1, 3, 4, 5]);
        assert_eq!(set.total(), 13)
    }

    #[test]
    fn test_argmax() {
        let set = MS0u32x4::from_slice(&[1, 5, 2, 8]);
        let expected = (3, 8);
        assert_eq!(set.argmax(), expected)
    }

    #[test]
    fn test_imax() {
        let set = MS0u32x4::from_slice(&[1, 5, 2, 8]);
        let expected = 3;
        assert_eq!(set.imax(), expected)
    }

    #[test]
    fn test_max() {
        let set = MS0u32x4::from_slice(&[1, 5, 2, 8]);
        let expected = 8;
        assert_eq!(set.max(), expected)
    }

    #[test]
    fn test_argmin() {
        let set = MS0u32x4::from_slice(&[1, 5, 2, 8]);
        let expected = (0, 1);
        assert_eq!(set.argmin(), expected)
    }

    #[test]
    fn test_imin() {
        let set = MS0u32x4::from_slice(&[1, 5, 2, 8]);
        let expected = 0;
        assert_eq!(set.imin(), expected)
    }

    #[test]
    fn test_min() {
        let set = MS0u32x4::from_slice(&[1, 5, 2, 8]);
        let expected = 1;
        assert_eq!(set.min(), expected)
    }

    #[test]
    fn test_choose() {
        let mut set = MS0u32x4::from_slice(&[2, 1, 3, 4]);
        let expected = MS0u32x4::from_slice(&[0, 0, 3, 0]);
        set.choose(2);
        assert_eq!(set, expected)
    }

    #[test]
    fn test_choose_random() {
        let mut result1 = MS0u32x4::from_slice(&[2, 1, 3, 4]);
        let expected1 = MS0u32x4::from_slice(&[0, 0, 3, 0]);
        let test_rng1 = &mut StdRng::seed_from_u64(5);
        result1.choose_random(test_rng1);
        assert_eq!(result1, expected1);

        let mut result2 = MS0u32x4::from_slice(&[2, 1, 3, 4]);
        let expected2 = MS0u32x4::from_slice(&[2, 0, 0, 0]);
        let test_rng2 = &mut StdRng::seed_from_u64(10);
        result2.choose_random(test_rng2);
        assert_eq!(result2, expected2);
    }

    #[test]
    fn test_choose_random_empty() {
        let mut result = MS0u32x4::from_slice(&[0, 0, 0, 0]);
        let expected = MS0u32x4::from_slice(&[0, 0, 0, 0]);
        let test_rng = &mut StdRng::seed_from_u64(1);
        result.choose_random(test_rng);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_collision_entropy() {
        let simple = MS0u32x4::from_slice(&[200, 0, 0, 0]);
        assert_eq!(simple.collision_entropy(), 0.0);

        let set = MS0u32x4::from_slice(&[2, 1, 1, 0]);
        let result = 1.415037499278844;
        assert_relative_eq!(set.collision_entropy(), result);
    }

    #[test]
    fn test_shannon_entropy1() {
        let a = MS0u32x4::from_slice(&[200, 0, 0, 0]);
        assert_eq!(a.shannon_entropy(), 0.0);

        let b = MS0u32x4::from_slice(&[2, 1, 1, 0]);
        let result = 1.0397207708399179;
        assert_relative_eq!(b.shannon_entropy(), result);
    }

    #[test]
    fn test_shannon_entropy2() {
        let set1 = MS0u32x4::from_slice(&[4, 6, 1, 6]);
        let entropy = set1.shannon_entropy();
        let result1 = 1.2422550455140853;
        assert_relative_eq!(entropy, result1);

        let set2 = MS0u32x4::from_slice(&[4, 6, 0, 6]);
        let entropy = set2.shannon_entropy();
        let result2 = 1.0821955300387673;
        assert_relative_eq!(entropy, result2);
    }
}
