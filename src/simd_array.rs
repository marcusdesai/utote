use generic_array::ArrayLength;
use packed_simd::*;
use rand::prelude::*;
use std::iter::FromIterator;
use typenum::{UInt, Unsigned};

use crate::{Multiset, SmallNum};


macro_rules! multiset_simd_array {
    ($simd:ty, $scalar:ty, $simd_f:ty, $simd_m:ty) => {
        impl<U, B> FromIterator<$scalar> for Multiset<$simd, UInt<U, B>>
            where
                UInt<U, B>: ArrayLength<$simd>
        {
            #[inline]
            fn from_iter<T: IntoIterator<Item=$scalar>>(iter: T) -> Self {
                let mut res = unsafe { Multiset::new_uninitialized() };
                let mut it = iter.into_iter();

                for i in 0..UInt::<U, B>::USIZE {
                    let mut elem_vec = <$simd>::ZERO;
                    for j in 0..<$simd>::lanes() {
                        if let Some(v) = it.next() {
                            elem_vec = elem_vec.replace(j, v)
                        }
                    }
                    unsafe { *res.data.get_unchecked_mut(i) = elem_vec }
                }

                res
            }
        }

        impl<U, B> Multiset<$simd, UInt<U, B>>
            where
                UInt<U, B>: ArrayLength<$simd>
        {
            #[inline]
            pub fn empty() -> Self {
                Self::repeat(<$scalar>::ZERO)
            }

            #[inline]
            pub fn repeat(elem: $scalar) -> Self {
                let mut res = unsafe { Multiset::new_uninitialized() };
                for i in 0..UInt::<U, B>::USIZE {
                    unsafe { *res.data.get_unchecked_mut(i) = <$simd>::splat(elem) }
                }
                res
            }

            #[inline]
            pub fn from_slice(slice: &[$scalar]) -> Self {
                assert_eq!(slice.len(), Self::len());
                Self::from_iter(slice.iter().cloned())
            }

            #[inline]
            pub const fn len() -> usize { <$simd>::lanes() * UInt::<U, B>::USIZE }

            #[inline]
            pub fn clear(&mut self) {
                self.data.iter_mut().for_each(|e| *e *= <$scalar>::ZERO);
            }

            #[inline]
            pub fn contains(self, elem: usize) -> bool {
                if elem < Self::len() {
                    let array_index = elem / <$simd>::lanes();
                    let vector_index = elem % <$simd>::lanes();
                    unsafe { self.data.get_unchecked(array_index).extract_unchecked(vector_index) > <$scalar>::ZERO }
                } else {
                    false
                }
            }

            #[inline]
            pub unsafe fn contains_unchecked(self, elem: usize) -> bool {
                let array_index = elem / <$simd>::lanes();
                let vector_index = elem % <$simd>::lanes();
                self.data.get_unchecked(array_index).extract_unchecked(vector_index) > <$scalar>::ZERO
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
            pub fn count_zero(&self) -> $scalar {
                self.fold(0, |acc, vec| acc + vec.eq(<$simd>::ZERO).bitmask().count_ones())
            }

            #[inline]
            pub fn count_non_zero(&self) -> $scalar {
                self.fold(0, |acc, vec| acc + vec.gt(<$simd>::ZERO).bitmask().count_ones())
            }

            #[inline]
            pub fn is_empty(&self) -> bool {
                self.data.iter().all(|vec| vec == &<$simd>::ZERO)
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
            pub fn total(&self) -> $scalar {
                self.fold(<$simd>::ZERO, |acc, vec| acc + vec).wrapping_sum()
            }

            /// Horizontal, really bad
            #[inline]
            pub fn argmax(&self) -> (usize, $scalar) {
                let mut the_max = unsafe { self.data.get_unchecked(0).extract_unchecked(0) };
                let mut the_i = 0;

                for arr_idx in 0..UInt::<U, B>::USIZE {
                    for i in 0..<$simd>::lanes() {
                        let val = unsafe { self.data.get_unchecked(arr_idx).extract_unchecked(i) };
                        if val > the_max {
                            the_max = val;
                            the_i = arr_idx * <$simd>::lanes() + i;
                        }
                    }
                }
                (the_i, the_max)
            }

            /// Horizontal, really bad
            #[inline]
            pub fn imax(&self) -> usize {
                self.argmax().0
            }

            /// Horizontal
            #[inline]
            pub fn max(&self) -> $scalar {
                self.fold(<$simd>::ZERO, |acc, vec| acc.max(vec)).max_element()
            }

            /// Horizontal, really bad
            #[inline]
            pub fn argmin(&self) -> (usize, $scalar) {
                let mut the_min = unsafe { self.data.get_unchecked(0).extract_unchecked(0) };
                let mut the_i = 0;

                for arr_idx in 0..UInt::<U, B>::USIZE {
                    for i in 0..<$simd>::lanes() {
                        let val = unsafe { self.data.get_unchecked(arr_idx).extract_unchecked(i) };
                        if val < the_min {
                            the_min = val;
                            the_i = i;
                        }
                    }
                }
                (the_i, the_min)
            }

            /// Horizontal, really bad
            #[inline]
            pub fn imin(&self) -> usize {
                self.argmin().0
            }

            /// Horizontal
            #[inline]
            pub fn min(&self) -> $scalar {
                self.fold(<$simd>::MAX, |acc, vec| acc.min(vec)).min_element()
            }

            #[inline]
            pub fn choose(&mut self, elem: usize) {
                let array_index = elem / <$simd>::lanes();
                let vector_index = elem % <$simd>::lanes();

                for i in 0..UInt::<U, B>::USIZE {
                    let data = unsafe { self.data.get_unchecked_mut(i) };
                    if i == array_index {
                        let mask = <$simd_m>::splat(false).replace(vector_index, true);
                        *data = mask.select(*data, <$simd>::ZERO)
                    } else {
                        *data *= <$scalar>::ZERO
                    }
                }
            }

            /// Horizontal, really bad
            #[inline]
            pub fn choose_random(&mut self, rng: &mut StdRng) {
                let choice_value = rng.gen_range(<$scalar>::ZERO, self.total() + <$scalar>::ONE);
                let mut vector_index: usize = 0;
                let mut acc: $scalar = <$scalar>::ZERO;
                let mut chosen: bool = false;
                for i in 0..UInt::<U, B>::USIZE {
                    let elem_vec = unsafe { self.data.get_unchecked_mut(i) };
                    if chosen {
                        *elem_vec *= <$scalar>::ZERO
                    } else {
                        'vec_loop: for j in 0..<$simd>::lanes() {
                            acc += unsafe { elem_vec.extract_unchecked(j) };
                            if acc > choice_value {
                                vector_index = j;
                                chosen = true;
                                break 'vec_loop
                            }
                        }
                        if chosen {
                            let mask = <$simd_m>::splat(false).replace(vector_index, true);
                            *elem_vec = mask.select(*elem_vec, <$simd>::ZERO)
                        } else {
                            *elem_vec *= <$scalar>::ZERO
                        }
                    }
                }
            }

            /// partial horizontal
            #[inline]
            pub fn collision_entropy(&self) -> f64 {
                let total: f64 = From::from(self.total());
                -self.fold(<$simd_f>::splat(0.0), |acc, vec| {
                    let data = <$simd_f>::from(vec);
                    acc + (data / total).powf(<$simd_f>::splat(2.0))
                }).sum().log2()
            }

            /// partial horizontal
            #[inline]
            pub fn shannon_entropy(&self) -> f64 {
                let total: f64 = From::from(self.total());
                -self.fold(<$simd_f>::splat(0.0), |acc, vec| {
                    let prob = <$simd_f>::from(vec) / total;
                    let data = prob * prob.ln();
                    acc + data.is_nan().select(<$simd_f>::ZERO, data)
                }).sum()
            }
        }
    }
}


multiset_simd_array!(u32x4, u32, f64x4, m32x4);
multiset_simd_array!(u32x2, u32, f64x2, m32x2);


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    type MS2u32x2 = Multiset<u32x2, typenum::U2>;

    #[test]
    fn test_empty() {
        let result = MS2u32x2::empty();
        let expected = MS2u32x2::from_iter(vec![0; 4].into_iter());
        assert_eq!(result, expected)
    }

    #[test]
    fn test_repeat() {
        let result = MS2u32x2::repeat(3);
        let expected = MS2u32x2::from_iter(vec![3; 4].into_iter());
        assert_eq!(result, expected)
    }

    #[test]
    fn test_len() {
        assert_eq!(MS2u32x2::len(), 4)
    }

    #[test]
    fn test_clear() {
        let mut set = MS2u32x2::repeat(3);
        set.clear();
        let expected = MS2u32x2::empty();
        assert_eq!(set, expected)
    }

    #[test]
    fn test_contains() {
        let set = MS2u32x2::from_slice(&[1, 0, 1, 0]);
        assert!(set.contains(0));
        assert!(!set.contains(1));
        assert!(set.contains(2));
        assert!(!set.contains(3));
        assert!(!set.contains(4))
    }

    #[test]
    fn test_contains_unchecked() {
        let set = MS2u32x2::from_slice(&[1, 0, 1, 0]);
        unsafe {
            assert!(set.contains_unchecked(0));
            assert!(!set.contains_unchecked(1));
            assert!(set.contains_unchecked(2));
            assert!(!set.contains_unchecked(3));
        }
    }

    #[test]
    fn test_intersection() {
        let a = MS2u32x2::from_slice(&[2, 0, 4, 0]);
        let b = MS2u32x2::from_slice(&[0, 0, 3, 1]);
        let c = MS2u32x2::from_slice(&[0, 0, 3, 0]);
        assert_eq!(c, a.intersection(&b))
    }

    #[test]
    fn test_union() {
        let a = MS2u32x2::from_slice(&[2, 0, 4, 0]);
        let b = MS2u32x2::from_slice(&[0, 0, 3, 1]);
        let c = MS2u32x2::from_slice(&[2, 0, 4, 1]);
        assert_eq!(c, a.union(&b))
    }

    #[test]
    fn test_count_zero() {
        let set = MS2u32x2::from_slice(&[0, 0, 3, 0]);
        assert_eq!(set.count_zero(), 3)
    }

    #[test]
    fn test_count_non_zero() {
        let set = MS2u32x2::from_slice(&[0, 2, 3, 0]);
        assert_eq!(set.count_non_zero(), 2)
    }

    #[test]
    fn test_is_empty() {
        let a = MS2u32x2::from_slice(&[2, 0, 4, 0]);
        let b = MS2u32x2::from_slice(&[0, 0, 0, 0]);
        assert!(!a.is_empty());
        assert!(b.is_empty());
    }

    #[test]
    fn test_is_singleton() {
        let a = MS2u32x2::from_slice(&[1, 0, 0, 0]);
        assert!(a.is_singleton());

        let b = MS2u32x2::from_slice(&[0, 0, 0, 5]);
        assert!(b.is_singleton());

        let c = MS2u32x2::from_slice(&[1, 0, 0, 5]);
        assert!(!c.is_singleton());

        let d = MS2u32x2::from_slice(&[0, 0, 0, 0]);
        assert!(!d.is_singleton());
    }

    #[test]
    fn test_is_subset() {
        let a = MS2u32x2::from_slice(&[2, 0, 4, 0]);
        let b = MS2u32x2::from_slice(&[2, 0, 4, 1]);
        assert!(a.is_subset(&b));
        assert!(!b.is_subset(&a));

        let c = MS2u32x2::from_slice(&[1, 3, 4, 5]);
        assert!(!a.is_subset(&c));
        assert!(!c.is_subset(&a));
    }

    #[test]
    fn test_is_superset() {
        let a = MS2u32x2::from_slice(&[2, 0, 4, 0]);
        let b = MS2u32x2::from_slice(&[2, 0, 4, 1]);
        assert!(!a.is_superset(&b));
        assert!(b.is_superset(&a));

        let c = MS2u32x2::from_slice(&[1, 3, 4, 5]);
        assert!(!a.is_superset(&c));
        assert!(!c.is_superset(&a));
    }

    #[test]
    fn test_is_any_lesser() {
        let a = MS2u32x2::from_slice(&[2, 0, 4, 0]);
        let b = MS2u32x2::from_slice(&[2, 0, 4, 1]);
        assert!(a.is_any_lesser(&b));
        assert!(!b.is_any_lesser(&a));

        let c = MS2u32x2::from_slice(&[1, 3, 4, 5]);
        assert!(a.is_any_lesser(&c));
        assert!(c.is_any_lesser(&a));
    }

    #[test]
    fn test_is_any_greater() {
        let a = MS2u32x2::from_slice(&[2, 0, 4, 0]);
        let b = MS2u32x2::from_slice(&[2, 0, 4, 1]);
        assert!(!a.is_any_greater(&b));
        assert!(b.is_any_greater(&a));

        let c = MS2u32x2::from_slice(&[1, 3, 4, 5]);
        assert!(a.is_any_greater(&c));
        assert!(c.is_any_greater(&a));
    }

    #[test]
    fn test_total() {
        let set = MS2u32x2::from_slice(&[1, 3, 4, 5]);
        assert_eq!(set.total(), 13)
    }

    #[test]
    fn test_argmax() {
        let set = MS2u32x2::from_slice(&[1, 5, 2, 8]);
        let expected = (3, 8);
        assert_eq!(set.argmax(), expected)
    }

    #[test]
    fn test_imax() {
        let set = MS2u32x2::from_slice(&[1, 5, 2, 8]);
        let expected = 3;
        assert_eq!(set.imax(), expected)
    }

    #[test]
    fn test_max() {
        let set = MS2u32x2::from_slice(&[1, 5, 2, 8]);
        let expected = 8;
        assert_eq!(set.max(), expected)
    }

    #[test]
    fn test_argmin() {
        let set = MS2u32x2::from_slice(&[1, 5, 2, 8]);
        let expected = (0, 1);
        assert_eq!(set.argmin(), expected)
    }

    #[test]
    fn test_imin() {
        let set = MS2u32x2::from_slice(&[1, 5, 2, 8]);
        let expected = 0;
        assert_eq!(set.imin(), expected)
    }

    #[test]
    fn test_min() {
        let set = MS2u32x2::from_slice(&[1, 5, 2, 8]);
        let expected = 1;
        assert_eq!(set.min(), expected)
    }

    #[test]
    fn test_choose() {
        let mut set = MS2u32x2::from_slice(&[2, 1, 3, 4]);
        let expected = MS2u32x2::from_slice(&[0, 0, 3, 0]);
        set.choose(2);
        assert_eq!(set, expected)
    }

    #[test]
    fn test_choose_random() {
        let mut result1 = MS2u32x2::from_slice(&[2, 1, 3, 4]);
        let expected1 = MS2u32x2::from_slice(&[0, 0, 3, 0]);
        let test_rng1 = &mut StdRng::seed_from_u64(5);
        result1.choose_random(test_rng1);
        assert_eq!(result1, expected1);

        let mut result2 = MS2u32x2::from_slice(&[2, 1, 3, 4]);
        let expected2 = MS2u32x2::from_slice(&[2, 0, 0, 0]);
        let test_rng2 = &mut StdRng::seed_from_u64(10);
        result2.choose_random(test_rng2);
        assert_eq!(result2, expected2);
    }

    #[test]
    fn test_choose_random_empty() {
        let mut result = MS2u32x2::from_slice(&[0, 0, 0, 0]);
        let expected = MS2u32x2::from_slice(&[0, 0, 0, 0]);
        let test_rng = &mut StdRng::seed_from_u64(1);
        result.choose_random(test_rng);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_collision_entropy() {
        let simple = MS2u32x2::from_slice(&[200, 0, 0, 0]);
        assert_eq!(simple.collision_entropy(), 0.0);

        let set = MS2u32x2::from_slice(&[2, 1, 1, 0]);
        let result = 1.415037499278844;
        assert_relative_eq!(set.collision_entropy(), result);
    }

    #[test]
    fn test_shannon_entropy1() {
        let a = MS2u32x2::from_slice(&[200, 0, 0, 0]);
        assert_eq!(a.shannon_entropy(), 0.0);

        let b = MS2u32x2::from_slice(&[2, 1, 1, 0]);
        let result = 1.0397207708399179;
        assert_relative_eq!(b.shannon_entropy(), result);
    }

    #[test]
    fn test_shannon_entropy2() {
        let set1 = MS2u32x2::from_slice(&[4, 6, 1, 6]);
        let entropy = set1.shannon_entropy();
        let result1 = 1.2422550455140853;
        assert_relative_eq!(entropy, result1);

        let set2 = MS2u32x2::from_slice(&[4, 6, 0, 6]);
        let entropy = set2.shannon_entropy();
        let result2 = 1.0821955300387673;
        assert_relative_eq!(entropy, result2);
    }
}
