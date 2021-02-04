macro_rules! tests_x4 {
    ($tests_name:ident, $scalar:ty, $ms_store:ty) => {
        #[cfg(test)]
        mod $tests_name {
            use super::*;
            use approx::assert_relative_eq;

            type MSType = Multiset<$scalar, $ms_store>;

            test_constructors!(MSType, 4);
            test_contains!(MSType, &[1, 0, 1, 0], &[0, 2], &[1, 3], &[4]);
            test_insert_remove_get!(MSType, &[1, 0, 1, 0], 2, 3, 1);
            test_intersection_union!(
                MSType,
                &[2, 0, 4, 0],
                &[0, 0, 3, 1],
                &[0, 0, 3, 0],
                &[2, 0, 4, 1]
            );
            test_count_zero!(MSType, &[0, 0, 3, 0], 3, 1);
            test_is_empty!(MSType, &[2, 0, 4, 0]);
            test_is_singleton!(MSType, &[0, 3, 0, 0], &[1, 2, 3, 4]);
            test_is_subset_superset!(MSType, &[2, 0, 4, 0], &[2, 0, 4, 1], &[1, 3, 4, 5]);
            test_total!(MSType, &[2, 1, 4, 3], 10);
            test_max_min!(MSType, &[1, 5, 2, 8], 3, 8, 0, 1);
            test_choices!(MSType, &[2, 1, 3, 4], &[0, 1, 0, 0]);
            test_entropy!(
                MSType,
                &[200, 0, 0, 0],
                &[2, 1, 1, 0],
                0.0,
                1.415037499278844,
                0.0,
                1.0397207708399179
            );
        }
    };
}

macro_rules! tests_x8 {
    ($tests_name:ident, $scalar:ty, $ms_store:ty) => {
        #[cfg(test)]
        mod $tests_name {
            use super::*;
            use approx::assert_relative_eq;

            type MSType = Multiset<$scalar, $ms_store>;

            test_constructors!(MSType, 8);
            test_contains!(
                MSType,
                &[1, 0, 1, 0, 3, 4, 0, 0],
                &[0, 2, 4, 5],
                &[1, 3, 6, 7],
                &[9]
            );
            test_insert_remove_get!(MSType, &[1, 0, 1, 0, 3, 4, 0, 0], 4, 1, 3);
            test_intersection_union!(
                MSType,
                &[2, 0, 4, 0, 7, 5, 0, 1],
                &[0, 0, 3, 1, 2, 2, 9, 0],
                &[0, 0, 3, 0, 2, 2, 0, 0],
                &[2, 0, 4, 1, 7, 5, 9, 1]
            );
            test_count_zero!(MSType, &[0, 0, 3, 0, 7, 4, 3, 1], 3, 5);
            test_is_empty!(MSType, &[2, 0, 4, 0, 0, 0, 1, 1]);
            test_is_singleton!(MSType, &[0, 3, 0, 0, 0, 0, 0, 0], &[1, 2, 3, 4, 1, 2, 3, 4]);
            test_is_subset_superset!(
                MSType,
                &[2, 0, 4, 0, 3, 1, 0, 5],
                &[2, 0, 4, 1, 3, 1, 0, 6],
                &[1, 3, 4, 5, 4, 0, 0, 0]
            );
            test_total!(MSType, &[2, 1, 4, 3, 6, 7, 0, 9], 32);
            test_max_min!(MSType, &[1, 5, 2, 8, 9, 6, 3, 4], 4, 9, 0, 1);
            test_choices!(MSType, &[2, 1, 3, 4, 0, 9, 5, 6], &[0, 1, 0, 0, 0, 0, 0, 0]);
            test_entropy!(
                MSType,
                &[200, 0, 0, 0, 0, 0, 0, 0],
                &[2, 1, 1, 0, 0, 0, 0, 0],
                0.0,
                1.415037499278844,
                0.0,
                1.0397207708399179
            );
        }
    };
}

macro_rules! test_constructors {
    ($typ:ty, $size:expr) => {
        #[test]
        fn test_empty() {
            let result = <$typ>::empty();
            let expected = <$typ>::from_iter(vec![0; $size].into_iter());
            assert_eq!(result, expected)
        }

        #[test]
        fn test_repeat() {
            let result = <$typ>::repeat(3);
            let expected = <$typ>::from_iter(vec![3; $size].into_iter());
            assert_eq!(result, expected)
        }

        #[test]
        fn test_len() {
            assert_eq!(<$typ>::len(), $size)
        }

        #[test]
        fn test_clear() {
            let mut set = <$typ>::repeat(3);
            set.clear();
            let expected = <$typ>::empty();
            assert_eq!(set, expected)
        }
    };
}

macro_rules! test_contains {
    ($typ:ty, $slice:expr, $contains:expr, $not_contains:expr, $out_of_bounds:expr) => {
        #[test]
        fn test_contains() {
            let set = <$typ>::from_slice($slice);
            $contains.iter().for_each(|elem| {
                assert!(set.contains(*elem));
            });

            $not_contains.iter().for_each(|elem| {
                assert!(!set.contains(*elem));
            });

            $out_of_bounds.iter().for_each(|elem| {
                assert!(!set.contains(*elem));
            });
        }

        #[test]
        fn test_contains_unchecked() {
            let set = <$typ>::from_slice($slice);
            unsafe {
                $contains.iter().for_each(|elem| {
                    assert!(set.contains_unchecked(*elem));
                });

                $not_contains.iter().for_each(|elem| {
                    assert!(!set.contains_unchecked(*elem));
                });
            }
        }
    };
}

macro_rules! test_insert_remove_get {
    ($typ:ty, $slice:expr, $elem:expr, $insert:expr, $get:expr) => {
        #[test]
        fn test_insert() {
            let mut set = <$typ>::from_slice($slice);
            set.insert($elem, $insert);
            assert_eq!(set.get($elem), Some($insert))
        }

        #[test]
        fn test_insert_unchecked() {
            let mut set = <$typ>::from_slice($slice);
            unsafe {
                set.insert($elem, $insert);
                assert_eq!(set.get_unchecked($elem), $insert)
            }
        }

        #[test]
        fn test_remove() {
            let mut set = <$typ>::from_slice($slice);
            set.remove($elem);
            assert_eq!(set.get($elem), Some(0))
        }

        #[test]
        fn test_remove_unchecked() {
            let mut set = <$typ>::from_slice($slice);
            unsafe {
                set.remove($elem);
                assert_eq!(set.get_unchecked($elem), 0)
            }
        }

        #[test]
        fn test_get() {
            let set = <$typ>::from_slice($slice);
            assert_eq!(set.get($elem), Some($get))
        }

        #[test]
        fn test_get_unchecked() {
            let set = <$typ>::from_slice($slice);
            unsafe { assert_eq!(set.get_unchecked($elem), $get) }
        }
    };
}

macro_rules! test_intersection_union {
    ($typ:ty, $slice1:expr, $slice2:expr, $intersection:expr, $union:expr) => {
        #[test]
        fn test_intersection() {
            let a = <$typ>::from_slice($slice1);
            let b = <$typ>::from_slice($slice2);
            let c = <$typ>::from_slice($intersection);
            assert_eq!(c, a.intersection(&b))
        }

        #[test]
        fn test_union() {
            let a = <$typ>::from_slice($slice1);
            let b = <$typ>::from_slice($slice2);
            let c = <$typ>::from_slice($union);
            assert_eq!(c, a.union(&b))
        }
    };
}

macro_rules! test_count_zero {
    ($typ:ty, $slice:expr, $count_zero:expr, $count_non_zero:expr) => {
        #[test]
        fn test_count_zero() {
            let set = <$typ>::from_slice($slice);
            assert_eq!(set.count_zero(), $count_zero)
        }

        #[test]
        fn test_count_non_zero() {
            let set = <$typ>::from_slice($slice);
            assert_eq!(set.count_non_zero(), $count_non_zero)
        }
    };
}

macro_rules! test_is_empty {
    ($typ:ty, $non_empty:expr) => {
        #[test]
        fn test_is_empty() {
            let a = <$typ>::from_slice($non_empty);
            let b = <$typ>::empty();
            assert!(!a.is_empty());
            assert!(b.is_empty());
        }
    };
}

macro_rules! test_is_singleton {
    ($typ:ty, $singleton:expr, $four_elem:expr) => {
        #[test]
        fn test_is_singleton() {
            let a = <$typ>::from_slice($singleton);
            assert!(a.is_singleton());

            let b = <$typ>::from_slice($four_elem);
            assert!(!b.is_singleton());

            let c = <$typ>::empty();
            assert!(!c.is_singleton());
        }
    };
}

macro_rules! test_is_subset_superset {
    ($typ:ty, $sub:expr, $super:expr, $neither:expr) => {
        #[test]
        fn test_is_subset() {
            let a = <$typ>::from_slice($sub);
            let b = <$typ>::from_slice($super);
            assert!(a.is_subset(&b));
            assert!(!b.is_subset(&a));

            let c = <$typ>::from_slice($neither);
            assert!(!a.is_subset(&c));
            assert!(!c.is_subset(&a));

            let d = <$typ>::from_slice($sub);
            assert!(a.is_subset(&d));
            assert!(d.is_subset(&a));
        }

        #[test]
        fn test_is_superset() {
            let a = <$typ>::from_slice($sub);
            let b = <$typ>::from_slice($super);
            assert!(!a.is_superset(&b));
            assert!(b.is_superset(&a));

            let c = <$typ>::from_slice($neither);
            assert!(!a.is_superset(&c));
            assert!(!c.is_superset(&a));

            let d = <$typ>::from_slice($sub);
            assert!(a.is_superset(&d));
            assert!(d.is_superset(&a));
        }

        #[test]
        fn test_is_proper_subset() {
            let a = <$typ>::from_slice($sub);
            let b = <$typ>::from_slice($super);
            assert!(a.is_proper_subset(&b));
            assert!(!b.is_proper_subset(&a));

            let c = <$typ>::from_slice($neither);
            assert!(!a.is_proper_subset(&c));
            assert!(!c.is_proper_subset(&a));

            let d = <$typ>::from_slice($sub);
            assert!(!a.is_proper_subset(&d));
            assert!(!d.is_proper_subset(&a));
        }

        #[test]
        fn test_is_proper_superset() {
            let a = <$typ>::from_slice($sub);
            let b = <$typ>::from_slice($super);
            assert!(!a.is_proper_superset(&b));
            assert!(b.is_proper_superset(&a));

            let c = <$typ>::from_slice($neither);
            assert!(!a.is_proper_superset(&c));
            assert!(!c.is_proper_superset(&a));

            let d = <$typ>::from_slice($sub);
            assert!(!a.is_proper_superset(&d));
            assert!(!d.is_proper_superset(&a));
        }

        #[test]
        fn test_is_any_lesser() {
            let a = <$typ>::from_slice($sub);
            let b = <$typ>::from_slice($super);
            assert!(a.is_any_lesser(&b));
            assert!(!b.is_any_lesser(&a));

            let c = <$typ>::from_slice($neither);
            assert!(a.is_any_lesser(&c));
            assert!(c.is_any_lesser(&a));

            let d = <$typ>::from_slice($sub);
            assert!(!a.is_any_lesser(&d));
            assert!(!d.is_any_lesser(&a));
        }

        #[test]
        fn test_is_any_greater() {
            let a = <$typ>::from_slice($sub);
            let b = <$typ>::from_slice($super);
            assert!(!a.is_any_greater(&b));
            assert!(b.is_any_greater(&a));

            let c = <$typ>::from_slice($neither);
            assert!(a.is_any_greater(&c));
            assert!(c.is_any_greater(&a));

            let d = <$typ>::from_slice($sub);
            assert!(!a.is_any_greater(&d));
            assert!(!d.is_any_greater(&a));
        }
    };
}

macro_rules! test_total {
    ($typ:ty, $slice:expr, $result:expr) => {
        #[test]
        fn test_total() {
            let set = <$typ>::from_slice($slice);
            assert_eq!(set.total(), $result)
        }
    };
}

macro_rules! test_max_min {
    ($typ:ty, $slice:expr, $max_idx:expr, $max_val:expr, $min_idx:expr, $min_val:expr) => {
        #[test]
        fn test_argmax() {
            let set = <$typ>::from_slice($slice);
            let expected = ($max_idx, $max_val);
            assert_eq!(set.argmax(), expected)
        }

        #[test]
        fn test_imax() {
            let set = <$typ>::from_slice($slice);
            let expected = $max_idx;
            assert_eq!(set.imax(), expected)
        }

        #[test]
        fn test_max() {
            let set = <$typ>::from_slice($slice);
            let expected = $max_val;
            assert_eq!(set.max(), expected)
        }

        #[test]
        fn test_argmin() {
            let set = <$typ>::from_slice($slice);
            let expected = ($min_idx, $min_val);
            assert_eq!(set.argmin(), expected)
        }

        #[test]
        fn test_imin() {
            let set = <$typ>::from_slice($slice);
            let expected = $min_idx;
            assert_eq!(set.imin(), expected)
        }

        #[test]
        fn test_min() {
            let set = <$typ>::from_slice($slice);
            let expected = $min_val;
            assert_eq!(set.min(), expected)
        }
    };
}

macro_rules! test_choices {
    ($typ:ty, $slice:expr, $choice:expr) => {
        #[test]
        fn test_choose() {
            let mut set = <$typ>::from_slice($slice);
            let expected = <$typ>::from_slice($choice);
            set.choose(1);
            assert_eq!(set, expected)
        }

        #[test]
        fn test_choose_random() {
            let mut result1 = <$typ>::from_slice($slice);
            let test_rng1 = &mut StdRng::seed_from_u64(thread_rng().next_u64());
            result1.choose_random(test_rng1);
            assert!(result1.is_singleton() && result1.is_subset(&<$typ>::from_slice($slice)));

            let mut result2 = <$typ>::from_slice($slice);
            let test_rng2 = &mut StdRng::seed_from_u64(thread_rng().next_u64());
            result2.choose_random(test_rng2);
            assert!(result1.is_singleton() && result1.is_subset(&<$typ>::from_slice($slice)));
        }

        #[test]
        fn test_choose_random_empty() {
            let mut result = <$typ>::empty();
            let expected = <$typ>::empty();
            let test_rng = &mut StdRng::seed_from_u64(thread_rng().next_u64());
            result.choose_random(test_rng);
            assert_eq!(result, expected);
        }
    };
}

macro_rules! test_entropy {
    ($typ:ty, $slice1:expr, $slice2:expr, $col1:expr, $col2:expr, $shan1:expr, $shan2:expr) => {
        #[test]
        fn test_collision_entropy() {
            let simple = <$typ>::from_slice($slice1);
            assert_eq!(simple.collision_entropy(), $col1);

            let set = <$typ>::from_slice($slice2);
            assert_relative_eq!(set.collision_entropy(), $col2, epsilon = f64::EPSILON);
        }

        #[test]
        fn test_shannon_entropy1() {
            let a = <$typ>::from_slice($slice1);
            assert_eq!(a.shannon_entropy(), $shan1);

            let b = <$typ>::from_slice($slice2);
            assert_relative_eq!(b.shannon_entropy(), $shan2, epsilon = f64::EPSILON);
        }
    };
}
