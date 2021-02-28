// Defines the body of partial ord implementations for all multiset implementations, only the
// header of the partial ord implementation differs between the various implementations.
macro_rules! partial_ord_body {
    () => {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            if self == other {
                Some(Ordering::Equal)
            } else if self.lt(other) {
                Some(Ordering::Less)
            } else if self.gt(other) {
                Some(Ordering::Greater)
            } else {
                None
            }
        }

        fn lt(&self, other: &Self) -> bool {
            self.is_proper_subset(other)
        }

        fn le(&self, other: &Self) -> bool {
            self.is_subset(other)
        }

        fn gt(&self, other: &Self) -> bool {
            self.is_proper_superset(other)
        }

        fn ge(&self, other: &Self) -> bool {
            self.is_superset(other)
        }
    };
}
