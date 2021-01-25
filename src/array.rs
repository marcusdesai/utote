use generic_array::{ArrayLength, GenericArray};
use std::mem;
use typenum::{UInt, Unsigned};

use crate::multiset::Multiset;


impl<N: Clone, U, B> Multiset<N, UInt<U, B>>
    where
        UInt<U, B>: ArrayLength<N>
{
    #[inline]
    pub(crate) unsafe fn new_uninitialized() -> Self {
        Multiset {
            data: mem::MaybeUninit::<GenericArray<N, UInt<U, B>>>::uninit().assume_init()
        }
    }

    #[inline]
    pub fn map<N2, F>(&self, mut f: F) -> Multiset<N2, UInt<U, B>>
        where
            N2: Clone,
            F: FnMut(N) -> N2,
            UInt<U, B>: ArrayLength<N2>,
    {
        let mut res = unsafe { Multiset::new_uninitialized() };
        for i in 0..UInt::<U, B>::USIZE {
            unsafe {
                let e = self.data.get_unchecked(i).clone();
                *res.data.get_unchecked_mut(i) = f(e)
            }
        }
        res
    }

    #[inline]
    pub fn fold<Acc, F>(&self, init: Acc, mut f: F) -> Acc
        where
            F: FnMut(Acc, N) -> Acc,
    {
        let mut res = init;
        for i in 0..UInt::<U, B>::USIZE {
            unsafe {
                let e = self.data.get_unchecked(i).clone();
                res = f(res, e)
            }
        }
        res
    }

    #[inline]
    pub fn zip_map<N2, N3, F>(&self, other: &Multiset<N2, UInt<U, B>>, mut f: F) -> Multiset<N3, UInt<U, B>>
        where
            N2: Clone,
            N3: Clone,
            F: FnMut(N, N2) -> N3,
            UInt<U, B>: ArrayLength<N2> + ArrayLength<N3>,
    {
        let mut res = unsafe { Multiset::new_uninitialized() };
        for i in 0..UInt::<U, B>::USIZE {
            unsafe {
                let e1 = self.data.get_unchecked(i).clone();
                let e2 = other.data.get_unchecked(i).clone();
                *res.data.get_unchecked_mut(i) = f(e1, e2)
            }
        }
        res
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    type MS4u8 = Multiset<u32, typenum::U4>;

    #[test]
    fn test_map() {
        let set = MS4u8::from_slice(&[1, 5, 2, 8]);
        let result = set.map(|e| e * 2);
        let expected = MS4u8::from_slice(&[2, 10, 4, 16]);
        assert_eq!(result, expected)
    }

    #[test]
    fn test_fold() {
        let set = MS4u8::from_slice(&[1, 5, 2, 8]);
        let result = set.fold(0, |acc, e| acc + e * 2);
        let expected = 32;
        assert_eq!(result, expected)
    }

    #[test]
    fn test_zip_map() {
        let set1 = MS4u8::from_slice(&[1, 5, 2, 8]);
        let set2 = MS4u8::from_slice(&[1, 5, 2, 8]);
        let result = set1.zip_map(&set2, |e1, e2| e1 + e2);
        let expected = MS4u8::from_slice(&[2, 10, 4, 16]);
        assert_eq!(result, expected)
    }
}