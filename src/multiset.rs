use std::mem;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use std::fmt::{Debug, Formatter, Result};
use std::hash::{Hash, Hasher};

/// Multiset! yay
pub struct Multiset<N, const SIZE: usize> {
    pub(crate) data: [N; SIZE],
}

// Safe because Multiset.data is private to Multiset
unsafe impl<N: Send, const SIZE: usize> Send for Multiset<N, SIZE> {}
unsafe impl<N: Sync, const SIZE: usize> Sync for Multiset<N, SIZE> {}

impl<N: Hash, const SIZE: usize> Hash for Multiset<N, SIZE> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data.hash(state)
    }
}

impl<N: Debug, const SIZE: usize> Debug for Multiset<N, SIZE> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("Multiset").field("data", &self.data).finish()
    }
}

impl<N: Clone, const SIZE: usize> Clone for Multiset<N, SIZE> {
    fn clone(&self) -> Self {
        Multiset { data: self.data.clone() }
    }
}

impl<N: Copy, const SIZE: usize> Copy for Multiset<N, SIZE> {}

impl<N: PartialEq, const SIZE: usize> PartialEq for Multiset<N, SIZE> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<N, const SIZE: usize> Eq for Multiset<N, SIZE>
    where Multiset<N, SIZE>: PartialEq, {}

pub struct MultisetIterator<N, const SIZE: usize> {
    pub(crate) multiset: Multiset<N, SIZE>,
    pub(crate) index: usize,
}

impl<N, const SIZE: usize> Add for Multiset<N, SIZE>
    where
        N: Add + Copy,
        <N as Add>::Output: Copy,
{
    type Output = Multiset<<N as Add>::Output, SIZE>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut res: Self::Output = unsafe { Multiset::new_uninitialized() };
        for i in 0..SIZE {
            unsafe {
                let e1 = self.data.get_unchecked(i);
                let e2 = rhs.data.get_unchecked(i);
                *res.data.get_unchecked_mut(i) = *e1 + *e2;
            }
        }
        res
    }
}

impl<N, const SIZE: usize> AddAssign for Multiset<N, SIZE>
    where
        N: AddAssign + Copy,
{
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..SIZE {
            unsafe {
                let e = rhs.data.get_unchecked(i);
                *self.data.get_unchecked_mut(i) += *e;
            }
        }
    }
}

impl<N, const SIZE: usize> Sub for Multiset<N, SIZE>
    where
        N: Sub + Copy,
        <N as Sub>::Output: Copy,
{
    type Output = Multiset<<N as Sub>::Output, SIZE>;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut res: Self::Output = unsafe { Multiset::new_uninitialized() };
        for i in 0..SIZE {
            unsafe {
                let e1 = self.data.get_unchecked(i);
                let e2 = rhs.data.get_unchecked(i);
                *res.data.get_unchecked_mut(i) = *e1 - *e2;
            }
        }
        res
    }
}

impl<N, const SIZE: usize> SubAssign for Multiset<N, SIZE>
    where
        N: SubAssign + Copy,
{
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..SIZE {
            unsafe {
                let e = rhs.data.get_unchecked(i);
                *self.data.get_unchecked_mut(i) -= *e;
            }
        }
    }
}

impl<N, const SIZE: usize> Mul for Multiset<N, SIZE>
    where
        N: Mul + Copy,
        <N as Mul>::Output: Copy,
{
    type Output = Multiset<<N as Mul>::Output, SIZE>;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut res: Self::Output = unsafe { Multiset::new_uninitialized() };
        for i in 0..SIZE {
            unsafe {
                let e1 = self.data.get_unchecked(i);
                let e2 = rhs.data.get_unchecked(i);
                *res.data.get_unchecked_mut(i) = *e1 * *e2;
            }
        }
        res
    }
}

impl<N, const SIZE: usize> MulAssign for Multiset<N, SIZE>
    where
        N: MulAssign + Copy,
{
    fn mul_assign(&mut self, rhs: Self) {
        for i in 0..SIZE {
            unsafe {
                let e = rhs.data.get_unchecked(i);
                *self.data.get_unchecked_mut(i) *= *e;
            }
        }
    }
}

impl<N, const SIZE: usize> Div for Multiset<N, SIZE>
    where
        N: Div + Copy,
        <N as Div>::Output: Copy,
{
    type Output = Multiset<<N as Div>::Output, SIZE>;

    fn div(self, rhs: Self) -> Self::Output {
        let mut res: Self::Output = unsafe { Multiset::new_uninitialized() };
        for i in 0..SIZE {
            unsafe {
                let e1 = self.data.get_unchecked(i);
                let e2 = rhs.data.get_unchecked(i);
                *res.data.get_unchecked_mut(i) = *e1 / *e2;
            }
        }
        res
    }
}

impl<N, const SIZE: usize> DivAssign for Multiset<N, SIZE>
    where
        N: DivAssign + Copy,
{
    fn div_assign(&mut self, rhs: Self) {
        for i in 0..SIZE {
            unsafe {
                let e = rhs.data.get_unchecked(i);
                *self.data.get_unchecked_mut(i) /= *e;
            }
        }
    }
}

impl<N: Copy, const SIZE: usize> Multiset<N, SIZE> {
    #[inline]
    pub(crate) unsafe fn new_uninitialized() -> Self {
        Multiset { data: mem::MaybeUninit::<[N; SIZE]>::uninit().assume_init() }
    }

    // Does not abstract over simd and scalar types.
    #[inline]
    pub(crate) fn fold<Acc, F>(&self, init: Acc, mut f: F) -> Acc
        where
            F: FnMut(Acc, N) -> Acc,
    {
        let mut res = init;
        for i in 0..SIZE {
            unsafe {
                let e = *self.data.get_unchecked(i);
                res = f(res, e)
            }
        }
        res
    }

    // Does not abstract over simd and scalar types.
    #[inline]
    pub(crate) fn zip_map<N2, N3, F>(&self, other: &Multiset<N2, SIZE>, mut f: F) -> Multiset<N3, SIZE>
        where
            N2: Copy,
            N3: Copy,
            F: FnMut(N, N2) -> N3,
    {
        let mut res = unsafe { Multiset::new_uninitialized() };
        self.data.iter().zip(other.data.iter()).enumerate().for_each(|(i, (a, b))| unsafe {
            *res.data.get_unchecked_mut(i) = f(*a, *b)
        });
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type MS4u32 = Multiset<u32, 4>;

    #[test]
    fn test_fold() {
        let set = MS4u32::from_slice(&[1, 5, 2, 8]);
        let result = set.fold(0, |acc, e| acc + e * 2);
        let expected = 32;
        assert_eq!(result, expected)
    }

    #[test]
    fn test_zip_map() {
        let set1 = MS4u32::from_slice(&[1, 5, 2, 8]);
        let set2 = MS4u32::from_slice(&[1, 5, 2, 8]);
        let result = set1.zip_map(&set2, |e1, e2| e1 + e2);
        let expected = MS4u32::from_slice(&[2, 10, 4, 16]);
        assert_eq!(result, expected)
    }
}
