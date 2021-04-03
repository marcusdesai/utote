use generic_array::{ArrayLength, GenericArray};
use generic_array::functional::FunctionalSequence;
use std::mem;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use typenum::Unsigned;
use std::fmt::{Debug, Formatter, Result};
use std::hash::{Hash, Hasher};

pub trait MultisetStorage<T> {
    type Storage;
}

impl<N, U: Unsigned> MultisetStorage<N> for U
    where
        U: ArrayLength<N>,
{
    type Storage = GenericArray<N, Self>;
}

/// Multiset! yay
pub struct Multiset<N, U: MultisetStorage<N>> {
    pub(crate) data: U::Storage,
}

// Safe because Multiset.data is private to Multiset
unsafe impl<N: Send, U: MultisetStorage<N>> Send for Multiset<N, U> {}
unsafe impl<N: Sync, U: MultisetStorage<N>> Sync for Multiset<N, U> {}

impl<N: Hash, U: Unsigned> Hash for Multiset<N, U>
    where
        U: MultisetStorage<N, Storage=GenericArray<N, U>> + ArrayLength<N>,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data.hash(state)
    }
}

impl<N: Debug, U: Unsigned> Debug for Multiset<N, U>
    where
        U: MultisetStorage<N, Storage=GenericArray<N, U>> + ArrayLength<N>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("Multiset").field("data", &self.data).finish()
    }
}

impl<N: Clone, U: Unsigned> Clone for Multiset<N, U>
    where
        U: MultisetStorage<N, Storage=GenericArray<N, U>> + ArrayLength<N>,
{
    fn clone(&self) -> Self {
        Multiset { data: self.data.clone() }
    }
}

impl<N: Copy, U: Unsigned> Copy for Multiset<N, U>
    where
        U::ArrayType: Copy,
        U: MultisetStorage<N, Storage=GenericArray<N, U>> + ArrayLength<N>, {}

impl<N, U: MultisetStorage<N>> PartialEq for Multiset<N, U>
    where
        <U as MultisetStorage<N>>::Storage: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<N, U: MultisetStorage<N>> Eq for Multiset<N, U>
    where Multiset<N, U>: PartialEq, {}

pub struct MultisetIterator<N, U: MultisetStorage<N>> {
    pub(crate) multiset: Multiset<N, U>,
    pub(crate) index: usize,
}

impl<N, U: Unsigned> Add for Multiset<N, U>
    where
        N: Add + Copy,
        <N as Add>::Output: Copy,
        U: ArrayLength<N> + ArrayLength<<N as Add>::Output>,
{
    type Output = Multiset<<N as Add>::Output, U>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut res: Self::Output = unsafe { Multiset::new_uninitialized() };
        for i in 0..U::USIZE {
            unsafe {
                let e1 = self.data.get_unchecked(i);
                let e2 = rhs.data.get_unchecked(i);
                *res.data.get_unchecked_mut(i) = *e1 + *e2;
            }
        }
        res
    }
}

impl<N, U: Unsigned> AddAssign for Multiset<N, U>
    where
        N: AddAssign + Copy,
        U: ArrayLength<N>,
{
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..U::USIZE {
            unsafe {
                let e = rhs.data.get_unchecked(i);
                *self.data.get_unchecked_mut(i) += *e;
            }
        }
    }
}

impl<N, U: Unsigned> Sub for Multiset<N, U>
    where
        N: Sub + Copy,
        <N as Sub>::Output: Copy,
        U: ArrayLength<N> + ArrayLength<<N as Sub>::Output>,
{
    type Output = Multiset<<N as Sub>::Output, U>;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut res: Self::Output = unsafe { Multiset::new_uninitialized() };
        for i in 0..U::USIZE {
            unsafe {
                let e1 = self.data.get_unchecked(i);
                let e2 = rhs.data.get_unchecked(i);
                *res.data.get_unchecked_mut(i) = *e1 - *e2;
            }
        }
        res
    }
}

impl<N, U: Unsigned> SubAssign for Multiset<N, U>
    where
        N: SubAssign + Copy,
        U: ArrayLength<N>,
{
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..U::USIZE {
            unsafe {
                let e = rhs.data.get_unchecked(i);
                *self.data.get_unchecked_mut(i) -= *e;
            }
        }
    }
}

impl<N, U: Unsigned> Mul for Multiset<N, U>
    where
        N: Mul + Copy,
        <N as Mul>::Output: Copy,
        U: ArrayLength<N> + ArrayLength<<N as Mul>::Output>,
{
    type Output = Multiset<<N as Mul>::Output, U>;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut res: Self::Output = unsafe { Multiset::new_uninitialized() };
        for i in 0..U::USIZE {
            unsafe {
                let e1 = self.data.get_unchecked(i);
                let e2 = rhs.data.get_unchecked(i);
                *res.data.get_unchecked_mut(i) = *e1 * *e2;
            }
        }
        res
    }
}

impl<N, U: Unsigned> MulAssign for Multiset<N, U>
    where
        N: MulAssign + Copy,
        U: ArrayLength<N>,
{
    fn mul_assign(&mut self, rhs: Self) {
        for i in 0..U::USIZE {
            unsafe {
                let e = rhs.data.get_unchecked(i);
                *self.data.get_unchecked_mut(i) *= *e;
            }
        }
    }
}

impl<N, U: Unsigned> Div for Multiset<N, U>
    where
        N: Div + Copy,
        <N as Div>::Output: Copy,
        U: ArrayLength<N> + ArrayLength<<N as Div>::Output>,
{
    type Output = Multiset<<N as Div>::Output, U>;

    fn div(self, rhs: Self) -> Self::Output {
        let mut res: Self::Output = unsafe { Multiset::new_uninitialized() };
        for i in 0..U::USIZE {
            unsafe {
                let e1 = self.data.get_unchecked(i);
                let e2 = rhs.data.get_unchecked(i);
                *res.data.get_unchecked_mut(i) = *e1 / *e2;
            }
        }
        res
    }
}

impl<N, U: Unsigned> DivAssign for Multiset<N, U>
    where
        N: DivAssign + Copy,
        U: ArrayLength<N>,
{
    fn div_assign(&mut self, rhs: Self) {
        for i in 0..U::USIZE {
            unsafe {
                let e = rhs.data.get_unchecked(i);
                *self.data.get_unchecked_mut(i) /= *e;
            }
        }
    }
}

impl<N: Copy, U: Unsigned> Multiset<N, U>
    where
        U: ArrayLength<N>,
{
    #[inline]
    pub(crate) unsafe fn new_uninitialized() -> Self {
        Multiset {
            data: mem::MaybeUninit::<GenericArray<N, U>>::uninit().assume_init(),
        }
    }

    #[inline]
    pub(crate) fn map<N2, F>(&self, f: F) -> Multiset<N2, U>
        where
            N2: Copy,
            F: FnMut(N) -> N2,
            U: ArrayLength<N2>,
            GenericArray<N, U>: Copy,
    {
        Multiset { data: self.data.map(f) }
    }

    #[inline]
    pub(crate) fn fold<Acc, F>(&self, init: Acc, f: F) -> Acc
        where
            F: FnMut(Acc, N) -> Acc,
            GenericArray<N, U>: Copy,
    {
        self.data.fold(init, f)
    }

    #[inline]
    pub(crate) fn zip_map<N2, N3, F>(&self, other: &Multiset<N2, U>, f: F) -> Multiset<N3, U>
        where
            N2: Copy,
            N3: Copy,
            F: FnMut(N, N2) -> N3,
            U: ArrayLength<N2> + ArrayLength<N3>,
            GenericArray<N, U>: Copy,
            GenericArray<N2, U>: Copy,
    {
        Multiset { data: self.data.zip((*other).data, f) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type MS4u32 = Multiset<u32, typenum::U4>;

    #[test]
    fn test_map() {
        let set = MS4u32::from_slice(&[1, 5, 2, 8]);
        let result = set.map(|e| e * 2);
        let expected = MS4u32::from_slice(&[2, 10, 4, 16]);
        assert_eq!(result, expected)
    }

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
