use generic_array::{ArrayLength, GenericArray};
use std::mem;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use typenum::bit::Bit;
use typenum::uint::{UInt, Unsigned, UTerm};
use std::fmt::{Debug, Formatter, Result};
use std::hash::{Hash, Hasher};

/// Trait enabling the sleight of hand using different typenum structs to define different storage
/// types.
pub trait MultisetStorage<T>: Unsigned {
    type Storage;
}

impl<N> MultisetStorage<N> for UTerm {
    type Storage = N;
}

impl<N, U: Unsigned, B: Bit> MultisetStorage<N> for UInt<U, B>
    where
        UInt<U, B>: ArrayLength<N>,
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

impl<N: Hash> Hash for Multiset<N, UTerm>
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data.hash(state);
    }
}

impl<N: Hash, U: Unsigned, B: Bit> Hash for Multiset<N, UInt<U, B>>
    where
        UInt<U, B>: MultisetStorage<N, Storage=GenericArray<N, UInt<U, B>>> + ArrayLength<N>,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data.hash(state)
    }
}

impl<N: Debug> Debug for Multiset<N, UTerm>
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("Multiset").field("data", &self.data).finish()
    }
}

impl<N: Debug, U: Unsigned, B: Bit> Debug for Multiset<N, UInt<U, B>>
    where
        UInt<U, B>: MultisetStorage<N, Storage=GenericArray<N, UInt<U, B>>> + ArrayLength<N>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("Multiset").field("data", &self.data).finish()
    }
}

impl<N: Clone> Clone for Multiset<N, UTerm>
{
    fn clone(&self) -> Self {
        Multiset { data: self.data.clone() }
    }
}

impl<N: Clone, U: Unsigned, B: Bit> Clone for Multiset<N, UInt<U, B>>
    where
        UInt<U, B>: MultisetStorage<N, Storage=GenericArray<N, UInt<U, B>>> + ArrayLength<N>,
{
    fn clone(&self) -> Self {
        Multiset { data: self.data.clone() }
    }
}

impl<N: Copy> Copy for Multiset<N, UTerm> {}

impl<N: Copy, U: Unsigned, B: Bit> Copy for Multiset<N, UInt<U, B>>
    where
        GenericArray<N, UInt<U, B>>: Copy,
        UInt<U, B>: MultisetStorage<N, Storage=GenericArray<N, UInt<U, B>>> + ArrayLength<N>, {}

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

impl<N> Add for Multiset<N, UTerm>
    where
        N: Add,
{
    type Output = Multiset<<N as Add>::Output, UTerm>;

    fn add(self, rhs: Self) -> Self::Output {
        Multiset { data: self.data + rhs.data }
    }
}

impl<N> AddAssign for Multiset<N, UTerm>
    where
        N: AddAssign,
{
    fn add_assign(&mut self, rhs: Self) {
        self.data += rhs.data
    }
}

impl<N> Sub for Multiset<N, UTerm>
    where
        N: Sub,
{
    type Output = Multiset<<N as Sub>::Output, UTerm>;

    fn sub(self, rhs: Self) -> Self::Output {
        Multiset { data: self.data - rhs.data }
    }
}

impl<N> SubAssign for Multiset<N, UTerm>
    where
        N: SubAssign,
{
    fn sub_assign(&mut self, rhs: Self) {
        self.data -= rhs.data
    }
}

impl<N> Mul for Multiset<N, UTerm>
    where
        N: Mul,
{
    type Output = Multiset<<N as Mul>::Output, UTerm>;

    fn mul(self, rhs: Self) -> Self::Output {
        Multiset { data: self.data * rhs.data }
    }
}

impl<N> MulAssign for Multiset<N, UTerm>
    where
        N: MulAssign,
{
    fn mul_assign(&mut self, rhs: Self) {
        self.data *= rhs.data
    }
}

impl<N> Div for Multiset<N, UTerm>
    where
        N: Div,
{
    type Output = Multiset<<N as Div>::Output, UTerm>;

    fn div(self, rhs: Self) -> Self::Output {
        Multiset { data: self.data / rhs.data }
    }
}

impl<N> DivAssign for Multiset<N, UTerm>
    where
        N: DivAssign,
{
    fn div_assign(&mut self, rhs: Self) {
        self.data /= rhs.data
    }
}

impl<N, U: Unsigned, B: Bit> Add for Multiset<N, UInt<U, B>>
    where
        N: Add + Copy,
        <N as Add>::Output: Copy,
        UInt<U, B>: ArrayLength<N> + ArrayLength<<N as Add>::Output>,
{
    type Output = Multiset<<N as Add>::Output, UInt<U, B>>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut res = unsafe { Multiset::new_uninitialized() };
        for i in 0..UInt::<U, B>::USIZE {
            unsafe {
                let e1 = self.data.get_unchecked(i);
                let e2 = rhs.data.get_unchecked(i);
                *res.data.get_unchecked_mut(i) = *e1 + *e2;
            }
        }
        res
    }
}

impl<N, U: Unsigned, B: Bit> AddAssign for Multiset<N, UInt<U, B>>
    where
        N: AddAssign + Copy,
        UInt<U, B>: ArrayLength<N>,
{
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..UInt::<U, B>::USIZE {
            unsafe {
                let e = rhs.data.get_unchecked(i);
                *self.data.get_unchecked_mut(i) += *e;
            }
        }
    }
}

impl<N, U: Unsigned, B: Bit> Sub for Multiset<N, UInt<U, B>>
    where
        N: Sub + Copy,
        <N as Sub>::Output: Copy,
        UInt<U, B>: ArrayLength<N> + ArrayLength<<N as Sub>::Output>,
{
    type Output = Multiset<<N as Sub>::Output, UInt<U, B>>;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut res = unsafe { Multiset::new_uninitialized() };
        for i in 0..UInt::<U, B>::USIZE {
            unsafe {
                let e1 = self.data.get_unchecked(i);
                let e2 = rhs.data.get_unchecked(i);
                *res.data.get_unchecked_mut(i) = *e1 - *e2;
            }
        }
        res
    }
}

impl<N, U: Unsigned, B: Bit> SubAssign for Multiset<N, UInt<U, B>>
    where
        N: SubAssign + Copy,
        UInt<U, B>: ArrayLength<N>,
{
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..UInt::<U, B>::USIZE {
            unsafe {
                let e = rhs.data.get_unchecked(i);
                *self.data.get_unchecked_mut(i) -= *e;
            }
        }
    }
}

impl<N, U: Unsigned, B: Bit> Mul for Multiset<N, UInt<U, B>>
    where
        N: Mul + Copy,
        <N as Mul>::Output: Copy,
        UInt<U, B>: ArrayLength<N> + ArrayLength<<N as Mul>::Output>,
{
    type Output = Multiset<<N as Mul>::Output, UInt<U, B>>;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut res = unsafe { Multiset::new_uninitialized() };
        for i in 0..UInt::<U, B>::USIZE {
            unsafe {
                let e1 = self.data.get_unchecked(i);
                let e2 = rhs.data.get_unchecked(i);
                *res.data.get_unchecked_mut(i) = *e1 * *e2;
            }
        }
        res
    }
}

impl<N, U: Unsigned, B: Bit> MulAssign for Multiset<N, UInt<U, B>>
    where
        N: MulAssign + Copy,
        UInt<U, B>: ArrayLength<N>,
{
    fn mul_assign(&mut self, rhs: Self) {
        for i in 0..UInt::<U, B>::USIZE {
            unsafe {
                let e = rhs.data.get_unchecked(i);
                *self.data.get_unchecked_mut(i) *= *e;
            }
        }
    }
}

impl<N, U: Unsigned, B: Bit> Div for Multiset<N, UInt<U, B>>
    where
        N: Div + Copy,
        <N as Div>::Output: Copy,
        UInt<U, B>: ArrayLength<N> + ArrayLength<<N as Div>::Output>,
{
    type Output = Multiset<<N as Div>::Output, UInt<U, B>>;

    fn div(self, rhs: Self) -> Self::Output {
        let mut res = unsafe { Multiset::new_uninitialized() };
        for i in 0..UInt::<U, B>::USIZE {
            unsafe {
                let e1 = self.data.get_unchecked(i);
                let e2 = rhs.data.get_unchecked(i);
                *res.data.get_unchecked_mut(i) = *e1 / *e2;
            }
        }
        res
    }
}

impl<N, U: Unsigned, B: Bit> DivAssign for Multiset<N, UInt<U, B>>
    where
        N: DivAssign + Copy,
        UInt<U, B>: ArrayLength<N>,
{
    fn div_assign(&mut self, rhs: Self) {
        for i in 0..UInt::<U, B>::USIZE {
            unsafe {
                let e = rhs.data.get_unchecked(i);
                *self.data.get_unchecked_mut(i) /= *e;
            }
        }
    }
}

impl<N: Copy, U: Unsigned, B: Bit> Multiset<N, UInt<U, B>>
    where
        UInt<U, B>: ArrayLength<N>,
{
    #[inline]
    pub(crate) unsafe fn new_uninitialized() -> Self {
        Multiset {
            data: mem::MaybeUninit::<GenericArray<N, UInt<U, B>>>::uninit().assume_init(),
        }
    }

    #[inline]
    pub fn map<N2, F>(&self, mut f: F) -> Multiset<N2, UInt<U, B>>
        where
            N2: Copy,
            F: FnMut(N) -> N2,
            UInt<U, B>: ArrayLength<N2>,
    {
        let mut res = unsafe { Multiset::new_uninitialized() };
        for i in 0..UInt::<U, B>::USIZE {
            unsafe {
                let e = *self.data.get_unchecked(i);
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
                let e = *self.data.get_unchecked(i);
                res = f(res, e)
            }
        }
        res
    }

    #[inline]
    pub fn zip_map<N2, N3, F>(
        &self,
        other: &Multiset<N2, UInt<U, B>>,
        mut f: F,
    ) -> Multiset<N3, UInt<U, B>>
        where
            N2: Copy,
            N3: Copy,
            F: FnMut(N, N2) -> N3,
            UInt<U, B>: ArrayLength<N2> + ArrayLength<N3>,
    {
        let mut res = unsafe { Multiset::new_uninitialized() };
        for i in 0..UInt::<U, B>::USIZE {
            unsafe {
                let e1 = *self.data.get_unchecked(i);
                let e2 = *other.data.get_unchecked(i);
                *res.data.get_unchecked_mut(i) = f(e1, e2)
            }
        }
        res
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
