use generic_array::{ArrayLength, GenericArray};
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};
use typenum::{UInt, Unsigned, U0};

/// Trait enabling the sleight of hand using different typenum structs to define different storage
/// types.
pub trait MultisetStorage<T> {
    type Storage;
}

impl<N> MultisetStorage<N> for U0 {
    type Storage = N;
}

impl<N, U, B> MultisetStorage<N> for UInt<U, B>
where
    UInt<U, B>: ArrayLength<N>,
{
    type Storage = GenericArray<N, Self>;
}

/// Multiset! yay
#[derive(Debug, Copy, Clone)]
pub struct Multiset<N, U: MultisetStorage<N>> {
    pub(crate) data: U::Storage,
}

impl<N, U: MultisetStorage<N>> PartialEq for Multiset<N, U>
where
    <U as MultisetStorage<N>>::Storage: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<N> AddAssign for Multiset<N, U0>
where
    N: AddAssign,
{
    fn add_assign(&mut self, rhs: Self) {
        self.data += rhs.data
    }
}

impl<N> SubAssign for Multiset<N, U0>
where
    N: SubAssign,
{
    fn sub_assign(&mut self, rhs: Self) {
        self.data -= rhs.data
    }
}

impl<N> MulAssign for Multiset<N, U0>
where
    N: MulAssign,
{
    fn mul_assign(&mut self, rhs: Self) {
        self.data *= rhs.data
    }
}

impl<N> DivAssign for Multiset<N, U0>
where
    N: DivAssign,
{
    fn div_assign(&mut self, rhs: Self) {
        self.data /= rhs.data
    }
}

impl<N, U, B> AddAssign for Multiset<N, UInt<U, B>>
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

impl<N, U, B> SubAssign for Multiset<N, UInt<U, B>>
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

impl<N, U, B> MulAssign for Multiset<N, UInt<U, B>>
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

impl<N, U, B> DivAssign for Multiset<N, UInt<U, B>>
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
