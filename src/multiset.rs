use generic_array::{ArrayLength, GenericArray};
use typenum::{U0, UInt};


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


pub struct Multiset<N, U: MultisetStorage<N>> {
    pub(crate) data: U::Storage
}
