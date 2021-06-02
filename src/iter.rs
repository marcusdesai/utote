use std::arch::x86_64::{
    __m128i, __m256i, _mm256_loadu_si256, _mm256_storeu_si256, _mm_loadu_si128, _mm_storeu_si128,
};
use std::ops::{Deref, DerefMut};
use std::slice::from_raw_parts_mut;

#[inline]
unsafe fn load_unaligned_i128<T>(src: &[T]) -> __m128i {
    let ptr = src.as_ptr() as *const __m128i;
    unsafe { _mm_loadu_si128(ptr) }
}

#[inline]
unsafe fn load_unaligned_i256<T>(src: &[T]) -> __m256i {
    let ptr = src.as_ptr() as *const __m256i;
    unsafe { _mm256_loadu_si256(ptr) }
}

#[inline]
unsafe fn store_unaligned_i128<T>(dst: &mut [T], src: __m128i) {
    let ptr = dst.as_mut_ptr() as *mut __m128i;
    unsafe { _mm_storeu_si128(ptr, src) }
}

#[inline]
unsafe fn store_unaligned_i256<T>(dst: &mut [T], src: __m256i) {
    let ptr = dst.as_mut_ptr() as *mut __m256i;
    unsafe { _mm256_storeu_si256(ptr, src) }
}

pub(crate) trait Bits: Sized {
    fn bits() -> usize;
}

macro_rules! impl_bits {
    ($($scalar:ty, $bits:expr),*) => {
        $(impl Bits for $scalar {
            #[inline]
            fn bits() -> usize {
                $bits
            }
        })*
    };
}

impl_bits!(u8, 8, u16, 16, u32, 32, u64, 64);

pub(crate) struct SimdIter<'a, T: 'a, const BITS: usize> {
    slice: &'a [T],
    rem: &'a [T],
}

type SimdIter128<'a, T> = SimdIter<'a, T, 128>;
type SimdIter256<'a, T> = SimdIter<'a, T, 256>;

impl<'a, T: Bits, const BITS: usize> SimdIter<'a, T, BITS> {
    const BITS: usize = BITS;

    #[inline]
    pub fn new(slice: &'a [T]) -> Self {
        let rem = slice.len() % (BITS / T::bits());
        let fst_len = slice.len() - rem;
        // SAFETY: 0 <= fst_len <= slice.len() by construction above
        unsafe {
            let fst = slice.get_unchecked(..fst_len);
            let rem = slice.get_unchecked(fst_len..);
            Self { slice: fst, rem }
        }
    }
}

impl<'a, T: Bits + Copy> Iterator for SimdIter128<'a, T> {
    type Item = __m128i;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.slice.is_empty() {
            if self.rem.is_empty() {
                None
            } else {
                let mut arr = [0u64; 2];
                let (p, buffer, s) = unsafe { arr.align_to_mut::<T>() };
                debug_assert_eq!(buffer.len(), Self::BITS / T::bits());
                debug_assert!(p.is_empty() && s.is_empty());

                let rem = std::mem::replace(&mut self.rem, &[]);
                buffer[..rem.len()].copy_from_slice(rem);
                Some(unsafe { load_unaligned_i128(buffer) })
            }
        } else {
            let (fst, snd) = self.slice.split_at(Self::BITS / T::bits());
            self.slice = snd;
            Some(unsafe { load_unaligned_i128(fst) })
        }
    }
}

impl<'a, T: Bits + Copy> Iterator for SimdIter256<'a, T> {
    type Item = __m256i;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.slice.is_empty() {
            if self.rem.is_empty() {
                None
            } else {
                let mut arr = [0u64; 4];
                let (p, buffer, s) = unsafe { arr.align_to_mut::<T>() };
                debug_assert_eq!(buffer.len(), Self::BITS / T::bits());
                debug_assert!(p.is_empty() && s.is_empty());

                let rem = std::mem::replace(&mut self.rem, &[]);
                buffer[..rem.len()].copy_from_slice(rem);
                Some(unsafe { load_unaligned_i256(buffer) })
            }
        } else {
            let (fst, snd) = self.slice.split_at(Self::BITS / T::bits());
            self.slice = snd;
            Some(unsafe { load_unaligned_i256(fst) })
        }
    }
}

#[derive(Debug)]
pub(crate) struct PadSlice<'a, T: 'a, const C: usize> {
    pad_slice: [u64; C],
    rem: &'a mut [T],
}

impl<'a, T, const C: usize> PadSlice<'a, T, C> {
    #[inline]
    fn new(rem: &'a mut [T]) -> Self {
        let mut pad_slice: [u64; C] = [0; C];
        let (_, buffer, _) = unsafe { pad_slice.align_to_mut::<T>() };
        buffer[..rem.len()].swap_with_slice(rem);
        Self { pad_slice, rem }
    }
}

impl<'a, T, const C: usize> Deref for PadSlice<'a, T, C> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        let (_, slice, _) = unsafe { self.pad_slice.align_to::<T>() };
        slice
    }
}

impl<'a, T, const C: usize> DerefMut for PadSlice<'a, T, C> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        let (_, slice, _) = unsafe { self.pad_slice.align_to_mut::<T>() };
        slice
    }
}

impl<'a, T, const C: usize> Drop for PadSlice<'a, T, C> {
    #[inline]
    fn drop(&mut self) {
        let (_, slice, _) = unsafe { self.pad_slice.align_to_mut::<T>() };
        self.rem.swap_with_slice(&mut slice[..self.rem.len()]);
    }
}

#[derive(Debug)]
pub(crate) enum SliceChunks<'a, T, const C: usize> {
    Pad(PadSlice<'a, T, C>),
    Chunk(&'a mut [T]),
}

type SliceChunks128<'a, T> = SliceChunks<'a, T, 2>;
type SliceChunks256<'a, T> = SliceChunks<'a, T, 4>;

impl<'a, T, const C: usize> Deref for SliceChunks<'a, T, C> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Pad(v) => Deref::deref(v),
            Self::Chunk(slice) => slice,
        }
    }
}

impl<'a, T, const C: usize> DerefMut for SliceChunks<'a, T, C> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Pad(v) => DerefMut::deref_mut(v),
            Self::Chunk(mut_slice) => mut_slice,
        }
    }
}

pub(crate) struct ChunksPadMut<'a, T: 'a, const BITS: usize> {
    slice: &'a mut [T],
    rem: &'a mut [T],
}

type ChunksPadMut128<'a, T> = ChunksPadMut<'a, T, 128>;
type ChunksPadMut256<'a, T> = ChunksPadMut<'a, T, 256>;

impl<'a, T: Bits, const BITS: usize> ChunksPadMut<'a, T, BITS> {
    const BITS: usize = BITS;

    #[inline]
    fn new(slice: &'a mut [T]) -> Self {
        let slice_len = slice.len();
        let rem_len = slice_len % (BITS / T::bits());
        let fst_len = slice_len - rem_len;
        // SAFETY: 0 <= fst_len <= slice.len() by construction above
        let ptr = slice.as_mut_ptr();
        unsafe {
            let fst = from_raw_parts_mut(ptr, fst_len);
            let rem = from_raw_parts_mut(ptr.add(fst_len), slice_len - fst_len);
            Self { slice: fst, rem }
        }
    }
}

impl<'a, T: Bits> Iterator for ChunksPadMut128<'a, T> {
    type Item = SliceChunks128<'a, T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.slice.is_empty() {
            if self.rem.is_empty() {
                None
            } else {
                let rem = std::mem::replace(&mut self.rem, &mut []);
                Some(SliceChunks::Pad(PadSlice::new(rem)))
            }
        } else {
            let tmp = std::mem::replace(&mut self.slice, &mut []);
            let (head, tail) = tmp.split_at_mut(Self::BITS / T::bits());
            self.slice = tail;
            Some(SliceChunks::Chunk(head))
        }
    }
}

impl<'a, T: Bits> Iterator for ChunksPadMut256<'a, T> {
    type Item = SliceChunks256<'a, T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.slice.is_empty() {
            if self.rem.is_empty() {
                None
            } else {
                let rem = std::mem::replace(&mut self.rem, &mut []);
                Some(SliceChunks::Pad(PadSlice::new(rem)))
            }
        } else {
            let tmp = std::mem::replace(&mut self.slice, &mut []);
            let (head, tail) = tmp.split_at_mut(Self::BITS / T::bits());
            self.slice = tail;
            Some(SliceChunks::Chunk(head))
        }
    }
}

trait SimdIters<T> {
    fn simd128(&self) -> SimdIter128<'_, T>;
    fn simd256(&self) -> SimdIter256<'_, T>;
    fn pad_mut128(&mut self) -> ChunksPadMut128<'_, T>;
    fn pad_mut256(&mut self) -> ChunksPadMut256<'_, T>;
}

impl<T> SimdIters<T> for [T]
where
    T: Bits + Copy,
{
    #[inline]
    fn simd128(&self) -> SimdIter128<'_, T> {
        SimdIter128::new(self)
    }

    #[inline]
    fn simd256(&self) -> SimdIter256<'_, T> {
        SimdIter256::new(self)
    }

    #[inline]
    fn pad_mut128(&mut self) -> ChunksPadMut128<'_, T> {
        ChunksPadMut128::new(self)
    }

    #[inline]
    fn pad_mut256(&mut self) -> ChunksPadMut256<'_, T> {
        ChunksPadMut256::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Debug;
    use std::ops::MulAssign;

    fn blah<T: Bits + Copy + Debug + MulAssign<T>>(slice: &mut [T], m: T) {
        slice.pad_mut128().for_each(|mut slc| {
            &mut slc.iter_mut().for_each(|e| *e *= m);
            println!("{:?}", slc);
        });
        println!("{:?}", slice);
    }

    #[test]
    fn test_pad_128() {
        blah(&mut [1u8, 2, 3, 4, 5], 2)
    }
}
