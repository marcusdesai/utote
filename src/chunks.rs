/*
Chunk Utils

We want to iterate over chunks of the underlying storage of the Multiset structs and process these
chunks as simd vectors. But to do this we need to guarantee that the size of these chunks is always
exactly the length of whatever simd vector we want to use. To avoid restricting the size of the
storage to some multiple of a power of two we need to be able to iterate over chunks and pad any
chunks which are not the length of the simd vector being used.

The structs, traits and impls in this file provide an abstraction for doing this. Since we only
need some specific functionality the abstraction provided doesn't attempt to provide a solution for
iterating generally over padded chunks.
 */

use num_traits::Zero;
use std::slice::from_raw_parts_mut;

// We can implement a much more exacting version of ChunksExact because we can ensure that it will
// only be used on slices of the correct length, and we can use const generics.

struct ChunksExact<'a, T: 'a, const C: usize> {
    slice: &'a [T],
}

impl<'a, T: 'a, const C: usize> ChunksExact<'a, T, C> {
    #[inline]
    pub fn new(slice: &'a [T]) -> Self {
        Self { slice }
    }
}

impl<'a, T, const C: usize> Iterator for ChunksExact<'a, T, C> {
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<&'a [T]> {
        if self.slice.is_empty() {
            None
        } else {
            let (fst, snd) = self.slice.split_at(C);
            self.slice = snd;
            Some(fst)
        }
    }
}

struct ChunksExactMut<'a, T: 'a, const C: usize> {
    slice: &'a mut [T],
}

impl<'a, T: 'a, const C: usize> ChunksExactMut<'a, T, C> {
    #[inline]
    pub fn new(slice: &'a mut [T]) -> Self {
        Self { slice }
    }
}

impl<'a, T, const C: usize> Iterator for ChunksExactMut<'a, T, C> {
    type Item = &'a mut [T];

    #[inline]
    fn next(&mut self) -> Option<&'a mut [T]> {
        if self.slice.is_empty() {
            None
        } else {
            let tmp = std::mem::replace(&mut self.slice, &mut []);
            let (head, tail) = tmp.split_at_mut(C);
            self.slice = tail;
            Some(head)
        }
    }
}

trait ChunksExactTrait<T> {
    fn strict_chunks_exact<const C: usize>(&self) -> ChunksExact<'_, T, C>;
    fn strict_chunks_exact_mut<const C: usize>(&mut self) -> ChunksExactMut<'_, T, C>;
}

impl<T> ChunksExactTrait<T> for [T] {
    #[inline]
    fn strict_chunks_exact<const C: usize>(&self) -> ChunksExact<'_, T, C> {
        ChunksExact::new(self)
    }

    #[inline]
    fn strict_chunks_exact_mut<const C: usize>(&mut self) -> ChunksExactMut<'_, T, C> {
        ChunksExactMut::new(self)
    }
}

struct ChunksPad<'a, T: 'a, const C: usize> {
    v: &'a [T],
    rem: [T; C],
}

impl<'a, T: 'a, const C: usize> ChunksPad<'a, T, C>
where
    T: Copy + Zero,
{
    #[inline]
    pub fn new(slice: &'a [T]) -> Self {
        let rem = slice.len() % C;
        let fst_len = slice.len() - rem;
        // SAFETY: 0 <= fst_len <= slice.len() by construction above
        unsafe {
            let fst = slice.get_unchecked(..fst_len);
            let snd = slice.get_unchecked(fst_len..);
            let mut remainder = [T::zero(); C];
            remainder[..snd.len()].copy_from_slice(snd);
            Self {
                v: fst,
                rem: remainder,
            }
        }
    }

    #[inline]
    pub fn remainder(&self) -> &[T] {
        &self.rem
    }

    #[inline]
    pub fn iter(&self) -> ChunksExact<'_, T, C> {
        self.v.strict_chunks_exact()
    }
}

struct ChunksPadMut<'a, T: 'a, const C: usize> {
    v: &'a mut [T],
    rem: &'a mut [T],
}

impl<'a, T: 'a, const C: usize> ChunksPadMut<'a, T, C>
where
    T: Copy + Zero,
{
    #[inline]
    fn new(slice: &'a mut [T]) -> Self {
        let len = slice.len();
        let rem = len % C;
        let fst_len = len - rem;
        // SAFETY: 0 <= fst_len <= slice.len() by construction above
        let ptr = slice.as_mut_ptr();
        unsafe {
            let fst = from_raw_parts_mut(ptr, fst_len);
            let snd = from_raw_parts_mut(ptr.add(fst_len), len - fst_len);
            Self { v: fst, rem: snd }
        }
    }

    #[inline]
    pub fn remainder_with<F: FnMut(&mut [T])>(self, mut f: F) {
        let mut remainder = [T::zero(); C];
        remainder[..self.rem.len()].swap_with_slice(self.rem);
        f(&mut remainder);
        self.rem.swap_with_slice(&mut remainder[..self.rem.len()]);
    }

    #[inline]
    pub fn iter_mut(&mut self) -> ChunksExactMut<'_, T, C> {
        self.v.strict_chunks_exact_mut()
    }
}

trait ChunkPadUtils<T> {
    fn zip_map_chunks_remainder<F, const C: usize>(&self, other: &Self, out: &mut Self, f: F)
    where
        F: FnMut(&[T], &[T], &mut [T]);
    fn zip_map_chunks_exact<F, const C: usize>(&self, other: &Self, out: &mut Self, f: F)
    where
        F: FnMut(&[T], &[T], &mut [T]);
    fn zip_map_chunks_mut_remainder<F, const C: usize>(&mut self, other: &Self, f: F)
    where
        F: FnMut(&mut [T], &[T]);
    fn zip_map_chunks_mut_exact<F, const C: usize>(&mut self, other: &Self, f: F)
    where
        F: FnMut(&mut [T], &[T]);
    fn zip_all_chunks_remainder<F, const C: usize>(&self, other: &Self, f: F) -> bool
    where
        F: Fn(&[T], &[T]) -> bool;
    fn zip_all_chunks_exact<F, const C: usize>(&self, other: &Self, f: F) -> bool
    where
        F: Fn(&[T], &[T]) -> bool;
    fn zip_any_chunks_remainder<F, const C: usize>(&self, other: &Self, f: F) -> bool
    where
        F: Fn(&[T], &[T]) -> bool;
    fn zip_any_chunks_exact<F, const C: usize>(&self, other: &Self, f: F) -> bool
    where
        F: Fn(&[T], &[T]) -> bool;
    fn fold_chunks_remainder<Acc, F, const C: usize>(&self, init: Acc, f: F) -> Acc
    where
        F: FnMut(Acc, &[T]) -> Acc;
    fn fold_chunks_exact<Acc, F, const C: usize>(&self, init: Acc, f: F) -> Acc
    where
        F: FnMut(Acc, &[T]) -> Acc;
    fn all_chunks_remainder<F, const C: usize>(&self, f: F) -> bool
    where
        F: Fn(&[T]) -> bool;
    fn all_chunks_exact<F, const C: usize>(&self, f: F) -> bool
    where
        F: Fn(&[T]) -> bool;
    fn any_chunks_remainder<F, const C: usize>(&self, f: F) -> bool
    where
        F: Fn(&[T]) -> bool;
    fn any_chunks_exact<F, const C: usize>(&self, f: F) -> bool
    where
        F: Fn(&[T]) -> bool;
}

impl<T> ChunkPadUtils<T> for [T]
where
    T: Copy + Zero,
{
    #[inline]
    fn zip_map_chunks_remainder<F, const C: usize>(&self, other: &Self, out: &mut Self, mut f: F)
    where
        F: FnMut(&[T], &[T], &mut [T]),
    {
        let self_chunks = ChunksPad::<'_, T, C>::new(self);
        let other_chunks = ChunksPad::<'_, T, C>::new(other);
        let mut out_chunks = ChunksPadMut::<'_, T, C>::new(out);

        out_chunks
            .iter_mut()
            .zip(self_chunks.iter().zip(other_chunks.iter()))
            .for_each(|(r, (a, b))| f(a, b, r));
        out_chunks
            .remainder_with(|slice| f(self_chunks.remainder(), other_chunks.remainder(), slice));
    }

    #[inline]
    fn zip_map_chunks_exact<F, const C: usize>(&self, other: &Self, out: &mut Self, mut f: F)
    where
        F: FnMut(&[T], &[T], &mut [T]),
    {
        out.strict_chunks_exact_mut::<C>()
            .zip(
                self.strict_chunks_exact::<C>()
                    .zip(other.strict_chunks_exact::<C>()),
            )
            .for_each(|(r, (a, b))| f(a, b, r));
    }

    #[inline]
    fn zip_map_chunks_mut_remainder<F, const C: usize>(&mut self, other: &Self, mut f: F)
    where
        F: FnMut(&mut [T], &[T]),
    {
        let mut self_chunks = ChunksPadMut::<'_, T, C>::new(self);
        let other_chunks = ChunksPad::<'_, T, C>::new(other);
        self_chunks
            .iter_mut()
            .zip(other_chunks.iter())
            .for_each(|(a, b)| f(a, b));
        self_chunks.remainder_with(|slice| f(slice, other_chunks.remainder()));
    }

    #[inline]
    fn zip_map_chunks_mut_exact<F, const C: usize>(&mut self, other: &Self, mut f: F)
    where
        F: FnMut(&mut [T], &[T]),
    {
        self.strict_chunks_exact_mut::<C>()
            .zip(other.strict_chunks_exact::<C>())
            .for_each(|(a, b)| f(a, b))
    }

    #[inline]
    fn zip_all_chunks_remainder<F, const C: usize>(&self, other: &Self, f: F) -> bool
    where
        F: Fn(&[T], &[T]) -> bool,
    {
        let self_chunks = ChunksPad::<'_, T, C>::new(self);
        let other_chunks = ChunksPad::<'_, T, C>::new(other);
        self_chunks
            .iter()
            .zip(other_chunks.iter())
            .all(|(a, b)| f(a, b))
            && { f(self_chunks.remainder(), other_chunks.remainder()) }
    }

    #[inline]
    fn zip_all_chunks_exact<F, const C: usize>(&self, other: &Self, f: F) -> bool
    where
        F: Fn(&[T], &[T]) -> bool,
    {
        self.strict_chunks_exact::<C>()
            .zip(other.strict_chunks_exact::<C>())
            .all(|(a, b)| f(a, b))
    }

    #[inline]
    fn zip_any_chunks_remainder<F, const C: usize>(&self, other: &Self, f: F) -> bool
    where
        F: Fn(&[T], &[T]) -> bool,
    {
        let self_chunks = ChunksPad::<'_, T, C>::new(self);
        let other_chunks = ChunksPad::<'_, T, C>::new(other);
        self_chunks
            .iter()
            .zip(other_chunks.iter())
            .any(|(a, b)| f(a, b))
            || { f(self_chunks.remainder(), other_chunks.remainder()) }
    }

    #[inline]
    fn zip_any_chunks_exact<F, const C: usize>(&self, other: &Self, f: F) -> bool
    where
        F: Fn(&[T], &[T]) -> bool,
    {
        self.strict_chunks_exact::<C>()
            .zip(other.strict_chunks_exact::<C>())
            .any(|(a, b)| f(a, b))
    }

    #[inline]
    fn fold_chunks_remainder<Acc, F, const C: usize>(&self, init: Acc, mut f: F) -> Acc
    where
        F: FnMut(Acc, &[T]) -> Acc,
    {
        let mut res = init;
        let self_chunks = ChunksPad::<'_, T, C>::new(self);
        for slice in self_chunks.iter() {
            res = f(res, slice);
        }
        res = f(res, self_chunks.remainder());
        res
    }

    #[inline]
    fn fold_chunks_exact<Acc, F, const C: usize>(&self, init: Acc, mut f: F) -> Acc
    where
        F: FnMut(Acc, &[T]) -> Acc,
    {
        let mut res = init;
        for slice in self.strict_chunks_exact::<C>() {
            res = f(res, slice);
        }
        res
    }

    #[inline]
    fn all_chunks_remainder<F, const C: usize>(&self, f: F) -> bool
    where
        F: Fn(&[T]) -> bool,
    {
        let self_chunks = ChunksPad::<'_, T, C>::new(self);
        self_chunks.iter().all(|slice| f(slice)) && f(self_chunks.remainder())
    }

    #[inline]
    fn all_chunks_exact<F, const C: usize>(&self, f: F) -> bool
    where
        F: Fn(&[T]) -> bool,
    {
        self.strict_chunks_exact::<C>().all(|slice| f(slice))
    }

    #[inline]
    fn any_chunks_remainder<F, const C: usize>(&self, f: F) -> bool
    where
        F: Fn(&[T]) -> bool,
    {
        let self_chunks = ChunksPad::<'_, T, C>::new(self);
        self_chunks.iter().any(|slice| f(slice)) || f(self_chunks.remainder())
    }

    #[inline]
    fn any_chunks_exact<F, const C: usize>(&self, f: F) -> bool
    where
        F: Fn(&[T]) -> bool,
    {
        self.strict_chunks_exact::<C>().any(|slice| f(slice))
    }
}

pub(crate) trait ChunkUtils<T> {
    fn zip_map_chunks<F, const C: usize>(&self, other: &Self, out: &mut Self, f: F)
    where
        F: FnMut(&[T], &[T], &mut [T]);
    fn zip_map_chunks_mut<F, const C: usize>(&mut self, other: &Self, f: F)
    where
        F: FnMut(&mut [T], &[T]);
    fn zip_all_chunks<F, const C: usize>(&self, other: &Self, f: F) -> bool
    where
        F: Fn(&[T], &[T]) -> bool;
    fn zip_any_chunks<F, const C: usize>(&self, other: &Self, f: F) -> bool
    where
        F: Fn(&[T], &[T]) -> bool;
    fn fold_chunks<Acc, F, const C: usize>(&self, init: Acc, f: F) -> Acc
    where
        F: FnMut(Acc, &[T]) -> Acc;
    fn all_chunks<F, const C: usize>(&self, f: F) -> bool
    where
        F: Fn(&[T]) -> bool;
    fn any_chunks<F, const C: usize>(&self, f: F) -> bool
    where
        F: Fn(&[T]) -> bool;
}

impl<T> ChunkUtils<T> for [T]
where
    T: Copy + Zero,
{
    #[inline]
    fn zip_map_chunks<F, const C: usize>(&self, other: &Self, out: &mut Self, f: F)
    where
        F: FnMut(&[T], &[T], &mut [T]),
    {
        if self.len() % C == 0 {
            self.zip_map_chunks_exact::<F, C>(other, out, f)
        } else {
            self.zip_map_chunks_remainder::<F, C>(other, out, f)
        }
    }

    #[inline]
    fn zip_map_chunks_mut<F, const C: usize>(&mut self, other: &Self, f: F)
    where
        F: FnMut(&mut [T], &[T]),
    {
        if self.len() % C == 0 {
            self.zip_map_chunks_mut_remainder::<F, C>(other, f)
        } else {
            self.zip_map_chunks_mut_exact::<F, C>(other, f)
        }
    }

    #[inline]
    fn zip_all_chunks<F, const C: usize>(&self, other: &Self, f: F) -> bool
    where
        F: Fn(&[T], &[T]) -> bool,
    {
        if self.len() % C == 0 {
            self.zip_all_chunks_exact::<F, C>(other, f)
        } else {
            self.zip_all_chunks_remainder::<F, C>(other, f)
        }
    }

    #[inline]
    fn zip_any_chunks<F, const C: usize>(&self, other: &Self, f: F) -> bool
    where
        F: Fn(&[T], &[T]) -> bool,
    {
        if self.len() % C == 0 {
            self.zip_any_chunks_exact::<F, C>(other, f)
        } else {
            self.zip_any_chunks_remainder::<F, C>(other, f)
        }
    }

    #[inline]
    fn fold_chunks<Acc, F, const C: usize>(&self, init: Acc, f: F) -> Acc
    where
        F: FnMut(Acc, &[T]) -> Acc,
    {
        if self.len() % C == 0 {
            self.fold_chunks_exact::<Acc, F, C>(init, f)
        } else {
            self.fold_chunks_remainder::<Acc, F, C>(init, f)
        }
    }

    #[inline]
    fn all_chunks<F, const C: usize>(&self, f: F) -> bool
    where
        F: Fn(&[T]) -> bool,
    {
        if self.len() % C == 0 {
            self.all_chunks_exact::<F, C>(f)
        } else {
            self.all_chunks_remainder::<F, C>(f)
        }
    }

    #[inline]
    fn any_chunks<F, const C: usize>(&self, f: F) -> bool
    where
        F: Fn(&[T]) -> bool,
    {
        if self.len() % C == 0 {
            self.any_chunks_exact::<F, C>(f)
        } else {
            self.any_chunks_remainder::<F, C>(f)
        }
    }
}

// SAFETY: `out` vec is constructed as an empty buffer, both zip_map_chunks functions
// guarantee, if successful, that `out` vec will be fully initialised once complete.
// let mut out = Vec::with_capacity(self.len());
// unsafe { out.set_len(self.len()) }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remainder() {
        const CHUNK: usize = 2;
        let slice = [1, 2, 3, 4, 5];
        let chunks = ChunksPad::<u16, CHUNK>::new(&slice);

        assert_eq!(&[5, 0], chunks.remainder());
    }

    #[test]
    fn test_iter() {
        const CHUNK: usize = 2;
        let slice = [1, 2, 3, 4, 5];
        let chunks = ChunksPad::<u16, CHUNK>::new(&slice);
        let mut iter = chunks.iter();

        assert_eq!(&[1, 2], iter.next().unwrap());
        assert_eq!(&[3, 4], iter.next().unwrap());
        assert_eq!(None, iter.next());
    }

    #[test]
    fn test_remainder_with() {
        const CHUNK: usize = 2;
        let mut slice = [1, 2, 3, 4, 5];
        let chunks = ChunksPadMut::<u16, CHUNK>::new(&mut slice);
        chunks.remainder_with(|slice| slice.iter_mut().for_each(|e| *e *= 2));

        assert_eq!(slice, [1, 2, 3, 4, 10]);
    }

    #[test]
    fn test_iter_mut() {
        const CHUNK: usize = 2;
        let mut slice = [1, 2, 3, 4, 5];
        let mut chunks = ChunksPadMut::<u16, CHUNK>::new(&mut slice);
        let iter = chunks.iter_mut();

        iter.for_each(|slice| slice.iter_mut().for_each(|e| *e *= 2));

        assert_eq!(slice, [2, 4, 6, 8, 5]);
    }

    #[test]
    fn test_zip_map_chunks_remainder() {
        const CHUNK: usize = 2;
        let this: [u16; 5] = [1, 2, 3, 4, 5];
        let other = [1, 1, 1, 1, 1];
        let mut out = [0, 0, 0, 0, 0];

        this.zip_map_chunks_remainder::<_, CHUNK>(
            &other,
            &mut out,
            |slice_this, slice_other, slice_out| {
                slice_out
                    .iter_mut()
                    .zip(slice_this.iter().zip(slice_other.iter()))
                    .for_each(|(r, (a, b))| {
                        *r = a + b;
                    })
            },
        );

        assert_eq!(out, [2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_zip_map_chunks_exact() {
        const CHUNK: usize = 2;
        let this: [u16; 4] = [1, 2, 3, 4];
        let other = [1, 1, 1, 1];
        let mut out = [0, 0, 0, 0];

        this.zip_map_chunks_exact::<_, CHUNK>(
            &other,
            &mut out,
            |slice_this, slice_other, slice_out| {
                slice_out
                    .iter_mut()
                    .zip(slice_this.iter().zip(slice_other.iter()))
                    .for_each(|(r, (a, b))| {
                        *r = a + b;
                    })
            },
        );

        assert_eq!(out, [2, 3, 4, 5]);
    }

    #[test]
    fn test_zip_map_chunks_mut_remainder() {
        const CHUNK: usize = 2;
        let mut this: [u16; 5] = [1, 2, 3, 4, 5];
        let other = [1, 1, 1, 1, 1];

        this.zip_map_chunks_mut_remainder::<_, CHUNK>(&other, |slice_this, slice_other| {
            slice_this
                .iter_mut()
                .zip(slice_other.iter())
                .for_each(|(a, b)| *a += b);
        });

        assert_eq!(this, [2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_zip_map_chunks_mut_exact() {
        const CHUNK: usize = 2;
        let mut this: [u16; 4] = [1, 2, 3, 4];
        let other = [1, 1, 1, 1];

        this.zip_map_chunks_mut_exact::<_, CHUNK>(&other, |slice_this, slice_other| {
            slice_this
                .iter_mut()
                .zip(slice_other.iter())
                .for_each(|(a, b)| *a += b);
        });

        assert_eq!(this, [2, 3, 4, 5]);
    }

    #[test]
    fn test_zip_all_chunks_remainder() {
        const CHUNK: usize = 2;
        let this: [u16; 5] = [1, 2, 3, 4, 5];
        let other = [1, 1, 1, 1, 1];

        let is_true =
            this.zip_all_chunks_remainder::<_, CHUNK>(&other, |slice_this, slice_other| {
                slice_this
                    .iter()
                    .zip(slice_other.iter())
                    .all(|(a, b)| a >= b)
            });

        assert!(is_true);

        let is_false =
            this.zip_all_chunks_remainder::<_, CHUNK>(&other, |slice_this, slice_other| {
                slice_this
                    .iter()
                    .zip(slice_other.iter())
                    .all(|(a, b)| a < b)
            });

        assert!(!is_false);
    }

    #[test]
    fn test_zip_all_chunks_exact() {
        const CHUNK: usize = 2;
        let this: [u16; 4] = [1, 2, 3, 4];
        let other = [1, 1, 1, 1];

        let is_true = this.zip_all_chunks_exact::<_, CHUNK>(&other, |slice_this, slice_other| {
            slice_this
                .iter()
                .zip(slice_other.iter())
                .all(|(a, b)| a >= b)
        });

        assert!(is_true);

        let is_false = this.zip_all_chunks_exact::<_, CHUNK>(&other, |slice_this, slice_other| {
            slice_this
                .iter()
                .zip(slice_other.iter())
                .all(|(a, b)| a < b)
        });

        assert!(!is_false);
    }

    #[test]
    fn test_zip_any_chunks_remainder() {
        const CHUNK: usize = 2;
        let this: [u16; 5] = [1, 2, 3, 4, 5];
        let other = [1, 1, 1, 1, 1];

        let is_true =
            this.zip_any_chunks_remainder::<_, CHUNK>(&other, |slice_this, slice_other| {
                slice_this
                    .iter()
                    .zip(slice_other.iter())
                    .any(|(a, b)| b >= a)
            });

        assert!(is_true);

        let is_false =
            this.zip_any_chunks_remainder::<_, CHUNK>(&other, |slice_this, slice_other| {
                slice_this
                    .iter()
                    .zip(slice_other.iter())
                    .any(|(a, b)| b > a)
            });

        assert!(!is_false);
    }

    #[test]
    fn test_zip_any_chunks_exact() {
        const CHUNK: usize = 2;
        let this: [u16; 4] = [1, 2, 3, 4];
        let other = [1, 1, 1, 1];

        let is_true = this.zip_any_chunks_exact::<_, CHUNK>(&other, |slice_this, slice_other| {
            slice_this
                .iter()
                .zip(slice_other.iter())
                .any(|(a, b)| b >= a)
        });

        assert!(is_true);

        let is_false = this.zip_any_chunks_exact::<_, CHUNK>(&other, |slice_this, slice_other| {
            slice_this
                .iter()
                .zip(slice_other.iter())
                .any(|(a, b)| b > a)
        });

        assert!(!is_false);
    }

    #[test]
    fn test_fold_chunks_remainder() {
        const CHUNK: usize = 2;
        let this: [u16; 5] = [1, 2, 3, 4, 5];

        let res =
            this.fold_chunks_remainder::<u16, _, CHUNK>(0, |acc, slice| acc + slice.len() as u16);
        assert_eq!(res, 6);
    }

    #[test]
    fn test_fold_chunks_exact() {
        const CHUNK: usize = 2;
        let this: [u16; 4] = [1, 2, 3, 4];

        let res = this.fold_chunks_exact::<u16, _, CHUNK>(0, |acc, slice| acc + slice.len() as u16);
        assert_eq!(res, 4);
    }

    #[test]
    fn test_all_chunks_remainder() {
        const CHUNK: usize = 2;
        let this: [u16; 5] = [1, 2, 3, 4, 5];

        let res = this.all_chunks_remainder::<_, CHUNK>(|slice| slice.len() == 2);
        assert!(res);
    }

    #[test]
    fn test_all_chunks_exact() {
        const CHUNK: usize = 2;
        let this: [u16; 4] = [1, 2, 3, 4];

        let res = this.all_chunks_exact::<_, CHUNK>(|slice| slice.len() == 2);
        assert!(res);
    }

    #[test]
    fn test_any_chunks_remainder() {
        const CHUNK: usize = 2;
        let this: [u16; 5] = [1, 2, 3, 4, 5];

        let res = this.any_chunks_remainder::<_, CHUNK>(|slice| slice.iter().any(|e| e > &4));
        assert!(res);
    }

    #[test]
    fn test_any_chunks_exact() {
        const CHUNK: usize = 2;
        let this: [u16; 4] = [1, 2, 3, 4];

        let res = this.any_chunks_exact::<_, CHUNK>(|slice| slice.iter().any(|e| e > &4));
        assert!(!res);
    }
}
