/*
Chunk Utils
 */

use crate::small_num::SmallNumConsts;
use std::slice::{Chunks, ChunksMut, from_raw_parts_mut};

struct ChunksPad<'a, T: 'a, const SIZE: usize> {
    v: &'a [T],
    rem: [T; SIZE]
}

impl<'a, T: 'a, const SIZE: usize> ChunksPad<'a, T, SIZE>
    where
        T: Copy + SmallNumConsts
{
    const REM: [T; SIZE] = [T::ZERO; SIZE];

    pub fn new(slice: &'a [T]) -> Self {
        let rem = slice.len() % SIZE;
        let fst_len = slice.len() - rem;
        // SAFETY: 0 <= fst_len <= slice.len() by construction above
        unsafe {
            let fst = slice.get_unchecked(..fst_len);
            let snd = slice.get_unchecked(fst_len..);
            let mut remainder = Self::REM;
            remainder[..snd.len()].copy_from_slice(snd);
            Self { v: fst, rem: remainder }
        }
    }

    pub fn remainder(&self) -> &[T] { &self.rem }

    pub fn iter(&self) -> Chunks<'_, T> {
        self.v.chunks(SIZE)
    }
}

struct ChunksPadMut<'a, T: 'a, const SIZE: usize> {
    v: &'a mut [T],
    rem: &'a mut [T],
}

impl<'a, T: 'a, const SIZE: usize> ChunksPadMut<'a, T, SIZE>
    where
        T: Copy + SmallNumConsts
{
    const REM: [T; SIZE] = [T::ZERO; SIZE];

    #[inline]
    fn new(slice: &'a mut [T]) -> Self {
        let len = slice.len();
        let rem = len % SIZE;
        let fst_len = len - rem;
        // SAFETY: 0 <= fst_len <= slice.len() by construction above
        let ptr = slice.as_mut_ptr();
        unsafe {
            let fst = from_raw_parts_mut(ptr, fst_len);
            let snd = from_raw_parts_mut(ptr.add(fst_len), len - fst_len);
            Self { v: fst, rem: snd }
        }
    }

    pub fn remainder_with<F: FnMut(&mut [T])>(self, mut f: F) {
        let mut remainder = Self::REM;
        remainder[..self.rem.len()].swap_with_slice(self.rem);
        f(&mut remainder);
        self.rem.swap_with_slice(&mut remainder[..self.rem.len()]);
    }

    pub fn iter_mut(&mut self) -> ChunksMut<'_, T> {
        self.v.chunks_mut(SIZE)
    }
}

pub(crate) trait ChunkUtils<T> {
    fn zip_map_chunks_remainder<F, const C: usize>(&self, other: &Self, out: &mut Self, f: F) where
        F: FnMut(&[T], &[T], &mut [T]);
    fn zip_map_chunks_exact<F, const C: usize>(&self, other: &Self, out: &mut Self, f: F) where
        F: FnMut(&[T], &[T], &mut [T]);
    fn zip_all_chunks_remainder<F, const C: usize>(&self, other: &Self, f: F) -> bool where
        F: Fn(&[T], &[T]) -> bool;
    fn zip_all_chunks_exact<F, const C: usize>(&self, other: &Self, f: F) -> bool where
        F: Fn(&[T], &[T]) -> bool;
    fn fold_chunks_remainder<Acc, F, const C: usize>(&self, init: Acc, f: F) -> Acc where
        F: FnMut(Acc, &[T]) -> Acc;
    fn fold_chunks_exact<Acc, F, const C: usize>(&self, init: Acc, f: F) -> Acc where
        F: FnMut(Acc, &[T]) -> Acc;
    fn all_chunks_remainder<F, const C: usize>(&self, f: F) -> bool where F: Fn(&[T]) -> bool;
    fn all_chunks_exact<F, const C: usize>(&self, f: F) -> bool where F: Fn(&[T]) -> bool;
    fn any_chunks_remainder<F, const C: usize>(&self, f: F) -> bool where F: Fn(&[T]) -> bool;
    fn any_chunks_exact<F, const C: usize>(&self, f: F) -> bool where F: Fn(&[T]) -> bool;
}

impl<T> ChunkUtils<T> for [T]
    where
        T: Copy + SmallNumConsts
{
    #[inline]
    fn zip_map_chunks_remainder<F, const C: usize>(&self, other: &Self, out: &mut Self, mut f: F)
        where
            F: FnMut(&[T], &[T], &mut [T]),
    {
        let self_chunks: ChunksPad<'_, T, C> = ChunksPad::new(self);
        let other_chunks: ChunksPad<'_, T, C> = ChunksPad::new(other);
        let mut res_chunks: ChunksPadMut<'_, T, C> = ChunksPadMut::new(out);

        res_chunks
            .iter_mut()
            .zip(self_chunks.iter().zip(other_chunks.iter()))
            .for_each(|(r, (a, b))| f(a, b, r));
        res_chunks.remainder_with(|slice| {
            f(self_chunks.remainder(), other_chunks.remainder(), slice)
        });
    }

    #[inline]
    fn zip_map_chunks_exact<F, const C: usize>(&self, other: &Self, out: &mut Self, mut f: F)
        where
            F: FnMut(&[T], &[T], &mut [T]),
    {
        out.chunks_mut(C)
            .zip(self.chunks(C).zip(other.chunks(C)))
            .for_each(|(r, (a, b))| f(a, b, r));
    }

    #[inline]
    fn zip_all_chunks_remainder<F, const C: usize>(&self, other: &Self, f: F) -> bool
        where
            F: Fn(&[T], &[T]) -> bool,
    {
        let self_chunks: ChunksPad<'_, T, C> = ChunksPad::new(self);
        let other_chunks: ChunksPad<'_, T, C> = ChunksPad::new(other);
        self_chunks.iter().zip(other_chunks.iter()).all(|(a, b)| f(a, b)) && {
            f(self_chunks.remainder(), other_chunks.remainder())
        }
    }

    #[inline]
    fn zip_all_chunks_exact<F, const C: usize>(&self, other: &Self, f: F) -> bool
        where
            F: Fn(&[T], &[T]) -> bool,
    {
        self.chunks(C).zip(other.chunks(C)).all(|(a, b)| f(a, b))
    }

    #[inline]
    fn fold_chunks_remainder<Acc, F, const C: usize>(&self, init: Acc, mut f: F) -> Acc
        where
            F: FnMut(Acc, &[T]) -> Acc,
    {
        let mut res = init;
        let self_chunks: ChunksPad<'_, T, C> = ChunksPad::new(self);
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
        for slice in self.chunks(C) {
            res = f(res, slice);
        }
        res
    }

    #[inline]
    fn all_chunks_remainder<F, const C: usize>(&self, f: F) -> bool
        where
            F: Fn(&[T]) -> bool,
    {
        let self_chunks: ChunksPad<'_, T, C> = ChunksPad::new(self);
        self_chunks.iter().all(|slice| f(slice)) && f(self_chunks.remainder())
    }

    #[inline]
    fn all_chunks_exact<F, const C: usize>(&self, f: F) -> bool
        where
            F: Fn(&[T]) -> bool,
    {
        self.chunks(C).all(|slice| f(slice))
    }

    #[inline]
    fn any_chunks_remainder<F, const C: usize>(&self, f: F) -> bool
        where
            F: Fn(&[T]) -> bool,
    {
        let self_chunks: ChunksPad<'_, T, C> = ChunksPad::new(self);
        self_chunks.iter().any(|slice| f(slice)) || f(self_chunks.remainder())
    }

    #[inline]
    fn any_chunks_exact<F, const C: usize>(&self, f: F) -> bool
        where
            F: Fn(&[T]) -> bool,
    {
        self.chunks(C).any(|slice| f(slice))
    }
}
