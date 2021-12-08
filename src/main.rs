use core::arch::x86_64::*;
use core::mem::size_of;

#[target_feature(enable = "sse4.2")]
unsafe fn _mm_mullo_epi8(a: __m128i, b: __m128i) -> __m128i {
    let dst_even = _mm_mullo_epi16(a, b);
    let dst_odd = _mm_mullo_epi16(_mm_srli_epi16(a, 8), _mm_srli_epi16(b, 8));
    _mm_or_si128(
        _mm_slli_epi16(dst_odd, 8),
        _mm_srli_epi16(_mm_slli_epi16(dst_even, 8), 8),
    )
}

#[target_feature(enable = "avx2")]
unsafe fn _mm256_mullo_epi8(a: __m256i, b: __m256i) -> __m256i {
    let dst_even = _mm256_mullo_epi16(a, b);
    let dst_odd = _mm256_mullo_epi16(_mm256_srli_epi16(a, 8), _mm256_srli_epi16(b, 8));
    _mm256_or_si256(
        _mm256_slli_epi16(dst_odd, 8),
        _mm256_and_si256(dst_even, _mm256_set1_epi16(0xFF)),
    )
}

#[allow(non_snake_case)]
#[inline]
pub const fn _MM_SHUFFLE(z: u32, y: u32, x: u32, w: u32) -> i32 {
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

#[inline]
#[target_feature(enable = "sse4.2")]
unsafe fn _mm_cmpgt_epu64(a: __m128i, b: __m128i) -> __m128i {
    let r = _mm_andnot_si128(_mm_xor_si128(b, a), _mm_sub_epi64(b, a));
    let r = _mm_or_si128(r, _mm_andnot_si128(b, a));
    _mm_shuffle_epi32(_mm_srai_epi32(r, 31), _MM_SHUFFLE(3, 3, 1, 1))
}

#[inline]
#[target_feature(enable = "sse4.2")]
unsafe fn _mm_cmplt_epu64(a: __m128i, b: __m128i) -> __m128i {
    _mm_cmpgt_epu64(b, a)
}

#[target_feature(enable = "sse4.2")]
unsafe fn _mm_cmpgt_epu32(a: __m128i, b: __m128i) -> __m128i {
    _mm_andnot_si128(_mm_cmpeq_epi32(_mm_min_epu32(a, b), a), _mm_set1_epi32(-1))
}

#[target_feature(enable = "sse4.2")]
unsafe fn _mm_cmpgt_epu16(a: __m128i, b: __m128i) -> __m128i {
    _mm_andnot_si128(_mm_cmpeq_epi16(_mm_min_epu16(a, b), a), _mm_set1_epi16(-1))
}

#[target_feature(enable = "sse4.2")]
unsafe fn _mm_cmpgt_epu8(a: __m128i, b: __m128i) -> __m128i {
    _mm_andnot_si128(_mm_cmpeq_epi8(_mm_min_epu8(a, b), a), _mm_set1_epi8(-1))
}

#[inline]
#[target_feature(enable = "sse4.2")]
unsafe fn load_unaligned_i128<T>(src: &[T]) -> __m128i {
    debug_assert!(size_of::<T>() * src.len() >= size_of::<__m128i>());
    let ptr = src.as_ptr() as *const __m128i;
    _mm_loadu_si128(ptr)
}

#[inline]
#[target_feature(enable = "sse4.2")]
unsafe fn store_unaligned_i128<T>(dst: &mut [T], src: __m128i) {
    debug_assert!(size_of::<T>() * dst.len() >= size_of::<__m128i>());
    let ptr = dst.as_mut_ptr() as *mut __m128i;
    _mm_storeu_si128(ptr, src)
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load_unaligned_i256<T>(src: &[T]) -> __m256i {
    debug_assert!(size_of::<T>() * src.len() >= size_of::<__m256i>());
    let ptr = src.as_ptr() as *const __m256i;
    unsafe { _mm256_loadu_si256(ptr) }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn store_unaligned_i256<T>(dst: &mut [T], src: __m256i) {
    debug_assert!(size_of::<T>() * dst.len() >= size_of::<__m256i>());
    let ptr = dst.as_mut_ptr() as *mut __m256i;
    unsafe { _mm256_storeu_si256(ptr, src) }
}

#[target_feature(enable = "avx2")]
unsafe fn simd_stuff() {
    // let x = &mut [1u16, 2, 3, 4, 5, 6, 7, 8];
    // let vec = load_unaligned_i128(x);
    //
    // let x2: &[u16] = &[2, 2, 2, 2, 2, 2, 2, 2];
    // let vec2 = load_unaligned_i128(x2);
    // println!("{:?}", vec2);
    //
    // let mul = _mm_mullo_epi16(vec, vec2);
    // store_unaligned_i128(x, mul);
    // println!("{:?}", x);
    //
    // let aa: &mut [u8] = &mut [3; 16];
    // let aa_vec = load_unaligned_i128(aa);
    // let res = _mm_mullo_epi8(aa_vec, aa_vec);
    // store_unaligned_i128(aa, res);
    // println!("{:?}", aa);
    //
    // let bb: &mut [u8] = &mut [20; 32];
    // let bb_vec = load_unaligned_i256(bb);
    // let res = _mm256_mullo_epi8(bb_vec, bb_vec);
    // store_unaligned_i256(bb, res);
    // // todo: make sure to note that simd ops overflow
    // println!("{:?}", bb);
    // println!("{:?}", u8::overflowing_mul(20, 20));
    //
    // println!("{}", _MM_SHUFFLE(3, 3, 1, 1));

    let cmp1: &[u64] = &[u64::MAX - 2343, u64::MAX];
    let cmp2: &[u64] = &[u64::MAX, 2];
    let out: &mut [i64] = &mut [0, 0];
    let cmp1_vec = load_unaligned_i128(cmp1);
    let cmp2_vec = load_unaligned_i128(cmp2);
    store_unaligned_i128(out, _mm_cmpgt_epu64(cmp1_vec, cmp2_vec));
    println!("{:?}", out);
}

fn main() {
    unsafe { simd_stuff() }
}
