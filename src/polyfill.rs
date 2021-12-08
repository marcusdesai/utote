pub use std::arch::x86_64::*;
use std::mem::size_of;

#[inline]
#[allow(non_snake_case)]
const fn _MM_SHUFFLE(z: u32, y: u32, x: u32, w: u32) -> i32 {
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

////////////////////////////////////////////////////////////////////////////////
// Load From / Store To Slices
////////////////////////////////////////////////////////////////////////////////

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn load_unaligned_i128<T>(src: &[T]) -> __m128i {
    debug_assert!(size_of::<T>() * src.len() >= size_of::<__m128i>());
    let ptr = src.as_ptr() as *const __m128i;
    _mm_loadu_si128(ptr)
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn store_unaligned_i128<T>(dst: &mut [T], src: __m128i) {
    debug_assert!(size_of::<T>() * dst.len() >= size_of::<__m128i>());
    let ptr = dst.as_mut_ptr() as *mut __m128i;
    _mm_storeu_si128(ptr, src)
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn load_unaligned_i256<T>(src: &[T]) -> __m256i {
    debug_assert!(size_of::<T>() * src.len() >= size_of::<__m256i>());
    let ptr = src.as_ptr() as *const __m256i;
    _mm256_loadu_si256(ptr)
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn store_unaligned_i256<T>(dst: &mut [T], src: __m256i) {
    debug_assert!(size_of::<T>() * dst.len() >= size_of::<__m256i>());
    let ptr = dst.as_mut_ptr() as *mut __m256i;
    _mm256_storeu_si256(ptr, src)
}

////////////////////////////////////////////////////////////////////////////////
// Comparison Ops - SSE
////////////////////////////////////////////////////////////////////////////////

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_cmple_epu8(a: __m128i, b: __m128i) -> __m128i {
    _mm_cmpeq_epi8(_mm_min_epu8(a, b), a)
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_cmpgt_epu8(a: __m128i, b: __m128i) -> __m128i {
    _mm_andnot_si128(_mm_cmpge_epu8(a, b), _mm_set1_epi64x(-1))
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_cmpge_epu8(a: __m128i, b: __m128i) -> __m128i {
    _mm_cmple_epu8(b, a)
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_cmplt_epu8(a: __m128i, b: __m128i) -> __m128i {
    _mm_cmpgt_epu8(b, a)
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_cmple_epu16(a: __m128i, b: __m128i) -> __m128i {
    _mm_cmpeq_epi16(_mm_min_epu16(a, b), a)
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_cmpgt_epu16(a: __m128i, b: __m128i) -> __m128i {
    _mm_andnot_si128(_mm_cmpge_epu16(a, b), _mm_set1_epi64x(-1))
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_cmpge_epu16(a: __m128i, b: __m128i) -> __m128i {
    _mm_cmple_epu16(b, a)
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_cmplt_epu16(a: __m128i, b: __m128i) -> __m128i {
    _mm_cmpgt_epu16(b, a)
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_cmple_epu32(a: __m128i, b: __m128i) -> __m128i {
    _mm_cmpeq_epi32(_mm_min_epu32(a, b), a)
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_cmpgt_epu32(a: __m128i, b: __m128i) -> __m128i {
    _mm_andnot_si128(_mm_cmpge_epu32(a, b), _mm_set1_epi64x(-1))
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_cmpge_epu32(a: __m128i, b: __m128i) -> __m128i {
    _mm_cmple_epu32(b, a)
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_cmplt_epu32(a: __m128i, b: __m128i) -> __m128i {
    _mm_cmpgt_epu32(b, a)
}

////////////////////////////////////////////////////////////////////////////////
// Comparison Ops - AVX
////////////////////////////////////////////////////////////////////////////////

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn _mm256_cmple_epu8(a: __m256i, b: __m256i) -> __m256i {
    _mm256_cmpeq_epi8(_mm256_min_epu8(a, b), a)
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn _mm256_cmpgt_epu8(a: __m256i, b: __m256i) -> __m256i {
    _mm256_and_si256(_mm256_cmple_epu8(a, b), _mm256_set1_epi64x(-1))
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn _mm256_cmpge_epu8(a: __m256i, b: __m256i) -> __m256i {
    _mm256_cmple_epu8(b, a)
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn _mm256_cmplt_epu8(a: __m256i, b: __m256i) -> __m256i {
    _mm256_cmpgt_epu8(b, a)
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn _mm256_cmple_epu16(a: __m256i, b: __m256i) -> __m256i {
    _mm256_cmpeq_epi16(_mm256_min_epu16(a, b), a)
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn _mm256_cmpgt_epu16(a: __m256i, b: __m256i) -> __m256i {
    _mm256_and_si256(_mm256_cmple_epu16(a, b), _mm256_set1_epi64x(-1))
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn _mm256_cmpge_epu16(a: __m256i, b: __m256i) -> __m256i {
    _mm256_cmple_epu16(b, a)
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn _mm256_cmplt_epu16(a: __m256i, b: __m256i) -> __m256i {
    _mm256_cmpgt_epu16(b, a)
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn _mm256_cmple_epu32(a: __m256i, b: __m256i) -> __m256i {
    _mm256_cmpeq_epi32(_mm256_min_epu32(a, b), a)
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn _mm256_cmpgt_epu32(a: __m256i, b: __m256i) -> __m256i {
    _mm256_and_si256(_mm256_cmple_epu32(a, b), _mm256_set1_epi64x(-1))
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn _mm256_cmpge_epu32(a: __m256i, b: __m256i) -> __m256i {
    _mm256_cmple_epu32(b, a)
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn _mm256_cmplt_epu32(a: __m256i, b: __m256i) -> __m256i {
    _mm256_cmpgt_epu32(b, a)
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn _mm256_cmpgt_epu64(a: __m256i, b: __m256i) -> __m256i {
    let r = _mm256_andnot_si256(_mm256_xor_si256(b, a), _mm256_sub_epi64(b, a));
    let r = _mm256_or_si256(r, _mm256_andnot_si256(b, a));
    _mm256_shuffle_epi32(_mm256_srai_epi32(r, 31), _MM_SHUFFLE(3, 3, 1, 1))
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn _mm256_cmple_epu64(a: __m256i, b: __m256i) -> __m256i {
    _mm256_andnot_si256(_mm256_cmpgt_epu64(a, b), _mm256_set1_epi64x(-1))
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn _mm256_cmplt_epu64(a: __m256i, b: __m256i) -> __m256i {
    _mm256_cmpgt_epu64(b, a)
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn _mm256_cmpge_epu64(a: __m256i, b: __m256i) -> __m256i {
    _mm256_cmple_epu64(b, a)
}

////////////////////////////////////////////////////////////////////////////////
// Min / Max
////////////////////////////////////////////////////////////////////////////////

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn _mm256_blendv_epi64(a: __m256i, b: __m256i, mask: __m256i) -> __m256i {
    _mm256_castpd_si256(_mm256_blendv_pd(
        _mm256_castsi256_pd(a), _mm256_castsi256_pd(b), _mm256_castsi256_pd(mask)))
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn _mm256_min_epu64(a: __m256i, b: __m256i) -> __m256i {
    _mm256_blendv_epi64(b, a, _mm256_xor_si256(
        _mm256_cmpgt_epi64(b, a), _mm256_xor_si256(a, b)))
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn _mm256_max_epu64(a: __m256i, b: __m256i) -> __m256i {
    _mm256_blendv_epi64(b, a, _mm256_xor_si256(
        _mm256_cmpgt_epi64(a, b), _mm256_xor_si256(a, b)))
}

////////////////////////////////////////////////////////////////////////////////
// Multiplication
////////////////////////////////////////////////////////////////////////////////
