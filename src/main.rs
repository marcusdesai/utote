use core::arch::x86_64::*;

#[target_feature(enable = "sse4.2")]
unsafe fn _mm_mullo_epi8(a: __m128i, b: __m128i) -> __m128i {
    let dst_even = _mm_mullo_epi16(a, b);
    let dst_odd = _mm_mullo_epi16(_mm_srli_epi16(a, 8),_mm_srli_epi16(b, 8));
    _mm_or_si128(_mm_slli_epi16(dst_odd, 8),
                 _mm_srli_epi16(_mm_slli_epi16(dst_even,8), 8))
}

#[target_feature(enable = "sse4.2")]
unsafe fn simd_stuff() {
    // let bytes: [i8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    // let v = bytes.align_to::<u16>();
    // println!("{:?}", v);

    let x = &mut [1u16, 2, 3, 4, 5, 6, 7, 8];
    let ptr = x.as_ptr() as *const __m128i;
    let vec = unsafe { _mm_loadu_si128(ptr) };

    let x2: &[u16] = &[2, 2, 2, 2, 2, 2, 2, 2];
    // println!("{:?}", x2.align_to::<u64>());
    let ptr2 = x2.as_ptr() as *const __m128i;
    let vec2 = _mm_loadu_si128(ptr2);
    println!("{:?}", vec2);

    let mul = _mm_mullo_epi16(vec, vec2);
    //
    let dst = x.as_mut_ptr() as *mut __m128i;
    _mm_storeu_si128(dst, mul);
    //
    println!("{:?}", x);

    let aa: &mut [u8] = &mut [3; 16];
    let src = aa.as_ptr() as *const __m128i;
    let aa_vec = _mm_loadu_si128(src);

    let res = _mm_mullo_epi8(aa_vec, aa_vec);
    let out_ptr = aa.as_mut_ptr() as *mut __m128i;
    _mm_storeu_si128(out_ptr, res);

    println!("{:?}", aa);
}

fn main() {
    unsafe { simd_stuff() }
}