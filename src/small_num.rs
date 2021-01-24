use packed_simd::{u8x2, u8x4, u8x8, u8x16, u16x2, u16x4, u16x8, u16x16, u32x2, u32x4,
                  u32x8, u32x16, f32x2, f32x4, f32x8, f32x16, f64x2, f64x4, f64x8, SimdVector};


pub(crate) trait SmallNum {
    const ZERO: Self;
    const ONE: Self;
    const MAX: Self;
}


macro_rules! impl_small_num {
    ($zero:expr, $one:expr, $max:expr, $t:ty) => {
        impl SmallNum for $t {
            const ZERO: $t = $zero;
            const ONE: $t = $one;
            const MAX: $t = $max;
        }
    };
}


macro_rules! impl_simd_small_num {
    ($($t:ty),*) => {
        $(impl SmallNum for $t
            where
                <$t as SimdVector>::Element: SmallNum
        {
            const ZERO: $t = <$t>::splat(<$t as SimdVector>::Element::ZERO);
            const ONE: $t = <$t>::splat(<$t as SimdVector>::Element::ONE);
            const MAX: $t = <$t>::splat(<$t as SimdVector>::Element::MAX);
        })*
    };
}


impl_small_num!(0, 1, u8::MAX, u8);
impl_small_num!(0, 1, u16::MAX, u16);
impl_small_num!(0, 1, u32::MAX, u32);
impl_small_num!(0, 1, u64::MAX, u64);
impl_small_num!(0.0, 1.0, f32::MAX, f32);
impl_small_num!(0.0, 1.0, f64::MAX, f64);


impl_simd_small_num!(
u8x2, u8x4, u8x8, u8x16, u16x2, u16x4, u16x8, u16x16, u32x2, u32x4, u32x8, u32x16,
f32x2, f32x4, f32x8, f32x16, f64x2, f64x4, f64x8
);
