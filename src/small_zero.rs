use packed_simd::{u8x2, u8x4, u8x8, u8x16, u16x2, u16x4, u16x8, u16x16, u32x2, u32x4,
                  u32x8, u32x16, f32x2, f32x4, f32x8, f32x16, f64x2, f64x4, f64x8};


pub(crate) trait SmallZero {
    const ZERO: Self;
}


macro_rules! impl_small_zero {
    ($v:expr, $($t:ty),*) => {
        $(impl SmallZero for $t {
            const ZERO: $t = $v;
        })*
    };
}


macro_rules! impl_simd_small_zero {
    ($v:expr, $($t:ty),*) => {
        $(impl SmallZero for $t {
            const ZERO: $t = <$t>::splat($v);
        })*
    };
}


impl_small_zero!(0, u8, u16, u32, u64);
impl_small_zero!(0.0, f32, f64);

impl_simd_small_zero!(0, u8x2, u8x4, u8x8, u8x16, u16x2, u16x4, u16x8, u16x16, u32x2, u32x4, u32x8, u32x16);
impl_simd_small_zero!(0.0, f32x2, f32x4, f32x8, f32x16, f64x2, f64x4, f64x8);
