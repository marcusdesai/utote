use packed_simd::*;

use crate::small_num::SmallNumConsts;

macro_rules! impl_simd_small_num {
    ($($t:ty),*) => {
        $(impl SmallNumConsts for $t
            where
                <$t as SimdVector>::Element: SmallNumConsts
        {
            const ZERO: $t = <$t>::splat(<$t as SimdVector>::Element::ZERO);
            const ONE: $t = <$t>::splat(<$t as SimdVector>::Element::ONE);
            const MAX: $t = <$t>::splat(<$t as SimdVector>::Element::MAX);
        })*
    };
}

impl_simd_small_num!(
    u8x2, u8x4, u8x8, u8x16, u8x32, u8x64, u16x2, u16x4, u16x8, u16x16, u16x32, u32x2, u32x4,
    u32x8, u32x16, u64x2, u64x4, u64x8, f32x2, f32x4, f32x8, f32x16, f64x2, f64x4, f64x8
);
