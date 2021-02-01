/// Defines basic constants which can be defined for small values.
pub(crate) trait SmallNumConsts {
    const ZERO: Self;
    const ONE: Self;
    const MAX: Self;
}

macro_rules! impl_small_num {
    ($zero:expr, $one:expr, $max:expr, $t:ty) => {
        impl SmallNumConsts for $t {
            const ZERO: $t = $zero;
            const ONE: $t = $one;
            const MAX: $t = $max;
        }
    };
}

impl_small_num!(0, 1, u8::MAX, u8);
impl_small_num!(0, 1, u16::MAX, u16);
impl_small_num!(0, 1, u32::MAX, u32);
impl_small_num!(0, 1, u64::MAX, u64);
impl_small_num!(0.0, 1.0, f32::MAX, f32);
impl_small_num!(0.0, 1.0, f64::MAX, f64);
