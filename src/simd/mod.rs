#[cfg(feature = "packed_simd")]
mod simd_array;
#[cfg(feature = "packed_simd")]
mod simd_native;
#[cfg(feature = "packed_simd")]
mod small_num_impl;

pub use simd_array::*;
pub use simd_native::*;
