pub mod multiset;
#[macro_use]
#[allow(unused_macros)]
mod tests;
mod array;
mod scalar_array;
#[cfg(feature = "packed_simd")]
mod simd;
#[cfg(feature = "packed_simd")]
mod simd_array;
mod small_num;
