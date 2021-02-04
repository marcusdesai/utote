/*!
The __Utote__ crate provides a statically allocated implementation of a multiset of unsigned
integers. The implementation utilises type level unsigned integers from the [__typenum__ crate](https://docs.rs/typenum/1.12.0/typenum/index.html),
along with the [__generic-array__ crate](https://docs.rs/generic-array/0.14.4/generic_array/index.html)
to enable generically sized static multisets.

# Examples

```
use utote::MSu8;
use typenum::U4;

let multiset = MSu8::<U4>::from_slice(&[1, 3, 5, 7]);

assert_eq!(multiset.total(), 16);
assert_eq!(multiset.get(1), Some(3));

```

Optionally the [__packed_simd__ crate](https://rust-lang.github.io/packed_simd/packed_simd_2/index.html)
can be enabled (requires nightly) to allow for multisets which are built either from an array of
SIMD vectors, or directly from a single SIMD vector. Although the compiler is very good at
auto-vectorising code, these capabilities are provided so that you can explicitly direct the
compiler to use SIMD vectors, if they are available.

# Examples

```
use utote::MSu8x2;
use typenum::U2;

let multiset = MSu8x2::<U2>::from_slice(&[1, 3, 5, 7]);

assert_eq!(multiset.total(), 16);
assert_eq!(multiset.get(1), Some(3));

```

*/

mod multiset;
pub use multiset::*;

#[macro_use]
#[allow(unused_macros)]
mod tests;
#[macro_use]
mod common;
mod small_num;

mod scalar;
pub use scalar::*;

#[cfg(feature = "packed_simd")]
mod simd;
#[cfg(feature = "packed_simd")]
pub use simd::*;
