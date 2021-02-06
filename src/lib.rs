/*!
The __Utote__ crate provides a statically allocated implementation of a multiset of unsigned
integers. The implementation utilises type level unsigned integers from the [__typenum__ crate](https://docs.rs/typenum/1.12.0/typenum/index.html),
along with the [__generic-array__ crate](https://docs.rs/generic-array/0.14.4/generic_array/index.html)
to enable generically sized static multisets.

Each multiset is an ordered collection of unsigned counters, where the index of each counter is the
element of the counter, and the value of that counter is the number of times that element occurrs
in the multiset. A count of zero at index `i` indicates that element `i` is not in the multiset.
Any multiset where all counters are zero is equivalent to the empty multiset.

## Examples

```
use utote::MSu8;
use typenum::U4;

// A multiset of 4 elements, which can be counted up to u8::MAX
let multiset = MSu8::<U4>::from_slice(&[1, 3, 5, 7]);

assert_eq!(multiset.total(), 16);
assert_eq!(multiset.get(1), Some(3));
```

# SIMD Features

Optionally the [__packed_simd__ crate](https://rust-lang.github.io/packed_simd/packed_simd_2/index.html)
can be enabled (requires nightly) to allow for multisets which are built either from an array of
SIMD vectors, or directly from a single SIMD vector.

```toml
[dependencies]
utote = { version = ..., features = ["packed_simd"] }
```

Although the compiler is is able to auto-vectorise code, these capabilities are provided so that
you can explicitly direct the compiler to use SIMD vectors, if they are available.

## Examples

```
use utote::MSu8x2;
use typenum::U2;

let multiset = MSu8x2::<U2>::from_slice(&[1, 3, 5, 7]);

assert_eq!(multiset.total(), 16);
assert_eq!(multiset.get(1), Some(3));
```

In addition to multisets which consist of arrays of SIMD values, utote also implements multisets
which use a SIMD vector directly for the multiset data. Inline with the above reasoning for SIMD
use, the purpose of these types is so that, in the case where an array of SIMD values is not
needed, the user can ensure that no code used in managing the arrays is used in the program.

```
use utote::{MSu16x4, MS0u16x4};
use typenum::U0;

// This type alias is equivalent to MS0u16x4
type MSDirectSimd = MSu16x4<U0>;

let multiset = MS0u16x4::from_slice(&[1, 3, 5, 7]);

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
