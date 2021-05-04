/*!
The __Utote__ crate provides a statically allocated implementation of a multiset of unsigned
integers.

Each multiset is an ordered collection of unsigned counters, where the index of each counter is the
element of the counter, and the value of that counter is the number of times that element occurrs
in the multiset. A count of zero at index `i` indicates that element `i` is not in the multiset.
Any multiset where all counters are zero is equivalent to the empty multiset.

## Examples

```
use utote::Multiset;

// A multiset of 4 elements, which can be counted up to u8::MAX
let multiset: Multiset<u8, 4> = Multiset::from([1, 3, 5, 7]);

assert_eq!(multiset.total(), 16);
assert_eq!(multiset.get(1), Some(&3));
```

# Cargo Features

- __simd__: Requires nightly rust toolchain. Enables SIMD implementations using the
`packed_simd` crate and unsatble features `const_generics` and `const_evaluatable_checked`.
- __rand__: Enables `choose_random` methods for multiset structs using the `rand` crate.

# Using SIMD

Optionally the [__packed_simd__ crate](https://rust-lang.github.io/packed_simd/packed_simd_2/index.html)
can be enabled (requires nightly) to allow for multisets which are built either from an array of
SIMD vectors, or directly from a single SIMD vector.

```toml
[dependencies]
utote = { version = ..., features = ["simd"] }
```
*/

// todo: Docs:
//  - Useful when you have, for example, the same 10 things to always store.
//  - Provides a particular flavour of multiset.
//  - simd feature turns on use of packed_simd, const_generics and const_evaluatable_checked.
//  - Will stay below 1.0.0 until simd & const_generics stabilized.

#![cfg_attr(
    feature = "simd",
    feature(const_generics, const_evaluatable_checked),
    allow(incomplete_features)
)]

mod multiset;
pub use multiset::*;
#[cfg(feature = "simd")]
mod chunks;
#[cfg(feature = "simd")]
mod simd;
