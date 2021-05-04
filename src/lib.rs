/*!
The __Utote__ crate provides statically allocated multisets of unsigned
integers.

A multiset, also known as a "bag", is an extension to the concept of set where
elements can occur multiple times. Utote calls the number of times an element
occurs its `count`, though the term `multiplicity` is more often used in
mathematical descriptions of multisets. All the common operations on sets have
implementations on multisets, so you can `intersect` and `union` multisets,
along with everything else you would expect to be able to do.

If you have a known number of things that you want to count and do set-like
operations on, then this crate will likely be useful to you.


Each multiset is an ordered collection of unsigned counters, where the index of each counter is the
element of the counter, and the value of that counter is the number of times that element occurs
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

# Useful For



# Cargo Features

- __simd__: Requires nightly rust toolchain. Enables SIMD implementations using
the [__packed_simd__ crate](https://docs.rs/packed_simd_2) crate and
unsatble features: [const_generics](https://github.com/rust-lang/rust/issues/44580)
and [const_evaluatable_checked](https://github.com/rust-lang/rust/issues/76560).
- __rand__: Enables [choose_random](`Multiset::choose_random`) methods for
multiset structs using the [__rand__ crate](https://docs.rs/rand).

# Performance

#### Wihtout SIMD

The largest factor in multiset performance without simd enabled is the size of
the multiset, the larger the slower it is. The simplest way to improve
performance is to keep them as small as possible.

#### Using SIMD

The most simple way to improve performance is to use the simd implementations
by turning on the `simd` feature of Utote. If you can use the nightly toolchain
then this should be utilised.

#### Multiset Sizes with SIMD

With simd enabled the adivce on multiset size changes. Since iteration is
significantly reduced as a factor in performance it can be beneficial to fit
the multisets to a multiple of the cache line size of the CPUs you expect to
run on (probably 64 bytes). Counter intuatively, this can mean gaining much
greater performance when increasing the size of the multiset. For example,
again with simd enabled, using a `Multiset<u16, 32>` may greatly improve
performance over using a `Multiset<u16, 4>`.
*/

// todo: Docs:
//  - Useful when you have, for example, the same 10 things to always store.
//  - Provides a particular flavour of multiset.

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
