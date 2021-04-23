# Utote

[![docs.rs](https://docs.rs/utote/badge.svg)](https://docs.rs/utote)
[![Crates.io](https://img.shields.io/crates/v/utote.svg)](https://crates.io/crates/utote)
 
*** TODO: add note of minimum supported rust version, perhaps to changelog? ***

Stack allocated uint multiset implementation on rust stable, with optional SIMD implementations available using rust 
nightly.

The SIMD implementation uses [packed_simd](https://rust-lang.github.io/packed_simd/packed_simd_2) (behind a feature 
flag). `packed_simd` was chosen over alternatives due to it's simplicity and based on the assumption that when 
[std::simd](https://github.com/rust-lang/stdsimd) is stabilised it will look similar in API structure to `packed_simd` 
as it is now.

The implementations in `utote` are built using macros rather than generics because there is no generic interface 
available for the SIMD types available from `packed_simd`. Although there are crates available that enable generic 
implementation of SIMD code they either lack features in comparison to `packed_simd`, increase the complexity of the 
code, or are unstable themselves. The other benefit of using macros is that the actual implementation code is 
straightforward.

Since multisets are essentially collections of counters + some useful methods on those counters, and to keep things 
simple, implementations are only provided for `uint` types. The current Multiset is thus quite low level, perhaps 
better serving as a backend for a higher level multiset that works for any given type.

Please see the [docs](https://docs.rs/utote) for the API and more information!

### Future Development

- Build an implementation of Multiset, for scalar and SIMD types, which uses Vec.
- Build a Multiset implementation for any `T` which uses the `uint` implementations as a backend to handle the element 
  counters.
- Once `std::simd` is more fully implemented it should be possible to replace most of the macros with generic 
  implementations. And once this feature is stabilised it will be possible to enable all the SIMD implementations on 
  the stable toolchain. 

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.

## Acknowledgements

The implementations in this crate are inspired by [generic-array](https://docs.rs/generic-array/0.14.4/generic_array), 
[nalgebra](https://docs.rs/nalgebra) and [simba](https://docs.rs/simba).

# Changelog

### 0.4.1
- Minor performance improvements
- make `empty` & `repeat` constructors const

## 0.4.0 (Breaking)
- Deprecate direct SIMD implementation
- Utilise const generics (removing generic-array & typenum)

### 0.3.5
- fix choose_random implementations

### 0.3.4
- impl Send & Sync for Multiset
- re-export typenum

### 0.3.3
- impl FromIterator of refs for Multisets
- re-export simd types and generic-array traits

### 0.3.2
- Move to manual implementations of common traits on Multiset
- Manually define type aliases

### 0.3.1
- Make rng generic in `choose_random`

## 0.3.0
- Made `rand` dependency optional
- Switched from `StdRng` to `SmallRng`
