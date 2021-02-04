# Utote

[![docs.rs](https://docs.rs/utote/badge.svg)](https://docs.rs/utote)
[![Crates.io](https://img.shields.io/crates/v/utote.svg)](https://crates.io/crates/utote)

Stack allocated uint multiset implementation, with optional SIMD implementations available.

> :warning: This crate is **not** stable: expect API changes with each release.

The simd implementation uses [packed_simd](https://rust-lang.github.io/packed_simd/packed_simd_2) (behind a feature 
flag). `packed_simd` was chosen over alternatives due to it's simplicity and base on the assumption that when 
[std::simd](https://github.com/rust-lang/stdsimd) is stabilised it will look similar in implementation to `packed_simd` 
as it is now.

The implementations in `utote` are built using macros rather than generics because there is no generic interface 
available for the SIMD types in `packed_simd`. Although there are crates available that enable generic implementation 
of SIMD code they either lack features in comparison to `packed_simd`, increase the complexity of the implementation, 
or are unstable themselves. Other than enabling the implementation, the benefit of using macros is that the actual 
implementation code is straightforward. 

Please see the [docs](https://docs.rs/utote) for the API and more information!

### Future Development

Once `const generics` and `std::simd` are more fully implemented it should be possible to replace most of the macros 
with generic implementations. And once these features are stabilised it will be possible to enable all the SIMD 
implementations in `utote` in the stable toolchain. 

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

The implementations in this crate are inspired by [nalgebra](https://docs.rs/nalgebra) and [simba](https://docs.rs/simba).
