# Utote

Stack allocated uint multiset, with optional SIMD implementations.

> :warning: This library is **not** stable: expect API changes with each release.

Segregated use of `packed_simd_2` behind feature flags, requires nightly. The scalar multiset implementation is usable on stable.

Inspired by nalgebra and simba.

Why macros?

Why not simba?

Why packed_simd?

The compiler is very good at auto-vectorising in micro-benchmarks, but use explicit simd along with compiler flags to ensure that vectorised code is being emitted. 

### Basic Example

```rust
use utote::Multiset;
```

### Future

- Utilise `const generics` when fully stable.
- Use `std::simd` when that is also stable.
- Use simba if features expand to capture everything available in `packed_simd` or `std::simd`.

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.

##### P.S.
May rename to Asimdtote.
