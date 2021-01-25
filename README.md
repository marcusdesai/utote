# Utote

Stack allocated uint multiset, with optional SIMD implementations.

Currently nightly-only, but will segregate use of `packed_simd_2` behind feature flags to enable use of the scalar multiset implementation on stable.

### Future

- Utilise `const generics` when fully stable.
- Use `std::simd` when that is also stable.

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
