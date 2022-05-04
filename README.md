# Utote

[![docs.rs](https://docs.rs/utote/badge.svg)](https://docs.rs/utote)
[![Crates.io](https://img.shields.io/crates/v/utote.svg)](https://crates.io/crates/utote)

High performance, stack allocated uint multiset implementation on rust stable, 
with optional simd implementations available using rust nightly.

**minimum supported rust version: 1.51**

[Docs](https://docs.rs/utote)

## Examples

```rust
use utote::Multiset;

fn main() {
    // A multiset of 5 elements, which can count up to u8::MAX
    let mut multiset = Multiset::from([0u8, 3, 4, 0, 5]);
    assert_eq!(multiset.total(), 12);
    
    let equivalent_multiset = Multiset::<u8, 5>::from([0, 3, 4, 0, 5]);
    assert_eq!(multiset, equivalent_multiset);
    
    multiset.insert(2, 6);
    assert_eq!(multiset, Multiset::from([0, 3, 6, 0, 5]));
    
    for elem in multiset.iter() {
      println!("{}", elem);
    }
    
    assert_eq!(multiset.contains(0), false);
    assert_eq!(multiset.contains(1), true);
}
```

Some common set-like operations:

```rust
use utote::Multiset;

fn main() {
    let ms_sub: Multiset<u32, 3> = Multiset::from([0, 1, 1]);
    let ms_super = Multiset::from([1, 1, 2]);
  
    assert_eq!(ms_sub.is_subset(&ms_super), true);
  
    assert_eq!(ms_sub.union(&ms_super), Multiset::from([1, 1, 2]));
  
    assert_eq!(ms_super.is_proper_superset(&ms_sub), true);
  
    // Any multiset where all counts are zero is equivalent to
    // the empty multiset.
    let all_zero: Multiset<u64, 2> = Multiset::from([0, 0]);
    assert_eq!(all_zero, Multiset::new());
}
```

### Implementation Notes

#### SIMD Feature
list of functions with manual simd implementations

The Utote Multiset has a single generic API but multiple equivalent scalar and 
simd implementations of various functions where the use of simd can enhance 
performance. The simd functionality is **nightly** only, while the scalar 
versions can be used on stable.

The nightly only simd implementation uses [packed_simd] and the unstable 
features: [const_generics] and [const_evaluatable_checked] (all behind the 
feature flag `"simd"`). `packed_simd` was chosen over alternatives due to its 
simplicity and based on the assumption that when [std::simd] is stabilised it 
will look similar in API structure to `packed_simd` as it is now.

Once const generics and portable simd support hit stable this crate will also 
become fully stable. Until these features are stabilised the version of Utote 
will stay below `1.0.0`.

#### Migration from 0.6.0 to 0.7.0

* empty -> new

Since multisets are essentially collections of counters + some useful methods 
on those counters, and to keep things simple, implementations are only provided 
for `uint` types. The current Multiset is thus quite low level, perhaps better 
serving as a backend for a higher level multiset that works for any given type.

Although it would be simple to implement `Deref<[T]>` for Multiset I decided 
against this for two reasons. Firstly to avoid suggestively exposing methods in 
the API for Multiset which could sort the counts. Since the order of the counts 
is intrinsic to the implementation working I wanted to avoid any confusion that 
this would be appropriate. Second, as most of the functional methods for 
Multiset will eventually be implemented on slice and then used from different 
multiset varieties implmenting deref to slice could cause confusion in the 
code.

[packed_simd]: https://docs.rs/packed_simd_2
[const_generics]: https://github.com/rust-lang/rust/issues/44580
[const_evaluatable_checked]: https://github.com/rust-lang/rust/issues/76560
[std::simd]: https://github.com/rust-lang/stdsimd

### Future Development

- Provide a heap allocated MultisetVec type which uses a Vec for storage rather 
  than an array.

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall 
be dual licensed as above, without any additional terms or conditions.

## Acknowledgements

The implementations in this crate are inspired by [generic-array](https://docs.rs/generic-array), 
[nalgebra](https://docs.rs/nalgebra) and [simba](https://docs.rs/simba).

# Changelog

## 0.7.0 (Breaking)

## 0.6.0 (Breaking)
- API changes
  - Rename `Multiset::argmax` => `Multiset::elem_count_max`
  - Rename `Multiset::argmin` => `Multiset::elem_count_min`
  - Rename `Multiset::imax` => `Multiset::elem_max`
  - Rename `Multiset::imin` => `Multiset::elem_min`
  - Rename `Multiset::max` => `Multiset::count_max`
  - Rename `Multiset::min` => `Multiset::count_min`
- Cleanup & expand documentation
- Ensure `PartialOrd` impl uses most efficient method
- Add `From` mut `Multiset` ref
- Fix simd impls of `is_any_lesser` & `is_any_greater`
- Remove unnecessary `SmallRng` uses

## 0.5.0 (Breaking)
- Provide uniform generic interface
- Re-implement scalar and simd backends
- Remove all type aliases
- Remove all simd types / considerations from the API 
- Remove some `const` constructors to enable stable generic interface
- Improve documentation
- Add `Rem` ops
- Add broadcast arithmetic ops
- Add `From` implementations
- Complete `FromIterator` and `IntoIterator` impl coverage
- Add `Index` and `IndexMut` implementations
- Simplify multiple functions
- Add functions: 
  - `difference`
  - `symmetric_difference`
  - `from_elements`
  - `is_disjoint`
  - `get_mut`
  - `get_unchecked_mut`
- Add dynamic dispatch on detected cpu features for simd backends, currently 
  supporting:
  - `AVX2`
  - `AVX`
  - `SSE4.2`

### 0.4.1
- Minor performance improvements
- make `empty` & `repeat` constructors const

## 0.4.0 (Breaking)
- Minimum rust version: 1.51
- Deprecate direct simd implementation
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
