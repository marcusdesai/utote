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

# Cargo Features

- __packed_simd__: Requires nightly rust toolchain. Enables SIMD implementations using the
`packed_simd` crate.
- __rand__: Enables `choose_random` methods for multiset structs using the `rand` crate.

# Using SIMD

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

// Re-Exports
pub use generic_array;
pub use typenum;
#[cfg(feature = "packed_simd")]
pub use packed_simd;

#[cfg(test)]
mod generic_tests {
    use super::*;
    use typenum::Unsigned;
    use packed_simd::u8x2;
    use generic_array::ArrayLength;

    fn generic<U: Unsigned + MultisetStorage<u8x2>>(x: MSu8x2<U>)
        where
            U: ArrayLength<u8x2>,
            U::ArrayType: Copy,
    {
        x.intersection(&x);
    }

    #[test]
    fn test_generic() {
        generic(MSu8x2::<typenum::U4>::repeat(2))
    }
}
