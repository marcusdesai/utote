//!
//! The __Utote__ crate provides statically allocated multisets of unsigned
//! integers. A nightly only `simd` feature can also be used to greatly improve
//! performance.
//!
//! A [multiset], also known as a "bag", is an extension to the concept of set
//! where elements can occur multiple times. Utote calls the number of times an
//! element occurs its `count`, though the term `multiplicity` is more often
//! used in mathematical descriptions of multisets. All the common operations
//! on sets have implementations on multisets, so you can `intersect` and
//! `union` multisets, along with everything else you would expect to be able
//! to do.
//!
//! [multiset]: https://en.wikipedia.org/wiki/Multiset
//!
//! If you have a known number of things that you want to count and do set-like
//! operations on, then this crate will likely be useful to you.
//!
//! # Recipes
//!
//! `Multiset` can only keep track of uint counters, but if you have a fixed
//! number of non-uint items then pairing a `HashMap` with a `Multiset` can
//! provide significant performance benefits.
//!
//! ```
//! use std::collections::HashMap;
//! use utote::Multiset;
//!
//! let mut item_map: HashMap<&str, usize> = HashMap::new();
//! item_map.insert("foo", 0);
//! item_map.insert("bar", 1);
//!
//! let multiset: Multiset<u16, 2> = Multiset::empty();
//!
//! // do a whole load of stuff to the multiset...
//!
//! let bar_count: &u16 = multiset.get(*item_map.get("bar").unwrap()).unwrap();
//! ```
//!
//! # SIMD and Generics
//!
//! Due to remaining rough edges in the const generic feature there is an extra
//! constraint required when using `Multiset` generically. The const values for
//! the number of lanes for the simd types of a counter type need to be
//! constrained. The constraint:
//! > `where [(); T::L128 * T::L256 * T::LF]: Sized`
//!
//! is the most simple to add.
//!
//!
//! # Cargo Features
//!
//! - __simd__: Requires nightly rust toolchain. Enables simd implementations
//! using the [__packed_simd__ crate](https://docs.rs/packed_simd_2) crate and
//! unsatble features: [const_generics](https://github.com/rust-lang/rust/issues/44580)
//! and [const_evaluatable_checked](https://github.com/rust-lang/rust/issues/76560).
//! - __rand__: Enables [`choose_random`](Multiset::choose_random) methods for
//! multiset structs using the [__rand__ crate](https://docs.rs/rand).
//!
//! # Performance
//!
//! The largest factor in multiset performance is the size of the multiset, the
//! larger the slower it is. The simplest way to improve performance is to keep
//! them as small as possible.
//!
//! #### Using SIMD
//!
//! The most simple way to improve performance is to use the simd
//! implementations by turning on the `simd` feature of Utote. If you can use
//! the nightly toolchain then this should be utilised.

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
