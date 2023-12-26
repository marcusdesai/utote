//!
//! The __Utote__ crate provides statically allocated multisets of unsigned
//! integers. A [multiset], also known as a "bag", is an extension to the
//! concept of set where elements can occur multiple times. Utote calls the
//! number of times an element occurs its `count`, though the term
//! `multiplicity` is more often used in pure math contexts. All the common
//! operations on sets have implementations on multisets, so you can `intersect`
//! and `union` multisets, along with everything else you would expect to be
//! able to do.
//!
//! If you have a known number of things that you want to count and do set-like
//! operations on, then this crate will likely be useful to you.
//!
//! Most methods have simple implementations which should
//! autovectorise well. A nightly only `simd` feature can also be used to
//! improve performance for methods that don't autovectorise. The following
//! methods currently have implementations when using the `simd` feature:
//! - [Multiset::is_disjoint]
//! - [Multiset::is_subset]
//! - [Multiset::is_superset]
//! - [Multiset::is_proper_subset]
//! - [Multiset::is_proper_superset]
//! - [Multiset::total]
//! - [Multiset::collision_entropy]
//! - [Multiset::partial_cmp] for impl of PartialOrd
//!
//! [multiset]: https://en.wikipedia.org/wiki/Multiset
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
//! let multiset: Multiset<u16, 2> = Multiset::new();
//!
//! // do a whole load of stuff to the multiset...
//!
//! let bar_count: &u16 = multiset.get(*item_map.get("bar").unwrap()).unwrap();
//! ```
//!
//! # Cargo Features
//!
//! - __simd__: Requires nightly rust toolchain. Enables simd implementations
//! using [std::simd](https://github.com/rust-lang/stdsimd).
//! - __rand__: Enables [`choose_random`](Multiset::choose_weighted) methods for
//! multiset structs using the [__rand__ crate](https://docs.rs/rand).
//!
//! # Performance
//!
//! The largest factor in multiset performance is the size of the multiset, the
//! larger the slower it is. The simplest way to improve performance is to keep
//! them as small as possible.
//!
//! ## Using SIMD
//!
//! The most simple way to improve performance is to use the simd
//! implementations by turning on the `simd` feature of Utote. If you can use
//! the nightly toolchain then this should be utilised.

#![no_std]
#![cfg_attr(
    feature = "simd",
    feature(portable_simd),
    allow(incomplete_features)
)]

mod counter;
#[cfg(feature = "serde")]
mod serde;
mod simd_iter;

pub use counter::Counter;
pub use simd_iter::Multiset;
