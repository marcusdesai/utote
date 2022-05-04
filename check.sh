#!/usr/bin/env sh

# Test simd and not simd feature separately as we provide alternate
# implementations in each.
cargo check --all-targets --features="rand serde"
cargo check --all-targets --features="rand serde simd"
cargo test --all-targets --features="rand serde"
cargo test --doc --features="rand serde"
cargo test --all-targets --features="rand serde simd"
