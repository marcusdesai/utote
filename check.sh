#!/usr/bin/env sh

# Test simd and not simd feature separately as we provide alternate
# implementations in each.
cargo check --all-targets --features="rand"
cargo check --all-targets --features="rand simd"
cargo test --all-targets --features="rand"
cargo test --all-targets --features="rand simd"
