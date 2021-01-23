use packed_simd::{i32x4};


fn main() {
    let a = i32x4::new(1, 2, 3, 4);
    let b = i32x4::new(5, 6, 7, 8);
    assert_eq!(a + b, i32x4::new(6, 8, 10, 12));
}