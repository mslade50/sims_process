//! Optional/debug-only standalone CLI (plan §4): NOT a permanent fail-open
//! fallback. Currently just runs the op self-checks so the rlib links outside
//! the Python extension and can be exercised without a wheel install.

use sims_kernel::{ops, rng};

fn main() {
    assert_eq!(ops::rint_i64(2.5), 2);
    assert_eq!(ops::rint_i64(3.5), 4);
    assert_eq!(ops::min_rank(&[71, 68, 70, 70]), vec![4, 1, 2, 2]);
    let mut a = rng::NormalStream::new(456);
    let mut b = rng::NormalStream::new(456);
    assert_eq!(a.next_normal(), b.next_normal());
    println!("sims_kernel {} — selftest OK", env!("CARGO_PKG_VERSION"));
}
