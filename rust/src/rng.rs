//! Seeded RNG for the draw loop.
//!
//! Statistical-equivalence target (plan §3/§5): we do NOT replicate numpy's
//! `SeedSequence` hash mixer or its 256-region ziggurat. Any good stream is
//! acceptable as long as it is *deterministic run-to-run* so shadow dual-runs
//! and CI are reproducible. We use PCG64 (XSL-RR) — the same family as numpy's
//! `default_rng` — seeded from the integer seed via `rand_pcg`'s own SeedableRng.
//!
//! The draw loop MUST stay serial on a single shared stream (plan §3): the
//! per-player consumption order is part of the deterministic contract.

use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use rand_pcg::Pcg64;

/// Deterministic standard-normal source over a single PCG64 stream.
pub struct NormalStream {
    rng: Pcg64,
}

impl NormalStream {
    /// Seed mirrors the Python `default_rng(seed)` integer (456/42/789). The
    /// resulting *stream* differs from numpy (different seeding + ziggurat); only
    /// determinism and distributional correctness are guaranteed.
    pub fn new(seed: u64) -> Self {
        // Pcg64 wants a 32-byte seed; widen the scalar seed deterministically.
        let mut state = [0u8; 32];
        state[..8].copy_from_slice(&seed.to_le_bytes());
        state[8..16].copy_from_slice(&seed.wrapping_mul(0x9E3779B97F4A7C15).to_le_bytes());
        state[16..24].copy_from_slice(&seed.rotate_left(17).to_le_bytes());
        state[24..].copy_from_slice(&(!seed).to_le_bytes());
        NormalStream {
            rng: Pcg64::from_seed(state),
        }
    }

    /// One standard-normal draw.
    #[inline]
    pub fn next_normal(&mut self) -> f64 {
        StandardNormal.sample(&mut self.rng)
    }

    /// Fill one simulation row of 4 category draws.
    #[inline]
    pub fn next_row4(&mut self) -> [f64; 4] {
        [
            self.next_normal(),
            self.next_normal(),
            self.next_normal(),
            self.next_normal(),
        ]
    }

    /// Uniform in [0, n) — for playoff/win tiebreaks (statistical; no Lemire port).
    #[inline]
    pub fn gen_index(&mut self, n: usize) -> usize {
        self.rng.gen_range(0..n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_same_seed() {
        let mut a = NormalStream::new(456);
        let mut b = NormalStream::new(456);
        for _ in 0..1000 {
            assert_eq!(a.next_normal(), b.next_normal());
        }
    }

    #[test]
    fn different_seeds_differ() {
        let mut a = NormalStream::new(42);
        let mut b = NormalStream::new(789);
        let mut same = 0;
        for _ in 0..100 {
            if a.next_normal() == b.next_normal() {
                same += 1;
            }
        }
        assert!(same < 5, "streams should diverge, matched {same}/100");
    }

    #[test]
    fn standard_normal_moments() {
        // Sanity: large sample mean ~0, std ~1.
        let mut s = NormalStream::new(123);
        let n = 200_000;
        let mut sum = 0.0;
        let mut sumsq = 0.0;
        for _ in 0..n {
            let x = s.next_normal();
            sum += x;
            sumsq += x * x;
        }
        let mean = sum / n as f64;
        let var = sumsq / n as f64 - mean * mean;
        assert!(mean.abs() < 0.02, "mean {mean}");
        assert!((var - 1.0).abs() < 0.03, "var {var}");
    }
}
