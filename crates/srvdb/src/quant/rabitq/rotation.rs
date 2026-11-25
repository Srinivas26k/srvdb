//! Seeded random orthogonal rotation (ADR 0008)
//!
//! The rotation is regenerated from its seed on every open, never stored:
//! a Gaussian matrix from a fixed generator (SplitMix64 + Box-Muller),
//! orthonormalised row by row with modified Gram-Schmidt in f64.
//! [`Rotation::fingerprint`] is saved with the codes, so if a seed ever
//! expands to a different matrix (another platform's `ln`/`cos`, a code
//! change) the codes are detected as stale and rebuilt.

/// SplitMix64: tiny, well-known, stable across releases (unlike `rand`)
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in (0, 1]
    fn unit(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 1.0) / (1u64 << 53) as f64
    }

    /// Standard normal (Box-Muller, one value per call)
    fn gaussian(&mut self) -> f64 {
        let (u, v) = (self.unit(), self.unit());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

/// A `dim x dim` orthogonal matrix, row-major
#[derive(Debug, Clone)]
pub(crate) struct Rotation {
    dim: usize,
    rows: Vec<f32>,
}

impl Rotation {
    /// Build the rotation for `seed`. O(dim^3) time, dim^2 floats of memory.
    pub(crate) fn new(dim: usize, seed: u64) -> Self {
        let mut rng = SplitMix64(seed);
        let mut rows: Vec<Vec<f64>> = Vec::with_capacity(dim);
        while rows.len() < dim {
            let mut v: Vec<f64> = (0..dim).map(|_| rng.gaussian()).collect();
            // Two passes of modified Gram-Schmidt keep rows orthogonal to
            // f64 precision even for dimensions in the thousands
            for _ in 0..2 {
                for r in &rows {
                    let d: f64 = r.iter().zip(&v).map(|(a, b)| a * b).sum();
                    v.iter_mut().zip(r).for_each(|(x, a)| *x -= d * a);
                }
            }
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            // A draw (numerically) inside the span is discarded and redrawn
            if norm > 1e-6 {
                rows.push(v.into_iter().map(|x| x / norm).collect());
            }
        }
        Self {
            dim,
            rows: rows.into_iter().flatten().map(|x| x as f32).collect(),
        }
    }

    /// `R x`. Every query and insert pays this O(dim^2) product, so each
    /// row is summed in eight independent lanes the compiler vectorizes
    /// (a single running sum is one serial chain of dependent adds).
    pub(crate) fn apply(&self, x: &[f32]) -> Vec<f32> {
        self.rows
            .chunks_exact(self.dim.max(1))
            .map(|row| dot(row, x))
            .collect()
    }

    /// CRC32C of the matrix bytes
    pub(crate) fn fingerprint(&self) -> u32 {
        crc32c::crc32c(bytemuck::cast_slice(&self.rows))
    }
}

/// Dot product over the common length, in eight lanes
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut lanes = [0.0f32; 8];
    let (ha, ta) = a.as_chunks::<8>();
    let (hb, tb) = b.as_chunks::<8>();
    for (ca, cb) in ha.iter().zip(hb) {
        for i in 0..8 {
            lanes[i] += ca[i] * cb[i];
        }
    }
    let tail: f32 = ta.iter().zip(tb).map(|(x, y)| x * y).sum();
    lanes.iter().sum::<f32>() + tail
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_orthogonal_and_deterministic() {
        let dim = 48;
        let r = Rotation::new(dim, 7);
        for i in 0..dim {
            for j in 0..dim {
                let d: f32 = (0..dim)
                    .map(|t| r.rows[i * dim + t] * r.rows[j * dim + t])
                    .sum();
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((d - want).abs() < 1e-5, "({i},{j}) = {d}");
            }
        }
        assert_eq!(r.fingerprint(), Rotation::new(dim, 7).fingerprint());
        assert_ne!(r.fingerprint(), Rotation::new(dim, 8).fingerprint());
    }

    #[test]
    fn apply_matches_plain_product() {
        for dim in [1usize, 7, 8, 37, 64] {
            let r = Rotation::new(dim, 3);
            let x: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.37).sin()).collect();
            for (i, y) in r.apply(&x).iter().enumerate() {
                let want: f32 = (0..dim).map(|t| r.rows[i * dim + t] * x[t]).sum();
                assert!((y - want).abs() < 1e-5, "dim {dim} row {i}");
            }
        }
    }
}
