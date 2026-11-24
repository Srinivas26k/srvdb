//! Estimator accuracy of the RaBitQ quantizer

use super::*;

fn vectors(n: usize, dim: usize, mut seed: u64) -> Vec<Vec<f32>> {
    (0..n)
        .map(|_| {
            (0..dim)
                .map(|_| {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((seed >> 33) as f32 / (1u64 << 30) as f32) - 1.0
                })
                .collect()
        })
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

/// (mean error, mean absolute error) of the estimate over pairs that are
/// correlated (query = data + noise), so true cosines are spread out
fn errors(bits: u8, dim: usize) -> (f32, f32) {
    let q = RabitqQuantizer::new(dim, bits, 3);
    let data = vectors(200, dim, 1);
    let noise = vectors(200, dim, 2);
    let (mut sum, mut abs) = (0.0, 0.0);
    for (i, x) in data.iter().enumerate() {
        let mut record = Vec::new();
        q.encode_into(x, &mut record);
        assert_eq!(record.len(), q.record_size());
        let query: Vec<f32> = x
            .iter()
            .zip(&noise[i])
            .map(|(a, b)| a + b * (i % 4) as f32 * 0.5)
            .collect();
        let err = q.prepare(&query).score(&record) - cosine(x, &query);
        sum += err;
        abs += err.abs();
    }
    (sum / 200.0, abs / 200.0)
}

#[test]
fn estimates_are_close_and_unbiased() {
    let dim = 64;
    let mut last = f32::MAX;
    for bits in SUPPORTED_BITS {
        let (mean, mae) = errors(bits, dim);
        assert!(mean.abs() < 0.03, "{bits} bits: bias {mean}");
        assert!(mae < last, "{bits} bits: error {mae} not below {last}");
        last = mae;
    }
    assert!(last < 0.02, "4-bit mean abs error {last}");
}

#[test]
fn self_similarity_is_one_and_odd_dimensions_pack() {
    for bits in SUPPORTED_BITS {
        let dim = 37;
        let q = RabitqQuantizer::new(dim, bits, 9);
        assert_eq!(q.code_bytes(), dim.div_ceil(8 / bits as usize));
        let x = &vectors(1, dim, 5)[0];
        let mut record = Vec::new();
        q.encode_into(x, &mut record);
        let s = q.prepare(x).score(&record);
        assert!((s - 1.0).abs() < 1e-4, "{bits} bits: {s}");
    }
}

#[test]
fn zero_vector_scores_zero() {
    let q = RabitqQuantizer::new(16, 2, 1);
    let mut record = Vec::new();
    q.encode_into(&[0.0; 16], &mut record);
    assert_eq!(q.prepare(&[1.0; 16]).score(&record), 0.0);
}

/// `<o_bar, o>`, the cosine between a vector and its grid point, matches
/// the values reported for RaBitQ / extended RaBitQ on rotated data
/// (1 bit: sqrt(2/pi) = 0.798; more bits approach 1)
#[test]
fn grid_alignment_matches_theory() {
    let dim = 128;
    for (bits, floor) in [(1u8, 0.79), (2, 0.93), (4, 0.99)] {
        let q = RabitqQuantizer::new(dim, bits, 3);
        let per = 8 / bits as usize;
        let mask = (1u8 << bits) - 1;
        let mut total = 0.0;
        for x in vectors(100, dim, 4) {
            let mut r = Vec::new();
            q.encode_into(&x, &mut r);
            let gy = 1.0 / f32::from_le_bytes([r[0], r[1], r[2], r[3]]);
            let gg: f32 = (0..dim)
                .map(|i| {
                    let c = (r[FACTOR_BYTES + i / per] >> ((i % per) * bits as usize)) & mask;
                    (c as f32 + q.offset()).powi(2)
                })
                .sum();
            total += gy / gg.sqrt();
        }
        let mean = total / 100.0;
        assert!(mean >= floor && mean <= 1.0, "{bits} bits: {mean}");
    }
}
