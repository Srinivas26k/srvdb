//! Query tables quantized to u8
//!
//! Each f32 nibble table `T_m` is shifted by its minimum and divided by
//! one step shared by all tables, `step = max_m(max T_m - min T_m) / 255`,
//! then rounded. The dot product is recovered as
//! `step * sum + sum_m min T_m`; rounding moves each entry by at most
//! `step / 2`, so the dot product is off by at most `M * step / 2` for
//! `M` tables ([`QueryLut::max_error`]). Rerank absorbs this error.

/// u8 tables for one query, 16 bytes per nibble position
pub(crate) struct QueryLut {
    bytes: Vec<u8>,
    step: f32,
    bias: f32,
}

impl QueryLut {
    /// Quantize f32 nibble tables (low nibble then high nibble per byte)
    pub(crate) fn new(tables: &[[f32; 16]]) -> Self {
        let mins: Vec<f32> = tables
            .iter()
            .map(|t| t.iter().fold(f32::INFINITY, |m, v| m.min(*v)))
            .collect();
        let range = tables
            .iter()
            .zip(&mins)
            .map(|(t, min)| t.iter().fold(0.0f32, |m, v| m.max(v - min)))
            .fold(0.0f32, f32::max);
        let step = if range.is_finite() && range > 0.0 {
            range / 255.0
        } else {
            0.0
        };
        let mut bytes = Vec::with_capacity(tables.len() * 16);
        for (t, min) in tables.iter().zip(&mins) {
            for v in t {
                let q = if step > 0.0 {
                    ((v - min) / step).round()
                } else {
                    0.0
                };
                bytes.push(q.clamp(0.0, 255.0) as u8);
            }
        }
        let bias = if step > 0.0 { mins.iter().sum() } else { 0.0 };
        Self { bytes, step, bias }
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Dot product estimate from an integer sum
    #[inline]
    pub(crate) fn estimate(&self, sum: u32) -> f32 {
        self.step * sum as f32 + self.bias
    }

    /// Bound on `|estimate - f32 dot product|` from u8 rounding; the
    /// `1.001` margin for f32 rounding is a heuristic (ADR 0017)
    pub(crate) fn max_error(&self) -> f32 {
        (self.bytes.len() / 16) as f32 * self.step / 2.0 * 1.001
    }
}
