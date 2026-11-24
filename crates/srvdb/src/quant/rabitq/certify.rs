//! Error intervals and certification of RaBitQ estimates (ADR 0017)
//!
//! For a unit data vector `o` with code direction `o_bar = g / |g|` and a
//! unit query `q`, the RaBitQ estimate `<o_bar, q> / <o_bar, o>` differs
//! from `<o, q>` by
//! `sqrt(1 - <o, q>^2) * sqrt(1 / <o_bar, o>^2 - 1) * Z`, where `Z` is one
//! coordinate of a uniformly random unit vector in `D - 1` dimensions
//! (arXiv:2405.12497, Theorem 3.2), so `P(|Z| > eps / sqrt(D - 1)) <=
//! 2 exp(-eps^2 / 2)`. The u8 query tables add at most
//! `|f| * QueryLut::max_error()` on top (ADR 0012).

/// Nominal lower bound on the probability, over a hypothetical uniformly
/// random rotation, that a certified search returns exactly the top k of
/// the vectors it scanned. Approximate (pseudo-random f32 rotation,
/// heuristic f32 slack); not a per-query guarantee, not a long-run rate
/// for a fixed seed, not joint across queries (ADR 0017).
pub const CERTIFIED_CONFIDENCE: f64 = 0.99;

/// Absolute slack for f32 rounding in the rotation, tables, factor and
/// `|g|`. A heuristic, not a derived bound (ADR 0017).
const SLOP: f32 = 1e-3;

/// `sqrt(1 / <o_bar, o>^2 - 1) / sqrt(D - 1)` for one record: the error
/// width per unit of `eps`. `<o_bar, o> = <g, y> / |g| = 1 / (f |g|)`, and
/// `|g|^2` is summed over the real dimensions of the packed codes.
pub(crate) fn unit_width(factor: f32, codes: &[u8], dim: usize, bits: u8) -> f32 {
    let per = 8 / bits as usize;
    let mask = (1u16 << bits) - 1;
    let offset = 0.5 - (1u32 << (bits - 1)) as f32;
    let norm_sq: f32 = (0..dim)
        .map(|i| {
            let c = (codes[i / per] as u16 >> ((i % per) * bits as usize)) & mask;
            let g = c as f32 + offset;
            g * g
        })
        .sum();
    let inv_cos_sq = factor * factor * norm_sq;
    ((inv_cos_sq - 1.0).max(0.0) / (dim.max(2) - 1) as f32).sqrt()
}

/// `eps` such that a union bound over `n` intervals fails with
/// probability at most `1 - CERTIFIED_CONFIDENCE`:
/// `n * 2 exp(-eps^2 / 2) = 1 - CERTIFIED_CONFIDENCE`
pub(crate) fn epsilon(n: usize) -> f32 {
    let n = n.max(1) as f64;
    (2.0 * (2.0 * n / (1.0 - CERTIFIED_CONFIDENCE)).ln()).sqrt() as f32
}

/// Interval rule for one query
#[derive(Debug, Clone, Copy)]
pub(crate) struct Bounds {
    pub(crate) eps: f32,
    /// `QueryLut::max_error()` of the query
    pub(crate) lut_error: f32,
}

impl Bounds {
    /// Half-width of the interval around estimate `est` of a vector with
    /// correction factor `factor` and unit width `width`. The
    /// `sqrt(1 - <o, q>^2)` term is bounded with the smallest `|<o, q>|`
    /// the same interval allows, so it holds on the same event.
    pub(crate) fn half_width(&self, est: f32, factor: f32, width: f32) -> f32 {
        let lut = factor.abs() * self.lut_error + SLOP;
        let e0 = self.eps * width;
        let s = (est.abs() - e0 - lut).clamp(0.0, 1.0);
        e0 * (1.0 - s * s).sqrt() + lut
    }
}

/// One scanned vector with its interval `[lo, hi]` around the estimate
#[derive(Debug, Clone, Copy)]
pub(crate) struct Candidate {
    pub(crate) id: u64,
    pub(crate) lo: f32,
    pub(crate) hi: f32,
}

impl Candidate {
    /// Interval `est -/+ half`; a non-finite estimate or width (a damaged
    /// or degenerate factor) gives `[-inf, +inf]`, so the vector is never
    /// certified and never pruned: it is rescored exactly
    pub(crate) fn new(id: u64, est: f32, half: f32) -> Self {
        if est.is_finite() && half.is_finite() {
            Self {
                id,
                lo: est - half,
                hi: est + half,
            }
        } else {
            Self {
                id,
                lo: f32::NEG_INFINITY,
                hi: f32::INFINITY,
            }
        }
    }
}

fn kth_largest_lo(c: &[Candidate], k: usize) -> f32 {
    let mut lo: Vec<f32> = c.iter().map(|x| x.lo).collect();
    let (_, kth, _) = lo.select_nth_unstable_by(k - 1, |a, b| b.total_cmp(a));
    *kth
}

/// Drop candidates that cannot reach the top `k`: with `tau` the k-th
/// largest lower bound, `k` vectors score at least `tau`, so a vector
/// whose upper bound is below `tau` is outside the top k. Pruning a
/// subset (one scan task) keeps a superset of what the full set keeps.
pub(crate) fn prune(mut c: Vec<Candidate>, k: usize) -> Vec<Candidate> {
    if k == 0 || c.len() <= k {
        return c;
    }
    let tau = kth_largest_lo(&c, k);
    c.retain(|x| x.hi >= tau);
    c
}

/// Candidates split into certified members of the top `k` and the
/// uncertain ones that must be rescored exactly; uncertain ones are
/// sorted by upper bound, best first, so a capped rerank keeps the most
/// promising
#[derive(Debug, Default)]
pub(crate) struct Split {
    pub(crate) certified: Vec<u64>,
    pub(crate) uncertain: Vec<u64>,
}

/// Certify: `x` is in the top `k` if its lower bound reaches `tau` and at
/// most `k - 1` other candidates have an upper bound above it (every
/// vector that beats `x` has `hi > lo_x >= tau`, so it is a candidate).
/// At most `k` are certified; ties beyond `k` stay uncertain.
pub(crate) fn split(c: Vec<Candidate>, k: usize) -> Split {
    let c = prune(c, k);
    if c.len() <= k {
        return Split {
            certified: c.iter().map(|x| x.id).collect(),
            uncertain: Vec::new(),
        };
    }
    let tau = kth_largest_lo(&c, k);
    let mut his: Vec<f32> = c.iter().map(|x| x.hi).collect();
    his.sort_by(|a, b| b.total_cmp(a));
    let mut certified = Vec::new();
    let mut uncertain = Vec::new();
    for x in &c {
        let above = his.partition_point(|h| *h > x.lo);
        let others = above - usize::from(x.hi > x.lo);
        if x.lo >= tau && others < k {
            certified.push(*x);
        } else {
            uncertain.push(*x);
        }
    }
    certified.sort_by(|a, b| b.lo.total_cmp(&a.lo));
    uncertain.extend(certified.iter().skip(k).copied());
    certified.truncate(k);
    uncertain.sort_by(|a, b| b.hi.total_cmp(&a.hi));
    Split {
        certified: certified.into_iter().map(|x| x.id).collect(),
        uncertain: uncertain.into_iter().map(|x| x.id).collect(),
    }
}

#[cfg(test)]
#[path = "certify_tests.rs"]
mod tests;
