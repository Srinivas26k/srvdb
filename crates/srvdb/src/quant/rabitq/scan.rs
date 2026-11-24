//! Parallel code scan with an optional cooperative deadline (ADR 0017)
//!
//! Rayon tasks of [`TASK_BLOCKS`] blocks each score their blocks with the
//! fast-scan kernel and reduce them with a caller-supplied function. With
//! a deadline, each task checks the clock before every block and stops
//! once it has passed; the scan reports how many live vectors it scored.

use super::certify::{self, Bounds, Candidate, Split};
use super::fastscan::{FastScanCodes, Kernel, QueryLut, BLOCK};
use crate::error::{Error, Result};
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

/// Blocks per parallel task (1024 vectors)
const TASK_BLOCKS: usize = 32;

/// One query against the in-memory codes
pub(crate) struct Scan<'a> {
    pub(crate) codes: &'a FastScanCodes,
    pub(crate) lut: &'a QueryLut,
    pub(crate) deleted: &'a HashSet<u64>,
    pub(crate) deadline: Option<Instant>,
}

/// What a scan kept, and how many live vectors it scored
pub(crate) struct Scanned<T> {
    pub(crate) items: Vec<T>,
    pub(crate) scanned: usize,
}

impl Scan<'_> {
    /// Live (not deleted) vectors in the codes
    pub(crate) fn live(&self) -> usize {
        let n = self.codes.len();
        n - self.deleted.iter().filter(|&&id| id < n as u64).count()
    }

    fn run<T: Send>(&self, keep: impl Fn(Vec<(u64, f32)>) -> Vec<T> + Sync) -> Scanned<T> {
        use rayon::prelude::*;
        let blocks = self.codes.len().div_ceil(BLOCK);
        let tasks = blocks.div_ceil(TASK_BLOCKS);
        let kernel = Kernel::detect();
        let scanned = AtomicUsize::new(0);
        let items = (0..tasks)
            .into_par_iter()
            .flat_map_iter(|task| {
                let mut out = Vec::with_capacity(TASK_BLOCKS * BLOCK);
                let mut scores = [0.0f32; BLOCK];
                let end = ((task + 1) * TASK_BLOCKS).min(blocks);
                for b in task * TASK_BLOCKS..end {
                    if self.deadline.is_some_and(|d| Instant::now() >= d) {
                        break;
                    }
                    let real = self.codes.block_scores(b, self.lut, kernel, &mut scores);
                    let first = (b * BLOCK) as u64;
                    out.extend(
                        scores[..real]
                            .iter()
                            .zip(first..)
                            .filter(|(_, id)| self.deleted.is_empty() || !self.deleted.contains(id))
                            .map(|(s, id)| (id, *s)),
                    );
                }
                scanned.fetch_add(out.len(), Ordering::Relaxed);
                keep(out)
            })
            .collect();
        Scanned {
            items,
            scanned: scanned.into_inner(),
        }
    }

    /// The `n` best ids by estimated similarity among scanned vectors
    pub(crate) fn top(&self, n: usize) -> Scanned<(u64, f32)> {
        let mut out = self.run(|scored| crate::distance::select_top_k(scored, n));
        out.items = crate::distance::select_top_k(out.items, n);
        out
    }

    /// Certified and uncertain members of the top `k` among scanned
    /// vectors, with the union bound taken over every live vector.
    ///
    /// # Errors
    /// `InvalidState` if the codes do not keep error widths.
    pub(crate) fn certified(&self, k: usize) -> Result<Scanned<Split>> {
        let bounds = Bounds {
            eps: certify::epsilon(self.live()),
            lut_error: self.lut.max_error(),
        };
        if self.codes.len() > 0 && self.codes.width(0).is_none() {
            return Err(Error::invalid_state(
                "RaBitQ codes were loaded without error widths",
            ));
        }
        let kept = self.run(|scored| {
            let candidates = scored
                .into_iter()
                .map(|(id, est)| {
                    let i = id as usize;
                    let width = self.codes.width(i).unwrap_or(f32::INFINITY);
                    let half = bounds.half_width(est, self.codes.factor(i), width);
                    Candidate {
                        id,
                        lo: est - half,
                        hi: est + half,
                    }
                })
                .collect();
            certify::prune(candidates, k)
        });
        Ok(Scanned {
            items: vec![certify::split(kept.items, k)],
            scanned: kept.scanned,
        })
    }
}
