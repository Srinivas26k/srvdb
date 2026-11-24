//! Memory budget (ADR 0018): the resident-memory report, and the check
//! every write makes before it appends

use super::*;
use crate::quant::rabitq::RabitqConfig;
use crate::storage::settings::IndexKind;
use crate::tuning::budget::{self, Tier};

/// Estimated resident memory of a database ([`SrvDB::memory_report`]).
///
/// Estimates follow the storage layouts in ADR 0018: mapped pages every
/// search reads plus heap structures. Pages searches do not read, redb's
/// cache and per-query buffers are not counted.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct MemoryReport {
    /// Index mode in use, as [`SrvDB::index_name`]
    pub mode: &'static str,
    /// RaBitQ bits per dimension, `None` for other modes
    pub bits: Option<u8>,
    /// Vector dimension
    pub dimension: usize,
    /// Stored records, deleted ones included (they use memory until
    /// `compact()`)
    pub count: u64,
    /// Vectors or codes, as searches read them
    pub vector_bytes: u64,
    /// Graph, rotation, codebooks, quantizer or centroids
    pub index_bytes: u64,
    /// Typed links between records (approximate)
    pub links_bytes: u64,
    /// Keyword index as loaded now (approximate)
    pub keyword_bytes: u64,
    /// Sum of the four parts above
    pub total_bytes: u64,
    /// The memory budget in bytes, if the database has one
    pub budget_bytes: Option<u64>,
    /// The share of the budget estimates may fill (80 %)
    pub limit_bytes: Option<u64>,
    /// Records the current mode holds within the limit; the next write
    /// past it re-encodes to fewer bits or is refused
    pub capacity: Option<u64>,
    /// Whether searches are exact (flat) rather than approximate
    pub exact: bool,
}

fn tier_of(kind: &IndexKind) -> Option<Tier> {
    match kind {
        IndexKind::Flat => Some(Tier::Flat),
        IndexKind::Rabitq(c) => Some(Tier::Rabitq(c.bits)),
        _ => None,
    }
}

/// The `memory_mb` option of `create`, range-checked
pub(super) fn memory_mb_from(options: &super::CreateOptions) -> Result<Option<u64>> {
    let Some(mb) = options.get("memory_mb") else {
        return Ok(None);
    };
    if !(1..=budget::MAX_MEMORY_MB).contains(&mb) {
        return Err(Error::invalid_input(format!(
            "memory_mb must be between 1 and {}, got {mb}",
            budget::MAX_MEMORY_MB
        )));
    }
    Ok(Some(mb))
}

impl SrvDB {
    /// Links and keyword bytes: resident whatever the vector mode
    fn other_resident_bytes(&self) -> (u64, u64) {
        let links = (self.graph.len() as u64).saturating_mul(budget::LINK_BYTES);
        (links, self.keyword_resident_bytes())
    }

    /// Vector and index bytes of the current mode at `n` records
    fn mode_bytes(&self, kind: &IndexKind, n: u64) -> (u64, u64) {
        let dim = self.dimension;
        let graph = self
            .hnsw_index
            .as_ref()
            .map_or(0, |h| budget::hnsw_bytes(h.config().m, n));
        match kind {
            IndexKind::Sq8 => budget::sq8_bytes(dim, n),
            IndexKind::Pq | IndexKind::HnswPq(_) => {
                let m = self
                    .quantized_storage
                    .as_ref()
                    .map_or(dim / 8, |q| q.q_size);
                let (codes, books) = budget::pq_bytes(dim, m, n);
                (codes, books.saturating_add(graph))
            }
            IndexKind::Rabitq(c) => (
                budget::tier_vector_bytes(Tier::Rabitq(c.bits), dim, n),
                budget::tier_fixed_bytes(Tier::Rabitq(c.bits), dim),
            ),
            IndexKind::Ivf { nlist, hnsw, .. } => (
                budget::tier_vector_bytes(Tier::Flat, dim, n),
                budget::hnsw_bytes(hnsw.m, n).saturating_add(4 * (dim as u64) * (*nlist as u64)),
            ),
            IndexKind::Flat | IndexKind::Hnsw(_) | IndexKind::Auto => {
                (budget::tier_vector_bytes(Tier::Flat, dim, n), graph)
            }
        }
    }

    /// Estimated resident memory of this database, per part, and where it
    /// stands against its memory budget (ADR 0018). Cheap: no I/O; the
    /// keyword part counts the index only if a search has loaded it.
    pub fn memory_report(&self) -> MemoryReport {
        let kind = self.current_settings().kind;
        let count = self.count();
        let (vector_bytes, index_bytes) = self.mode_bytes(&kind, count);
        let (links_bytes, keyword_bytes) = self.other_resident_bytes();
        let limit = self.memory_budget.map(budget::limit_bytes);
        let capacity = match (limit, tier_of(&kind)) {
            (Some(limit), Some(tier)) => Some(budget::capacity(
                tier,
                self.dimension,
                links_bytes.saturating_add(keyword_bytes),
                limit,
            )),
            _ => None,
        };
        MemoryReport {
            mode: kind.name(),
            bits: self.rabitq_config.map(|c| c.bits),
            dimension: self.dimension,
            count,
            vector_bytes,
            index_bytes,
            links_bytes,
            keyword_bytes,
            total_bytes: vector_bytes
                .saturating_add(index_bytes)
                .saturating_add(links_bytes)
                .saturating_add(keyword_bytes),
            budget_bytes: self.memory_budget.map(|mb| mb.saturating_mul(1 << 20)),
            limit_bytes: limit,
            capacity,
            exact: kind == IndexKind::Flat,
        }
    }

    /// Before appending `adding` records: keep the mode if it still fits
    /// the budget, re-encode to the best lower rung that does, or refuse.
    /// Nothing is written when this fails.
    pub(super) fn fit_budget(&mut self, adding: usize) -> Result<()> {
        let Some(memory_mb) = self.memory_budget else {
            return Ok(());
        };
        let kind = self.current_settings().kind;
        let Some(current) = tier_of(&kind) else {
            return Err(Error::Internal(format!(
                "a memory budget is set on a {} database",
                kind.name()
            )));
        };
        let n = self.count().saturating_add(adding as u64);
        let (links, keyword) = self.other_resident_bytes();
        let other = links.saturating_add(keyword);
        let limit = budget::limit_bytes(memory_mb);
        match budget::select(current, self.dimension, n, other, limit) {
            Some(t) if t == current => Ok(()),
            Some(t) => self.step_down(t, n),
            None => Err(Error::MemoryBudget(format!(
                "{n} records need {} bytes at 1 bit per dimension (plus {other} bytes \
                 of links and keyword index), more than {limit} bytes (80 % of \
                 memory_mb = {memory_mb}); nothing was written. Create the database \
                 with a larger memory_mb, compact() after deletes, or use a fixed \
                 index mode",
                budget::tier_bytes(Tier::Rabitq(1), self.dimension, n)
            ))),
        }
    }

    /// Re-encode every stored vector at `tier` and record the new mode
    /// before anything is appended (crash analysis in ADR 0018)
    fn step_down(&mut self, tier: Tier, n: u64) -> Result<()> {
        let Tier::Rabitq(bits) = tier else {
            return Err(Error::Internal("stepped down to flat".into()));
        };
        log::info!("memory budget: re-encoding to {bits}-bit RaBitQ for {n} records");
        let start = std::time::Instant::now();
        self.attach_rabitq(RabitqConfig::new(bits)?)?;
        self.record_settings()?;
        log::info!("memory budget: re-encoded in {:?}", start.elapsed());
        Ok(())
    }
}
