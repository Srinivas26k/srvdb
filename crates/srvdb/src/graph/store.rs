//! Links on disk: the `links` table in `metadata.db` (ADR 0016)
//!
//! Only the commit and compaction transactions write this table, so links
//! change on disk exactly when the commit point does. Every such
//! transaction also records a *stamp* of the committed state (count and
//! tombstones). A SrvDB version before links commits without updating the
//! stamp, so stored links whose stamp no longer matches may point at the
//! wrong records and are refused on open.

use super::{validate_type, Limits, LinkGraph, Op};
use crate::error::Error;
use anyhow::Result;
use redb::{
    Database, Durability, ReadableDatabase, ReadableTable, ReadableTableMetadata, TableDefinition,
    WriteTransaction,
};
use std::collections::HashSet;

/// (source, type, target) -> ()
const LINKS: TableDefinition<(u64, &str, u64), ()> = TableDefinition::new("links");
/// Shared with `storage::metadata` (same names and types)
const STATE: TableDefinition<&str, u64> = TableDefinition::new("state");
const TOMBSTONES: TableDefinition<u64, ()> = TableDefinition::new("tombstones");
const COMMITTED_COUNT: &str = "committed_count";
const FORMAT_KEY: &str = "links_format";
const STAMP_KEY: &str = "links_stamp";
/// Version of the `links` table layout
const FORMAT: u64 = 1;

fn mix(mut x: u64) -> u64 {
    // splitmix64 finalizer: spreads ids so the sum below is a cheap,
    // order-independent hash of the tombstone set (not cryptographic;
    // it only has to notice changes made by an older SrvDB)
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Stamp of a committed state: the record count and the tombstone set
pub(crate) fn stamp(count: u64, tombstones: impl IntoIterator<Item = u64>) -> u64 {
    let (mut n, mut sum) = (0u64, 0u64);
    for id in tombstones {
        n += 1;
        sum = sum.wrapping_add(mix(id));
    }
    mix(count ^ 0x5EED) ^ mix(n.wrapping_add(sum))
}

fn corrupt(msg: String) -> anyhow::Error {
    Error::Corrupt(msg).into()
}

/// Load and validate every stored link, enforcing `limits` as it reads.
/// Anything inconsistent is `Corrupt`; nothing is dropped silently.
pub(crate) fn load(
    db: &Database,
    count: u64,
    deleted: &HashSet<u64>,
    limits: Limits,
) -> Result<LinkGraph> {
    let txn = db.begin_read()?;
    let table = match txn.open_table(LINKS) {
        Ok(table) => table,
        // A database from before links (SrvDB 0.3 and earlier)
        Err(redb::TableError::TableDoesNotExist(_)) => return Ok(LinkGraph::with_limits(limits)),
        Err(e) => return Err(e.into()),
    };
    let state = txn.open_table(STATE)?;
    let format = state.get(FORMAT_KEY)?.map(|v| v.value());
    if format.is_some_and(|v| v != FORMAT) {
        return Err(corrupt(format!(
            "metadata.db holds links in format {}, but this SrvDB reads format \
             {FORMAT}; open it with the newer SrvDB that wrote it",
            format.unwrap_or_default()
        )));
    }
    if table.len()? == 0 {
        return Ok(LinkGraph::with_limits(limits));
    }
    let stored = state.get(STAMP_KEY)?.map(|v| v.value());
    if stored != Some(stamp(count, deleted.iter().copied())) {
        return Err(corrupt(
            "the links in metadata.db were written before another SrvDB \
             version (one without links, 0.3 or earlier) added, deleted or \
             compacted records, so they may point at the wrong records. \
             Remove them with SrvDB::drop_links(path) (Python: \
             srvdb.drop_links(path)) and link again; vectors and metadata \
             are not affected"
                .to_string(),
        ));
    }
    let mut graph = LinkGraph::with_limits(limits);
    for entry in table.iter()? {
        let (key, _) = entry?;
        let (src, ty, dst) = key.value();
        let bad_ids = src >= count || dst >= count || src == dst;
        let dead = deleted.contains(&src) || deleted.contains(&dst);
        if bad_ids || dead || validate_type(ty).is_err() {
            return Err(corrupt(format!(
                "metadata.db holds an invalid link ({src} -> {dst}, {count} \
                 records committed); restore the database from a backup or \
                 remove all links with SrvDB::drop_links(path)"
            )));
        }
        graph.insert_checked(src, ty, dst).map_err(|limit| {
            corrupt(format!(
                "metadata.db holds more links than allowed ({limit}); remove \
                 them with SrvDB::drop_links(path)"
            ))
        })?;
    }
    Ok(graph)
}

/// Record the format and the stamp of the state this transaction commits
fn write_stamp(txn: &WriteTransaction) -> Result<()> {
    let tombstones: Vec<u64> = txn
        .open_table(TOMBSTONES)?
        .iter()?
        .map(|e| Ok(e?.0.value()))
        .collect::<Result<_>>()?;
    let mut state = txn.open_table(STATE)?;
    let count = state.get(COMMITTED_COUNT)?.map_or(0, |v| v.value());
    state.insert(STAMP_KEY, stamp(count, tombstones))?;
    state.insert(FORMAT_KEY, FORMAT)?;
    Ok(())
}

/// Write pending changes and the stamp inside the commit transaction
/// (called on every commit, also without changes)
pub(crate) fn apply(txn: &WriteTransaction, ops: &[Op]) -> Result<()> {
    {
        let mut table = txn.open_table(LINKS)?;
        for op in ops {
            match op {
                Op::Add(s, t, d) => {
                    table.insert((*s, &**t, *d), ())?;
                }
                Op::Remove(s, t, d) => {
                    table.remove((*s, &**t, *d))?;
                }
            }
        }
    }
    write_stamp(txn)
}

/// Replace the stored links with `graph` (compaction renumbering) and
/// stamp the compacted state, inside the compaction transaction
pub(crate) fn rewrite(txn: &WriteTransaction, graph: &LinkGraph) -> Result<()> {
    {
        let mut table = txn.open_table(LINKS)?;
        table.retain(|_, _| false)?;
        for (s, t, d) in graph.all() {
            table.insert((s, &*t, d), ())?;
        }
    }
    write_stamp(txn)
}

/// Durably delete every link and stamp the current state; returns how
/// many were removed (`SrvDB::drop_links`)
pub(crate) fn clear(db: &Database) -> Result<u64> {
    let mut txn = db.begin_write()?;
    txn.set_durability(Durability::Immediate)?;
    let removed = {
        let mut table = txn.open_table(LINKS)?;
        let n = table.len()?;
        table.retain(|_, _| false)?;
        n
    };
    write_stamp(&txn)?;
    txn.commit()?;
    Ok(removed)
}

#[cfg(test)]
#[path = "store_tests.rs"]
mod tests;
