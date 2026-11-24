//! Links on disk: the `links` table in `metadata.db` (ADR 0016)
//!
//! Only the commit and compaction transactions of `MetadataStore` write
//! this table, so links change on disk exactly when the commit point does.

use super::{validate_type, LinkGraph, Op};
use crate::error::Error;
use anyhow::Result;
use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition, WriteTransaction};
use std::collections::HashSet;

/// (source, type, target) -> ()
const LINKS: TableDefinition<(u64, &str, u64), ()> = TableDefinition::new("links");
/// Shared with `storage::metadata` (same name and types)
const STATE: TableDefinition<&str, u64> = TableDefinition::new("state");
const FORMAT_KEY: &str = "links_format";
/// Version of the `links` table layout
const FORMAT: u64 = 1;

/// Load and validate every stored link. Links touching a tombstoned record
/// are dropped and queued for removal; anything else invalid is `Corrupt`.
pub(crate) fn load(db: &Database, count: u64, deleted: &HashSet<u64>) -> Result<LinkGraph> {
    let txn = db.begin_read()?;
    match txn.open_table(STATE) {
        Ok(state) => {
            if let Some(v) = state.get(FORMAT_KEY)?.map(|v| v.value()) {
                if v != FORMAT {
                    return Err(Error::Corrupt(format!(
                        "metadata.db holds links in format {v}, but this SrvDB reads \
                         format {FORMAT}; open it with the newer SrvDB that wrote it"
                    ))
                    .into());
                }
            }
        }
        Err(redb::TableError::TableDoesNotExist(_)) => {}
        Err(e) => return Err(e.into()),
    }
    let table = match txn.open_table(LINKS) {
        Ok(table) => table,
        // A database from before links (SrvDB 0.4 and earlier)
        Err(redb::TableError::TableDoesNotExist(_)) => return Ok(LinkGraph::default()),
        Err(e) => return Err(e.into()),
    };
    let mut graph = LinkGraph::default();
    let mut stale = Vec::new();
    for entry in table.iter()? {
        let (key, _) = entry?;
        let (src, ty, dst) = key.value();
        if src >= count || dst >= count || src == dst || validate_type(ty).is_err() {
            return Err(Error::Corrupt(format!(
                "metadata.db holds an invalid link ({src} -> {dst}, {} records \
                 committed); restore the database from a backup",
                count
            ))
            .into());
        }
        if deleted.contains(&src) || deleted.contains(&dst) {
            stale.push((src, ty.to_string(), dst));
        } else {
            graph.insert_loaded(src, ty, dst);
        }
    }
    if !stale.is_empty() {
        log::warn!(
            "dropping {} links to deleted records (written by an older SrvDB)",
            stale.len()
        );
        graph
            .pending
            .extend(stale.into_iter().map(|(s, t, d)| Op::Remove(s, t, d)));
    }
    Ok(graph)
}

/// Write pending changes inside the commit transaction
pub(crate) fn apply(txn: &WriteTransaction, ops: &[Op]) -> Result<()> {
    if ops.is_empty() {
        return Ok(());
    }
    let mut table = txn.open_table(LINKS)?;
    for op in ops {
        match op {
            Op::Add(s, t, d) => {
                table.insert((*s, t.as_str(), *d), ())?;
            }
            Op::Remove(s, t, d) => {
                table.remove((*s, t.as_str(), *d))?;
            }
        }
    }
    txn.open_table(STATE)?.insert(FORMAT_KEY, FORMAT)?;
    Ok(())
}

/// Replace the stored links with `graph` (compaction renumbering), inside
/// the compaction transaction
pub(crate) fn rewrite(txn: &WriteTransaction, graph: &LinkGraph) -> Result<()> {
    let mut table = txn.open_table(LINKS)?;
    table.retain(|_, _| false)?;
    for (s, t, d) in graph.all() {
        table.insert((s, t.as_str(), d), ())?;
    }
    txn.open_table(STATE)?.insert(FORMAT_KEY, FORMAT)?;
    Ok(())
}

#[cfg(test)]
#[path = "store_tests.rs"]
mod tests;
