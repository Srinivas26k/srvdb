//! Metadata store and commit point (`metadata.db`, backed by redb)
//!
//! Durability (ADR 0003): metadata writes use `Durability::None`, which is
//! cheap and becomes durable only with a later durable commit. `commit`
//! records the committed vector count in one `Durability::Immediate`
//! transaction; that is the moment `persist()` makes data permanent. After a
//! crash redb rolls back to the last durable commit, so metadata and the
//! committed count always agree.

use anyhow::{Context, Result};
use redb::{Database, Durability, ReadableTable, ReadableTableMetadata, TableDefinition};
use std::path::Path;

const METADATA_TABLE: TableDefinition<u64, &str> = TableDefinition::new("metadata");
const STATE_TABLE: TableDefinition<&str, u64> = TableDefinition::new("state");
const COMMITTED_COUNT: &str = "committed_count";

pub struct MetadataStore {
    db: Database,
}

impl MetadataStore {
    pub fn new(db_path: &str) -> Result<Self> {
        let dir = Path::new(db_path);
        let db_file = dir.join("metadata.db");
        if !db_file.exists() {
            Self::create(dir, &db_file)?;
        }
        let db = Database::create(&db_file).context("failed to open metadata.db")?;
        Ok(Self { db })
    }

    /// Build a new metadata.db under a temporary name and rename it into
    /// place, so a crash during creation never leaves a partial file at
    /// the real name (redb cannot open one).
    fn create(dir: &Path, db_file: &Path) -> Result<()> {
        let tmp = dir.join(".metadata.db.tmp");
        if tmp.exists() {
            std::fs::remove_file(&tmp)?; // left by a crash during creation
        }
        {
            let db = Database::create(&tmp).context("failed to create metadata.db")?;
            let txn = db.begin_write()?;
            txn.open_table(METADATA_TABLE)?;
            txn.open_table(STATE_TABLE)?;
            txn.commit()?;
        }
        std::fs::rename(&tmp, db_file).context("failed to create metadata.db")?;
        super::atomic::sync_dir(dir)
    }

    /// Run `write` in a non-durable transaction (made durable by `commit`).
    fn write(&self, write: impl FnOnce(&mut redb::Table<u64, &str>) -> Result<()>) -> Result<()> {
        let mut txn = self.db.begin_write()?;
        txn.set_durability(Durability::None);
        {
            let mut table = txn.open_table(METADATA_TABLE)?;
            write(&mut table)?;
        }
        txn.commit()?;
        Ok(())
    }

    pub fn set(&mut self, id: u64, metadata: &str) -> Result<()> {
        self.write(|table| {
            table.insert(id, metadata)?;
            Ok(())
        })
    }

    /// Write many entries in one transaction
    pub fn set_batch(&mut self, items: &[(u64, String)]) -> Result<()> {
        self.write(|table| {
            for (id, metadata) in items {
                table.insert(*id, metadata.as_str())?;
            }
            Ok(())
        })
    }

    pub fn get(&self, id: u64) -> Result<Option<String>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(METADATA_TABLE)?;
        Ok(table.get(id)?.map(|v| v.value().to_string()))
    }

    pub fn get_batch(&self, ids: &[u64]) -> Result<Vec<Option<String>>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(METADATA_TABLE)?;
        ids.iter()
            .map(|id| Ok(table.get(*id)?.map(|v| v.value().to_string())))
            .collect()
    }

    pub fn delete(&mut self, id: u64) -> Result<()> {
        self.write(|table| {
            table.remove(id)?;
            Ok(())
        })
    }

    /// All (id, metadata) entries
    pub fn all(&self) -> Result<Vec<(u64, String)>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(METADATA_TABLE)?;
        table
            .iter()?
            .map(|entry| {
                let (k, v) = entry?;
                Ok((k.value(), v.value().to_string()))
            })
            .collect()
    }

    pub fn count(&self) -> Result<usize> {
        let txn = self.db.begin_read()?;
        Ok(txn.open_table(METADATA_TABLE)?.len()? as usize)
    }

    /// Durably commit: every earlier write, plus `count` as the committed
    /// number of vectors, survives a crash once this returns.
    pub fn commit(&mut self, count: u64) -> Result<()> {
        let mut txn = self.db.begin_write()?;
        txn.set_durability(Durability::Immediate);
        txn.open_table(STATE_TABLE)?
            .insert(COMMITTED_COUNT, count)?;
        txn.commit().context("failed to commit metadata.db")?;
        Ok(())
    }

    /// The committed vector count, or `None` for a database written before
    /// v0.3 (which had no commit record).
    pub fn committed_count(&self) -> Result<Option<u64>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(STATE_TABLE)?;
        Ok(table.get(COMMITTED_COUNT)?.map(|v| v.value()))
    }

    /// Durably remove metadata for ids at or above `count` (writes that were
    /// never committed by a durable `commit`).
    pub fn truncate_from(&mut self, count: u64) -> Result<()> {
        let mut txn = self.db.begin_write()?;
        txn.set_durability(Durability::Immediate);
        txn.open_table(METADATA_TABLE)?
            .retain_in(count.., |_, _| false)?;
        txn.commit()?;
        Ok(())
    }
}

#[cfg(test)]
#[path = "metadata_tests.rs"]
mod tests;
