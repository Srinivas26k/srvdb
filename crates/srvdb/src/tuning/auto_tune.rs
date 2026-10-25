//! # Auto mode
//!
//! Auto mode chooses the *index* for the data size and machine: exact flat
//! search for small datasets, an HNSW graph beyond that. It never changes
//! how vectors are stored: compression (SQ8, PQ) loses precision and cannot
//! be undone, so it is only ever an explicit choice. An HNSW graph is
//! derived data, saved by `persist()` and attached again on open.

use crate::strategy::{self, AutoTunerConfig, IndexMode};
use crate::{SrvDB, VectorEngine};

/// Choose the index for the current data and machine (called when auto
/// mode is selected).
pub fn apply_auto_strategy(db: &mut SrvDB) {
    if let Err(e) = apply_with(db, &AutoTunerConfig::new(detect_available_ram_gb())) {
        // Choosing an index is an optimisation; flat search still works
        log::warn!("auto mode: keeping flat search: {e}");
    }
}

/// Re-check the choice as data grows (called by `persist()` in auto mode).
pub fn check_and_migrate(db: &mut SrvDB) -> anyhow::Result<()> {
    if db.current_mode != IndexMode::Auto {
        return Ok(());
    }
    apply_with(db, &AutoTunerConfig::new(detect_available_ram_gb()))
}

/// Build an HNSW graph when the strategy asks for one and there is none.
/// An existing graph is never dropped, and storage is never converted.
pub(crate) fn apply_with(db: &mut SrvDB, config: &AutoTunerConfig) -> anyhow::Result<()> {
    let count = db.count() as usize;
    let choice = strategy::decide_mode(config, count);
    log::info!(
        "auto mode: {:.1} GB RAM, {count} vectors, chose {choice:?}",
        config.available_ram_gb
    );
    if choice == IndexMode::Hnsw && db.hnsw_index.is_none() && db.vector_storage.is_some() {
        log::info!("auto mode: building HNSW graph for {count} vectors");
        db.attach_hnsw(crate::HNSWConfig::default())?;
        db.current_mode = IndexMode::Auto; // attach_hnsw selects Hnsw mode
    }
    Ok(())
}

/// Detect total system RAM in GB.
///
/// Uses /proc/meminfo on Linux for zero-dependency detection.
/// Fallback to generic safe default (4GB) on other OS.
fn detect_available_ram_gb() -> f64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if line.starts_with("MemTotal:") {
                    // Line format: "MemTotal:        16316252 kB"
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(kb) = parts[1].parse::<u64>() {
                            return kb as f64 / 1024.0 / 1024.0;
                        }
                    }
                }
            }
        }
    }

    // Fallback for non-Linux or read failure
    4.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DatabaseConfig, Vector};
    use tempfile::TempDir;

    fn config(ram_gb: f64) -> AutoTunerConfig {
        AutoTunerConfig {
            available_ram_gb: ram_gb,
            dataset_size_threshold: 100,
            ..Default::default()
        }
    }

    fn filled(dir: &TempDir, n: usize) -> SrvDB {
        let mut db = SrvDB::new_with_config(
            dir.path().to_str().unwrap(),
            DatabaseConfig::new(128).unwrap(),
        )
        .unwrap();
        let vectors: Vec<Vector> = (0..n)
            .map(|i| {
                Vector::new(
                    (0..128)
                        .map(|j| ((i * 7 + j * 13) % 29) as f32 - 14.0)
                        .collect(),
                )
            })
            .collect();
        let metas = vec!["{}".to_string(); n];
        db.add_batch(&vectors, &metas).unwrap();
        db.current_mode = IndexMode::Auto;
        db
    }

    #[test]
    fn large_dataset_gets_an_hnsw_graph_that_survives_reopen() {
        let dir = TempDir::new().unwrap();
        {
            let mut db = filled(&dir, 150);
            apply_with(&mut db, &config(16.0)).unwrap();
            assert_eq!(db.hnsw_index.as_ref().map(|h| h.len()), Some(150));
            assert_eq!(db.current_mode, IndexMode::Auto);
            db.persist().unwrap();
        }
        let db = SrvDB::new_with_config(
            dir.path().to_str().unwrap(),
            DatabaseConfig::new(128).unwrap(),
        )
        .unwrap();
        assert_eq!(db.hnsw_index.as_ref().map(|h| h.len()), Some(150));
    }

    #[test]
    fn auto_mode_never_converts_storage() {
        for ram in [1.0, 3.0, 6.0, 16.0] {
            let dir = TempDir::new().unwrap();
            let mut db = filled(&dir, 150);
            apply_with(&mut db, &config(ram)).unwrap();
            db.persist().unwrap();
            assert!(db.vector_storage.is_some(), "{ram} GB");
            assert!(db.scalar_storage.is_none(), "{ram} GB");
            assert!(!dir.path().join("sq8_vectors.bin").exists(), "{ram} GB");
        }
    }

    #[test]
    fn small_dataset_stays_flat() {
        let dir = TempDir::new().unwrap();
        let mut db = filled(&dir, 50);
        apply_with(&mut db, &config(16.0)).unwrap();
        assert!(db.hnsw_index.is_none());
    }
}
