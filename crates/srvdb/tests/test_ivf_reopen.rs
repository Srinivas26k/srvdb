//! A trained IVF index is reloaded on open instead of being lost

use srvdb::{DatabaseConfig, IVFConfig, IndexMode, SrvDB, Vector, VectorEngine};
use tempfile::TempDir;

const DIM: usize = 128;

fn vector(seed: u64) -> Vector {
    let mut state = (seed + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    Vector::new(
        (0..DIM)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
            })
            .collect(),
    )
}

fn open(dir: &TempDir) -> SrvDB {
    SrvDB::new_with_config(
        dir.path().to_str().unwrap(),
        DatabaseConfig::new(DIM).unwrap(),
    )
    .unwrap()
}

fn add(db: &mut SrvDB, ids: std::ops::Range<u64>) {
    let vectors: Vec<Vector> = ids.clone().map(vector).collect();
    let metas: Vec<String> = ids.map(|i| format!(r#"{{"i": {i}}}"#)).collect();
    db.add_batch(&vectors, &metas).unwrap();
}

fn trained(dir: &TempDir, n: u64) -> SrvDB {
    let mut db = open(dir);
    add(&mut db, 0..n);
    db.configure_ivf(IVFConfig {
        nlist: 8,
        nprobe: 2,
        ..Default::default()
    })
    .unwrap();
    db.set_mode(IndexMode::Ivf);
    db.train_ivf().unwrap();
    db
}

fn top_ids(db: &SrvDB) -> Vec<Vec<u64>> {
    (9000..9020)
        .map(|q| {
            db.search(&vector(q), 5)
                .unwrap()
                .iter()
                .map(|r| r.id)
                .collect()
        })
        .collect()
}

#[test]
fn trained_index_survives_reopen() {
    let dir = TempDir::new().unwrap();
    let before = {
        let mut db = trained(&dir, 800);
        db.persist().unwrap();
        top_ids(&db)
    };
    let db = open(&dir);
    assert_eq!(db.current_mode, IndexMode::Ivf);
    // Same partitions and graphs: identical approximate results (nprobe=2)
    assert_eq!(top_ids(&db), before);
}

#[test]
fn vectors_committed_after_the_index_file_are_indexed_on_open() {
    let dir = TempDir::new().unwrap();
    {
        let mut db = trained(&dir, 500);
        db.persist().unwrap();
        let older = std::fs::read(dir.path().join("ivf.index")).unwrap();
        add(&mut db, 500..600);
        db.persist().unwrap();
        drop(db);
        std::fs::write(dir.path().join("ivf.index"), older).unwrap();
    }
    let db = open(&dir);
    assert_eq!(db.current_mode, IndexMode::Ivf);
    for id in [510, 555, 599] {
        assert_eq!(db.search(&vector(id), 1).unwrap()[0].id, id);
    }
}
