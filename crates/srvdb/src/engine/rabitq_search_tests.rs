//! The rerank of a certified or deadline search stops between batches
//! once the deadline has passed

use super::*;
use crate::quant::rabitq::RabitqConfig;
use crate::{Vector, VectorEngine};
use tempfile::TempDir;

#[test]
fn rescore_checks_the_deadline_between_batches() {
    let dir = TempDir::new().unwrap();
    let mut db = SrvDB::new_rabitq(
        dir.path().to_str().unwrap(),
        8,
        RabitqConfig::new(1).unwrap(),
    )
    .unwrap();
    let vecs: Vec<Vector> = (0..600)
        .map(|i| Vector::new((0..8).map(|d| ((i * 7 + d) % 13) as f32 + 1.0).collect()))
        .collect();
    db.add_batch(&vecs, &vec!["{}".to_string(); 600]).unwrap();
    let (_, vectors, _) = db.parts().unwrap();
    let ids: Vec<u64> = (0..600).collect();
    let query = vecs[0].data.clone();

    let (all, complete) = db.rescore(vectors, &query, &ids, None).unwrap();
    assert!(complete);
    assert_eq!(all.len(), 600);

    // An expired deadline still rescores the first batch, then stops
    let (some, complete) = db
        .rescore(vectors, &query, &ids, Some(Instant::now()))
        .unwrap();
    assert!(!complete);
    assert_eq!(some.len(), 256);
}
