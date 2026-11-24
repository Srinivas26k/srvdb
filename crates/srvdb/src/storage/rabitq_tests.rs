//! Tests for RaBitQ code storage: reopen, validation and damage

use super::*;
use tempfile::TempDir;

const DIM: usize = 40;

fn data(n: usize) -> Vec<Vec<f32>> {
    let mut s = 11u64;
    (0..n)
        .map(|_| {
            (0..DIM)
                .map(|_| {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((s >> 33) as f32 / (1u64 << 30) as f32) - 1.0
                })
                .collect()
        })
        .collect()
}

fn filled(dir: &Path, config: &RabitqConfig, n: usize) -> RabitqStorage {
    let mut s = RabitqStorage::create(dir, DIM, config).unwrap();
    let d = data(n);
    s.append(d.iter().map(|v| v.as_slice())).unwrap();
    s.flush().unwrap();
    s
}

fn err(r: Result<RabitqStorage>) -> Error {
    Error::from(r.err().expect("open should fail"))
}

#[test]
fn reopens_with_same_settings_and_scores_alike() {
    let dir = TempDir::new().unwrap();
    let config = RabitqConfig::new(2).unwrap();
    let before = filled(dir.path(), &config, 600);
    let query = &data(600)[17];
    let want = before.search(query, 5, &HashSet::new());
    drop(before);
    let after = RabitqStorage::open(dir.path(), DIM, &config).unwrap();
    assert_eq!(after.count(), 600);
    assert_eq!(after.search(query, 5, &HashSet::new()), want);
    assert_eq!(want[0].0, 17);
    let deleted: HashSet<u64> = [17].into_iter().collect();
    assert_ne!(after.search(query, 1, &deleted)[0].0, 17);
}

#[test]
fn other_settings_are_refused() {
    let dir = TempDir::new().unwrap();
    let config = RabitqConfig::new(4).unwrap();
    drop(filled(dir.path(), &config, 10));
    for other in [
        RabitqConfig::new(1).unwrap(),
        config.with_seed(config.seed + 1),
    ] {
        let e = err(RabitqStorage::open(dir.path(), DIM, &other));
        assert!(matches!(e, Error::InvalidState(_)), "{e:?}");
    }
    let e = err(RabitqStorage::open(dir.path(), DIM + 1, &config));
    assert!(matches!(e, Error::InvalidState(_)), "{e:?}");
    let empty = TempDir::new().unwrap();
    let e = err(RabitqStorage::open(empty.path(), DIM, &config));
    assert!(matches!(e, Error::InvalidState(_)), "{e:?}");
}

#[test]
fn damaged_files_are_detected() {
    let dir = TempDir::new().unwrap();
    let config = RabitqConfig::new(1).unwrap();
    drop(filled(dir.path(), &config, 300));

    let codes = dir.path().join(FILE_NAME);
    let mut bytes = std::fs::read(&codes).unwrap();
    bytes[24 + 100] ^= 0x40;
    std::fs::write(&codes, &bytes).unwrap();
    let e = err(RabitqStorage::open(dir.path(), DIM, &config));
    assert!(matches!(e, Error::ChecksumMismatch(_)), "{e:?}");

    let params = dir.path().join(PARAMS_FILE);
    let mut bytes = std::fs::read(&params).unwrap();
    *bytes.last_mut().unwrap() ^= 0x01;
    std::fs::write(&params, &bytes).unwrap();
    let e = err(RabitqStorage::open(dir.path(), DIM, &config));
    assert!(matches!(e, Error::ChecksumMismatch(_)), "{e:?}");

    // create() starts over cleanly
    let fresh = RabitqStorage::create(dir.path(), DIM, &config).unwrap();
    assert_eq!(fresh.count(), 0);
}
