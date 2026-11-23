use super::*;
use tempfile::TempDir;

const RECORD: usize = 96; // does not divide BLOCK_BYTES, so records straddle blocks

fn data(records: usize) -> Vec<u8> {
    (0..records * RECORD)
        .map(|i| (i * 31 % 251) as u8)
        .collect()
}

#[test]
fn incremental_append_matches_one_pass() {
    let bytes = data(30_000); // about 2.7 blocks
    let mut sums = BlockSums::new(RECORD);
    for chunk in bytes.chunks(RECORD * 777) {
        sums.append(chunk);
    }
    assert_eq!(sums, BlockSums::compute(RECORD, 30_000, &bytes));
    assert_eq!(sums.crcs.len(), 3);
}

#[test]
fn truncate_matches_fresh_table() {
    let bytes = data(30_000);
    for keep in [0, 1, 10_923, 10_922, 25_000] {
        let mut sums = BlockSums::compute(RECORD, 30_000, &bytes);
        sums.truncate(keep, &bytes);
        assert_eq!(sums, BlockSums::compute(RECORD, keep, &bytes), "{keep}");
    }
}

#[test]
fn verify_detects_any_flipped_byte() {
    let path = Path::new("vectors.bin");
    let mut bytes = data(30_000);
    let sums = BlockSums::compute(RECORD, 30_000, &bytes);
    sums.verify(path, &bytes).unwrap();
    for at in [0, BLOCK_BYTES as usize + 5, bytes.len() - 1] {
        bytes[at] ^= 0x01;
        let err = Error::from(sums.verify(path, &bytes).unwrap_err());
        assert!(matches!(err, Error::ChecksumMismatch(_)), "{err:?}");
        bytes[at] ^= 0x01;
    }
    let err = Error::from(sums.verify(path, &bytes[..100]).unwrap_err());
    assert!(matches!(err, Error::Corrupt(_)), "{err:?}");
}

#[test]
fn sidecar_round_trips_and_rejects_mismatch() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("vectors.bin");
    assert!(BlockSums::load(&path, RECORD).unwrap().is_none());
    let sums = BlockSums::compute(RECORD, 100, &data(100));
    sums.save(&path).unwrap();
    assert_eq!(BlockSums::load(&path, RECORD).unwrap(), Some(sums));
    let err = Error::from(BlockSums::load(&path, RECORD + 1).unwrap_err());
    assert!(matches!(err, Error::Corrupt(_)), "{err:?}");

    let side = sidecar(&path);
    let mut raw = std::fs::read(&side).unwrap();
    *raw.last_mut().unwrap() ^= 0xFF;
    std::fs::write(&side, raw).unwrap();
    let err = Error::from(BlockSums::load(&path, RECORD).unwrap_err());
    assert!(matches!(err, Error::ChecksumMismatch(_)), "{err:?}");
}
