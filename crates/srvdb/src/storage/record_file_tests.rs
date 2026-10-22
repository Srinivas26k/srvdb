//! RecordFile tests

use super::*;
use tempfile::TempDir;

const DIM: usize = 4;
const REC: usize = 4; // one u8 code per dimension, like SQ8

fn open(dir: &TempDir) -> RecordFile {
    RecordFile::open(&dir.path().join("records.bin"), DIM, REC, 1).unwrap()
}

#[test]
fn appended_records_are_readable_immediately() {
    let dir = TempDir::new().unwrap();
    let mut file = open(&dir);
    assert_eq!(file.append(&[1, 2, 3, 4]).unwrap(), 0);
    assert_eq!(file.append(&[5, 6, 7, 8, 9, 10, 11, 12]).unwrap(), 1);
    assert_eq!(file.count(), 3);
    assert_eq!(file.get(1).unwrap(), &[5, 6, 7, 8]);
    assert_eq!(file.range(1, 2).unwrap(), &[5, 6, 7, 8, 9, 10, 11, 12]);
    assert!(file.get(3).is_none());
    assert!(file.range(2, 2).is_none());
}

#[test]
fn growth_keeps_earlier_records() {
    let dir = TempDir::new().unwrap();
    let mut file = open(&dir);
    for i in 0..5000u32 {
        file.append(&(i as u8).to_le_bytes().repeat(REC)).unwrap();
    }
    assert_eq!(file.count(), 5000);
    assert_eq!(file.get(4321).unwrap(), &[(4321u32 as u8); REC]);
}

#[test]
fn synced_records_survive_reopen_and_unsynced_do_not_count() {
    let dir = TempDir::new().unwrap();
    {
        let mut file = open(&dir);
        file.append(&[1; REC]).unwrap();
        file.sync().unwrap();
        file.append(&[2; REC]).unwrap(); // never synced
    }
    let file = open(&dir);
    assert_eq!(file.count(), 1);
    assert_eq!(file.get(0).unwrap(), &[1; REC]);
}

#[test]
fn stray_bytes_after_a_crash_never_shift_new_ids() {
    // v0.2 bug: bytes past the committed count stayed in the file and new
    // records were appended after them, so ids pointed at the wrong data.
    let dir = TempDir::new().unwrap();
    {
        let mut file = open(&dir);
        file.append(&[1; REC]).unwrap();
        file.sync().unwrap();
        file.append(&[9; REC * 3]).unwrap(); // uncommitted tail
    }
    let mut file = open(&dir);
    assert_eq!(file.append(&[2; REC]).unwrap(), 1);
    assert_eq!(file.get(1).unwrap(), &[2; REC]);
}

#[test]
fn truncate_drops_uncommitted_records() {
    let dir = TempDir::new().unwrap();
    {
        let mut file = open(&dir);
        file.append(&[1; REC * 3]).unwrap();
        file.sync().unwrap();
        file.truncate_to(2).unwrap();
        assert_eq!(file.count(), 2);
        assert!(file.get(2).is_none());
    }
    let len = std::fs::metadata(dir.path().join("records.bin"))
        .unwrap()
        .len();
    assert_eq!(len, (VectorHeader::SIZE + 2 * REC) as u64);
    assert_eq!(open(&dir).count(), 2);
}

#[test]
fn truncate_beyond_count_is_corruption() {
    let dir = TempDir::new().unwrap();
    let mut file = open(&dir);
    file.append(&[1; REC]).unwrap();
    let err = file.truncate_to(5).unwrap_err();
    assert!(matches!(Error::from(err), Error::Corrupt(_)));
}

#[test]
fn reopening_with_another_dimension_is_rejected() {
    let dir = TempDir::new().unwrap();
    drop(open(&dir));
    let err = RecordFile::open(&dir.path().join("records.bin"), 8, 8, 1)
        .err()
        .unwrap();
    assert!(matches!(Error::from(err), Error::InvalidInput(_)));
}

#[test]
fn partial_records_are_rejected() {
    let dir = TempDir::new().unwrap();
    let mut file = open(&dir);
    assert!(file.append(&[1, 2, 3]).is_err());
    assert_eq!(file.count(), 0);
}
