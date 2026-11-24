//! Format 2 of `index_settings.bin`: the memory budget (ADR 0018)

use super::*;
use crate::error::Error;
use tempfile::TempDir;

fn budgeted(kind: IndexKind, memory_mb: u64) -> IndexSettings {
    IndexSettings {
        dimension: 64,
        kind,
        memory_mb: Some(memory_mb),
    }
}

/// The payload format number of an envelope
fn payload_format(bytes: &[u8]) -> u16 {
    let record: Record = format::decode(format::INDEX_SETTINGS, bytes).unwrap();
    record.format
}

#[test]
fn a_budget_round_trips_as_format_2() {
    let dir = TempDir::new().unwrap();
    for kind in [
        IndexKind::Flat,
        IndexKind::Rabitq(RabitqConfig::new(1).unwrap()),
    ] {
        let settings = budgeted(kind, 50);
        assert_eq!(payload_format(&encode(&settings).unwrap()), 2);
        save(dir.path(), &settings).unwrap();
        assert_eq!(load(dir.path()).unwrap(), Some(settings));
    }
}

/// Migration: a format-1 image as the previous encoder wrote it (no budget
/// field) reads as "no budget", and without a budget format 1 is still
/// written, so earlier 0.4 builds keep reading those databases
#[test]
fn format_1_images_read_without_a_budget_and_stay_format_1() {
    let v1 = Record {
        format: 1,
        kind: 6,
        dimension: 64,
        bits: 2,
        rerank: 5,
        seed: 1,
        ..Default::default()
    };
    let bytes = format::encode(format::INDEX_SETTINGS, &v1).unwrap();
    let settings = decode(&bytes).unwrap();
    assert_eq!(settings.memory_mb, None);
    assert!(matches!(settings.kind, IndexKind::Rabitq(c) if c.bits == 2));
    let rewritten = encode(&settings).unwrap();
    assert_eq!(payload_format(&rewritten), 1);
    assert_eq!(rewritten, bytes);
}

#[test]
fn bad_budgets_are_corrupt() {
    let v2 = |kind: u8, memory_mb: u64, m: u64| {
        let base = Record {
            format: 2,
            kind,
            dimension: 64,
            m,
            ef_construction: m * 10,
            ef_search: m * 10,
            ..Default::default()
        };
        format::encode(format::INDEX_SETTINGS, &RecordV2 { base, memory_mb }).unwrap()
    };
    let cases = [
        v2(0, 0, 0),             // format 2 without a budget
        v2(0, (1 << 20) + 1, 0), // above 1 TiB
        v2(1, 50, 16),           // HNSW cannot have a budget
        v2(2, 50, 0),            // nor SQ8
    ];
    for bytes in cases {
        let err = Error::from(decode(&bytes).unwrap_err());
        assert!(matches!(err, Error::Corrupt(_)), "{err:?}");
    }
    // A format-2 image cut short of its budget field
    let short = Record {
        format: 2,
        dimension: 64,
        ..Default::default()
    };
    let bytes = format::encode(format::INDEX_SETTINGS, &short).unwrap();
    let err = Error::from(decode(&bytes).unwrap_err());
    assert!(matches!(err, Error::Corrupt(_)), "{err:?}");
}

/// A payload with bytes after the record is damage, not a valid file
#[test]
fn trailing_bytes_are_corrupt_in_both_formats() {
    let v1 = Record {
        format: 1,
        dimension: 64,
        ..Default::default()
    };
    let padded = format::encode(format::INDEX_SETTINGS, &(v1.clone(), 7u8)).unwrap();
    let err = Error::from(decode(&padded).unwrap_err());
    assert!(
        matches!(err, Error::Corrupt(ref m) if m.contains("unexpected")),
        "{err:?}"
    );

    let v2 = RecordV2 {
        base: Record { format: 2, ..v1 },
        memory_mb: 50,
    };
    let padded = format::encode(format::INDEX_SETTINGS, &(v2, 7u8)).unwrap();
    let err = Error::from(decode(&padded).unwrap_err());
    assert!(
        matches!(err, Error::Corrupt(ref m) if m.contains("unexpected")),
        "{err:?}"
    );
}
