use super::*;
use tempfile::TempDir;

fn hnsw() -> HnswParams {
    HnswParams {
        m: 24,
        ef_construction: 300,
        ef_search: 80,
    }
}

fn all_kinds() -> Vec<IndexKind> {
    vec![
        IndexKind::Flat,
        IndexKind::Hnsw(hnsw()),
        IndexKind::Sq8,
        IndexKind::Pq,
        IndexKind::HnswPq(hnsw()),
        IndexKind::Ivf {
            nlist: 64,
            nprobe: 8,
            hnsw: hnsw(),
        },
        IndexKind::Rabitq(RabitqConfig::new(2).unwrap().with_rerank(7).with_seed(9)),
        IndexKind::Auto,
    ]
}

fn kind_of(result: Result<IndexSettings>) -> Error {
    Error::from(result.unwrap_err())
}

#[test]
fn every_kind_round_trips_through_the_file() {
    let dir = TempDir::new().unwrap();
    assert_eq!(load(dir.path()).unwrap(), None);
    for kind in all_kinds() {
        let settings = IndexSettings {
            dimension: 128,
            kind,
        };
        save(dir.path(), &settings).unwrap();
        assert_eq!(load(dir.path()).unwrap(), Some(settings));
    }
}

#[test]
fn damage_is_a_checksum_mismatch_and_truncation_is_corrupt() {
    let settings = IndexSettings {
        dimension: 128,
        kind: IndexKind::Hnsw(hnsw()),
    };
    let bytes = encode(&settings).unwrap();
    let mut flipped = bytes.clone();
    *flipped.last_mut().unwrap() ^= 0x01;
    assert!(matches!(
        kind_of(decode(&flipped)),
        Error::ChecksumMismatch(_)
    ));
    assert!(matches!(
        kind_of(decode(&bytes[..bytes.len() - 1])),
        Error::Corrupt(_)
    ));
    let mut magic = bytes.clone();
    magic[0] = b'X';
    assert!(matches!(kind_of(decode(&magic)), Error::Corrupt(_)));
    let mut version = bytes;
    version[8] = 99;
    assert!(matches!(kind_of(decode(&version)), Error::Corrupt(_)));
}

type Edit = Box<dyn FnOnce(&mut Record)>;

fn record_with(edit: impl FnOnce(&mut Record)) -> Vec<u8> {
    let mut r = to_record(&IndexSettings {
        dimension: 128,
        kind: IndexKind::Flat,
    });
    edit(&mut r);
    format::encode(format::INDEX_SETTINGS, &r).unwrap()
}

#[test]
fn unknown_kinds_formats_and_bad_parameters_are_corrupt() {
    let cases: Vec<Edit> = vec![
        Box::new(|r| r.kind = 42),
        Box::new(|r| r.format = 2),
        Box::new(|r| r.dimension = 0),
        Box::new(|r| r.m = 16),   // flat with HNSW parameters
        Box::new(|r| r.kind = 1), // hnsw with m = 0
        Box::new(|r| {
            r.kind = 5;
            r.m = 16;
            r.ef_construction = 200;
            r.ef_search = 50;
            r.nlist = 4;
            r.nprobe = 5; // more probes than partitions
        }),
        Box::new(|r| {
            r.kind = 6;
            r.bits = 3;
        }),
    ];
    for edit in cases {
        let err = kind_of(decode(&record_with(edit)));
        assert!(matches!(err, Error::Corrupt(_)), "{err:?}");
    }
}
