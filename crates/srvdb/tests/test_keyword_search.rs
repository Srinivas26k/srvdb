//! BM25 keyword search (ADR 0015): scores against a reference computed
//! here, reopen, deletes, compaction and damaged index files.

use srvdb::{Error, Filter, SrvDB, Vector, VectorEngine};
use tempfile::TempDir;

const DIM: usize = 128;

const DOCS: [&str; 6] = [
    "The quick brown fox jumps over the lazy dog",
    "A quick brown dog outpaces a quick fox",
    "Lazy afternoons: the dog sleeps",
    "Rust error E0308: mismatched types",
    "Foxes and dogs, a field guide",
    "nothing relevant here",
];

fn meta(i: usize, text: &str) -> String {
    serde_json::json!({ "text": text, "group": i % 2 }).to_string()
}

fn vector(i: usize) -> Vector {
    Vector::new((0..DIM).map(|j| ((i * 3 + j) % 7) as f32 + 1.0).collect())
}

fn filled(dir: &TempDir) -> SrvDB {
    let mut db = SrvDB::new_with_config(dir.path().to_str().unwrap(), dim_config()).unwrap();
    let vectors: Vec<Vector> = (0..DOCS.len()).map(vector).collect();
    let metas: Vec<String> = DOCS.iter().enumerate().map(|(i, t)| meta(i, t)).collect();
    db.add_batch(&vectors, &metas).unwrap();
    db.enable_keyword_index("text").unwrap();
    db
}

fn dim_config() -> srvdb::DatabaseConfig {
    srvdb::DatabaseConfig::new(DIM).unwrap()
}

fn tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Textbook BM25 (k1 = 1.2, b = 0.75, Lucene idf) over `docs`
fn reference(docs: &[(u64, &str)], query: &str) -> Vec<(u64, f64)> {
    let docs: Vec<(u64, Vec<String>)> = docs.iter().map(|(id, t)| (*id, tokens(t))).collect();
    let n = docs.len() as f64;
    let avgdl = docs.iter().map(|d| d.1.len()).sum::<usize>() as f64 / n;
    let mut q = tokens(query);
    q.sort();
    q.dedup();
    let mut out = Vec::new();
    for (id, words) in &docs {
        let mut score = 0.0;
        for term in &q {
            let df = docs.iter().filter(|d| d.1.contains(term)).count() as f64;
            let tf = words.iter().filter(|w| *w == term).count() as f64;
            if tf == 0.0 {
                continue;
            }
            let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
            let dl = words.len() as f64;
            score += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * dl / avgdl));
        }
        if score > 0.0 {
            out.push((*id, score));
        }
    }
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    out
}

fn assert_matches(got: &[srvdb::SearchResult], want: &[(u64, f64)]) {
    assert_eq!(got.len(), want.len(), "{got:?} vs {want:?}");
    for (g, (id, score)) in got.iter().zip(want) {
        assert_eq!(g.id, *id);
        assert!(
            (f64::from(g.score) - score).abs() < 1e-4,
            "{} vs {score}",
            g.score
        );
    }
}

fn all_docs() -> Vec<(u64, &'static str)> {
    DOCS.iter()
        .enumerate()
        .map(|(i, t)| (i as u64, *t))
        .collect()
}

#[test]
fn bm25_matches_the_reference() {
    let dir = TempDir::new().unwrap();
    let db = filled(&dir);
    for query in [
        "quick fox",
        "lazy dog",
        "E0308",
        "dog dog fox QUICK",
        "absent",
    ] {
        let got = db.keyword_search(query, 10, None).unwrap();
        assert_matches(&got, &reference(&all_docs(), query));
    }
    let top = db.keyword_search("quick fox", 2, None).unwrap();
    assert_eq!(top.len(), 2);
    assert!(top[0].metadata.as_deref().unwrap().contains("quick"));
}

#[test]
fn filter_restricts_keyword_results() {
    let dir = TempDir::new().unwrap();
    let db = filled(&dir);
    let filter = Filter::from_json(r#"{"group": 1}"#).unwrap();
    let got = db.keyword_search("dog", 10, Some(&filter)).unwrap();
    assert!(!got.is_empty());
    assert!(got.iter().all(|r| !r.id.is_multiple_of(2)));
}

#[test]
fn invalid_input_and_disabled_index_are_rejected() {
    let dir = TempDir::new().unwrap();
    let mut db = SrvDB::new_with_config(dir.path().to_str().unwrap(), dim_config()).unwrap();
    let err = db.keyword_search("fox", 3, None).unwrap_err();
    assert!(matches!(err, Error::InvalidState(_)), "{err:?}");
    db.enable_keyword_index("text").unwrap();
    assert!(matches!(
        db.keyword_search("fox", 0, None),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        db.keyword_search(" ,.; ", 3, None),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        db.enable_keyword_index(""),
        Err(Error::InvalidInput(_))
    ));
    assert!(db.keyword_search("fox", 3, None).unwrap().is_empty());
}

#[test]
fn deletes_adds_and_reopen_keep_scores_exact() {
    let dir = TempDir::new().unwrap();
    let mut db = filled(&dir);
    db.delete(1).unwrap();
    db.add(&vector(9), &meta(9, "quick quick quick fox"))
        .unwrap();
    let mut live: Vec<(u64, &str)> = all_docs().into_iter().filter(|d| d.0 != 1).collect();
    live.push((6, "quick quick quick fox"));
    assert_matches(
        &db.keyword_search("quick fox", 10, None).unwrap(),
        &reference(&live, "quick fox"),
    );
    db.persist().unwrap();
    drop(db);

    let db = SrvDB::new_with_config(dir.path().to_str().unwrap(), dim_config()).unwrap();
    assert_eq!(db.keyword_field().unwrap().as_deref(), Some("text"));
    assert_matches(
        &db.keyword_search("quick fox", 10, None).unwrap(),
        &reference(&live, "quick fox"),
    );
}

#[test]
fn unpersisted_writes_are_dropped_on_reopen_and_lagging_file_catches_up() {
    let dir = TempDir::new().unwrap();
    let mut db = filled(&dir);
    db.persist().unwrap();
    // Not persisted: lost on reopen together with its metadata
    db.add(&vector(7), &meta(7, "zebra")).unwrap();
    drop(db);
    let mut db = SrvDB::new_with_config(dir.path().to_str().unwrap(), dim_config()).unwrap();
    assert!(db.keyword_search("zebra", 3, None).unwrap().is_empty());

    // Persisted metadata, but an index file from before the add (as after
    // a crash between the metadata commit and the index write)
    let old = std::fs::read(dir.path().join("keyword.index")).unwrap();
    db.add(&vector(8), &meta(8, "zebra")).unwrap();
    db.delete(0).unwrap();
    db.persist().unwrap();
    drop(db);
    std::fs::write(dir.path().join("keyword.index"), old).unwrap();
    let db = SrvDB::new_with_config(dir.path().to_str().unwrap(), dim_config()).unwrap();
    let zebra = db.keyword_search("zebra", 3, None).unwrap();
    assert_eq!(zebra.iter().map(|r| r.id).collect::<Vec<_>>(), [6]);
    assert!(db.keyword_search("jumps", 3, None).unwrap().is_empty());
}

#[test]
fn compaction_renumbers_the_keyword_index() {
    let dir = TempDir::new().unwrap();
    let mut db = filled(&dir);
    db.persist().unwrap();
    db.delete(0).unwrap();
    db.delete(2).unwrap();
    assert_eq!(db.compact().unwrap(), 2);
    let live: Vec<(u64, &str)> = [1usize, 3, 4, 5]
        .iter()
        .enumerate()
        .map(|(new, old)| (new as u64, DOCS[*old]))
        .collect();
    assert_matches(
        &db.keyword_search("dog fox", 10, None).unwrap(),
        &reference(&live, "dog fox"),
    );
    drop(db);
    let db = SrvDB::new_with_config(dir.path().to_str().unwrap(), dim_config()).unwrap();
    assert_matches(
        &db.keyword_search("dog fox", 10, None).unwrap(),
        &reference(&live, "dog fox"),
    );
}

#[test]
fn damaged_index_file_is_reported_and_rebuilt_on_enable() {
    let dir = TempDir::new().unwrap();
    let mut db = filled(&dir);
    db.persist().unwrap();
    drop(db);
    let path = dir.path().join("keyword.index");
    let mut bytes = std::fs::read(&path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0xFF;
    std::fs::write(&path, &bytes).unwrap();

    let db = SrvDB::new_with_config(dir.path().to_str().unwrap(), dim_config()).unwrap();
    let err = db.keyword_search("fox", 3, None).unwrap_err();
    assert!(matches!(err, Error::ChecksumMismatch(_)), "{err:?}");
    drop(db);
    std::fs::write(&path, b"garbage").unwrap();
    let mut db = SrvDB::new_with_config(dir.path().to_str().unwrap(), dim_config()).unwrap();
    assert!(matches!(
        db.keyword_search("fox", 3, None),
        Err(Error::Corrupt(_))
    ));
    db.enable_keyword_index("text").unwrap();
    assert_matches(
        &db.keyword_search("fox", 10, None).unwrap(),
        &reference(&all_docs(), "fox"),
    );
    db.persist().unwrap();
    drop(db);
    let db = SrvDB::new_with_config(dir.path().to_str().unwrap(), dim_config()).unwrap();
    assert_matches(
        &db.keyword_search("fox", 10, None).unwrap(),
        &reference(&all_docs(), "fox"),
    );
}

#[test]
fn another_field_rebuilds_the_index() {
    let dir = TempDir::new().unwrap();
    let mut db = filled(&dir);
    db.add(&vector(1), r#"{"title": ["fox", "den"]}"#).unwrap();
    db.enable_keyword_index("title").unwrap();
    let got = db.keyword_search("fox", 10, None).unwrap();
    assert_eq!(got.iter().map(|r| r.id).collect::<Vec<_>>(), [6]);
}
