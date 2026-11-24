//! Unit tests for filter parsing and evaluation

use super::Filter;
use crate::Error;

fn f(json: &str) -> Filter {
    Filter::from_json(json).expect("valid filter")
}

fn rejected(json: &str) -> String {
    match Filter::from_json(json) {
        Err(Error::InvalidInput(msg)) => msg,
        other => panic!("expected InvalidInput for {json}, got {other:?}"),
    }
}

#[test]
fn equality_and_numbers_by_value() {
    let filter = f(r#"{"genre": "jazz", "year": 1959}"#);
    assert!(filter.matches(r#"{"genre": "jazz", "year": 1959.0}"#));
    assert!(!filter.matches(r#"{"genre": "jazz", "year": 1960}"#));
    assert!(!filter.matches(r#"{"genre": "jazz"}"#));
    assert!(f(r#"{"flag": true}"#).matches(r#"{"flag": true}"#));
    assert!(!f(r#"{"flag": true}"#).matches(r#"{"flag": 1}"#));
}

#[test]
fn comparison_operators() {
    let filter = f(r#"{"year": {"$gte": 1950, "$lt": 1960}}"#);
    assert!(filter.matches(r#"{"year": 1950}"#));
    assert!(filter.matches(r#"{"year": 1959.5}"#));
    assert!(!filter.matches(r#"{"year": 1960}"#));
    assert!(!filter.matches(r#"{"year": "1955"}"#));
    assert!(f(r#"{"n": {"$gt": "b"}}"#).matches(r#"{"n": "c"}"#));
    assert!(f(r#"{"n": {"$lte": 2}}"#).matches(r#"{"n": 2}"#));
}

#[test]
fn membership_and_missing_fields() {
    let inn = f(r#"{"tag": {"$in": ["a", "b"]}}"#);
    let nin = f(r#"{"tag": {"$nin": ["a", "b"]}}"#);
    let ne = f(r#"{"tag": {"$ne": "a"}}"#);
    assert!(inn.matches(r#"{"tag": "b"}"#));
    assert!(!inn.matches(r#"{}"#));
    assert!(nin.matches(r#"{"tag": "c"}"#));
    assert!(nin.matches(r#"{}"#));
    assert!(ne.matches(r#"{}"#));
    assert!(!ne.matches(r#"{"tag": "a"}"#));
    assert!(!f(r#"{"tag": {"$in": []}}"#).matches(r#"{"tag": "a"}"#));
}

#[test]
fn logic_and_nested_paths() {
    let filter = f(r#"{"$or": [{"meta.lang": "en"}, {"$not": {"meta.year": {"$lt": 2000}}}]}"#);
    assert!(filter.matches(r#"{"meta": {"lang": "en", "year": 1990}}"#));
    assert!(filter.matches(r#"{"meta": {"lang": "de", "year": 2005}}"#));
    assert!(!filter.matches(r#"{"meta": {"lang": "de", "year": 1990}}"#));
    let and = f(r#"{"$and": [{"a": 1}, {"b": 2}]}"#);
    assert!(and.matches(r#"{"a": 1, "b": 2}"#));
    assert!(!and.matches(r#"{"a": 1, "b": 3}"#));
}

#[test]
fn invalid_metadata_matches_nothing() {
    assert!(!f(r#"{"a": {"$ne": 1}}"#).matches("not json"));
}

#[test]
fn bad_filters_are_rejected_with_a_reason() {
    assert!(rejected("[1]").contains("JSON object"));
    assert!(rejected("{").contains("not valid JSON"));
    assert!(rejected("{}").contains("at least one condition"));
    assert!(rejected(r#"{"$xor": []}"#).contains("unknown filter operator '$xor'"));
    assert!(rejected(r#"{"a": {"$regex": "x"}}"#).contains("unknown operator '$regex'"));
    assert!(rejected(r#"{"a": {}}"#).contains("empty operator object"));
    assert!(rejected(r#"{"a": [1, 2]}"#).contains("$in"));
    assert!(rejected(r#"{"a": {"$gt": true}}"#).contains("number or a string"));
    assert!(rejected(r#"{"a": {"$eq": [1]}}"#).contains("$eq"));
    assert!(rejected(r#"{"a": {"$in": 3}}"#).contains("list"));
    assert!(rejected(r#"{"$and": []}"#).contains("non-empty list"));
    assert!(rejected(r#"{"a..b": 1}"#).contains("empty part"));
}

#[test]
fn messages_do_not_echo_operand_values() {
    let msg = rejected(r#"{"a": {"$gt": ["secret-value-x"]}}"#);
    assert!(!msg.contains("secret-value-x"));
}

#[test]
fn large_integers_compare_exactly() {
    // 2^53 + 1 is not representable as f64; it would equal 2^53 there
    let big = r#"{"n": 9007199254740993}"#;
    let eq = f(big);
    assert!(eq.matches(big));
    assert!(!eq.matches(r#"{"n": 9007199254740992}"#));
    assert!(!eq.matches(r#"{"n": 9007199254740992.0}"#));
    assert!(f(r#"{"n": {"$gt": 9007199254740992}}"#).matches(big));
    let inn = f(r#"{"n": {"$in": [9007199254740993]}}"#);
    assert!(inn.matches(big));
    assert!(!inn.matches(r#"{"n": 9007199254740992}"#));
    let max = r#"{"n": 18446744073709551615}"#;
    assert!(f(r#"{"n": {"$in": [18446744073709551615]}}"#).matches(max));
    let lt = f(r#"{"n": {"$lt": -9223372036854775807}}"#);
    assert!(lt.matches(r#"{"n": -9223372036854775808}"#));
}

#[test]
fn membership_uses_the_same_equality_as_eq() {
    let inn = f(r#"{"n": {"$in": [1, 2.5, "x", true, null]}}"#);
    let hits = ["1.0", "2.5", "\"x\"", "true", "null"];
    for hit in hits {
        assert!(inn.matches(&format!(r#"{{"n": {hit}}}"#)), "{hit}");
    }
    for miss in ["1.5", "\"1\"", "false", "[1]"] {
        assert!(!inn.matches(&format!(r#"{{"n": {miss}}}"#)), "{miss}");
    }
    assert!(f(r#"{"n": {"$nin": [1]}}"#).matches(r#"{"n": [1]}"#));
}

fn join(items: impl Iterator<Item = String>) -> String {
    items.collect::<Vec<_>>().join(",")
}

#[test]
fn list_and_node_caps_are_enforced() {
    let values = |n: usize| join((0..n).map(|i| i.to_string()));
    let max = Filter::MAX_LIST_LEN;
    let ok = format!(r#"{{"a": {{"$in": [{}]}}}}"#, values(max));
    assert!(Filter::from_json(&ok).is_ok());
    let msg = rejected(&format!(r#"{{"a": {{"$nin": [{}]}}}}"#, values(max + 1)));
    assert!(msg.contains("at most 10000"), "{msg}");

    let conds = |n: usize| join((0..n).map(|i| format!(r#"{{"a": {i}}}"#)));
    // The $or node itself counts as one condition
    let ok = format!(r#"{{"$or": [{}]}}"#, conds(Filter::MAX_NODES - 1));
    assert!(Filter::from_json(&ok).is_ok());
    let msg = rejected(&format!(r#"{{"$or": [{}]}}"#, conds(Filter::MAX_NODES)));
    assert!(msg.contains("at most 1000 are allowed"), "{msg}");
    let msg = rejected(&format!(
        r#"{{"$and": [{}]}}"#,
        conds(Filter::MAX_NODES + 1)
    ));
    assert!(msg.contains("at most 1000 conditions"), "{msg}");
}

#[test]
fn debug_redacts_operand_values() {
    let filter = f(r#"{"a.b": "secret-1", "n": {"$in": ["secret-2"]}, "$not": {"c": {"$gt": 7}}}"#);
    let shown = format!("{filter:?}");
    assert!(!shown.contains("secret"), "{shown}");
    assert!(!shown.contains('7'), "{shown}");
    assert!(shown.contains("a.b $eq <redacted>"), "{shown}");
    assert!(shown.contains("n $in <redacted>"), "{shown}");
}
