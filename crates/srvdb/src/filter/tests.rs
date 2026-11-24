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
