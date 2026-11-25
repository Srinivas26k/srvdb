//! Exact comparison and hashing of scalar JSON values
//!
//! Integers compare as integers (so 2^53 + 1 differs from 2^53); a float
//! with no fractional part equals the same integer; other floats compare as
//! `f64`. `Key` gives every scalar one hashable form with the same equality,
//! so `$in` / `$nin` membership is a hash lookup.

use serde_json::{Number, Value};
use std::cmp::Ordering;

/// Hashable form of a scalar; equal keys mean equal values
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) enum Key {
    Null,
    Bool(bool),
    Int(i128),
    /// A float with a fractional part (or beyond the i128 range), as bits
    Float(u64),
    Str(String),
}

/// Floats below this magnitude with no fractional part convert to i128 exactly
const I128_LIMIT: f64 = 1.7e38;

fn number(n: &Number) -> Key {
    if let Some(i) = n.as_i64() {
        return Key::Int(i.into());
    }
    if let Some(u) = n.as_u64() {
        return Key::Int(u.into());
    }
    // serde_json numbers are always i64, u64 or a finite f64
    let f = n.as_f64().unwrap_or(0.0);
    if f.fract() == 0.0 && f.abs() < I128_LIMIT {
        // Exact: |f| < 2^127 and f is integral
        Key::Int(f as i128)
    } else {
        Key::Float(f.to_bits())
    }
}

/// The key of a scalar, or None for an array or object
pub(super) fn key(v: &Value) -> Option<Key> {
    Some(match v {
        Value::Null => Key::Null,
        Value::Bool(b) => Key::Bool(*b),
        Value::Number(n) => number(n),
        Value::String(s) => Key::Str(s.clone()),
        Value::Array(_) | Value::Object(_) => return None,
    })
}

/// Scalars by key; arrays and objects never equal a filter operand
pub(super) fn equal(a: &Value, b: &Value) -> bool {
    matches!((key(a), key(b)), (Some(x), Some(y)) if x == y)
}

/// Numbers with numbers, strings with strings; other pairs are unordered
pub(super) fn compare(a: &Value, b: &Value) -> Option<Ordering> {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => match (number(x), number(y)) {
            (Key::Int(i), Key::Int(j)) => Some(i.cmp(&j)),
            (Key::Float(f), Key::Float(g)) => f64::from_bits(f).partial_cmp(&f64::from_bits(g)),
            // A fractional or huge float never ties with an integer, and
            // every float above 2^53 is integral, so f64 ordering is exact
            (Key::Int(i), Key::Float(g)) => (i as f64).partial_cmp(&f64::from_bits(g)),
            (Key::Float(f), Key::Int(j)) => f64::from_bits(f).partial_cmp(&(j as f64)),
            _ => None,
        },
        (Value::String(x), Value::String(y)) => Some(x.cmp(y)),
        _ => None,
    }
}
