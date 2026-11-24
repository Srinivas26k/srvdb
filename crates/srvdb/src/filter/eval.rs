//! Testing a parsed filter against one metadata value

use super::{Cond, Node};
use serde_json::Value;
use std::cmp::Ordering;

impl Node {
    pub(super) fn eval(&self, meta: &Value) -> bool {
        match self {
            Node::And(nodes) => nodes.iter().all(|n| n.eval(meta)),
            Node::Or(nodes) => nodes.iter().any(|n| n.eval(meta)),
            Node::Not(node) => !node.eval(meta),
            Node::Field { path, cond } => cond.eval(lookup(meta, path)),
        }
    }
}

impl Cond {
    /// `field` is None when the record has no such field
    fn eval(&self, field: Option<&Value>) -> bool {
        let Some(a) = field else {
            return matches!(self, Cond::Ne(_) | Cond::Nin(_));
        };
        match self {
            Cond::Eq(b) => equal(a, b),
            Cond::Ne(b) => !equal(a, b),
            Cond::Gt(b) => compare(a, b) == Some(Ordering::Greater),
            Cond::Gte(b) => matches!(compare(a, b), Some(Ordering::Greater | Ordering::Equal)),
            Cond::Lt(b) => compare(a, b) == Some(Ordering::Less),
            Cond::Lte(b) => matches!(compare(a, b), Some(Ordering::Less | Ordering::Equal)),
            Cond::In(list) => list.iter().any(|b| equal(a, b)),
            Cond::Nin(list) => !list.iter().any(|b| equal(a, b)),
        }
    }
}

fn lookup<'a>(meta: &'a Value, path: &[String]) -> Option<&'a Value> {
    path.iter().try_fold(meta, |v, key| v.as_object()?.get(key))
}

/// Numbers compare by value (`1 == 1.0`); everything else structurally
fn equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x.as_f64() == y.as_f64(),
        _ => a == b,
    }
}

/// Numbers with numbers, strings with strings; other pairs are unordered
fn compare(a: &Value, b: &Value) -> Option<Ordering> {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x.as_f64()?.partial_cmp(&y.as_f64()?),
        (Value::String(x), Value::String(y)) => Some(x.cmp(y)),
        _ => None,
    }
}
