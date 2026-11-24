//! Metadata filters: a fixed set of operators over JSON metadata
//!
//! A [`Filter`] is parsed once from a JSON object and then tested against
//! the metadata of each vector. The grammar (ADR 0014):
//!
//! - `{"field": value}`: the field equals `value` (string, number, bool
//!   or null; integers compare exactly, and `1` equals `1.0`)
//! - `{"field": {"$op": value, ...}}` with `$eq`, `$ne`, `$gt`, `$gte`,
//!   `$lt`, `$lte`, `$in`, `$nin`; several operators on one field must
//!   all hold
//! - `{"$and": [filter, ...]}`, `{"$or": [filter, ...]}`, `{"$not": filter}`
//! - several keys in one object must all hold
//! - `"a.b"` names the field `b` inside the object `a`
//!
//! A missing field matches only `$ne` and `$nin`. `$gt`, `$gte`, `$lt` and
//! `$lte` compare numbers with numbers and strings with strings; any other
//! pair does not match. A filter holds at most [`Filter::MAX_NODES`]
//! conditions and each `$in` / `$nin` list at most [`Filter::MAX_LIST_LEN`]
//! values.

mod eval;
mod parse;
#[cfg(test)]
mod tests;
mod value;

use std::collections::HashSet;
use std::fmt;

use serde_json::Value;
use value::Key;

/// Operator names accepted on a field, in the order error messages list them
pub(crate) const FIELD_OPS: &[&str] = &["$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin"];
/// Operator names accepted at the top of a filter object
pub(crate) const LOGIC_OPS: &[&str] = &["$and", "$or", "$not"];
const MAX_NODES: usize = Filter::MAX_NODES;
const MAX_LIST_LEN: usize = Filter::MAX_LIST_LEN;

/// A validated metadata filter; build it with [`Filter::from_json`].
///
/// Cheap to test against many records and safe to share between threads.
/// `Debug` shows the structure (fields and operators) but never the
/// operand values.
#[derive(Clone, PartialEq)]
pub struct Filter {
    root: Node,
}

#[derive(Clone, PartialEq)]
enum Node {
    And(Vec<Node>),
    Or(Vec<Node>),
    Not(Box<Node>),
    Field { path: Vec<String>, cond: Cond },
}

#[derive(Clone, PartialEq)]
enum Cond {
    Eq(Value),
    Ne(Value),
    Gt(Value),
    Gte(Value),
    Lt(Value),
    Lte(Value),
    In(HashSet<Key>),
    Nin(HashSet<Key>),
}

impl Filter {
    /// Most conditions (field tests plus `$and` / `$or` / `$not`) in one filter
    pub const MAX_NODES: usize = 1_000;
    /// Most values in one `$in` or `$nin` list
    pub const MAX_LIST_LEN: usize = 10_000;

    /// Parse a filter from a JSON object (grammar in the module docs).
    ///
    /// # Errors
    /// [`crate::Error::InvalidInput`] if the text is not a JSON object, an
    /// object or `$and` / `$or` list is empty, an operator is not in the
    /// fixed set, a field name has an empty part, an operand has the wrong
    /// type (for example a list given to `$eq`, or a bool to `$gt`), or the
    /// filter exceeds [`Filter::MAX_NODES`] or [`Filter::MAX_LIST_LEN`].
    /// Messages name the operator or field, never the operand values.
    pub fn from_json(json: &str) -> crate::Result<Self> {
        let value: Value = serde_json::from_str(json)
            .map_err(|e| crate::Error::invalid_input(format!("filter is not valid JSON: {e}")))?;
        Ok(Self {
            root: parse::filter(&value)?,
        })
    }

    /// Whether the metadata JSON text satisfies the filter. Text that is
    /// not valid JSON matches nothing (so `$not` of anything is false).
    pub fn matches(&self, metadata: &str) -> bool {
        serde_json::from_str::<Value>(metadata).is_ok_and(|v| self.root.eval(&v))
    }
}

impl fmt::Debug for Filter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Filter").field(&self.root).finish()
    }
}

/// Structure only: operand values may be user data and are never shown
impl fmt::Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Node::And(n) => f.debug_tuple("$and").field(n).finish(),
            Node::Or(n) => f.debug_tuple("$or").field(n).finish(),
            Node::Not(n) => f.debug_tuple("$not").field(n).finish(),
            Node::Field { path, cond } => {
                let op = match cond {
                    Cond::Eq(_) => "$eq",
                    Cond::Ne(_) => "$ne",
                    Cond::Gt(_) => "$gt",
                    Cond::Gte(_) => "$gte",
                    Cond::Lt(_) => "$lt",
                    Cond::Lte(_) => "$lte",
                    Cond::In(_) => "$in",
                    Cond::Nin(_) => "$nin",
                };
                write!(f, "{} {op} <redacted>", path.join("."))
            }
        }
    }
}
