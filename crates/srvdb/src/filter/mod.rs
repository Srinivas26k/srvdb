//! Metadata filters: a fixed set of operators over JSON metadata
//!
//! A [`Filter`] is parsed once from a JSON object and then tested against
//! the metadata of each vector. The grammar (ADR 0014):
//!
//! - `{"field": value}`: the field equals `value` (string, number, bool
//!   or null; numbers compare by value, so `1` equals `1.0`)
//! - `{"field": {"$op": value, ...}}` with `$eq`, `$ne`, `$gt`, `$gte`,
//!   `$lt`, `$lte`, `$in`, `$nin`; several operators on one field must
//!   all hold
//! - `{"$and": [filter, ...]}`, `{"$or": [filter, ...]}`, `{"$not": filter}`
//! - several keys in one object must all hold
//! - `"a.b"` names the field `b` inside the object `a`
//!
//! A missing field matches only `$ne` and `$nin`. `$gt`, `$gte`, `$lt` and
//! `$lte` compare numbers with numbers and strings with strings; any other
//! pair does not match.

mod eval;
mod parse;
#[cfg(test)]
mod tests;

use serde_json::Value;

/// Operator names accepted on a field, in the order error messages list them
pub(crate) const FIELD_OPS: &[&str] = &["$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin"];
/// Operator names accepted at the top of a filter object
pub(crate) const LOGIC_OPS: &[&str] = &["$and", "$or", "$not"];

/// A validated metadata filter; build it with [`Filter::from_json`].
///
/// Cheap to test against many records and safe to share between threads.
#[derive(Debug, Clone, PartialEq)]
pub struct Filter {
    root: Node,
}

#[derive(Debug, Clone, PartialEq)]
enum Node {
    And(Vec<Node>),
    Or(Vec<Node>),
    Not(Box<Node>),
    Field { path: Vec<String>, cond: Cond },
}

#[derive(Debug, Clone, PartialEq)]
enum Cond {
    Eq(Value),
    Ne(Value),
    Gt(Value),
    Gte(Value),
    Lt(Value),
    Lte(Value),
    In(Vec<Value>),
    Nin(Vec<Value>),
}

impl Filter {
    /// Parse a filter from a JSON object (grammar in the module docs).
    ///
    /// # Errors
    /// [`crate::Error::InvalidInput`] if the text is not a JSON object, an
    /// object or `$and` / `$or` list is empty, an operator is not in the
    /// fixed set, a field name has an empty part, or an operand has the
    /// wrong type (for example a list given to `$eq`, or a bool to `$gt`).
    /// Messages name the operator or field, never the operand values.
    pub fn from_json(json: &str) -> crate::Result<Self> {
        let value: Value = serde_json::from_str(json)
            .map_err(|e| crate::Error::invalid_input(format!("filter is not valid JSON: {e}")))?;
        Ok(Self {
            root: parse::object(&value)?,
        })
    }

    /// Whether the metadata JSON text satisfies the filter. Text that is
    /// not valid JSON matches nothing (so `$not` of anything is false).
    pub fn matches(&self, metadata: &str) -> bool {
        serde_json::from_str::<Value>(metadata).is_ok_and(|v| self.root.eval(&v))
    }
}
