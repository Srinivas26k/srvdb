//! Parsing and validating a filter object into a `Node` tree

use super::value::key;
use super::{Cond, Node, FIELD_OPS, LOGIC_OPS, MAX_LIST_LEN, MAX_NODES};
use crate::error::{Error, Result};
use serde_json::{Map, Value};
use std::collections::HashSet;

/// Parse a whole filter, rejecting one with more than `MAX_NODES` conditions
pub(super) fn filter(value: &Value) -> Result<Node> {
    let root = object(value)?;
    let nodes = root.size();
    if nodes > MAX_NODES {
        return Err(Error::invalid_input(format!(
            "filter has {nodes} conditions; at most {MAX_NODES} are allowed \
             (use $in for many values of one field)"
        )));
    }
    Ok(root)
}

impl Node {
    /// Number of nodes in the tree
    fn size(&self) -> usize {
        match self {
            Node::And(n) | Node::Or(n) => 1 + n.iter().map(Node::size).sum::<usize>(),
            Node::Not(n) => 1 + n.size(),
            Node::Field { .. } => 1,
        }
    }
}

/// Parse one filter object; several keys are combined with AND
fn object(value: &Value) -> Result<Node> {
    let Value::Object(map) = value else {
        return Err(Error::invalid_input(
            "a filter must be a JSON object such as {\"field\": value}",
        ));
    };
    if map.is_empty() {
        return Err(Error::invalid_input(
            "a filter object must hold at least one condition; pass no filter to match everything",
        ));
    }
    let nodes = map
        .iter()
        .map(|(key, value)| entry(key, value))
        .collect::<Result<Vec<_>>>()?;
    Ok(all(nodes))
}

/// One node, or the AND of several
fn all(mut nodes: Vec<Node>) -> Node {
    if nodes.len() == 1 {
        nodes.remove(0)
    } else {
        Node::And(nodes)
    }
}

fn entry(key: &str, value: &Value) -> Result<Node> {
    match key {
        "$and" => Ok(Node::And(list(key, value)?)),
        "$or" => Ok(Node::Or(list(key, value)?)),
        "$not" => Ok(Node::Not(Box::new(object(value)?))),
        _ if key.starts_with('$') => Err(Error::invalid_input(format!(
            "unknown filter operator '{key}' at the top level; use one of {} or a field name",
            LOGIC_OPS.join(", ")
        ))),
        _ => field(key, value),
    }
}

/// The operand of `$and` / `$or`: a non-empty list of filter objects
fn list(op: &str, value: &Value) -> Result<Vec<Node>> {
    match value {
        // Checked before parsing the items, so a huge list costs nothing
        Value::Array(items) if items.len() > MAX_NODES => Err(Error::invalid_input(format!(
            "{op} has {} filters; at most {MAX_NODES} conditions are allowed",
            items.len()
        ))),
        Value::Array(items) if !items.is_empty() => items.iter().map(object).collect(),
        _ => Err(Error::invalid_input(format!(
            "{op} needs a non-empty list of filter objects"
        ))),
    }
}

fn field(name: &str, value: &Value) -> Result<Node> {
    let path: Vec<String> = name.split('.').map(str::to_string).collect();
    if path.iter().any(String::is_empty) {
        return Err(Error::invalid_input(format!(
            "filter field name '{name}' has an empty part; use names like \"a\" or \"a.b\""
        )));
    }
    let conds = match value {
        Value::Object(ops) => operators(name, ops)?,
        Value::Array(_) => {
            return Err(Error::invalid_input(format!(
                "filter field '{name}' is given a list; use {{\"$in\": [...]}} to match any of several values"
            )))
        }
        other => vec![Cond::Eq(other.clone())],
    };
    Ok(all(conds
        .into_iter()
        .map(|cond| Node::Field {
            path: path.clone(),
            cond,
        })
        .collect()))
}

fn operators(name: &str, ops: &Map<String, Value>) -> Result<Vec<Cond>> {
    if ops.is_empty() {
        return Err(Error::invalid_input(format!(
            "filter field '{name}' has an empty operator object; use one of {}",
            FIELD_OPS.join(", ")
        )));
    }
    ops.iter().map(|(op, v)| operator(name, op, v)).collect()
}

fn operator(name: &str, op: &str, v: &Value) -> Result<Cond> {
    let bad =
        |want: &str| Error::invalid_input(format!("{op} on filter field '{name}' needs {want}"));
    let scalar = |v: &Value| !matches!(v, Value::Array(_) | Value::Object(_));
    let ordered = matches!(v, Value::Number(_) | Value::String(_));
    let values = || match v {
        Value::Array(items) if items.len() > MAX_LIST_LEN => Err(Error::invalid_input(format!(
            "{op} on filter field '{name}' has {} values; at most {MAX_LIST_LEN} are allowed",
            items.len()
        ))),
        Value::Array(items) => items
            .iter()
            .map(key)
            .collect::<Option<HashSet<_>>>()
            .ok_or_else(|| bad("a list of strings, numbers, bools or nulls")),
        _ => Err(bad("a list of strings, numbers, bools or nulls")),
    };
    Ok(match op {
        "$eq" | "$ne" if !scalar(v) => return Err(bad("a string, number, bool or null")),
        "$eq" => Cond::Eq(v.clone()),
        "$ne" => Cond::Ne(v.clone()),
        "$gt" | "$gte" | "$lt" | "$lte" if !ordered => return Err(bad("a number or a string")),
        "$gt" => Cond::Gt(v.clone()),
        "$gte" => Cond::Gte(v.clone()),
        "$lt" => Cond::Lt(v.clone()),
        "$lte" => Cond::Lte(v.clone()),
        "$in" => Cond::In(values()?),
        "$nin" => Cond::Nin(values()?),
        _ => {
            return Err(Error::invalid_input(format!(
                "unknown operator '{op}' on filter field '{name}'; use one of {}",
                FIELD_OPS.join(", ")
            )))
        }
    })
}
