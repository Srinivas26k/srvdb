//! Parsing and validating a filter object into a `Node` tree

use super::{Cond, Node, FIELD_OPS, LOGIC_OPS};
use crate::error::{Error, Result};
use serde_json::{Map, Value};

/// Parse one filter object; several keys are combined with AND
pub(super) fn object(value: &Value) -> Result<Node> {
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
        Value::Array(items) if items.iter().all(scalar) => Ok(items.clone()),
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
