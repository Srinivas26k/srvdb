//! Tokenizer and field extraction for keyword search (ADR 0015)

/// Runs longer than this many characters are dropped (hashes, base64)
pub(crate) const MAX_TOKEN_CHARS: usize = 64;

/// Split `text` into lowercase tokens: maximal runs of alphanumeric
/// characters (Unicode), each lowercased; overlong runs are dropped.
pub(crate) fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|run| !run.is_empty() && run.chars().count() <= MAX_TOKEN_CHARS)
        .map(str::to_lowercase)
        .collect()
}

/// The text of `field` in a metadata JSON object, or `None` if the record
/// is not JSON, lacks the field or the value is not a string or a list of
/// strings. `"a.b"` names a nested field.
pub(crate) fn field_text(meta: &str, field: &str) -> Option<String> {
    // A record that is not JSON has no field to index; that is not an error
    let value: serde_json::Value = serde_json::from_str(meta).ok()?;
    let mut node = &value;
    for part in field.split('.') {
        node = node.as_object()?.get(part)?;
    }
    match node {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Array(items) => {
            let parts: Option<Vec<&str>> = items.iter().map(|v| v.as_str()).collect();
            Some(parts?.join(" "))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_on_non_alphanumeric_and_lowercases() {
        assert_eq!(
            tokenize("Hello, WORLD! error_code-42"),
            ["hello", "world", "error", "code", "42"]
        );
    }

    #[test]
    fn is_unicode_aware() {
        assert_eq!(tokenize("Ärger über Straße"), ["ärger", "über", "straße"]);
        assert_eq!(tokenize("ΣΟΦΙΑ мир"), ["σοφια", "мир"]);
    }

    #[test]
    fn drops_overlong_runs() {
        let long = "a".repeat(MAX_TOKEN_CHARS + 1);
        assert_eq!(tokenize(&format!("x {long} y")), ["x", "y"]);
    }

    #[test]
    fn reads_strings_lists_and_nested_fields() {
        assert_eq!(field_text(r#"{"t":"a b"}"#, "t").as_deref(), Some("a b"));
        assert_eq!(
            field_text(r#"{"t":["a","b"]}"#, "t").as_deref(),
            Some("a b")
        );
        assert_eq!(
            field_text(r#"{"d":{"t":"x"}}"#, "d.t").as_deref(),
            Some("x")
        );
        assert_eq!(field_text(r#"{"t":3}"#, "t"), None);
        assert_eq!(field_text(r#"{"t":["a",1]}"#, "t"), None);
        assert_eq!(field_text("not json", "t"), None);
    }
}
