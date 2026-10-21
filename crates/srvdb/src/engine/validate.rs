//! Input validation for the public API
//!
//! Every write and query is checked here before it reaches storage, so
//! bad input fails with a clear error instead of corrupting results.

use crate::error::{Error, Result};

/// Reject a vector whose length differs from the database dimension or
/// that contains NaN or infinity (which would break similarity ordering).
pub(super) fn vector(values: &[f32], dimension: usize) -> Result<()> {
    if values.len() != dimension {
        return Err(Error::DimensionMismatch {
            expected: dimension,
            actual: values.len(),
        });
    }
    if let Some(position) = values.iter().position(|v| !v.is_finite()) {
        return Err(Error::invalid_input(format!(
            "vector contains NaN or infinity at position {position}"
        )));
    }
    Ok(())
}

/// `k` must ask for at least one result.
pub(super) fn k(k: usize) -> Result<()> {
    if k == 0 {
        return Err(Error::invalid_input("k must be at least 1"));
    }
    Ok(())
}

/// Metadata is stored as JSON text and must parse as JSON.
pub(super) fn metadata(meta: &str) -> Result<()> {
    serde_json::from_str::<serde::de::IgnoredAny>(meta)
        .map(|_| ())
        .map_err(|e| Error::invalid_input(format!("metadata is not valid JSON: {e}")))
}

/// Validate a whole batch before any of it is written.
pub(super) fn batch(vectors: &[&[f32]], metas: &[String], dimension: usize) -> Result<()> {
    if vectors.len() != metas.len() {
        return Err(Error::invalid_input(format!(
            "got {} vectors but {} metadata entries; they must match",
            vectors.len(),
            metas.len()
        )));
    }
    for (i, (values, meta)) in vectors.iter().zip(metas).enumerate() {
        vector(values, dimension).map_err(|e| at_index(e, i))?;
        metadata(meta).map_err(|e| at_index(e, i))?;
    }
    Ok(())
}

/// Prefix an error with the batch position it came from.
fn at_index(err: Error, index: usize) -> Error {
    match err {
        Error::InvalidInput(msg) => Error::InvalidInput(format!("item {index}: {msg}")),
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_wrong_dimension() {
        let err = vector(&[0.0; 3], 4).unwrap_err();
        assert!(matches!(
            err,
            Error::DimensionMismatch {
                expected: 4,
                actual: 3
            }
        ));
    }

    #[test]
    fn rejects_nan_and_infinity() {
        assert!(matches!(
            vector(&[0.0, f32::NAN], 2),
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(
            vector(&[f32::INFINITY, 0.0], 2),
            Err(Error::InvalidInput(_))
        ));
        assert!(vector(&[0.5, -0.5], 2).is_ok());
    }

    #[test]
    fn rejects_zero_k() {
        assert!(k(0).is_err());
        assert!(k(1).is_ok());
    }

    #[test]
    fn metadata_must_be_json() {
        assert!(metadata(r#"{"a": 1}"#).is_ok());
        assert!(metadata("[1, 2]").is_ok());
        assert!(metadata("not json").is_err());
    }

    #[test]
    fn batch_reports_failing_item() {
        let good = [0.0f32; 2];
        let bad = [f32::NAN, 0.0];
        let vectors: Vec<&[f32]> = vec![&good, &bad];
        let metas = vec!["{}".to_string(), "{}".to_string()];
        let err = batch(&vectors, &metas, 2).unwrap_err().to_string();
        assert!(err.contains("item 1"), "{err}");
        assert!(batch(&vectors, &metas[..1], 2).is_err());
    }
}
