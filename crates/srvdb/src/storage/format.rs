//! Envelope for every serialized SrvDB file (ADR 0003)
//!
//! Layout, little-endian:
//!
//! | offset | size | field                                  |
//! |--------|------|----------------------------------------|
//! | 0      | 4    | magic `SRVD`                           |
//! | 4      | 4    | kind (`HNSW`, `IVFX`, `SQ8Q`, `PQCB`)  |
//! | 8      | 2    | format version of the payload          |
//! | 10     | 2    | reserved (zero)                        |
//! | 12     | 8    | payload length                         |
//! | 20     | 4    | CRC32 of the payload                   |
//! | 24     | ...  | payload (postcard)                     |
//!
//! Anything that does not match (wrong magic or kind, newer version,
//! truncation, checksum mismatch, undecodable payload) is `Error::Corrupt`.

use crate::error::Error;
use anyhow::Result;
use serde::{de::DeserializeOwned, Serialize};

pub(crate) const MAGIC: &[u8; 4] = b"SRVD";
const HEADER: usize = 24;

pub(crate) const HNSW_GRAPH: [u8; 4] = *b"HNSW";
pub(crate) const IVF_INDEX: [u8; 4] = *b"IVFX";
pub(crate) const SQ8_QUANTIZER: [u8; 4] = *b"SQ8Q";
pub(crate) const PQ_CODEBOOKS: [u8; 4] = *b"PQCB";

/// Current payload version for every kind
pub(crate) const VERSION: u16 = 1;

/// Serialize `value` and wrap it in the envelope
pub(crate) fn encode<T: Serialize>(kind: [u8; 4], value: &T) -> Result<Vec<u8>> {
    let payload = postcard::to_allocvec(value)
        .map_err(|e| Error::Internal(format!("failed to serialize {}: {e}", name(kind))))?;
    let mut out = Vec::with_capacity(HEADER + payload.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&kind);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&[0, 0]);
    out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    out.extend_from_slice(&crc32fast::hash(&payload).to_le_bytes());
    out.extend_from_slice(&payload);
    Ok(out)
}

/// Check the envelope and deserialize its payload
pub(crate) fn decode<T: DeserializeOwned>(kind: [u8; 4], bytes: &[u8]) -> Result<T> {
    let corrupt = |why: String| Error::Corrupt(format!("{} file: {why}", name(kind)));
    if bytes.len() < HEADER || !is_enveloped(bytes) {
        return Err(corrupt("not a SrvDB file or truncated header".into()).into());
    }
    if bytes[4..8] != kind {
        return Err(corrupt(format!("holds {}", name(bytes[4..8].try_into()?))).into());
    }
    let version = u16::from_le_bytes([bytes[8], bytes[9]]);
    if version > VERSION {
        return Err(corrupt(format!(
            "format version {version} is newer than this SrvDB supports ({VERSION})"
        ))
        .into());
    }
    let len = u64::from_le_bytes(bytes[12..20].try_into()?) as usize;
    let crc = u32::from_le_bytes(bytes[20..24].try_into()?);
    let payload = &bytes[HEADER..];
    if payload.len() != len {
        return Err(corrupt(format!(
            "expected {len} payload bytes, found {}",
            payload.len()
        ))
        .into());
    }
    if crc32fast::hash(payload) != crc {
        return Err(corrupt("checksum mismatch".into()).into());
    }
    postcard::from_bytes(payload).map_err(|e| corrupt(e.to_string()).into())
}

/// Whether `bytes` start with the envelope magic (files from v0.2 do not)
pub(crate) fn is_enveloped(bytes: &[u8]) -> bool {
    bytes.starts_with(MAGIC)
}

fn name(kind: [u8; 4]) -> String {
    String::from_utf8_lossy(&kind).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corrupt(result: Result<Vec<f32>>) -> bool {
        matches!(result.map_err(Error::from), Err(Error::Corrupt(_)))
    }

    #[test]
    fn round_trips() {
        let value = vec![1.0f32, -2.5, 3.25];
        let bytes = encode(SQ8_QUANTIZER, &value).unwrap();
        assert!(is_enveloped(&bytes));
        assert_eq!(decode::<Vec<f32>>(SQ8_QUANTIZER, &bytes).unwrap(), value);
    }

    #[test]
    fn rejects_damage_and_mismatches() {
        let bytes = encode(SQ8_QUANTIZER, &vec![1.0f32, 2.0]).unwrap();

        let mut flipped = bytes.clone();
        *flipped.last_mut().unwrap() ^= 0xFF;
        assert!(corrupt(decode(SQ8_QUANTIZER, &flipped)));

        assert!(corrupt(decode(SQ8_QUANTIZER, &bytes[..bytes.len() - 1])));
        assert!(corrupt(decode(PQ_CODEBOOKS, &bytes)));
        assert!(corrupt(decode(SQ8_QUANTIZER, b"plain bytes, no envelope")));

        let mut newer = bytes;
        newer[8] = 99;
        assert!(corrupt(decode(SQ8_QUANTIZER, &newer)));
    }
}
