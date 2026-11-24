//! Envelope for every serialized SrvDB file (ADR 0003)
//!
//! Layout, little-endian:
//!
//! | offset | size | field                                  |
//! |--------|------|----------------------------------------|
//! | 0      | 4    | magic `SRVD`                           |
//! | 4      | 4    | kind (`HNSW`, `IVFX`, `SQ8Q`, `PQCB`, `RBQP`, `IDXS`, ...) |
//! | 8      | 2    | format version of the payload          |
//! | 10     | 2    | reserved (zero)                        |
//! | 12     | 8    | payload length                         |
//! | 20     | 4    | checksum of the payload                |
//! | 24     | ...  | payload (postcard)                     |
//!
//! The checksum is CRC32C since envelope version 2 (ADR 0007); version 1
//! files (SrvDB 0.3) used CRC32 (IEEE) and are still read.
//!
//! A checksum mismatch is `Error::ChecksumMismatch`; anything else that
//! does not match (wrong magic or kind, newer version, truncation,
//! undecodable payload) is `Error::Corrupt`.

use crate::error::Error;
use anyhow::Result;
use serde::{de::DeserializeOwned, Serialize};

pub(crate) const MAGIC: &[u8; 4] = b"SRVD";
const HEADER: usize = 24;

pub(crate) const HNSW_GRAPH: [u8; 4] = *b"HNSW";
pub(crate) const IVF_INDEX: [u8; 4] = *b"IVFX";
pub(crate) const SQ8_QUANTIZER: [u8; 4] = *b"SQ8Q";
pub(crate) const PQ_CODEBOOKS: [u8; 4] = *b"PQCB";
pub(crate) const CHECKSUMS: [u8; 4] = *b"CRCS";
pub(crate) const RABITQ_PARAMS: [u8; 4] = *b"RBQP";
pub(crate) const INDEX_SETTINGS: [u8; 4] = *b"IDXS";
pub(crate) const KEYWORD_INDEX: [u8; 4] = *b"KWIX";

/// Current envelope version for every kind: 2 = CRC32C checksum
pub(crate) const VERSION: u16 = 2;
/// Envelope version written by SrvDB 0.3: CRC32 (IEEE) checksum
const VERSION_CRC32: u16 = 1;

/// Checksum of `payload` as envelope `version` defines it
fn checksum(version: u16, payload: &[u8]) -> u32 {
    if version == VERSION_CRC32 {
        crc32fast::hash(payload)
    } else {
        crc32c::crc32c(payload)
    }
}

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
    out.extend_from_slice(&checksum(VERSION, &payload).to_le_bytes());
    out.extend_from_slice(&payload);
    Ok(out)
}

/// Check the envelope and deserialize its payload (trailing payload bytes
/// are ignored; see [`decode_exact`])
pub(crate) fn decode<T: DeserializeOwned>(kind: [u8; 4], bytes: &[u8]) -> Result<T> {
    let payload = checked_payload(kind, bytes)?;
    postcard::from_bytes(payload).map_err(|e| corrupt(kind, e.to_string()).into())
}

/// As [`decode`], but the value must use the whole payload: a payload
/// with bytes after it is `Corrupt`
pub(crate) fn decode_exact<T: DeserializeOwned>(kind: [u8; 4], bytes: &[u8]) -> Result<T> {
    let payload = checked_payload(kind, bytes)?;
    let (value, rest) =
        postcard::take_from_bytes(payload).map_err(|e| corrupt(kind, e.to_string()))?;
    if !rest.is_empty() {
        return Err(corrupt(
            kind,
            format!("{} unexpected bytes after the payload", rest.len()),
        )
        .into());
    }
    Ok(value)
}

fn corrupt(kind: [u8; 4], why: String) -> Error {
    Error::Corrupt(format!("{} file: {why}", name(kind)))
}

/// The payload of a valid envelope of `kind`
fn checked_payload(kind: [u8; 4], bytes: &[u8]) -> Result<&[u8]> {
    let corrupt = |why: String| corrupt(kind, why);
    if bytes.len() < HEADER || !is_enveloped(bytes) {
        return Err(corrupt("not a SrvDB file or truncated header".into()).into());
    }
    if bytes[4..8] != kind {
        return Err(corrupt(format!("holds {}", name(bytes[4..8].try_into()?))).into());
    }
    let version = u16::from_le_bytes([bytes[8], bytes[9]]);
    if version == 0 || version > VERSION {
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
    if checksum(version, payload) != crc {
        return Err(Error::ChecksumMismatch(format!(
            "{} file: payload does not match its checksum; delete it if it is an \
             index (it is rebuilt) or restore the database from a backup",
            name(kind)
        ))
        .into());
    }
    Ok(payload)
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
        let err = Error::from(decode::<Vec<f32>>(SQ8_QUANTIZER, &flipped).unwrap_err());
        assert!(matches!(err, Error::ChecksumMismatch(_)), "{err:?}");

        assert!(corrupt(decode(SQ8_QUANTIZER, &bytes[..bytes.len() - 1])));
        assert!(corrupt(decode(PQ_CODEBOOKS, &bytes)));
        assert!(corrupt(decode(SQ8_QUANTIZER, b"plain bytes, no envelope")));

        let mut newer = bytes;
        newer[8] = 99;
        assert!(corrupt(decode(SQ8_QUANTIZER, &newer)));
    }

    #[test]
    fn writes_crc32c() {
        let bytes = encode(SQ8_QUANTIZER, &vec![1.0f32]).unwrap();
        assert_eq!(u16::from_le_bytes([bytes[8], bytes[9]]), 2);
        let crc = u32::from_le_bytes(bytes[20..24].try_into().unwrap());
        assert_eq!(crc, crc32c::crc32c(&bytes[HEADER..]));
    }

    /// Envelope v1 files from SrvDB 0.3 (CRC32 IEEE) still decode
    #[test]
    fn reads_v1_envelopes_from_0_3() {
        let sq: crate::quant::scalar::ScalarQuantizer = decode(
            SQ8_QUANTIZER,
            include_bytes!("../../tests/data/v0.3/sq8/scalar_quantizer.bin"),
        )
        .unwrap();
        assert_eq!(sq.dimension, 128);
        let graph = include_bytes!("../../tests/data/v0.3/hnsw/hnsw.graph");
        assert_eq!(u16::from_le_bytes([graph[8], graph[9]]), 1);
        let index = crate::hnsw::HNSWIndex::from_bytes(graph).unwrap();
        assert_eq!(index.len(), 40);

        // A damaged v1 file is still caught by its CRC32
        let mut damaged = graph.to_vec();
        *damaged.last_mut().unwrap() ^= 0x01;
        let err = Error::from(crate::hnsw::HNSWIndex::from_bytes(&damaged).err().unwrap());
        assert!(matches!(err, Error::ChecksumMismatch(_)), "{err:?}");
    }
}
