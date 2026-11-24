//! What `SrvDB::create` is asked to build (ADR 0013): an index kind from
//! a fixed set and short named options from a per-kind allow-list

use crate::error::{Error, Result};
use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;

/// The index a new database is created with
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum IndexChoice {
    /// Exact scan of full-precision vectors
    Flat,
    /// HNSW graph over full-precision vectors
    #[default]
    Hnsw,
    /// IVF partitions, each searched through an HNSW graph
    Ivf,
    /// Scalar quantization, one byte per value
    Sq8,
    /// Product quantization codes
    Pq,
    /// Full vectors searched through RaBitQ codes with exact rerank
    Rabitq,
    /// Mode chosen from the hardware and data size
    Auto,
}

/// Every index name, in the order error messages list them
const NAMES: [(&str, IndexChoice); 7] = [
    ("flat", IndexChoice::Flat),
    ("hnsw", IndexChoice::Hnsw),
    ("ivf", IndexChoice::Ivf),
    ("sq8", IndexChoice::Sq8),
    ("pq", IndexChoice::Pq),
    ("rabitq", IndexChoice::Rabitq),
    ("auto", IndexChoice::Auto),
];

const HNSW: &[&str] = &["m", "ef_construction", "ef_search"];
const IVF: &[&str] = &["nlist", "nprobe", "m", "ef_construction", "ef_search"];
const TRAINED: &[&str] = &["training"];
const RABITQ: &[&str] = &["bits", "rerank", "seed"];

impl IndexChoice {
    /// The lowercase name used by `FromStr` and in messages
    pub fn name(self) -> &'static str {
        NAMES
            .iter()
            .find(|(_, choice)| *choice == self)
            .map_or("unknown", |(name, _)| name)
    }

    /// The option names this index accepts (empty for flat and auto)
    pub fn options(self) -> &'static [&'static str] {
        match self {
            Self::Hnsw => HNSW,
            Self::Ivf => IVF,
            Self::Sq8 | Self::Pq => TRAINED,
            Self::Rabitq => RABITQ,
            Self::Flat | Self::Auto => &[],
        }
    }
}

impl fmt::Display for IndexChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for IndexChoice {
    type Err = Error;

    /// Parse an index name, case-insensitively.
    ///
    /// # Errors
    /// `InvalidInput` naming the valid indexes.
    fn from_str(s: &str) -> Result<Self> {
        let lower = s.to_ascii_lowercase();
        NAMES
            .iter()
            .find(|(name, _)| *name == lower)
            .map(|(_, choice)| *choice)
            .ok_or_else(|| {
                let valid: Vec<&str> = NAMES.iter().map(|(name, _)| *name).collect();
                Error::invalid_input(format!(
                    "unknown index '{s}'; choose one of: {}",
                    valid.join(", ")
                ))
            })
    }
}

/// The index and options for [`crate::SrvDB::create`].
///
/// Options are checked by name against [`IndexChoice::options`] when set;
/// their values are range-checked by `create`.
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct CreateOptions {
    index: IndexChoice,
    values: BTreeMap<&'static str, u64>,
    training: Option<Vec<Vec<f32>>>,
    exist_ok: bool,
}

impl CreateOptions {
    /// Options for `index`, with every parameter at its default
    pub fn new(index: IndexChoice) -> Self {
        Self {
            index,
            ..Self::default()
        }
    }

    /// The index to create
    pub fn index(&self) -> IndexChoice {
        self.index
    }

    /// Set the numeric option `name` (for example `m` or `nlist`).
    ///
    /// # Errors
    /// `InvalidInput` if the index does not take an option of that name;
    /// the message lists the valid ones.
    pub fn set(&mut self, name: &str, value: u64) -> Result<&mut Self> {
        let key = self.accept(name)?;
        if key == "training" {
            return Err(Error::invalid_input(
                "option 'training' takes vectors, not a number",
            ));
        }
        self.values.insert(key, value);
        Ok(self)
    }

    /// Set the `training` option: sample vectors an SQ8 or PQ quantizer is
    /// trained on. They are not added to the database. Without them the
    /// quantizer is trained on the first batch added.
    ///
    /// # Errors
    /// `InvalidInput` if the index is not `sq8` or `pq`.
    pub fn set_training(&mut self, vectors: Vec<Vec<f32>>) -> Result<&mut Self> {
        self.accept("training")?;
        self.training = Some(vectors);
        Ok(self)
    }

    /// With `true`, `create` opens an existing database whose recorded
    /// settings match instead of failing (like `os.makedirs(exist_ok=True)`)
    pub fn set_exist_ok(&mut self, exist_ok: bool) -> &mut Self {
        self.exist_ok = exist_ok;
        self
    }

    /// Whether an existing, matching database is opened by `create`
    pub fn exist_ok(&self) -> bool {
        self.exist_ok
    }

    /// The value of a numeric option, if it was set
    pub fn get(&self, name: &str) -> Option<u64> {
        self.values.get(name).copied()
    }

    /// The training vectors, if they were set
    pub fn training(&self) -> Option<&[Vec<f32>]> {
        self.training.as_deref()
    }

    fn accept(&self, name: &str) -> Result<&'static str> {
        let valid = self.index.options();
        valid.iter().copied().find(|v| *v == name).ok_or_else(|| {
            let listed = if valid.is_empty() {
                "it takes no options".to_string()
            } else {
                format!("valid options: {}", valid.join(", "))
            };
            Error::invalid_input(format!(
                "option '{name}' does not apply to index '{}'; {listed}",
                self.index
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip_case_insensitively() {
        for (name, choice) in NAMES {
            assert_eq!(name.to_uppercase().parse::<IndexChoice>().unwrap(), choice);
            assert_eq!(choice.name(), name);
        }
        let err = "hnswx".parse::<IndexChoice>().unwrap_err().to_string();
        assert!(
            err.contains("flat, hnsw, ivf, sq8, pq, rabitq, auto"),
            "{err}"
        );
    }

    #[test]
    fn options_outside_the_allow_list_are_refused() {
        let mut hnsw = CreateOptions::new(IndexChoice::Hnsw);
        hnsw.set("m", 32).unwrap();
        assert_eq!(hnsw.get("m"), Some(32));
        let err = hnsw.set("bits", 4).unwrap_err().to_string();
        assert!(err.contains("m, ef_construction, ef_search"), "{err}");
        assert!(hnsw.set_training(vec![vec![0.0]]).is_err());

        let mut flat = CreateOptions::new(IndexChoice::Flat);
        let err = flat.set("m", 1).unwrap_err().to_string();
        assert!(err.contains("takes no options"), "{err}");

        let mut sq8 = CreateOptions::new(IndexChoice::Sq8);
        assert!(sq8.set("training", 1).is_err());
        sq8.set_training(vec![vec![1.0]]).unwrap();
        assert_eq!(sq8.training().map(<[_]>::len), Some(1));
    }
}
