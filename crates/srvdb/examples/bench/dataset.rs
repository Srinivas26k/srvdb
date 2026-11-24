//! Benchmark datasets: seeded synthetic clusters or user-supplied .fvecs files.
//!
//! Nothing is downloaded. Real datasets (SIFT1M, GloVe) are read from a path
//! the user provides and are never committed to the repository.

use std::io::Read;
use std::path::Path;

/// Base vectors plus held-out queries.
pub struct Dataset {
    pub name: String,
    pub dim: usize,
    pub base: Vec<Vec<f32>>,
    pub queries: Vec<Vec<f32>>,
}

/// SplitMix64: a tiny deterministic generator so results do not depend on
/// the `rand` crate's stream across versions.
pub struct SplitMix(u64);

impl SplitMix {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Standard normal via Box-Muller.
    pub fn gauss(&mut self) -> f32 {
        let u1 = self.unit().max(f64::MIN_POSITIVE);
        let u2 = self.unit();
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }
}

/// Clustered Gaussians: `clusters` random unit centres, each point is a
/// centre plus isotropic noise. Queries are drawn the same way.
pub fn synthetic(dim: usize, n: usize, nq: usize, seed: u64) -> Dataset {
    let clusters = (n / 100).clamp(8, 1024);
    let mut rng = SplitMix::new(seed);
    let centres: Vec<Vec<f32>> = (0..clusters)
        .map(|_| (0..dim).map(|_| rng.gauss()).collect())
        .collect();
    let noise = 0.6;
    let point = |rng: &mut SplitMix| -> Vec<f32> {
        let c = &centres[(rng.next_u64() % clusters as u64) as usize];
        c.iter().map(|x| x + noise * rng.gauss()).collect()
    };
    let base = (0..n).map(|_| point(&mut rng)).collect();
    let queries = (0..nq).map(|_| point(&mut rng)).collect();
    Dataset {
        name: format!("synthetic-gauss(dim={dim},n={n},clusters={clusters},seed={seed})"),
        dim,
        base,
        queries,
    }
}

/// Read up to `limit` vectors from an .fvecs file (little-endian i32 dim,
/// then dim f32 values, repeated).
pub fn read_fvecs(path: &Path, limit: usize) -> Result<Vec<Vec<f32>>, String> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .and_then(|mut f| f.read_to_end(&mut bytes))
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let mut out = Vec::new();
    let mut pos = 0;
    while pos + 4 <= bytes.len() && out.len() < limit {
        let d = i32::from_le_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]]);
        if d <= 0 {
            return Err(format!(
                "{}: bad dimension {d} at byte {pos}",
                path.display()
            ));
        }
        let d = d as usize;
        let end = pos + 4 + d * 4;
        if end > bytes.len() {
            return Err(format!(
                "{}: truncated record at byte {pos}",
                path.display()
            ));
        }
        let v: Vec<f32> = bytes[pos + 4..end]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        out.push(v);
        pos = end;
    }
    if let Some(a) = out.first() {
        let dim = a.len();
        if out.iter().any(|v| v.len() != dim) {
            return Err(format!("{}: mixed dimensions", path.display()));
        }
    }
    Ok(out)
}

/// Load base and query .fvecs files supplied by the user.
pub fn from_fvecs(base: &Path, query: &Path, n: usize, nq: usize) -> Result<Dataset, String> {
    let base_v = read_fvecs(base, n)?;
    let queries = read_fvecs(query, nq)?;
    let dim = base_v.first().map(Vec::len).ok_or("base file is empty")?;
    if queries.first().map(Vec::len) != Some(dim) {
        return Err("query dimension differs from base dimension".into());
    }
    let file = |p: &Path| {
        p.file_name()
            .map_or(String::new(), |f| f.to_string_lossy().into())
    };
    Ok(Dataset {
        name: format!(
            "fvecs(base={},query={},n={})",
            file(base),
            file(query),
            base_v.len()
        ),
        dim,
        base: base_v,
        queries,
    })
}

/// Write vectors as .fvecs (little-endian i32 dim, then dim f32 values).
pub fn write_fvecs(path: &Path, vecs: &[Vec<f32>]) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(vecs.iter().map(|v| 4 + v.len() * 4).sum());
    for v in vecs {
        let d = i32::try_from(v.len()).map_err(|_| "dimension too large".to_string())?;
        bytes.extend_from_slice(&d.to_le_bytes());
        for x in v {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
    }
    std::fs::write(path, bytes).map_err(|e| format!("cannot write {}: {e}", path.display()))
}

/// Export base and query vectors to `dir` as base.fvecs and query.fvecs.
pub fn export(ds: &Dataset, dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    write_fvecs(&dir.join("base.fvecs"), &ds.base)?;
    write_fvecs(&dir.join("query.fvecs"), &ds.queries)
}
