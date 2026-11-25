//! Command-line options for the benchmark harness.

use std::path::PathBuf;

pub const ALL_MODES: [&str; 8] = [
    "flat", "sq8", "pq", "hnsw", "ivf", "rabitq1", "rabitq2", "rabitq4",
];

pub const USAGE: &str = "\
usage: cargo run --release -p srvdb --example bench -- [options]

  --dim N            synthetic dimension (default 384)
  --n N              base vectors (default 10000)
  --queries N        query vectors (default 200)
  --k N              neighbours per query (default 10)
  --seed N           synthetic dataset seed (default 42)
  --modes LIST       comma list of flat,sq8,pq,hnsw,ivf,rabitq1,rabitq2,rabitq4
                     (default all)
  --ef-search N      HNSW/IVF search list size (default 100)
  --nlist N          IVF partitions (default sqrt(n))
  --nprobe N         IVF partitions probed (default 8)
  --rerank N         RaBitQ rerank factor (default 40/20/10 for 1/2/4 bits)
  --base-fvecs PATH  load base vectors from an .fvecs file instead
  --query-fvecs PATH load query vectors from an .fvecs file (with --base-fvecs)
  --filter-pct P     search only a seeded random P% of vectors (filtered
                     search vs post-filtering; default off)
  --out PATH         write JSON results here (default bench-results.json)
  --work-dir PATH    directory for temporary databases (default system temp)
  --export-fvecs DIR write base.fvecs and query.fvecs to DIR and exit
                     (used by benchmarks/compare to share the dataset)
";

#[derive(Clone, Debug)]
pub struct Args {
    pub dim: usize,
    pub n: usize,
    pub queries: usize,
    pub k: usize,
    pub seed: u64,
    pub modes: Vec<String>,
    pub ef_search: usize,
    pub nlist: Option<usize>,
    pub nprobe: usize,
    pub rerank: Option<usize>,
    pub base_fvecs: Option<PathBuf>,
    pub query_fvecs: Option<PathBuf>,
    pub filter_pct: Option<f64>,
    pub out: PathBuf,
    pub work_dir: Option<PathBuf>,
    /// Write the dataset as .fvecs files to this directory and exit.
    pub export_fvecs: Option<PathBuf>,
    /// Internal: run a single mode and print one JSON line (child process).
    pub child: Option<String>,
}

fn num<T: std::str::FromStr>(flag: &str, v: Option<String>) -> Result<T, String> {
    let v = v.ok_or(format!("{flag} needs a value"))?;
    v.parse()
        .map_err(|_| format!("{flag}: '{v}' is not a valid number"))
}

pub fn parse(raw: &[String]) -> Result<Args, String> {
    let mut a = Args {
        dim: 384,
        n: 10_000,
        queries: 200,
        k: 10,
        seed: 42,
        modes: ALL_MODES.iter().map(|s| s.to_string()).collect(),
        ef_search: 100,
        nlist: None,
        nprobe: 8,
        rerank: None,
        base_fvecs: None,
        query_fvecs: None,
        filter_pct: None,
        out: PathBuf::from("bench-results.json"),
        work_dir: None,
        export_fvecs: None,
        child: None,
    };
    let mut it = raw.iter().cloned();
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--dim" => a.dim = num(&flag, it.next())?,
            "--n" => a.n = num(&flag, it.next())?,
            "--queries" => a.queries = num(&flag, it.next())?,
            "--k" => a.k = num(&flag, it.next())?,
            "--seed" => a.seed = num(&flag, it.next())?,
            "--ef-search" => a.ef_search = num(&flag, it.next())?,
            "--nlist" => a.nlist = Some(num(&flag, it.next())?),
            "--nprobe" => a.nprobe = num(&flag, it.next())?,
            "--rerank" => a.rerank = Some(num(&flag, it.next())?),
            "--modes" => {
                let v = it.next().ok_or("--modes needs a value")?;
                a.modes = v.split(',').map(|s| s.trim().to_lowercase()).collect();
            }
            "--base-fvecs" => a.base_fvecs = it.next().map(PathBuf::from),
            "--query-fvecs" => a.query_fvecs = it.next().map(PathBuf::from),
            "--filter-pct" => a.filter_pct = Some(num(&flag, it.next())?),
            "--out" => a.out = it.next().map(PathBuf::from).ok_or("--out needs a value")?,
            "--work-dir" => a.work_dir = it.next().map(PathBuf::from),
            "--export-fvecs" => a.export_fvecs = it.next().map(PathBuf::from),
            "--child" => a.child = it.next(),
            "-h" | "--help" => return Err(String::new()),
            other => return Err(format!("unknown option {other}")),
        }
    }
    if let Some(bad) = a.modes.iter().find(|m| !ALL_MODES.contains(&m.as_str())) {
        return Err(format!("unknown mode '{bad}'; choose from {ALL_MODES:?}"));
    }
    if a.k == 0 || a.n == 0 || a.queries == 0 {
        return Err("--k, --n and --queries must be at least 1".into());
    }
    if a.filter_pct.is_some_and(|p| !(p > 0.0 && p <= 100.0)) {
        return Err("--filter-pct must be in (0, 100]".into());
    }
    if a.base_fvecs.is_some() != a.query_fvecs.is_some() {
        return Err("--base-fvecs and --query-fvecs must be given together".into());
    }
    Ok(a)
}
