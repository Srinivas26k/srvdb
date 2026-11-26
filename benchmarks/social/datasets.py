"""Dataset download, verification and loading for the social benchmark.

Network access lives here, in benchmarks/, never in the SrvDB library
(AGENTS.md section 9). Files are cached under benchmarks/social/.data/.
Every file is checked against the size the server reports and, for
Hugging Face LFS files, against the published SHA256 before use.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / ".data"
ANN = "http://ann-benchmarks.com/{name}.hdf5"
HF_REPO = "KShivendu/dbpedia-entities-openai-1M"
HF_API = f"https://huggingface.co/api/datasets/{HF_REPO}"
HF_FILE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{{rev}}/{{path}}"

DATASETS = {
    "dbpedia-openai-100k": {
        "kind": "hf",
        "dim": 1536,
        "n": 100_000,
        "queries": 1000,
        "source": f"https://huggingface.co/datasets/{HF_REPO}",
        "licence": "MIT (Hugging Face dataset card)",
        "model": "OpenAI text-embedding-ada-002",
    },
    "nytimes-256-angular": {
        "kind": "ann",
        "dim": 256,
        "queries": 1000,
        "source": ANN.format(name="nytimes-256-angular"),
        "licence": "derived from UCI Bag of Words (NYTimes), CC BY 4.0",
    },
    "glove-100-angular": {
        "kind": "ann",
        "dim": 100,
        "queries": 1000,
        "source": ANN.format(name="glove-100-angular"),
        "licence": "GloVe vectors, PDDL 1.0",
    },
    "synthetic-768": {
        "kind": "synthetic",
        "dim": 768,
        "n": 50_000,
        "queries": 1000,
        "source": "seeded clustered Gaussian (numpy, seed 42), offline",
        "licence": "generated",
    },
}


# ann-benchmarks.com rejects the default Python-urllib agent (HTTP 403).
AGENT = {"User-Agent": "srvdb-benchmark/1.0"}


def _open(req):
    if isinstance(req, str):
        req = urllib.request.Request(req)
    for key, value in AGENT.items():
        req.add_header(key, value)
    return urllib.request.urlopen(req, timeout=60)


def _download(url: str, tmp: Path, attempts: int = 20) -> None:
    """Stream url into tmp, resuming with HTTP Range after a stall or drop."""
    for attempt in range(attempts):
        have = tmp.stat().st_size if tmp.exists() else 0
        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with _open(req) as r, tmp.open("ab" if r.status == 206 else "wb") as f:
                while block := r.read(1 << 20):
                    f.write(block)
            return
        except (TimeoutError, OSError) as e:
            if attempt == attempts - 1:
                raise RuntimeError(f"download of {url} failed: {e}") from e


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _fetch(url: str, dest: Path, size: int | None, sha: str | None) -> dict:
    """Download url to dest once; verify size (and sha256 when published)."""
    start = time.time()
    downloaded = not dest.exists()
    if downloaded:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        _download(url, tmp)
        tmp.rename(dest)
    actual = dest.stat().st_size
    if size is not None and actual != size:
        raise RuntimeError(f"{dest.name}: size {actual} != expected {size}")
    digest = _sha256(dest)
    if sha is not None and digest != sha:
        raise RuntimeError(f"{dest.name}: sha256 does not match the source")
    return {
        "file": dest.name,
        "bytes": actual,
        "sha256": digest,
        "size_verified": size is not None,
        "sha256_verified_against_source": sha is not None,
        "downloaded_this_run": downloaded,
        "fetch_seconds": round(time.time() - start, 1),
    }


def _remote_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    with _open(req) as r:
        n = r.headers.get("Content-Length")
    return int(n) if n else None


def _normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)


def _clustered(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((100, dim)).astype(np.float32)
    x = centers[rng.integers(0, 100, n)]
    return x + 0.5 * rng.standard_normal((n, dim)).astype(np.float32)


def exact_topk(base: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Exact cosine top-k by numpy brute force (inputs L2-normalised)."""
    out = np.empty((len(queries), k), dtype=np.int64)
    for i in range(0, len(queries), 100):
        s = queries[i : i + 100] @ base.T
        top = np.argpartition(-s, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(s, top, 1), axis=1)
        out[i : i + 100] = np.take_along_axis(top, order, 1)
    return out


def _hf_files() -> tuple[str, list[dict]]:
    """Pinned revision and LFS metadata (size, sha256) of the first shards."""
    with _open(HF_API) as r:
        info = json.load(r)
    rev = info["sha"]
    shards = sorted(
        s["rfilename"] for s in info["siblings"] if s["rfilename"].endswith(".parquet")
    )
    body = json.dumps({"paths": shards[:4]}).encode()
    req = urllib.request.Request(
        f"{HF_API}/paths-info/{rev}",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with _open(req) as r:
        meta = json.load(r)
    return rev, sorted(meta, key=lambda m: m["path"])


def _load_hf(spec: dict) -> tuple[np.ndarray, dict]:
    import pyarrow.parquet as pq

    rev, files = _hf_files()
    need = spec["n"] + spec["queries"]
    parts, fetched = [], []
    for m in files:
        lfs = m.get("lfs") or {}
        dest = DATA / "dbpedia" / Path(m["path"]).name
        url = HF_FILE.format(rev=rev, path=m["path"])
        fetched.append(
            _fetch(url, dest, lfs.get("size", m.get("size")), lfs.get("oid"))
        )
        col = pq.read_table(dest, columns=["openai"]).column("openai")
        parts.append(np.asarray(col.to_pylist(), dtype=np.float32))
        if sum(len(p) for p in parts) >= need:
            break
    return np.concatenate(parts)[:need], {"revision": rev, "files": fetched}


def prepare(name: str, k: int = 10):
    """Return (base, queries, ground-truth ids, provenance) for a dataset."""
    spec = DATASETS[name]
    prov = {key: v for key, v in spec.items() if key != "kind"}
    nq = spec["queries"]
    if spec["kind"] == "ann":
        import h5py

        url = spec["source"]
        dest = DATA / f"{name}.hdf5"
        prov["files"] = [_fetch(url, dest, _remote_size(url), None)]
        with h5py.File(dest, "r") as h:
            base = _normalise(h["train"][:])
            q = _normalise(h["test"][:nq])
            gt = h["neighbors"][:nq, :k].astype(np.int64)
        prov |= {
            "n": len(base),
            "ground_truth": "ann-benchmarks neighbors (first queries)",
        }
        return base, q, gt, prov
    if spec["kind"] == "synthetic":
        x = _clustered(spec["n"] + nq, spec["dim"], 42)
    else:
        x, extra = _load_hf(spec)
        prov |= extra | {
            "subset": f"first {spec['n']} rows base, next {nq} rows queries"
        }
    x = _normalise(x)
    base, q = x[: spec["n"]], x[spec["n"] :]
    prov["ground_truth"] = "numpy exact cosine brute force"
    return base, q, exact_topk(base, q, k), prov
