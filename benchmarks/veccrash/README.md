# VecCrash: crash-consistency benchmark for embedded vector stores

VecCrash asks one question of each library: if the process is killed
while it is writing, does the data the library already said was durable
come back intact when the store is reopened?

Findings are in [REPORT.md](REPORT.md); raw numbers are in
[results.json](results.json) and per-trial records in `trials/`.

## Method

1. **Writer** (`writer.py`, a child process) creates an empty store and
   adds batches of `--batch` vectors (default 100) of dimension `--dim`
   (default 128). Ids are known (batch `b` holds ids `1 + b*batch` to
   `(b+1)*batch`) and values are seeded float32 normals, so any batch can
   be regenerated exactly. After each batch it calls the library's
   durability point. Only after that call returns does it append the
   batch number to an ack log and `fsync` the ack log. The ack log is the
   exact set of data the library promised was durable. Batch 0 is acked
   before the writer signals ready, so every trial has promised data.
2. **Parent** (`run.py`) first runs three uncut writers to calibrate how
   long a writer runs after ready. Each trial then waits for ready, sleeps
   a seeded uniform random delay in [0, 1.1 x that time] and sends
   `SIGKILL` to the writer's process group. Delays therefore land during
   adds, during durability calls and (a few) after the writer finished.
3. **Verifier** (`verify.py`, a fresh process) reopens the store through
   the library's normal read API and classifies the trial, in priority
   order:

| Outcome | Meaning | Violation? |
|---|---|---|
| `CORRUPT_OPEN` | reopening or reading raises, crashes or hangs (120 s) | yes |
| `SILENT_CORRUPTION` | opens, but an id is unknown or a vector differs from what was written | yes |
| `LOST_ACKED` | an acknowledged id is missing | yes |
| `EXTRA_PARTIAL` | unacknowledged data is visible (whole or partial batch) | no |
| `OK` | exactly the acknowledged batches, values intact | no |

Losing unacknowledged data is always allowed. `EXTRA_PARTIAL` is
expected whenever the kill lands after a durability call returned but
before the ack log write, or for libraries that persist on `add`.
Values are compared with tolerance `1e-6` (all libraries store float32).
Proportions carry Wilson 95% score intervals.

SrvDB 0.4 has no public call that returns a stored vector, so for SrvDB
the verifier checks each visible id's metadata, then runs an exact
`search_batch` with each expected vector and requires its own id back at
cosine similarity > 0.99999. The flat index makes this exact; for the
HNSW variant a graph miss would be counted as silent corruption, so that
variant is conservative against SrvDB.

## Libraries and durability points

Each library is used through its documented persistence API, with the
ordinary save pattern from its documentation.

| Key | Pattern | Durability point | Documented guarantee | Docs |
|---|---|---|---|---|
| `srvdb` | `srvdb.create(path, 128, "flat")`, `add`, `save()` | `save()` | durable after `save()` returns (crash-safe commit) | `python/srvdb/_low_level.pyi`, https://srinivas26k.github.io/srvdb/ |
| `srvdb-hnsw` | same, default HNSW index | `save()` | same | same |
| `chroma` | `chromadb.PersistentClient(path).create_collection`, `add` | `add()` returns | data is persisted automatically to `path` | https://docs.trychroma.com/docs/run-chroma/clients |
| `lancedb` | `lancedb.connect(path)`, `create_table`, `table.add` | `add()` returns (commits a table version) | each write commits a new version | https://docs.lancedb.com/ |
| `faiss` | `IndexIDMap(IndexFlatL2)`, `add_with_ids`, `faiss.write_index(index, path)` | `write_index` | none (writes the file in place) | https://github.com/facebookresearch/faiss/wiki/Getting-started |
| `faiss-atomic` | as `faiss`, but write to `path.tmp`, `fsync`, `os.replace`, `fsync` dir | the rename | application-provided | same |
| `hnswlib` | `Index.init_index`, `add_items`, `save_index(path)` | `save_index` | none | https://github.com/nmslib/hnswlib |
| `usearch` | `Index(ndim, "l2sq", "f32")`, `add`, `save(path)` | `save` | none stated | https://unum-cloud.github.io/USearch/python/ |
| `sqlite-vec` | SQLite `journal_mode=WAL`, `vec0` virtual table, `INSERT`, `commit()` | `commit()` | SQLite transactions are atomic and durable | https://alexgarcia.xyz/sqlite-vec/python.html, https://www.sqlite.org/wal.html |

Versions and licences are recorded in `results.json`
(`environment.libraries`); all are permissive (MIT, Apache-2.0 or
BSD-style; hnswlib 0.8.0 declares no licence in its package metadata, its
repository LICENSE is Apache-2.0). Competitors are installed only in the
benchmark venv; none is
a SrvDB dependency.

## Fairness rules

- Every library gets the same data, dimension, batch size, number of
  batches, kill-time distribution and verifier.
- Each library uses its recommended pattern; no library gets extra
  `fsync` calls it does not make itself, except the explicitly separate
  `faiss-atomic` variant.
- FAISS, hnswlib and usearch make no crash-safety promise for their save
  calls. Their results are labelled "no documented guarantee; typical
  application pattern" (marked `*` in the chart). A violation there is a
  property of the usual application pattern, not a broken promise.
  `faiss-atomic` shows that the fix is the application's job.
- The kill window is calibrated per library, so slow and fast libraries
  are both killed across their whole write phase. Time per trial is
  recorded (`median_trial_s`) for transparency only; VecCrash makes no
  speed claims, and timings were taken on a shared, loaded machine.

## Caveats

- **Process crash, not power loss.** `SIGKILL` ends the process, but the
  kernel page cache survives, so data that was written but never
  `fsync`ed is still visible to the verifier. VecCrash therefore cannot
  detect missing `fsync` calls. Simulating power loss needs root (for
  example dropping caches, `dm-flakey` or a VM snapshot) and is future
  work.
- Kill timing is wall-clock sampling. Short critical sections (for
  example a 2 MB `write_index`) are hit rarely; a zero count is not proof
  of safety. The Wilson upper bound says how rare a failure could still be.
- The machine was shared with another benchmark during the run.

## Reproduce

```bash
uv venv -p 3.12 .veccrash-venv
VIRTUAL_ENV=$PWD/.veccrash-venv uv pip install maturin numpy chromadb lancedb pyarrow \
    faiss-cpu hnswlib usearch sqlite-vec matplotlib
VIRTUAL_ENV=$PWD/.veccrash-venv .veccrash-venv/bin/maturin develop --release
.veccrash-venv/bin/python benchmarks/veccrash/run.py --trials 400 --workers 4 --batches 20
.veccrash-venv/bin/python benchmarks/veccrash/render.py
```

The exact command, commit, hardware, OS, filesystem and versions of a
run are written to `results.json`. `--workdir` (default
`/data/build/veccrash/work`) holds the trial stores; each is deleted after
verification. Use a real disk filesystem, not tmpfs.
