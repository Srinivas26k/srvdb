# ADR 0013: Simple verbs: `create`, `open`, `save`, `close`, `delete`

- **Status:** Accepted
- **Date:** 2026-11-24

## Context

Through 0.3 a Python user met five ways to get a database:
`Database(path, dimension, mode)` and the static constructors `new_hnsw`,
`new_scalar_quantized`, `new_product_quantized` and `new_rabitq`, plus
`srvdb.open(path, dimension, mode)` (#37, unreleased). Every call created
the database if it was missing, so a typo in a path silently made a new,
empty database, and reopening required repeating the dimension and mode.
The reference pages were titled `srvdb.Database...`, and the long
constructor names wrapped badly in the sidebar. Issue #57 asks for a few
plain verbs; ADR 0011 (#50) made it possible by recording the index
settings on disk, so a database can be reopened by path alone.

Constraints: AGENTS.md section 4 (validation in the core, bindings only
convert), section 6 (typed errors, Python exception mapping), ADR 0003
(`persist()` is the only commit point), ADR 0005 (one open instance per
directory, held by a lock).

## How other embedded databases do it

| Library | Create / open | Index choice | Durability |
|---------|---------------|--------------|------------|
| Chroma | `PersistentClient(path)`, `create_collection` / `get_collection` / `get_or_create_collection`, `delete_collection` | metadata dict (`hnsw:space`, ...) | writes persisted automatically |
| LanceDB | `lancedb.connect(path)`, `create_table(name, data, exist_ok=...)`, `open_table`, `drop_table` | `table.create_index(...)` by keyword | every write is a new version on disk |
| FAISS | `faiss.index_factory(d, "HNSW32")`, `write_index` / `read_index` | a factory string naming the index | none: the caller writes the file |
| Qdrant (local) | `QdrantClient(path=...)`, `create_collection` / `collection_exists` / `delete_collection` | `hnsw_config`, `quantization_config` objects | persisted by the client |

What we take: verbs like Chroma and LanceDB (`create` fails on an existing
database, `open` never creates, `delete` drops it, `exist_ok` for
get-or-create like LanceDB and `os.makedirs`); the index chosen by name, as
in `faiss.index_factory`, with short keyword options instead of config
objects. What we do not take: hidden autosave. SrvDB's guarantee (ADR 0003)
is that data is durable exactly when `persist()` returns, so saving stays
explicit, and a context manager gives the convenience without making a
half-finished batch durable.

## Decision

### Python verbs

```python
db = srvdb.create(path, dimension, index="hnsw", *, exist_ok=False, **options)
db = srvdb.open(path)
db.save()            # persist() stays as an alias
db.close()
srvdb.delete(path)
with srvdb.open(path) as db: ...
```

- **`create`** makes a new database and records its settings before it
  returns. `index` is one of `flat`, `hnsw`, `ivf`, `sq8`, `pq`, `rabitq`,
  `auto`. Each index has a fixed allow-list of options: hnsw `m`,
  `ef_construction`, `ef_search`; ivf `nlist`, `nprobe` and the partition
  graph's `m`, `ef_construction`, `ef_search`; sq8/pq `training`; rabitq
  `bits`, `rerank`, `seed`. An option that does not apply raises
  `ValueError` listing the valid ones; values are range-checked before
  anything is written. The path must not exist or be an empty directory
  (`FileExistsError`). With `exist_ok=True` an existing database is opened
  if its recorded index, dimension and every option given match; a
  difference raises `ValueError` (`SettingsMismatch` / `DimensionMismatch`).
  Training vectors are not compared: an existing quantizer is kept.
- **Default index `hnsw`.** The issue proposed `flat`; the user asked for
  `hnsw`, which scales past a few thousand vectors without a later
  migration. Flat remains one keyword away and is exact.
- **`open(path)`** takes only the path and calls `SrvDB::open_existing`.
  It never creates anything; a missing database is `FileNotFoundError`.
  The `dimension`/`mode` arguments of #37 are removed (never released).
- **`save()`** is the documented commit; `persist()` is kept as an alias,
  not deprecated, because 0.3 code and the Rust trait use it.
- **`close()`** drops the core database and its lock without saving.
  It is idempotent; any other method then raises `ValueError` ("closed"),
  as Python file objects do.
- **Context manager.** A clean exit saves and closes. An exception closes
  without saving, so a half-finished batch is never made durable, and the
  exception propagates. Durability stays explicit: nothing saves on a
  timer, on garbage collection or on `close()`.
- **`delete(path)`**, not `remove`: it mirrors `create`, and `os.remove`
  suggests a file. It lives at module level, so it does not collide with
  `db.delete(ids)`, which deletes vectors. The reference pages say so.

### Safety of `delete`

`SrvDB::delete_database` removes a directory only if every entry is a file
SrvDB writes (its known file names, `.crc` sidecars, `.compact` staging
copies, `.<name>.tmp` atomic-write temporaries; no subdirectories) and at
least one file starts with a SrvDB magic number (`SRVD` envelope or `SvDB`
vector header). A path that is itself a symbolic link is refused (checked
with `symlink_metadata`), and magic numbers are read only from regular
files, so a FIFO cannot hang the check. It takes the database lock first,
so an open database is refused with `Locked` (`BlockingIOError`). With the
lock held, the directory is renamed to a hidden sibling tombstone
`.<name>.srvdb-deleting-<pid>`; the entries are checked again there, the
files are removed, on Unix the lock file is unlinked while still locked,
the lock is released and the tombstone is removed. Nobody can open or lock
the database at its old path once the rename has happened, which closes
the window in which a second writer could take a fresh lock in a
directory being emptied, and later steps cannot be redirected through a
changed path component. A crash part way can leave the tombstone; it is
removed by hand.

### Failure safety of `create`

`create` builds the database in a hidden sibling directory
(`.<name>.srvdb-creating-<pid>-<n>`) and renames it into place only when it
is complete and its settings are recorded; an existing empty target
directory is replaced. On any error the staging directory, which only this
call created, is removed, so a failed create leaves nothing at `path` and a
retry succeeds. Missing parent directories are kept.

### Core API (Rust)

Validation is in the core so Rust users get the same checks:
`SrvDB::create(path, dimension, &CreateOptions)`, `CreateOptions`
(`new(IndexChoice)`, `set(name, u64)`, `set_training`, `set_exist_ok`),
`IndexChoice` (`FromStr`, `options()`), `SrvDB::delete_database(path)`,
`SrvDB::index_name()`, `SrvDB::dimension()`. Both new public types are
`#[non_exhaustive]`. `Error` gains `NotFound` and `AlreadyExists`
(Python `FileNotFoundError` and `FileExistsError`, both `OSError`
subclasses); `open_existing` on a missing database now returns `NotFound`
instead of `InvalidInput` (unreleased behaviour).

The binding converts the keyword dict (`training` to vectors, everything
else to `u64`) and maps errors; it holds no option names of its own.

### `Database(...)` and the `new_*` constructors

They keep working unchanged in 0.4 and are no longer documented: no
reference page, guide, README or example shows them, and `Database`
remains importable for type hints. No `DeprecationWarning` yet, so 0.3
code runs quietly; whether to deprecate them is decided for 0.5 with user
feedback. Their create-if-missing behaviour is unchanged.

## Alternatives considered

- **Keep `srvdb.open(path, dimension, mode)` as create-or-open.** One verb,
  but a mistyped path creates an empty database and the settings must be
  repeated on every open. Rejected for separate `create`/`open` plus
  `exist_ok`.
- **Config objects (`HnswConfig(m=...)`) instead of keywords.** Typed, but
  heavier than Chroma or FAISS for the common case; the Rust API keeps
  typed options and Python gets keywords checked by the same code.
- **Autosave on close or garbage collection (Chroma-like).** Convenient,
  but makes the moment data becomes durable invisible and would save a
  half-finished batch after an exception; it contradicts ADR 0003.
- **`srvdb.remove` / `srvdb.drop`.** `drop` reads as SQL; `remove` as a file
  operation. `delete` pairs with `create`.
- **`shutil.rmtree` in Python.** Would delete any directory a typo points
  at and ignores the lock; rejected for a checked core function.

## Consequences

- Five verbs cover the lifecycle; the reference is titled by them
  (`srvdb.create`, `srvdb.open`, `srvdb.delete`, `db.add`, `db.save`, ...),
  and the per-constructor pages are gone.
- A database must be created once before it is opened; scripts that
  should run repeatedly use `create(..., exist_ok=True)`.
- Two `Error` variants and two public types to maintain; the allow-list of
  options lives in `engine/create_options.rs` and must grow with new index
  parameters.
- `delete` must learn every new file name SrvDB writes
  (`storage/identity.rs`), or it will refuse databases that contain them.
