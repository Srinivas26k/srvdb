# SrvDB 0.3 fixtures

Written by the 0.3 on-disk format code (commit e55aa0a on `release/0.4`,
before checksums: vector header version 4, file envelope version 1). Used by
`tests/test_checksums.rs` and the `storage::format` unit tests to show that
0.3 databases still open and are upgraded.

## How they were produced

A throwaway example in `crates/srvdb/examples/` built 40 vectors of
dimension 128. Vector `i` has component `j` equal to
`((i*7 + j*3) % 11) - 5.0 + j*0.1 + i*0.01`, with metadata `{"i": i}`. For
each mode it called `SrvDB::new_with_config` (dimension 128), `add_batch`,
then `persist()`:

- `hnsw/`: `config.index_type = IndexType::HNSW`
- `sq8/`: `config.quantization.enabled = true`,
  `config.quantization.mode = QuantizationMode::Scalar`

Then, by hand:

- removed the `srvdb.lock` files and `sq8/metadata.db` (the SQ8 test opens
  only the storage files)
- truncated the vector files to the committed records, dropping the
  preallocated growth chunk (all zeros, never committed):
  `truncate -s 20504 hnsw/vectors.bin` (24 + 40 * 512) and
  `truncate -s 5144 sq8/sq8_vectors.bin` (24 + 40 * 128)

`hnsw/metadata.db` is 536 KB because redb preallocates its file; it holds
only the 40 metadata entries and the committed count.

To regenerate: check out e55aa0a, recreate the example from this
description, run it with the output directory as its argument, and apply
the same trimming.
