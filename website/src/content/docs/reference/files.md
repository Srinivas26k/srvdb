---
title: On-disk files
description: What each file in a SrvDB database directory holds.
---

A database is one directory. Copy the whole directory (while it is closed) to
back it up.

| File | Contents | Derived? |
|------|----------|----------|
| `metadata.db` | Metadata, commit record, deletes ([redb](https://github.com/cberner/redb)) | No |
| `vectors.bin` | Full-precision vectors (flat, HNSW, IVF, auto) | No |
| `sq8_vectors.bin`, `scalar_quantizer.bin` | SQ8 codes and scaling | No |
| `quantized_vectors.bin`, `codebooks.bin` | PQ codes and codebooks | No |
| `hnsw.graph` | HNSW graph | Yes: rebuilt if missing |
| `ivf.index` | Trained IVF index | Yes: retrain with `train_ivf()` |
| `srvdb.lock` | Lock held while the database is open | — |

Index files are derived data: they can always be rebuilt from the vectors.
Every file is validated on open; a damaged file is reported as an error, never
read as garbage. Do not edit these files by hand.
