---
title: Memory budget
description: How SrvDB estimates resident memory per vector in each index mode, and how a memory budget picks between flat and 4, 2 or 1-bit RaBitQ as the database grows.
---

A memory budget caps how much RAM a database's searches need. You set it
once, at create time, with `index="auto"` and `memory_mb`. SrvDB then
keeps the most precise mode whose estimated footprint fits, and moves to
fewer bits per dimension when growth would break the budget. The design
is recorded in
[ADR 0018](https://github.com/Srinivas26k/srvdb/blob/master/docs/adr/0018-memory-budget.md).

## What counts as resident

Resident memory is what a search must keep in RAM to run at full speed:
heap structures, plus mapped file pages that every search reads. Pages
searches do not read can be dropped by the operating system and are not
counted. For `D` dimensions and `n` stored records (deleted ones too,
until `compact()`):

| Mode | Bytes per vector | Fixed bytes |
|------|------------------|-------------|
| flat | `4D` | 0 |
| sq8 | `D` | `8D` |
| pq | `D/8` (or `D/4`) | `1024 D` |
| RaBitQ, `b` bits | `ceil(D b / 8) + 8` | `4 D²` (the rotation) |
| HNSW graph | `16 m + 96` on top of the vectors | 0 |
| links | 64 per link | 0 |
| keyword index | 64 per (document, term) pair, plus 64 per document | 0 |

In RaBitQ mode the full vectors stay on disk and are read only for the
few candidates being reranked, so they are not counted. The codes are
padded to blocks of 32 vectors. The rotation is the same size whatever
the count, which matters for small databases of large dimension.

The RaBitQ figures are checked against measured heap allocations in the
test suite. The HNSW figure is an upper estimate, and the links and
keyword figures are rough.

## How the mode is chosen

1. The limit is 80 % of the budget. The other 20 % covers what the model
   does not count, and the slack a growing buffer can hold.
2. The candidates are, from most to least precise: flat (exact), then
   RaBitQ with 4, 2 and 1 bits. All of them keep the full vectors, so
   moving down is a rebuild of derived data, never a loss of vectors.
3. Before each add, SrvDB takes the first candidate, at or below the
   current one, that fits the count after the add.

SQ8 and PQ are not candidates. They keep no full-precision vectors, so
they could never be re-encoded to fewer bits later.

When even 1 bit does not fit, the add raises `ValueError` and nothing is
written. The mode never moves back up on its own.

## Crash safety

Moving down rebuilds the RaBitQ codes from `vectors.bin`, then records
the new mode in `index_settings.bin` with an atomic rename, and only then
appends the new vectors. If the process dies in between, the next open
either rebuilds the codes or repeats the step. A crash test kills a
writer at random moments while it steps down.
