# ADR 0009: LangChain and LlamaIndex adapters as optional extras

- **Status:** Accepted
- **Date:** 2026-11-24

## Context

Most RAG applications are built on LangChain or LlamaIndex, and both
expect a vector store that implements their interface (issue #26, roadmap
item F1). Users would otherwise each write their own glue around
`srvdb.Database`. The constraints come from AGENTS.md: the core stays
dependency-free and makes no network calls; new optional integrations need
an ADR and must not become dependencies of the core package; every new
dependency needs a compatible licence.

## Decision

1. **Pure-Python adapters in `srvdb.integrations`.**
   `srvdb.integrations.langchain.SrvDBVectorStore` subclasses
   `langchain_core.vectorstores.VectorStore`, and
   `srvdb.integrations.llamaindex.SrvDBVectorStore` subclasses
   `BasePydanticVectorStore`. Both wrap a `srvdb.Database` (opened with
   `srvdb.open(...)`); nothing changes in the Rust core or the PyO3
   bindings.
2. **Optional extras, no core dependency.** The frameworks are declared
   only as `srvdb[langchain]` and `srvdb[llamaindex]`. `dependencies`
   stays empty, `import srvdb` imports no framework (tested), and
   importing an adapter without its extra raises an `ImportError` naming
   the extra to install.
3. **Version floors:** `langchain-core>=0.3` and `llama-index-core>=0.12`.
   These are the first lines with the current interfaces the adapters use
   (LangChain: `Document.id`, `get_by_ids`, pydantic 2; LlamaIndex:
   `BasePydanticVectorStore`, `delete_nodes`, `MetadataFilters` with
   nested filters and `NOT`). Tested against langchain-core 1.8.4 and
   llama-index-core 0.14.26. There are no upper bounds; CI catches breaks.
4. **Filtering after the search.** SrvDB has no native metadata filter
   yet, so the adapters run the vector search, drop hits whose metadata
   does not match, and double the number of candidates until `k` matches
   are found or every vector has been seen. Results are exact with respect
   to the filter, but a very selective filter on a large database costs
   close to a full scan plus a metadata read per candidate. LlamaIndex
   `delete(ref_doc_id)` and `delete_nodes(filters=...)` scan every record.
   When native filtered search lands (issue #25), the adapters switch to
   it without an API change.
5. **Persist on every write.** Each adapter add or delete calls
   `persist()` before returning, so anything a framework reports as stored
   survives a crash (ADR 0003). Frameworks have no commit step, so this is
   the only safe default.

## Licence review

Both extras were installed into a clean Python 3.13 venv on 2026-11-24
and checked with `uvx pip-licenses --from=mixed`: 78 packages. Both
frameworks are MIT. Most of the packages they pull in are MIT, Apache-2.0,
BSD or ISC, or dual-licensed with one of those (for example `packaging`,
`sniffio` and `dirtyjson`). The following are **outside the AGENTS.md
allow-list** and are listed here for the maintainer to decide. All of them
are permissive or weak-copyleft, and none is a dependency of the core
package:

| Package | Version | Licence | Pulled in by |
|---|---|---|---|
| certifi | 2026.11.20 | MPL-2.0 | requests/httpx (both) |
| tqdm | 4.70.1 | MPL-2.0 AND MIT | llama-index-core, nltk |
| orjson | 3.13.0 | MPL-2.0 AND (Apache-2.0 OR MIT) | langsmith (langchain) |
| typing_extensions | 4.16.0 | PSF-2.0 | both frameworks, pydantic |
| aiohappyeyeballs | 2.7.1 | PSF-2.0 | aiohttp (llamaindex) |
| defusedxml | 0.7.1 | PSF-2.0 | nltk (llamaindex) |
| greenlet | 3.5.6 | MIT AND PSF-2.0 | SQLAlchemy, platform-conditional (llamaindex) |
| regex | 2026.11.23 | Apache-2.0 AND CNRI-Python | nltk, tiktoken (llamaindex) |
| pillow | 13.0.0 | MIT-CMU | llama-index-core |
| numpy | 2.5.5 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | llama-index-core |

(The "Pulled in by" column comes from `uv pip tree --invert` on that venv.)
The adapters themselves add no network calls. `langchain-core` installs the
`langsmith` client, which only sends traces if the application turns
tracing on through environment variables. SrvDB never does.

## Alternatives considered

- **Separate packages (`srvdb-langchain`, `llama-index-vector-stores-srvdb`).**
  This is the frameworks' own convention and gives fully separate release
  cycles, but it means more packages to publish and version against
  `srvdb`. Extras keep a single wheel. A separate package can still be split
  out later without breaking imports.
- **Upstream the adapters into the framework repositories.** Their
  maintainers would decide the review timeline and API, and SrvDB features
  would reach users only after a framework release. This can be done later
  on top of this code.
- **Native filtered search first.** That would block adoption on a core
  change. Filtering after the search is correct now and is replaced
  transparently by #25.
- **Batching writes or leaving `persist()` to the caller.** This is faster
  for many small adds, but a framework would report data as stored when it
  is not yet durable. It can be added later as an opt-in flag.

## Consequences

- RAG users get a drop-in store with no core change and no new core
  dependency.
- The adapters must track two moving framework APIs. Their tests skip
  without the extras, so CI needs a job that installs them.
- Filtered queries and LlamaIndex deletes by document are O(n) in the worst
  case until #25.
- Persisting on every write makes bulk ingestion through the adapters
  slower than calling `Database.add` directly in batches.
- The maintainer accepted the transitive licences above (2026-11-24). The
  allow-list in AGENTS.md section 9 binds the core crates, the bindings and
  the published wheel. Optional extras may pull in permissive or weak-copyleft
  packages (MPL-2.0, PSF-2.0, MIT-CMU, CNRI-Python) because SrvDB neither
  bundles nor redistributes them; users install them only by asking for the
  extra.
