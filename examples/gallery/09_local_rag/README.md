# 09 Local RAG

The retrieval half of a RAG application, offline, over real text: the
SrvDB documentation pages in this repository
(`website/src/content/docs/{start,concepts,guides}`), split into chunks at
`##` headings. Each chunk is stored with its page, heading and text as
metadata, and keyword search is enabled on the text.

A small labelled set of questions (each with the page that answers it)
measures hit@k, the share of questions whose page is among the top k
chunks, for vector, keyword (`text=`) and hybrid search. For each
question the script prints the best chunk: the context you would pass to
a local LLM.

```bash
python examples/gallery/09_local_rag/local_rag.py [--fast]
python examples/gallery/09_local_rag/local_rag.py --model /path/to/all-MiniLM-L6-v2
```

By default embeddings are TF-IDF + TruncatedSVD (128 dimensions), with no
download. `--model DIR` loads a sentence-transformers model you have
already downloaded (`pip install sentence-transformers`); the script sets
`HF_HUB_OFFLINE=1` so nothing is fetched. Getting the model is your
explicit step, never the script's.

Output: one line per question, hit@1..5 per method, and
`output/09_local_rag.png` with hit@k against k, one line per method.

The question set is small (15 questions, 5 with `--fast`), so one question
moves hit@k by several points; it shows how to measure, not a benchmark.
For generation, see the [local RAG guide](https://srinivas26k.github.io/srvdb/guides/local-rag/).
