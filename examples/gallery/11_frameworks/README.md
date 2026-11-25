# 11 LangChain and LlamaIndex

The same synthetic corpus stored three times: with `db.add`, through
`srvdb.integrations.langchain.SrvDBVectorStore` and through
`srvdb.integrations.llamaindex.SrvDBVectorStore`. Embeddings are character
n-gram TF-IDF + TruncatedSVD (128 dimensions), wrapped as a LangChain
`Embeddings` class and passed to LlamaIndex nodes directly, so nothing is
downloaded.

For every query the script checks that each adapter returns the same
top-5 ids as the direct search, runs one metadata-filtered query through
each framework (the filter runs inside SrvDB), and times the queries.

```bash
pip install "srvdb[langchain]" "srvdb[llamaindex]"
python examples/gallery/11_frameworks/frameworks.py [--fast]
```

A framework that is not installed is skipped with a message.

Output: the filtered results, the agreement with direct search, mean
latency per query, and `output/11_frameworks.png` with one latency bar
per path (direct, LangChain, LlamaIndex): the adapters' overhead over the
direct call. See the
[framework guide](https://srinivas26k.github.io/srvdb/guides/frameworks/).
