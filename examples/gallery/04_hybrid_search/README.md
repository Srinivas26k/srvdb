# 04 Hybrid search

A synthetic support knowledge base: articles from eight topics, each with
a unique ticket code like `KB-10421`. Embeddings are TF-IDF reduced to 64
dimensions with TruncatedSVD, which captures topics but blurs rare codes.
`db.enable_keyword_search("text")` indexes the article text for BM25.

Three ways to search, on two query sets:

| Call | Method |
|------|--------|
| `db.search(vector, k)` | vector only |
| `db.search(None, k, text=query)` | keyword (BM25) only |
| `db.search(vector, k, text=query)` | hybrid, fused with Reciprocal Rank Fusion |

- topic queries (three topic words): precision@5 by topic;
- code queries (a ticket code plus a topic word): hit@5 for that article.

```bash
python examples/gallery/04_hybrid_search/hybrid_search.py [--fast]
```

Output: precision and hit rate per method and `output/04_hybrid_search.png`,
grouped bars (vector, keyword, hybrid) for each query set.

What to look for: vector search alone rarely finds an exact code; keyword
search does. Rank fusion gives equal weight to both lists, so when the
vector list is confidently wrong (here, many articles share the topic
word) the fused result can land between the two. That is a real property
of RRF, worth measuring on your own queries before choosing a method.
See the [keyword and hybrid search guide](https://srinivas26k.github.io/srvdb/guides/keyword-search/).
