---
title: Build a local RAG app
description: Index your own documents with a local embedding model and retrieve context for an LLM, fully offline.
---

Retrieval-augmented generation (RAG) finds the passages most relevant to a
question and gives them to a language model as context. Here every piece runs
on your machine: the embedding model, SrvDB and (optionally) the LLM.

## Install

```bash
pip install srvdb sentence-transformers
```

`all-MiniLM-L6-v2` is a small, fast embedding model (384 dimensions). It is
downloaded once, then works offline.

## 1. Split documents into chunks

```python
from pathlib import Path

def chunks(text, size=800, overlap=100):
    step = size - overlap
    return [text[i:i + size] for i in range(0, max(len(text) - overlap, 1), step)]

passages = []
for path in Path("notes").glob("**/*.md"):
    for n, chunk in enumerate(chunks(path.read_text(encoding="utf-8"))):
        passages.append({"id": f"{path}#{n}", "source": str(path), "text": chunk})
```

## 2. Embed and store

```python
import json
import srvdb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
db = srvdb.Database("./rag-db", dimension=384)

vectors = model.encode([p["text"] for p in passages], normalize_embeddings=True)
db.upsert(
    ids=[p["id"] for p in passages],
    embeddings=vectors.tolist(),
    metadatas=[json.dumps({"source": p["source"], "text": p["text"]}) for p in passages],
)
db.persist()
```

`upsert` makes the script safe to re-run: changed chunks replace the old ones.

## 3. Retrieve

```python
def retrieve(question, k=4):
    q = model.encode(question, normalize_embeddings=True).tolist()
    return [json.loads(db.get(doc_id)) | {"score": score} for doc_id, score in db.search(q, k)]

for hit in retrieve("How do I rotate the API keys?"):
    print(f'{hit["score"]:.2f}  {hit["source"]}')
```

## 4. Ask a local LLM

Any local model works. With [Ollama](https://ollama.com) (`pip install ollama`):

```python
import ollama

question = "How do I rotate the API keys?"
context = "\n\n".join(hit["text"] for hit in retrieve(question))
reply = ollama.chat(model="llama3.2", messages=[
    {"role": "system", "content": "Answer only from the context.\n\n" + context},
    {"role": "user", "content": question},
])
print(reply["message"]["content"])
```

## Tips

- **Pick a mode later.** Start with `flat` (exact). For large collections,
  see [Choosing an index mode](/srvdb/guides/index-modes/), and measure
  recall against `flat` on your own data.
- **Keep text in metadata** (as above) or keep a separate store keyed by id.
- **Persist after each batch**, not after every chunk.
- Integrations for LangChain and LlamaIndex are
  [coming soon](/srvdb/project/integrations/).
