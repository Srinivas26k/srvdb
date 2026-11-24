"""Optional adapters that expose a SrvDB database to AI frameworks.

Each adapter lives in its own module and imports its framework only when
that module is imported, so ``import srvdb`` never pulls in a framework:

- ``srvdb.integrations.langchain`` (``pip install "srvdb[langchain]"``)
- ``srvdb.integrations.llamaindex`` (``pip install "srvdb[llamaindex]"``)

The adapters make no network calls; embeddings come from whatever model
the application passes in.
"""
