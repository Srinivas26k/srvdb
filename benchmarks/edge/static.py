"""Facts taken from documentation and source, not measured.

Each entry names its source so a reader can check it. Telemetry notes
were read from the installed package source of the measured version;
the strace connect counts in results.json are the measured evidence.
"""

from __future__ import annotations

GH = "https://" + "github.com"

STATIC = {
    "srvdb": {
        "runtime": "in-process library",
        "doc": "https://srinivas26k.github.io/srvdb/",
        "telemetry": "none: the library makes no network calls (AGENTS.md section 9)",
        "aarch64_ci": ".github/workflows/ci.yml job test-rust-arm64 'Rust core tests "
        "(aarch64, NEON)' on ubuntu-24.04-arm; job wheel-arm64 builds the aarch64 wheel",  # noqa: E501
    },
    "chroma": {
        "runtime": "in-process with chromadb.PersistentClient (a client/server mode also exists)",  # noqa: E501
        "doc": "https://docs.trychroma.com/docs/run-chroma/clients",
        "telemetry": "setting anonymized_telemetry defaults to True (chromadb/config.py); "  # noqa: E501
        "in chromadb 1.5.9 the Posthog client's capture() is a no-op "
        "(chromadb/telemetry/product/posthog.py)",
    },
    "lancedb": {
        "runtime": "in-process library (embedded, local directory)",
        "doc": "https://docs.lancedb.com/",
        "telemetry": "no phone-home found; lancedb/otel.py only bridges metrics into a "
        "MeterProvider the application supplies",
    },
    "faiss": {
        "runtime": "in-process library",
        "doc": f"{GH}/facebookresearch/faiss/wiki/Getting-started",
        "telemetry": "none found in package source",
    },
    "hnswlib": {
        "runtime": "in-process library",
        "doc": f"{GH}/nmslib/hnswlib",
        "telemetry": "none found in package source",
    },
    "usearch": {
        "runtime": "in-process library",
        "doc": "https://unum-cloud.github" + ".io/USearch/python/",
        "telemetry": "none found in package source",
    },
    "sqlite-vec": {
        "runtime": "in-process SQLite extension",
        "doc": "https://alexgarcia.xyz/sqlite-vec/python.html",
        "telemetry": "none found in package source",
    },
}
