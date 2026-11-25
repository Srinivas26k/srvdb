"""Deterministic, offline bag-of-words embedding for adapter tests."""

import hashlib

DIM = 128


def embed(text):
    """Hash each lower-cased word into one of DIM buckets."""
    vec = [0.0] * DIM
    for word in text.lower().split():
        digest = hashlib.sha256(word.encode()).digest()
        vec[int.from_bytes(digest[:4], "little") % DIM] += 1.0
    if not any(vec):
        vec[0] = 1.0
    return vec
