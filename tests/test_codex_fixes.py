"""Regression tests for two SAFE Codex-found retrieval bugs (ride into v1.2).

Each test FAILS before its fix and PASSES after. All CPU-only, GPU-free, no
network, no model download: they exercise the dep-free dev paths only.

  1. HashingEmbedder used Python's per-process-salted builtin hash() on
     strings, so a persisted hashing index built in one process and queried in
     another landed docs and queries in different feature spaces. The fix uses
     a deterministic stdlib (blake2b) hash. (NOTE: the served Qwen3-Embedding
     dense index is real embeddings and never touches hash(); unaffected.)
  2. HybridRetriever.retrieve() silently returned an empty result for an
     unknown `mode` (e.g. a trailing-space typo) instead of erroring.

The third fix (mathlas.__version__ vs pyproject) is in test_version_metadata.py.
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest


# --------------------------------------------------------------------- #
# 1. deterministic embedding hash (process-independent)
# --------------------------------------------------------------------- #
_VEC_PROBE = (
    "import numpy as np;"
    "from mathlas.embed import HashingEmbedder;"
    "v = HashingEmbedder(dim=256).encode(['contraction mapping fixed point'])[0];"
    "print(' '.join(f'{x:.6f}' for x in v))"
)


def _encode_in_subprocess(seed: str):
    import os

    env = dict(os.environ)
    env["PYTHONHASHSEED"] = seed
    env["OMP_NUM_THREADS"] = "2"
    env["CUDA_VISIBLE_DEVICES"] = ""
    out = subprocess.run(
        [sys.executable, "-c", _VEC_PROBE],
        env=env, capture_output=True, text=True, check=True,
    ).stdout.split()
    return np.array([float(x) for x in out], dtype=np.float32)


def test_hashing_encode_survives_process_restart():
    """The SAME text -> the SAME embedded vector across two fresh interpreters
    with DIFFERENT PYTHONHASHSEEDs. With the old builtin hash() the per-process
    salt sent tokens to different buckets/signs, so a persisted index and a
    later query landed in different feature spaces. Drives the real encode()."""
    v_a = _encode_in_subprocess("0")
    v_b = _encode_in_subprocess("123456789")
    assert v_a.shape == v_b.shape == (256,)
    assert np.count_nonzero(v_a) > 1  # the basis actually carries tokens
    assert np.allclose(v_a, v_b, atol=0.0)




# --------------------------------------------------------------------- #
# 2. unknown retrieve mode raises (instead of empty result)
# --------------------------------------------------------------------- #
def _retriever():
    from mathlas.embed import HashingEmbedder
    from mathlas.retrieve.corpus import Document
    from mathlas.retrieve.hybrid import HybridRetriever

    docs = [
        Document(doc_id="0", slogan="a contraction mapping has a unique fixed point",
                 statement="x", name="Banach"),
        Document(doc_id="1", slogan="the fundamental group of the circle is Z",
                 statement="x", name="Pi1 S1"),
        Document(doc_id="2", slogan="every bounded monotone sequence converges",
                 statement="x", name="MCT"),
    ]
    return HybridRetriever(docs, embedder=HashingEmbedder())


QUERY = "contraction mapping unique fixed point"
