"""MATHLAS_ENCODER laptop tier (the 0.6B-built sibling index). Pinned contracts:

  * tier parsing: unset/empty/"8b" -> flagship (path unchanged); "0.6b"/"0.6B"/
    "06b" -> the ``_06b.npz`` sibling; anything else raises (a typo must never
    silently serve the wrong embedding space);
  * path mapping: the sibling is REQUIRED to exist (honest FileNotFoundError
    naming scripts/build_06b_index.py), an already-``_06b`` path passes through;
  * the MCP server serves the sibling index end-to-end under
    MATHLAS_ENCODER=0.6b (synthetic hashing-embedder indexes, no model
    download), including composed with MATHLAS_QUANTIZED;
  * MATHLAS_STATEMENT_INDEX + MATHLAS_ENCODER=0.6b is refused (the statement
    channel is 8B-space; mixing encoders would be garbage).

Real-index recall/latency numbers for the tier live in docs/QUANTIZED_TIER.md
(produced by scripts/eval_06b_tier.py); these tests pin the selection
mechanism only.
"""
import json
import os

import numpy as np
import pytest

from mathlas import server
from mathlas.embed import HashingEmbedder
from mathlas.retrieve.quantized import build_quantized_artifacts


def _write_index(path, n=24, dim=256, token_prefix="tok"):
    """A small dev index npz shaped like the production one, whose matrix is a
    HashingEmbedder over the slogans (so query-time embedding matches)."""
    emb = HashingEmbedder(dim=dim)
    meta = [{"doc_id": f"d{i}", "slogan": f"unique {token_prefix}{i} alt{i}",
             "statement": f"statement {i}"} for i in range(n)]
    M = emb.encode([m["slogan"] for m in meta]).astype(np.float16)
    np.savez(path, matrix=M, dim=dim, model="hashing", embedder="hashing",
             meta=json.dumps(meta))
    return path


# --------------------------------------------------------------------- #
# tier parsing + path mapping
# --------------------------------------------------------------------- #
def test_encoder_tier_default_and_aliases(monkeypatch):
    monkeypatch.delenv("MATHLAS_ENCODER", raising=False)
    assert server._encoder_tier() == "8b"
    for raw in ("", "  ", "8b", "8B"):
        monkeypatch.setenv("MATHLAS_ENCODER", raw)
        assert server._encoder_tier() == "8b"
    for raw in ("0.6b", "0.6B", "06b", " 0.6b "):
        monkeypatch.setenv("MATHLAS_ENCODER", raw)
        assert server._encoder_tier() == "0.6b"


def test_encoder_tier_rejects_typo(monkeypatch):
    monkeypatch.setenv("MATHLAS_ENCODER", "0.6")
    with pytest.raises(ValueError, match="MATHLAS_ENCODER"):
        server._encoder_tier()


def test_encoder_index_path_mapping(tmp_path):
    base = os.path.join(tmp_path, "index.npz")
    sib = os.path.join(tmp_path, "index_06b.npz")
    # 8b: untouched, regardless of what exists
    assert server._encoder_index_path(base, "8b") == base
    # 0.6b: sibling required
    with pytest.raises(FileNotFoundError, match="build_06b_index"):
        server._encoder_index_path(base, "0.6b")
    _write_index(sib)
    assert server._encoder_index_path(base, "0.6b") == sib
    # an explicit _06b path passes through unchanged
    assert server._encoder_index_path(sib, "0.6b") == sib


# --------------------------------------------------------------------- #
# end-to-end through the MCP server's retriever builder
# --------------------------------------------------------------------- #
def test_server_serves_06b_sibling(tmp_path, monkeypatch):
    """MATHLAS_ENCODER=0.6b swaps the served index for the _06b sibling; the
    two indexes carry DIFFERENT corpora here so the swap is observable."""
    base = _write_index(os.path.join(tmp_path, "dev.npz"), token_prefix="eight")
    sib = _write_index(os.path.join(tmp_path, "dev_06b.npz"),
                       token_prefix="small")
    monkeypatch.setenv("MATHLAS_INDEX", base)
    monkeypatch.delenv("MATHLAS_SEED", raising=False)
    monkeypatch.delenv("MATHLAS_QUANTIZED", raising=False)
    monkeypatch.delenv("MATHLAS_STATEMENT_INDEX", raising=False)
    monkeypatch.delenv("MATHLAS_RERANK", raising=False)
    server._RETRIEVER_CACHE.clear()
    try:
        monkeypatch.setenv("MATHLAS_ENCODER", "0.6b")
        r = server._build_retriever(None, 0)
        assert r.index_path == sib
        hits = r.retrieve("unique small7", k=2, mode="dense")
        assert hits and hits[0].meta["doc_id"] == "d7"
        # default tier still serves the base index
        monkeypatch.setenv("MATHLAS_ENCODER", "8b")
        server._RETRIEVER_CACHE.clear()
        r = server._build_retriever(None, 0)
        assert r.index_path == base
        hits = r.retrieve("unique eight4", k=2, mode="dense")
        assert hits and hits[0].meta["doc_id"] == "d4"
    finally:
        server._RETRIEVER_CACHE.clear()


def test_server_06b_composes_with_quantized(tmp_path, monkeypatch):
    base = _write_index(os.path.join(tmp_path, "dev.npz"), token_prefix="eight")
    sib = _write_index(os.path.join(tmp_path, "dev_06b.npz"),
                       token_prefix="small")
    build_quantized_artifacts(sib, log=lambda *_: None)
    monkeypatch.setenv("MATHLAS_INDEX", base)
    monkeypatch.setenv("MATHLAS_ENCODER", "0.6b")
    monkeypatch.setenv("MATHLAS_QUANTIZED", "binary")
    monkeypatch.delenv("MATHLAS_SEED", raising=False)
    monkeypatch.delenv("MATHLAS_STATEMENT_INDEX", raising=False)
    monkeypatch.delenv("MATHLAS_RERANK", raising=False)
    server._RETRIEVER_CACHE.clear()
    try:
        r = server._build_retriever(None, 0)
        assert r.index_path == sib
        assert r._quant is not None and r._quant.kind == "binary"
        hits = r.retrieve("unique small3", k=2, mode="dense")
        assert hits and hits[0].meta["doc_id"] == "d3"
    finally:
        server._RETRIEVER_CACHE.clear()


def test_server_06b_missing_sibling_is_honest(tmp_path, monkeypatch):
    base = _write_index(os.path.join(tmp_path, "dev.npz"))
    monkeypatch.setenv("MATHLAS_INDEX", base)
    monkeypatch.setenv("MATHLAS_ENCODER", "0.6b")
    monkeypatch.delenv("MATHLAS_SEED", raising=False)
    server._RETRIEVER_CACHE.clear()
    try:
        with pytest.raises(FileNotFoundError, match="build_06b_index"):
            server._build_retriever(None, 0)
    finally:
        server._RETRIEVER_CACHE.clear()


def test_server_06b_rejects_statement_index(tmp_path, monkeypatch):
    base = _write_index(os.path.join(tmp_path, "dev.npz"))
    sib = _write_index(os.path.join(tmp_path, "dev_06b.npz"))
    monkeypatch.setenv("MATHLAS_INDEX", base)
    monkeypatch.setenv("MATHLAS_ENCODER", "0.6b")
    monkeypatch.setenv("MATHLAS_STATEMENT_INDEX", sib)  # any path: refused first
    monkeypatch.delenv("MATHLAS_SEED", raising=False)
    monkeypatch.delenv("MATHLAS_QUANTIZED", raising=False)
    server._RETRIEVER_CACHE.clear()
    try:
        with pytest.raises(ValueError, match="MATHLAS_STATEMENT_INDEX"):
            server._build_retriever(None, 0)
    finally:
        server._RETRIEVER_CACHE.clear()
