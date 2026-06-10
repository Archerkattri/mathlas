"""Quantized dense tier (mathlas/retrieve/quantized.py) — the laptop-class
serving path. Pinned contracts:

  * int8 per-dim symmetric quantization round-trips within scale/2 per element;
  * sign packing round-trips exactly;
  * the chunked packed-bit Hamming kernel equals a brute-force popcount;
  * two-stage retrieval (binary Hamming shortlist -> exact rescore) returns
    EXACTLY the brute-force fp32 top-k on a synthetic corpus when the
    shortlist covers the corpus, and the int8 path matches brute force too;
  * ``HybridRetriever.from_index(..., quantized=...)`` serves end-to-end from
    the quantized artifacts without touching the fp16 matrix, and refuses a
    request for artifacts that were never built (honest error, no fallback).

Real-index numbers (recall deltas on the 3.68M production index) live in
docs/QUANTIZED_TIER.md and are produced by scripts/eval_quantized_tier.py;
these tests pin the mechanism on synthetic data so the suite needs no 30 GB
download.
"""
import json
import os

import numpy as np
import pytest

from mathlas.embed import HashingEmbedder
from mathlas.retrieve.hybrid import HybridRetriever
from mathlas.retrieve.quantized import (
    QuantizedDenseIndex, build_quantized_artifacts, dequantize_int8,
    hamming_distances, npz_member_memmap, pack_signs, quant_paths,
    quantize_matrix_int8)

RNG = np.random.default_rng(0)


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _make_index(tmp_path, n=64, dim=64):
    """A small dev index npz shaped like the production one (fp16 matrix +
    in-npz meta JSON), plus its quantized sidecars."""
    M = _unit(RNG.standard_normal((n, dim))).astype(np.float16)
    meta = [{"doc_id": f"d{i}", "slogan": f"slogan {i}", "statement": f"stmt {i}"}
            for i in range(n)]
    path = os.path.join(tmp_path, "index.npz")
    np.savez(path, matrix=M, dim=dim, model="dev", meta=json.dumps(meta))
    build_quantized_artifacts(path, chunk_rows=17, log=lambda *_: None)
    return path, _unit(M.astype(np.float32))


# --------------------------------------------------------------------- #
# quantization primitives
# --------------------------------------------------------------------- #
def test_int8_roundtrip_error_bound():
    M = _unit(RNG.standard_normal((200, 128))).astype(np.float32)
    q8, scale = quantize_matrix_int8(M)
    back = dequantize_int8(q8, scale)
    # symmetric rounding: each element within half a quantization step
    assert np.all(np.abs(back - M) <= scale / 2 + 1e-7)
    # cosine of dequantized rows vs originals stays essentially 1
    cos = np.sum(_unit(back) * M, axis=1)
    assert cos.min() > 0.999


def test_int8_scale_never_zero():
    M = np.zeros((4, 8), dtype=np.float32)
    q8, scale = quantize_matrix_int8(M)
    assert np.all(scale > 0) and np.all(q8 == 0)
    assert np.all(dequantize_int8(q8, scale) == 0)


def test_pack_signs_roundtrip():
    M = RNG.standard_normal((50, 64)).astype(np.float32)
    bits = pack_signs(M)
    assert bits.shape == (50, 8) and bits.dtype == np.uint8
    assert np.array_equal(np.unpackbits(bits, axis=1).astype(bool), M > 0)


def test_hamming_kernel_matches_bruteforce():
    a = RNG.integers(0, 256, size=(300, 16), dtype=np.uint8)
    q = RNG.integers(0, 256, size=16, dtype=np.uint8)
    got = hamming_distances(a, q, chunk_rows=37)   # force multiple chunks
    want = np.unpackbits(np.bitwise_xor(a, q), axis=1).sum(axis=1)
    assert np.array_equal(got, want.astype(np.int32))


def test_npz_member_memmap_equals_load(tmp_path):
    path, M32 = _make_index(tmp_path)
    mm = npz_member_memmap(path)
    assert mm.dtype == np.float16 and mm.shape == (64, 64)
    assert np.array_equal(np.asarray(mm), np.load(path)["matrix"])


# --------------------------------------------------------------------- #
# two-stage retrieval == brute force
# --------------------------------------------------------------------- #
def test_int8_search_matches_bruteforce(tmp_path):
    """The int8 path is an EXACT brute-force dot over the dequantized rows
    (the quantization loss itself is measured on the real index, see
    docs/QUANTIZED_TIER.md — here we pin that search adds no further error)."""
    path, M32 = _make_index(tmp_path)
    idx = QuantizedDenseIndex(path, kind="int8")
    D = dequantize_int8(np.asarray(idx._q8), idx._scale)
    for qi in (0, 7, 33):
        q = M32[qi] + 0.05 * RNG.standard_normal(64).astype(np.float32)
        q = _unit(q)
        want = list(np.argsort(-(D @ q))[:10])
        assert idx.search(q, 10) == [int(i) for i in want]


def test_binary_rescore_matches_bruteforce_full_shortlist(tmp_path):
    path, M32 = _make_index(tmp_path)
    # shortlist >= corpus -> the rescore IS exact brute force (int8-dequant)
    idx = QuantizedDenseIndex(path, kind="binary", shortlist=64)
    D = dequantize_int8(np.asarray(idx._q8), idx._scale)
    for qi in (3, 11, 60):
        q = _unit(M32[qi] + 0.05 * RNG.standard_normal(64).astype(np.float32))
        want = list(np.argsort(-(D @ q))[:5])
        assert idx.search(q, 5) == [int(i) for i in want]


def test_binary_fp16_rescore_path(tmp_path):
    path, M32 = _make_index(tmp_path)
    # drop the int8 sidecar -> rescore="auto" falls back to fp16 memmap rows
    os.remove(quant_paths(path)["int8"])
    os.remove(quant_paths(path)["int8_scale"])
    idx = QuantizedDenseIndex(path, kind="binary", shortlist=64)
    assert idx._fp16 is not None and idx._q8 is None
    q = _unit(M32[5])
    want = list(np.argsort(-(M32 @ q))[:5])
    assert idx.search(q, 5) == [int(i) for i in want]


def test_binary_shortlist_contains_true_neighbor(tmp_path):
    path, M32 = _make_index(tmp_path)
    idx = QuantizedDenseIndex(path, kind="binary", shortlist=8)
    for qi in range(0, 64, 9):           # exact-row queries: Hamming dist 0
        assert idx.search(M32[qi], 1) == [qi]


def test_missing_artifacts_is_honest_error(tmp_path):
    M = _unit(RNG.standard_normal((8, 16))).astype(np.float16)
    path = os.path.join(tmp_path, "bare.npz")
    np.savez(path, matrix=M, dim=16, meta=json.dumps(
        [{"doc_id": str(i), "slogan": "s", "statement": "t"} for i in range(8)]))
    with pytest.raises(FileNotFoundError, match="build_quantized_artifacts"):
        QuantizedDenseIndex(path, kind="int8")
    with pytest.raises(FileNotFoundError, match="build_quantized_artifacts"):
        QuantizedDenseIndex(path, kind="binary")


# --------------------------------------------------------------------- #
# HybridRetriever.from_index(quantized=...) end-to-end
# --------------------------------------------------------------------- #
def _embedded_index(tmp_path, n=40):
    """Index whose matrix matches a HashingEmbedder over the slogans, so a
    query embedded at retrieve() time lives in the same space. Each slogan
    carries TWO distinguishing tokens (dim=256): hash() is randomized per
    process (PYTHONHASHSEED), and with one token a single bucket collision
    can make two doc vectors identical -> flaky ties."""
    emb = HashingEmbedder(dim=256)
    meta = [{"doc_id": f"d{i}", "slogan": f"unique slogan token{i} alt{i}",
             "statement": f"statement {i}"} for i in range(n)]
    M = emb.encode([m["slogan"] for m in meta]).astype(np.float16)
    path = os.path.join(tmp_path, "dev.npz")
    np.savez(path, matrix=M, dim=256, model="hashing", embedder="hashing",
             meta=json.dumps(meta))
    build_quantized_artifacts(path, log=lambda *_: None)
    return path, emb


@pytest.mark.parametrize("kind", ["int8", "binary"])
def test_from_index_quantized_end_to_end(tmp_path, kind):
    path, emb = _embedded_index(tmp_path)
    r = HybridRetriever.from_index(path, embedder=emb, quantized=kind,
                                   bm25_cache=None)
    assert r._quant is not None and r._emb is None    # fp16 never loaded
    assert r.index_dim == 256
    hits = r.retrieve("unique slogan token7", k=3)
    assert hits and hits[0].meta["doc_id"] == "d7"
    # dense-only mode exercises the quantized channel alone
    hits = r.retrieve("unique slogan token12", k=3, mode="dense")
    assert hits[0].meta["doc_id"] == "d12"


def test_from_index_quantized_matches_default_topk(tmp_path):
    """int8-quantized serving returns the same dense top-k as the default
    fp32-in-RAM path on the same index (the no-surprise contract)."""
    path, emb = _embedded_index(tmp_path)
    r_full = HybridRetriever.from_index(path, embedder=emb, bm25_cache=None)
    r_q8 = HybridRetriever.from_index(path, embedder=emb, quantized="int8",
                                      bm25_cache=None)
    for q in ("unique slogan token3", "token19 slogan", "something unrelated"):
        full = [c.meta["doc_id"] for c in r_full.retrieve(q, k=5, mode="dense")]
        q8 = [c.meta["doc_id"] for c in r_q8.retrieve(q, k=5, mode="dense")]
        assert full == q8


def test_from_index_quantized_rejects_statement_index(tmp_path):
    path, emb = _embedded_index(tmp_path)
    with pytest.raises(ValueError, match="statement_index"):
        HybridRetriever.from_index(path, embedder=emb, quantized="int8",
                                   statement_index=path, bm25_cache=None)


def test_server_env_var_serves_quantized(tmp_path, monkeypatch):
    """MATHLAS_QUANTIZED=binary makes the MCP server serve the laptop tier."""
    from mathlas import server
    path, _ = _embedded_index(tmp_path)
    monkeypatch.setenv("MATHLAS_INDEX", path)
    monkeypatch.setenv("MATHLAS_QUANTIZED", "binary")
    monkeypatch.delenv("MATHLAS_SEED", raising=False)
    server._RETRIEVER_CACHE.clear()
    try:
        r = server._build_retriever(None, 0)
        assert r._quant is not None and r._quant.kind == "binary"
        assert r.retrieve("unique slogan token4", k=2)[0].meta["doc_id"] == "d4"
        monkeypatch.setenv("MATHLAS_QUANTIZED", "int4")   # unsupported
        server._RETRIEVER_CACHE.clear()
        with pytest.raises(ValueError, match="MATHLAS_QUANTIZED"):
            server._build_retriever(None, 0)
    finally:
        server._RETRIEVER_CACHE.clear()


def test_quantized_index_row_count_guard(tmp_path):
    """A quantized index whose row count disagrees with the meta is refused."""
    path, emb = _embedded_index(tmp_path)
    docs_meta = [{"doc_id": "only-one", "slogan": "s", "statement": "t"}]
    bad = os.path.join(tmp_path, "bad.npz")
    M = np.load(path)["matrix"]
    np.savez(bad, matrix=M, dim=64, meta=json.dumps(docs_meta))
    build_quantized_artifacts(bad, log=lambda *_: None)
    with pytest.raises(ValueError, match="rows"):
        HybridRetriever.from_index(bad, embedder=emb, quantized="int8",
                                   bm25_cache=None)
