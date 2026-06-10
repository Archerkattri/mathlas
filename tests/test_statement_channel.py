"""Statement channel (dual dense) + its server env wiring.

The second dense channel embeds every doc by its cleaned STATEMENT text and is
folded into the dense ranking by per-doc max-sim (HybridRetriever._dense_rank).
Full-corpus numbers in docs/RETRIEVAL_UPGRADE_NOTES.md; these tests pin the
mechanics on a tiny hashing-embedder index:

  * a statement-shaped query that the slogan channel cannot see ranks via the
    statement channel (the whole point of the channel);
  * default behaviour (no statement matrix) is unchanged;
  * row-misaligned matrices are refused loudly;
  * from_index(statement_index=...) loads the sibling npz end-to-end;
  * MATHLAS_STATEMENT_INDEX wires it into the MCP server ("auto" resolves
    beside the index; a missing path is an honest FileNotFoundError; unset
    keeps today's single-channel behaviour).
"""
import json
import os

import numpy as np
import pytest

from mathlas.embed import HashingEmbedder
from mathlas.retrieve.hybrid import HybridRetriever


def _dual_index(tmp_path, n=24):
    """Dev npz pair: slogan matrix + row-aligned statement matrix. Slogans and
    statements use DISJOINT token sets so each channel only sees its own
    surface form (the cross-representation gap in miniature)."""
    emb = HashingEmbedder(dim=256)
    meta = [{"doc_id": f"d{i}", "slogan": f"slogan slotok{i}",
             "statement": f"statement stmtok{i}"} for i in range(n)]
    M = emb.encode([m["slogan"] for m in meta]).astype(np.float16)
    S = emb.encode([m["statement"] for m in meta]).astype(np.float16)
    path = os.path.join(tmp_path, "dev.npz")
    np.savez(path, matrix=M, dim=256, model="hashing", embedder="hashing",
             meta=json.dumps(meta))
    spath = os.path.join(tmp_path, "index_full_statement.npz")
    np.savez(spath, matrix=S, dim=256, model="hashing", embedder="hashing",
             channel="statement", n_docs=n)
    return path, spath, emb


def test_statement_channel_bridges_statement_queries(tmp_path):
    path, spath, emb = _dual_index(tmp_path)
    base = HybridRetriever.from_index(path, embedder=emb, bm25_cache=None,
                                      with_bm25=False)
    dual = HybridRetriever.from_index(path, embedder=emb, bm25_cache=None,
                                      with_bm25=False, statement_index=spath)
    q = "stmtok7"                       # statement surface form only
    assert dual._stmt_emb is not None and base._stmt_emb is None
    hits = dual.retrieve(q, k=3, mode="dense")
    assert hits and hits[0].meta["doc_id"] == "d7"
    base_top = [c.meta["doc_id"] for c in base.retrieve(q, k=3, mode="dense")]
    assert base_top[:1] != ["d7"]       # slogan channel alone cannot see it
    # and slogan-shaped queries still rank via the slogan channel
    assert dual.retrieve("slotok11", k=3, mode="dense")[0].meta["doc_id"] == "d11"


def test_statement_matrix_row_mismatch_refused(tmp_path):
    path, spath, emb = _dual_index(tmp_path)
    bad = np.asarray(np.load(spath)["matrix"])[:-1]
    with pytest.raises(ValueError, match="statement"):
        HybridRetriever.from_index(path, embedder=emb, bm25_cache=None,
                                   with_bm25=False,
                                   statement_index=_save_bad(tmp_path, bad))


def _save_bad(tmp_path, matrix):
    p = os.path.join(tmp_path, "bad_statement.npz")
    np.savez(p, matrix=matrix, dim=matrix.shape[1])
    return p


def test_lean_loader_exact_and_copy_free(tmp_path):
    """The OOM fix (2026-06-10): from_index streams npz matrices into ONE
    unit-norm fp32 array instead of materialising np.load + astype + divide
    temps (which OOM-killed the full dual-channel load on a 251 GB box).
    Pin (a) the lean loader's values equal the old math exactly and (b) the
    fp32 in-place adoption contract of _ensure_unit_rows."""
    from mathlas.retrieve.hybrid import _ensure_unit_rows, _load_unit_fp32_matrix
    rng = np.random.default_rng(7)
    M = rng.standard_normal((301, 32)).astype(np.float16)   # odd row count
    path = os.path.join(tmp_path, "m.npz")
    np.savez(path, matrix=M, dim=32)
    lean = _load_unit_fp32_matrix(path)
    n = np.linalg.norm(M.astype(np.float32), axis=1, keepdims=True)
    n[n == 0] = 1.0
    old = (M.astype(np.float32) / n).astype(np.float32)     # the old math
    assert lean.dtype == np.float32 and np.array_equal(lean, old)
    # fp32 writable input is adopted (normalised in place, no copy) ...
    x = rng.standard_normal((10, 4)).astype(np.float32)
    assert _ensure_unit_rows(x) is x
    # ... while fp16 input still gets the old copy behaviour
    y16 = rng.standard_normal((10, 4)).astype(np.float16)
    out = _ensure_unit_rows(y16)
    assert out.dtype == np.float32 and out is not y16
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-6)


def test_server_env_statement_index(tmp_path, monkeypatch):
    from mathlas import server
    path, spath, _ = _dual_index(tmp_path)
    monkeypatch.setenv("MATHLAS_INDEX", path)
    monkeypatch.delenv("MATHLAS_SEED", raising=False)
    monkeypatch.delenv("MATHLAS_QUANTIZED", raising=False)
    try:
        # unset -> single channel, exactly today's behaviour
        monkeypatch.delenv("MATHLAS_STATEMENT_INDEX", raising=False)
        server._RETRIEVER_CACHE.clear()
        assert server._build_retriever(None, 0)._stmt_emb is None
        # explicit path -> dual channel served
        monkeypatch.setenv("MATHLAS_STATEMENT_INDEX", spath)
        server._RETRIEVER_CACHE.clear()
        r = server._build_retriever(None, 0)
        assert r._stmt_emb is not None
        assert r.retrieve("stmtok3", k=2, mode="dense")[0].meta["doc_id"] == "d3"
        # "auto" -> the sibling index_full_statement.npz beside the index
        monkeypatch.setenv("MATHLAS_STATEMENT_INDEX", "auto")
        server._RETRIEVER_CACHE.clear()
        assert server._build_retriever(None, 0)._stmt_emb is not None
        # missing path -> honest error naming the builder
        monkeypatch.setenv("MATHLAS_STATEMENT_INDEX",
                           os.path.join(tmp_path, "nope.npz"))
        server._RETRIEVER_CACHE.clear()
        with pytest.raises(FileNotFoundError, match="MATHLAS_STATEMENT_INDEX"):
            server._build_retriever(None, 0)
    finally:
        server._RETRIEVER_CACHE.clear()
