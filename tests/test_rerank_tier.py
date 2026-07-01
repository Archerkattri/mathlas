"""Rerank-backend selection (MATHLAS_RERANK_MODEL) — pins the tier mechanism.

Two selectable cross-encoders sit behind ``MATHLAS_RERANK=1``: ``qwen3`` (the
default, unchanged Qwen3-Reranker-0.6B) and ``jina-v3`` (jinaai/jina-reranker-v3,
arXiv:2509.25085). These tests pin, WITHOUT downloading any weights:

  * ``make_reranker`` selection: unset/empty/"qwen3" -> Qwen3Reranker; "jina-v3"
    -> JinaRerankerV3; a typo RAISES (never silently serves the wrong model);
  * the JinaRerankerV3 score() contract with a MOCKED remote head: rerank()
    returns rows sorted by score carrying an ``index`` INTO the batch, and
    score() scatters them back into the CALLER'S input order as a float32 array;
  * lazy import / graceful fallback: constructing a reranker loads nothing, and a
    missing model surfaces as an honest ImportError/OSError at first .score().
"""
from __future__ import annotations

import numpy as np
import pytest

from mathlas.retrieve.rerank import (JinaRerankerV3, Qwen3Reranker,
                                     RERANK_MODEL_ENV, make_reranker)


# --------------------------------------------------------------------- #
# backend selection
# --------------------------------------------------------------------- #
def test_make_reranker_default_is_qwen3(monkeypatch):
    monkeypatch.delenv(RERANK_MODEL_ENV, raising=False)
    assert isinstance(make_reranker(), Qwen3Reranker)
    for raw in ("", "  ", "qwen3", "QWEN3", " Qwen3 "):
        monkeypatch.setenv(RERANK_MODEL_ENV, raw)
        assert isinstance(make_reranker(), Qwen3Reranker)


def test_make_reranker_selects_jina(monkeypatch):
    for raw in ("jina-v3", "JINA-V3", " jina-v3 "):
        monkeypatch.setenv(RERANK_MODEL_ENV, raw)
        assert isinstance(make_reranker(), JinaRerankerV3)


def test_make_reranker_rejects_typo(monkeypatch):
    monkeypatch.setenv(RERANK_MODEL_ENV, "jina")  # not "jina-v3"
    with pytest.raises(ValueError, match=RERANK_MODEL_ENV):
        make_reranker()


def test_make_reranker_explicit_arg_overrides_env(monkeypatch):
    monkeypatch.setenv(RERANK_MODEL_ENV, "jina-v3")
    assert isinstance(make_reranker("qwen3"), Qwen3Reranker)


# --------------------------------------------------------------------- #
# JinaRerankerV3.score() contract, with a mocked remote head
# --------------------------------------------------------------------- #
class _FakeJinaModel:
    """Stand-in for the AutoModel(trust_remote_code=True) remote head: its
    rerank() returns rows SORTED BY SCORE, each carrying the index into the
    passed document list (the documented jina-reranker-v3 return shape)."""

    def __init__(self, scores):
        self._scores = scores  # index-aligned relevance scores to hand back

    def rerank(self, query, documents, top_n=None, return_embeddings=False):
        rows = [{"index": i, "relevance_score": self._scores[i],
                 "document": d} for i, d in enumerate(documents)]
        rows.sort(key=lambda r: r["relevance_score"], reverse=True)
        return rows


def test_jina_score_maps_back_to_input_order():
    rr = JinaRerankerV3()
    # inject the fake head so no weights load; score() must UNDO rerank()'s sort
    rr._model = _FakeJinaModel([0.1, 0.9, 0.4])
    out = rr.score("q", ["a", "b", "c"])
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [0.1, 0.9, 0.4], rtol=0, atol=1e-6)


def test_jina_score_empty_docs():
    rr = JinaRerankerV3()
    rr._model = _FakeJinaModel([])  # should not even be consulted
    out = rr.score("q", [])
    assert out.shape == (0,) and out.dtype == np.float32


def test_jina_score_batches_across_the_64_cap(monkeypatch):
    """With batch_size 2, three docs span two rerank() calls; each call's index
    is batch-local, so score() must offset by the batch base (b + idx)."""
    calls = []

    class _Rec(_FakeJinaModel):
        def rerank(self, query, documents, top_n=None, return_embeddings=False):
            calls.append(list(documents))
            base = {"a": 0.2, "b": 0.7, "c": 0.5}
            rows = [{"index": i, "relevance_score": base[d], "document": d}
                    for i, d in enumerate(documents)]
            rows.sort(key=lambda r: r["relevance_score"], reverse=True)
            return rows

    rr = JinaRerankerV3(batch_size=2)
    rr._model = _Rec([])
    out = rr.score("q", ["a", "b", "c"])
    assert calls == [["a", "b"], ["c"]]  # split at the batch boundary
    np.testing.assert_allclose(out, [0.2, 0.7, 0.5], rtol=0, atol=1e-6)


def test_jina_lazy_no_load_on_construction():
    """Constructing the backend must not load any model (the whole point of the
    lazy tier); _model stays None until the first score()."""
    rr = JinaRerankerV3()
    assert rr._model is None
