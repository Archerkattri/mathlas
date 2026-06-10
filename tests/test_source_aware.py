"""Source-aware retrieval (source_filter / source_weights) — opt-in knobs.

Motivation (RESULTS.md §3b, 2026-06-10): adding 2.34M web-mined Dolma docs to
the served 3.68M index crowded canonical targets out of the paper-level top-20
on the TheoremSearch-110 (corpus-only 13.6% -> 11.8%). These tests pin:

  * the canonical source classifier (URL/identifier string -> key);
  * DEFAULT PATH BYTE-IDENTICAL: with both knobs off, retrieve() returns
    exactly the pre-feature dense+BM25+RRF ranking (order AND float scores,
    asserted against an independent reimplementation of the fusion);
  * filter semantics: exclude/include act IN-CHANNEL (depth preserved — an
    excluded source never takes a channel slot);
  * weight semantics: score(d) = w_src(d) * RRF(d); w=0 == exclude;
  * typo safety: unknown source keys raise, never silently no-op;
  * MCP server plumbing of both params (12 tools stay 12).
"""
from __future__ import annotations

import numpy as np
import pytest

from mathlas.embed import HashingEmbedder
from mathlas.retrieve.bm25 import BM25Index
from mathlas.retrieve.corpus import KNOWN_SOURCE_KEYS, Document, source_key
from mathlas.retrieve.hybrid import HybridRetriever, rrf_fuse


# --------------------------------------------------------------------- #
# the canonical source classifier
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("doc_id,source,want", [
    ("dolma:1cf0#42", "arXiv (dolma-v1_7)", "dolma"),   # doc_id prefix wins
    ("12345", "arXiv (dolma-v1_7)", "dolma"),           # 'dolma' before 'arxiv'!
    ("85188", "https://stacks.math.columbia.edu/tag/06UU", "stacks"),
    ("9", "https://proofwiki.org/wiki/Banach_Fixed-Point_Theorem", "proofwiki"),
    ("7", "https://arxiv.org/abs/2301.00001", "arxiv"),
    ("8", "arXiv:2301.00001v2", "arxiv"),
    ("x", "https://math.uchicago.edu/~amathew/CRing.pdf", "other"),
    ("y", None, "other"),
])
def test_source_key_classifier(doc_id, source, want):
    assert source_key(doc_id, source) == want
    assert want in KNOWN_SOURCE_KEYS


# --------------------------------------------------------------------- #
# synthetic two-source corpus
# --------------------------------------------------------------------- #
def _doc(i, slogan, *, dolma=False):
    if dolma:
        return Document(doc_id=f"dolma:aa#{i}", slogan=slogan, statement=slogan,
                        source="arXiv (dolma-v1_7)")
    return Document(doc_id=str(i), slogan=slogan, statement=slogan,
                    name=f"Theorem {i}",
                    source=f"https://arxiv.org/abs/23{i:02d}.0000{i}")


def _corpus():
    """Docs crafted so that for the query below, a DOLMA doc ranks first by
    default in both channels and an ARXIV doc ranks right behind it."""
    return [
        _doc(0, "every contraction mapping on a complete metric space has a "
                "unique fixed point", dolma=True),
        _doc(1, "a contraction mapping on a complete metric space has a "
                "unique fixed point theorem"),
        _doc(2, "compact operators on a hilbert space admit a spectral "
                "decomposition", dolma=True),
        _doc(3, "the fundamental group of the circle is the integers"),
        _doc(4, "every bounded monotone sequence of reals converges"),
    ]


QUERY = "every contraction mapping on a complete metric space has a unique fixed point"


def test_default_path_is_byte_identical_to_plain_rrf():
    """Pin: knobs off => exactly the pre-feature ranking (order AND scores),
    against an independent reimplementation of dense+BM25+RRF."""
    docs = _corpus()
    emb = HashingEmbedder()
    retr = HybridRetriever(docs, embedder=emb)

    # independent fusion: dense = cosine argsort, sparse = BM25, RRF k=10
    mat = emb.encode([d.embed_text for d in docs])
    q = emb.encode([QUERY], is_query=True)[0]
    depth = max(5, retr.channel_depth)
    dense = [int(i) for i in np.argsort(-(mat @ q))[:depth]]
    sparse = [i for i, _ in BM25Index([d.sparse_text for d in docs])
              .search(QUERY, depth)]
    fused = rrf_fuse([dense, sparse], rrf_k=retr.rrf_k)
    want = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:5]

    got = retr.retrieve(QUERY, k=5)
    assert [(docs.index(next(d for d in docs if d.doc_id == c.meta["doc_id"])),
             c.score) for c in got] == [(i, pytest.approx(s, abs=0)) for i, s in want]

    # explicit no-ops resolve to the same thing: {} disables, all-1.0 == off
    got_noop = retr.retrieve(QUERY, k=5, source_filter={},
                             source_weights={"dolma": 1.0, "arxiv": 1.0})
    assert [(c.meta["doc_id"], c.score) for c in got_noop] == \
           [(c.meta["doc_id"], c.score) for c in got]


def test_filter_exclude_and_include():
    retr = HybridRetriever(_corpus(), embedder=HashingEmbedder())
    base = retr.retrieve(QUERY, k=5)
    assert source_key(None, base[0].source) == "dolma"  # dolma wins by default

    no_dolma = retr.retrieve(QUERY, k=5, source_filter={"exclude": ["dolma"]})
    assert no_dolma and all(
        source_key(c.meta["doc_id"], c.source) != "dolma" for c in no_dolma)
    # in-channel: the 3 arxiv docs all still surface (depth preserved)
    assert len(no_dolma) == 3
    assert no_dolma[0].meta["doc_id"] == "1"  # next-best takes the top slot

    only_dolma = retr.retrieve(QUERY, k=5, source_filter={"include": ["dolma"]})
    assert only_dolma and all(
        source_key(c.meta["doc_id"], c.source) == "dolma" for c in only_dolma)
    assert len(only_dolma) == 2


def test_weights_reorder_and_zero_excludes():
    retr = HybridRetriever(_corpus(), embedder=HashingEmbedder())
    base = retr.retrieve(QUERY, k=5)
    assert base[0].meta["doc_id"] == "dolma:aa#0"
    base_scores = {c.meta["doc_id"]: c.score for c in base}

    # w=0.25: the dolma top hit is demoted below the arxiv runner-up, and its
    # score is EXACTLY 0.25 * its unweighted fused score (the documented math).
    down = retr.retrieve(QUERY, k=5, source_weights={"dolma": 0.25})
    assert down[0].meta["doc_id"] == "1"
    got = {c.meta["doc_id"]: c.score for c in down}
    assert got["dolma:aa#0"] == pytest.approx(0.25 * base_scores["dolma:aa#0"])
    assert got["1"] == pytest.approx(base_scores["1"])  # w_arxiv stays 1.0

    # w=0 drops the source entirely (== exclude, at the fused-pool level)
    gone = retr.retrieve(QUERY, k=5, source_weights={"dolma": 0.0})
    assert all(source_key(c.meta["doc_id"], c.source) != "dolma" for c in gone)


def test_instance_defaults_and_per_call_override():
    retr = HybridRetriever(_corpus(), embedder=HashingEmbedder(),
                           source_weights={"dolma": 0.25})
    assert retr.retrieve(QUERY, k=5)[0].meta["doc_id"] == "1"  # default applied
    # per-call {} explicitly disables the instance default
    assert retr.retrieve(QUERY, k=5,
                         source_weights={})[0].meta["doc_id"] == "dolma:aa#0"


def test_unknown_source_keys_raise():
    retr = HybridRetriever(_corpus(), embedder=HashingEmbedder())
    with pytest.raises(ValueError, match="unknown source key"):
        retr.retrieve(QUERY, k=3, source_filter={"exclude": ["dolmaa"]})
    with pytest.raises(ValueError, match="unknown source key"):
        retr.retrieve(QUERY, k=3, source_weights={"web": 0.5})
    with pytest.raises(ValueError, match=">= 0"):
        retr.retrieve(QUERY, k=3, source_weights={"dolma": -0.5})
    with pytest.raises(ValueError, match="include"):
        retr.retrieve(QUERY, k=3, source_filter={"includ": ["dolma"]})
    with pytest.raises(ValueError):  # bad config fails at construction too
        HybridRetriever(_corpus(), source_weights={"dolmaa": 0.5})


# --------------------------------------------------------------------- #
# MCP server plumbing
# --------------------------------------------------------------------- #
def test_server_search_existing_math_plumbs_source_args(tmp_path, monkeypatch):
    from mathlas import server

    monkeypatch.setenv("MATHLAS_SEED", "1")
    monkeypatch.setenv("MATHLAS_FINDINGS", str(tmp_path / "findings.jsonl"))
    monkeypatch.setitem(server._RETRIEVER_CACHE, "<seed>",
                        HybridRetriever(_corpus(), embedder=HashingEmbedder()))

    out = server.tool_search_existing_math(QUERY, k=5)
    assert out["candidates"][0]["source"] == "arXiv (dolma-v1_7)"

    out = server.tool_search_existing_math(
        QUERY, k=5, source_filter={"exclude": ["dolma"]})
    assert out["candidates"] and all(
        "dolma" not in (c["source"] or "") for c in out["candidates"])

    out = server.tool_search_existing_math(
        QUERY, k=5, source_weights={"dolma": 0.25})
    assert "arxiv.org" in out["candidates"][0]["source"]

    # tool count unchanged (12), and a bad key surfaces as an in-band error
    assert len(server.tool_names()) == 12
    res = server._dispatch("tools/call", {
        "name": "search_existing_math",
        "arguments": {"query": QUERY, "source_filter": {"exclude": ["webby"]}},
    })
    assert res.get("isError") and "unknown source key" in res["content"][0]["text"]
