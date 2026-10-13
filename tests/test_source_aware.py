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

from urllib.parse import urlparse

import numpy as np
import pytest

from mathlas.embed import HashingEmbedder
from mathlas.retrieve.bm25 import BM25Index
from mathlas.retrieve.corpus import KNOWN_SOURCE_KEYS, Document, source_key
from mathlas.retrieve.hybrid import HybridRetriever, rrf_fuse


# --------------------------------------------------------------------- #
# the canonical source classifier
# --------------------------------------------------------------------- #


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
    top_host = urlparse(out["candidates"][0]["source"]).hostname
    assert top_host == "arxiv.org"

    # tool count unchanged (12), and a bad key surfaces as an in-band error
    assert len(server.tool_names()) == 12
    res = server._dispatch("tools/call", {
        "name": "search_existing_math",
        "arguments": {"query": QUERY, "source_filter": {"exclude": ["webby"]}},
    })
    assert res.get("isError") and "unknown source key" in res["content"][0]["text"]
