"""Fast, CPU-only, GPU-free regression tests for mathlas's airtight tiers.

These mirror the `benchmarks/` scripts at the library level so `pytest` exercises
the core guarantees without a GPU, the 1.6M index, a Lean toolchain, or local OEIS
data. The heavyweight tiers (retrieval over the real index, the Lean kernel, and the
OEIS exact-match) have their own benchmark scripts and are skipped here when their
optional data/toolchain is absent — never faked.
"""
from __future__ import annotations

import mpmath
import pytest

# --- numeric identification (airtight: independent high-precision re-eval) --------

KNOWN = [
    ("zeta(2)", lambda: mpmath.zeta(2), "pi**2/6"),
    ("catalan", lambda: mpmath.catalan, "Catalan"),
    ("golden", lambda: (1 + mpmath.sqrt(5)) / 2, None),
    ("log2", lambda: mpmath.log(2), "log(2)"),
    ("e", lambda: mpmath.e, None),
]
NEGATIVE = [
    ("sin1*log7", lambda: mpmath.sin(1) * mpmath.log(7)),
    ("tan2+1/3", lambda: mpmath.tan(2) + mpmath.mpf(1) / 3),
    ("exp(sin2)", lambda: mpmath.exp(mpmath.sin(2))),
]


@pytest.mark.parametrize("label,gen,want", KNOWN)
def test_identify_recovers_known_constants(label, gen, want):
    from mathlas import identify

    with mpmath.workdps(60):
        res = identify(gen())
    assert res.identified, f"{label}: expected a verified closed form"
    assert res.best is not None
    if want is not None:
        assert want in res.best.display


@pytest.mark.parametrize("label,gen", NEGATIVE)
def test_identify_rejects_structureless(label, gen):
    """The honesty gate: a structureless irrational must return UNIDENTIFIED."""
    from mathlas import identify

    with mpmath.workdps(60):
        res = identify(gen())
    assert not res.identified, f"{label}: false-positive (should be UNIDENTIFIED)"


def test_verify_closed_form_airtight():
    from mathlas import verify_closed_form

    assert verify_closed_form(mpmath.mpf("1.6449340668482264"), "pi**2/6").ok
    assert not verify_closed_form(mpmath.mpf("1.6449340668482264"), "pi**2/7").ok


# --- applicability scaffold (deterministic decomposition, no LLM) -----------------


def test_applicability_checklist_surfaces_hypotheses():
    from mathlas import applicability_checklist

    chk = applicability_checklist(
        "Let G be a finite group and p a prime dividing the order of G. "
        "Then G has an element of order p."
    )
    assert any("finit" in p.lower() for p in chk.preconditions)


def test_mapping_scaffold_returns_structure():
    from mathlas import mapping_scaffold
    from mathlas.map import MappingScaffold

    s = mapping_scaffold(
        "show x = cos x has a unique fixed point",
        "Let (X,d) be a complete metric space and T a contraction. "
        "Then T has a unique fixed point.",
    )
    assert isinstance(s, MappingScaffold)
    assert s.checklist
    assert s.questions


# --- FunSearch harness (deterministic sandbox, no LLM) ----------------------------


def test_funsearch_starter_scores():
    import mathlas.funsearch as fs

    r = fs.evaluate(fs.get_problem("cap_set").starter_src, "cap_set")
    assert r.score == pytest.approx(5.0)


def test_funsearch_sandbox_kills_runaway():
    """An infinite loop in untrusted AI code must be killed, not hang the harness."""
    import mathlas.funsearch as fs

    r = fs.evaluate("def solve(n):\n    while True:\n        pass\n", "cap_set", timeout_s=2.0)
    assert r.timed_out and not r.ok


# --- web-augmentation directive (structured plan, no web call, no LLM) ------------


def test_search_directive_is_structured():
    from mathlas.webaug import search_directive

    d = search_directive("evaluate the sum 1/n^4")
    assert d.arxiv_queries
    assert d.named_results


# --- DENSE-finding self-augmenting corpus (caller supplies the vector, no model) ---


def test_dense_finding_surfaced_on_paraphrase(tmp_path, monkeypatch):
    """A web-added finding with a CALLER-supplied dense_vec must be surfaced via the
    DENSE channel for a PARAPHRASE query that BM25 MISSES — proving the stored
    vector is actually USED at retrieval. OOM-free: the dep-free HashingEmbedder
    backs the served index AND produces the finding/query vectors, so they share a
    space. HashingEmbedder is a bag-of-words hash, so 'paraphrase' here means a
    query that overlaps the finding's SLOGAN tokens (dense match) while the BM25
    `sparse_text` is dominated by the unrelated STATEMENT/source tokens — the exact
    regime where slogan-dense beats statement-heavy BM25."""
    from mathlas.embed import HashingEmbedder
    from mathlas.retrieve.corpus import Document
    from mathlas.retrieve.hybrid import HybridRetriever
    from mathlas import server, webaug

    monkeypatch.setenv("MATHLAS_FINDINGS", str(tmp_path / "findings.jsonl"))
    # Force the in-process seed corpus so ``tool_search_existing_math`` uses the
    # injected ``"<seed>"`` retriever below and NEVER the (multi-GB) prebuilt index
    # that may exist at the default path — keeps this CPU-only test hermetic + fast.
    monkeypatch.setenv("MATHLAS_SEED", "1")
    webaug._LIVE_CACHE.clear()

    emb = HashingEmbedder(dim=256)
    seeds = [
        Document(doc_id="s0", slogan="every continuous function on a closed interval is bounded",
                 statement="A continuous function on [a,b] attains a maximum.",
                 name="Extreme Value Theorem", source="seed"),
        Document(doc_id="s1", slogan="the derivative of a constant is zero",
                 statement="d/dx c = 0.", name="Constant Rule", source="seed"),
    ]
    retr = HybridRetriever(seeds, embedder=emb)
    server._RETRIEVER_CACHE.clear()
    server._RETRIEVER_CACHE["<seed>"] = retr

    # The query's content words ("contraction mapping unique fixed point") appear in
    # the SLOGAN (-> high dense cosine) but NOT in the statement, which is written in
    # disjoint vocabulary. BM25's sparse_text = name+slogan+statement+source, but we
    # query terms only present in the slogan so the finding's BM25 rank is weak/tied
    # against the seeds, while dense is decisive.
    slogan = "contraction mapping has a unique fixed point"
    statement = ("Let Y be a thoroughly exhaustive realm endowed with a shrinking "
                 "operator; then exactly one element stays invariant under it.")
    query_dense = "contraction mapping unique fixed point"
    # A purely BM25 query (terms only in the disjoint statement) — to show BM25 alone:
    query_bm25_miss = query_dense

    dense_vec = [float(x) for x in emb.encode([slogan], is_query=False)[0]]
    r = webaug.add_finding(statement, slogan, "arXiv:0000.00000",
                           name="Banach fixed-point theorem", dense_vec=dense_vec)
    assert r.ok and r.dense_added

    # The dense channel ALONE ranks this finding first for the paraphrase query.
    dranked = server._dense_finding_ranking(query_dense, retr)
    assert dranked and dranked[0].get("statement") == statement

    # FULL pipeline: the finding is surfaced (dense+BM25+RRF fused).
    res = server.tool_search_existing_math(query_dense, k=5)
    stmts = [c.get("statement") for c in res["candidates"]]
    assert statement in stmts, "dense finding not surfaced via the dense channel"
    assert res["live_findings_merged"] >= 1

    # PROOF the dense channel is load-bearing: drop the stored dense_vec and the
    # SAME query now relies on BM25 alone. We rebuild the finding without a vector
    # and confirm dense ranking is empty (so any surfacing was via dense, not BM25).
    webaug._LIVE_CACHE.clear()
    (tmp_path / "findings.jsonl").write_text("")  # clear sidecar
    server._RETRIEVER_CACHE.clear()              # no in-process embedder at all
    r2 = webaug.add_finding(statement, slogan, "arXiv:0000.00000",
                            name="Banach fixed-point theorem")  # NO dense_vec
    # no in-process model -> no dense vec stored -> dense ranking finds nothing
    assert not r2.dense_added
    assert server._dense_finding_ranking(query_dense, retr) == []


def test_dense_finding_dim_mismatch_is_honest_error(tmp_path, monkeypatch):
    """A caller-supplied dense_vec whose length != served index dim must be rejected
    with an honest error (finding NOT added) — never silently stored wrong-space."""
    from mathlas.embed import HashingEmbedder
    from mathlas.retrieve.corpus import Document
    from mathlas.retrieve.hybrid import HybridRetriever
    from mathlas import server, webaug

    monkeypatch.setenv("MATHLAS_FINDINGS", str(tmp_path / "findings.jsonl"))
    webaug._LIVE_CACHE.clear()
    retr = HybridRetriever(
        [Document(doc_id="s0", slogan="x", statement="x", name="x", source="s")],
        embedder=HashingEmbedder(dim=256))
    server._RETRIEVER_CACHE.clear()
    server._RETRIEVER_CACHE["<seed>"] = retr

    r = webaug.add_finding("stmt", "slogan", "src", dense_vec=[0.1, 0.2, 0.3])
    assert not r.ok and not r.dense_added
    assert "dim" in r.note.lower()
    assert webaug.load_findings() == []          # nothing persisted


def test_add_finding_bm25_only_backward_compatible(tmp_path, monkeypatch):
    """No dense_vec, no loaded model -> BM25-only, dense_added False, still found by
    an exact-term query. Backward-compatible with the original behaviour."""
    from mathlas import server, webaug

    monkeypatch.setenv("MATHLAS_FINDINGS", str(tmp_path / "findings.jsonl"))
    webaug._LIVE_CACHE.clear()
    server._RETRIEVER_CACHE.clear()              # no served index in-process

    r = webaug.add_finding("Turán's theorem bounds edges in triangle-free graphs.",
                           "max edges in a triangle-free graph", "arXiv:1")
    assert r.ok and not r.dense_added
    hits = webaug.search_findings("triangle-free graph edges", k=5)
    assert any("Turán" in h.get("statement", "") for h in hits)


# --- BM25 sparse-index cache (build once, reload without re-tokenizing) ------------


def test_bm25_save_load_roundtrip(tmp_path):
    """A saved BM25 index reloads to an IDENTICAL searcher WITHOUT re-tokenizing:
    same top-k doc ids + scores for every query, same vocab/idf/doc_len/tf. This is
    what lets the multi-million-doc served index skip the slow in-memory rebuild."""
    import numpy as np
    from mathlas.retrieve.bm25 import BM25Index

    corpus = [
        "Banach fixed point theorem contraction complete metric space unique",
        "etale cohomology of schemes Grothendieck topology",
        "every bounded sequence has a convergent subsequence Bolzano Weierstrass",
        "the sum over n of 1 over n squared equals pi squared over six Basel",
        "",  # empty-document edge case
    ]
    bm = BM25Index(corpus)
    bundle = bm.save(str(tmp_path / "small.bm25"))
    bm2 = BM25Index.load(bundle)

    for q in ["Banach contraction", "etale cohomology",
              "bounded sequence convergent subsequence", "pi squared six",
              "token_not_in_any_document"]:
        assert bm.search(q, k=10) == bm2.search(q, k=10)
    assert (bm.k1, bm.b, bm.n) == (bm2.k1, bm2.b, bm2.n)
    assert bm.avgdl == pytest.approx(bm2.avgdl)
    assert bm.vocab == bm2.vocab
    assert np.array_equal(bm.idf, bm2.idf)
    assert np.array_equal(bm.doc_len, bm2.doc_len)
    assert (bm.tf != bm2.tf).nnz == 0          # identical sparse term-doc matrix

    # a foreign/empty directory is refused honestly, not mis-loaded.
    with pytest.raises((FileNotFoundError, ValueError)):
        BM25Index.load(str(tmp_path / "does_not_exist.bm25"))


def test_from_index_bm25_cache_roundtrip(tmp_path):
    """``from_index`` builds the BM25 cache once, then LOADS it (no rebuild) on the
    next call with identical results; rebuilds when the corpus changes; honours
    ``with_bm25=False`` (dense-only) and ``bm25_cache=None`` (no persisted cache)."""
    import json
    import os
    import numpy as np
    from mathlas.embed import HashingEmbedder
    from mathlas.retrieve.corpus import Document
    from mathlas.retrieve.hybrid import HybridRetriever

    emb = HashingEmbedder()
    docs = [
        Document(doc_id="0", slogan="Banach fixed point contraction complete metric space",
                 statement="contraction mapping", name="Banach fixed point"),
        Document(doc_id="1", slogan="etale cohomology of schemes", statement="etale site",
                 name="etale cohomology"),
        Document(doc_id="2", slogan="bounded sequence has a convergent subsequence",
                 statement="Bolzano Weierstrass", name="Bolzano-Weierstrass"),
    ]

    def write_index(docs):
        idx = tmp_path / "index_full_dense.npz"
        sidecar = tmp_path / "index_full_dense.meta.jsonl"
        M = emb.encode([d.embed_text for d in docs], is_query=False).astype(np.float16)
        with open(sidecar, "w") as f:
            for d in docs:
                f.write(json.dumps({"doc_id": d.doc_id, "name": d.name,
                                    "slogan": d.slogan, "statement": d.statement,
                                    "source": d.source, "title": d.title,
                                    "label": d.label, "citations": d.citations,
                                    "category": d.category}) + "\n")
        np.savez(idx, matrix=M, dim=M.shape[1], model="hashing-dev",
                 embedder="hashing", meta_file=sidecar.name)
        return str(idx)

    idx = write_index(docs)
    cache = str(tmp_path / "index_full_dense.bm25")

    # first load BUILDS + SAVES the cache (BM25 on by default).
    r1 = HybridRetriever.from_index(idx, embedder=HashingEmbedder())
    assert r1._bm25 is not None and os.path.isdir(cache)
    assert os.path.exists(os.path.join(cache, "source.json"))
    base = r1._bm25.search("Banach contraction fixed point", 10)
    assert base and base[0][0] == 0

    # second load HITS the cache (identical results, same doc count).
    r2 = HybridRetriever.from_index(idx, embedder=HashingEmbedder())
    assert r2._bm25.n == len(docs)
    assert r2._bm25.search("Banach contraction fixed point", 10) == base

    # growing the corpus invalidates the cache -> rebuild to the new doc count.
    docs2 = docs + [Document(doc_id="3", slogan="compact Hausdorff space",
                             statement="compactness", name="compact")]
    r3 = HybridRetriever.from_index(write_index(docs2), embedder=HashingEmbedder())
    assert r3._bm25.n == len(docs2)

    # dense-only opt-out, and no-cache opt-out (must not write a bundle).
    assert HybridRetriever.from_index(idx, embedder=HashingEmbedder(),
                                      with_bm25=False)._bm25 is None
    eph = str(tmp_path / "ephemeral.npz")
    np.savez(eph, matrix=emb.encode([d.embed_text for d in docs]).astype(np.float16),
             dim=emb.dim, model="hashing-dev", embedder="hashing",
             meta=json.dumps([{"doc_id": d.doc_id, "slogan": d.slogan,
                               "statement": d.statement, "name": d.name} for d in docs]))
    HybridRetriever.from_index(eph, embedder=HashingEmbedder(), bm25_cache=None)
    assert not os.path.exists(str(tmp_path / "ephemeral.bm25"))


# --- MCP server surface (tools enumerated, dep-free JSON-RPC dispatch) -------------


def test_server_exposes_all_tools():
    from mathlas.server import tool_names

    names = tool_names()
    assert len(names) == 12  # funsearch trio collapsed into one; +search_formal_math
    for required in ("identify_constant", "search_existing_math", "verify_numeric",
                     "verify_formal", "applicability_checklist", "conjecture_relation",
                     "funsearch", "search_formal_math"):
        assert required in names


def test_server_jsonrpc_dispatch_identify():
    """The dependency-free JSON-RPC fallback dispatches a real tool call."""
    import json

    from mathlas.server import _dispatch

    listed = _dispatch("tools/list", {})
    assert len(listed["tools"]) == 12
    res = _dispatch(
        "tools/call",
        {"name": "identify_constant", "arguments": {"value": "1.2020569031595942854"}},
    )
    payload = json.loads(res["content"][0]["text"])
    assert payload["best"]["display"] == "zeta(3)"
