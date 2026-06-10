"""Tests for mathlas.formal_search (Loogle/LeanSearch proxy) + the server's
tool-surface changes that shipped with it (the `search_formal_math` tool, the
collapsed `funsearch` tool, titles/outputSchema on the fallback server).

The HTTP layer is mocked at the single network choke point
(``formal_search._http_json``) — no test here touches the network except the
one OPT-IN live test, skipped unless ``MATHLAS_LIVE=1``.
"""

import json
import os
import time
import urllib.error

import pytest

import mathlas.formal_search as fs


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Point the on-disk TTL cache at a per-test temp dir so (a) tests never
    read/write the user's real ~/.cache/mathlas and (b) every test starts with
    an empty cache."""
    monkeypatch.setenv("MATHLAS_CACHE_DIR", str(tmp_path / "mathlas-cache"))
    monkeypatch.delenv("MATHLAS_NO_CACHE", raising=False)


# ---- canned payloads matching the real services' wire shapes ----------------

LOOGLE_PAYLOAD = {
    "count": 2,
    "hits": [
        {
            "name": "Real.sqrt_nonneg",
            "type": "∀ (x : ℝ), 0 ≤ Real.sqrt x",
            "module": "Mathlib.Analysis.SpecialFunctions.Sqrt",
            "doc": "sqrt is nonneg",
        },
        {
            "name": "Real.sqrt_eq_zero",
            "type": "...",
            "module": "Mathlib.Foo",
            "doc": None,
        },
    ],
    "error": None,
    "header": "<b>2 hits</b>",
}

# LeanSearch returns ONE list per query; each row {"result": {...}, "distance": d}
# (shape verified against the live endpoint 2026-06-09).
LEANSEARCH_PAYLOAD = [
    [
        {
            "result": {
                "module_name": ["Mathlib", "Data", "Int", "Init"],
                "kind": "definition",
                "name": ["Int", "succ"],
                "signature": "(a : ℤ)",
                "type": "ℤ → ℤ",
                "value": ":=\n  a + 1",
                "docstring": None,
                "informal_name": "Successor function on integers",
                "informal_description": "Maps an integer a to a + 1.",
            },
            "distance": 0.259,
        },
    ]
]


def _mock_http(
    monkeypatch, loogle=None, leansearch=None, loogle_exc=None, leansearch_exc=None
):
    def fake(url, payload=None, timeout_s=15.0):
        if url.startswith(fs.LOOGLE_URL):
            if loogle_exc is not None:
                raise loogle_exc
            return loogle
        if url.startswith(fs.LEANSEARCH_URL):
            if leansearch_exc is not None:
                raise leansearch_exc
            return leansearch
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(fs, "_http_json", fake)


# ---- mocked-HTTP tests -------------------------------------------------------


def test_loogle_normalization(monkeypatch):
    _mock_http(monkeypatch, loogle=LOOGLE_PAYLOAD)
    out = fs.search_loogle("Real.sqrt", k=10)
    assert out["available"] is True and out["backend"] == "loogle"
    assert out["count"] == 2 and len(out["hits"]) == 2
    h = out["hits"][0]
    assert h["name"] == "Real.sqrt_nonneg"
    assert h["provenance"] == "external:loogle" and h["source"] == "loogle"
    # docs URL derived from module path + anchored decl name
    assert h["url"].startswith(
        "https://leanprover-community.github.io/mathlib4_docs/"
        "Mathlib/Analysis/SpecialFunctions/Sqrt.html#"
    )


def test_leansearch_normalization(monkeypatch):
    _mock_http(monkeypatch, leansearch=LEANSEARCH_PAYLOAD)
    out = fs.search_leansearch("successor of an integer", k=5)
    assert out["available"] is True
    h = out["hits"][0]
    assert h["name"] == "Int.succ"  # name parts joined
    assert h["module"] == "Mathlib.Data.Int.Init"  # module parts joined
    assert h["provenance"] == "external:leansearch"
    assert h["doc"] == "Maps an integer a to a + 1."  # informal fallback
    assert h["distance"] == pytest.approx(0.259)


def test_unavailable_is_honest(monkeypatch):
    _mock_http(
        monkeypatch,
        loogle_exc=urllib.error.URLError("down"),
        leansearch_exc=TimeoutError("timed out"),
    )
    out = fs.search_formal_math("anything", backend="auto")
    assert out["hits"] == []  # NO fabricated hits
    for b in ("loogle", "leansearch"):
        assert out["backends"][b]["available"] is False
        assert "unreachable" in out["backends"][b]["error"]
    assert "UNAVAILABLE" in out["note"]


def test_merge_interleaves_and_dedupes(monkeypatch):
    # same decl name from both services -> appears once in the merged list
    loogle = {
        "count": 1,
        "hits": [
            {
                "name": "Int.succ",
                "type": "t",
                "module": "Mathlib.Data.Int.Init",
                "doc": None,
            }
        ],
        "error": None,
    }
    _mock_http(monkeypatch, loogle=loogle, leansearch=LEANSEARCH_PAYLOAD)
    out = fs.search_formal_math("succ", k=10, backend="auto")
    names = [h["name"] for h in out["hits"]]
    assert names.count("Int.succ") == 1
    assert out["backends"]["loogle"]["available"] is True
    assert out["backends"]["leansearch"]["available"] is True


def test_one_backend_down_other_still_serves(monkeypatch):
    _mock_http(
        monkeypatch,
        loogle_exc=urllib.error.HTTPError(
            fs.LOOGLE_URL, 502, "Bad Gateway", None, None
        ),
        leansearch=LEANSEARCH_PAYLOAD,
    )
    out = fs.search_formal_math("succ", backend="auto")
    assert out["backends"]["loogle"]["available"] is False
    assert [h["source"] for h in out["hits"]] == ["leansearch"]


def test_unknown_backend_is_actionable():
    out = fs.search_formal_math("x", backend="mathlib4")
    assert out["hits"] == [] and "unknown backend" in out["error"]
    assert "loogle" in out["error"] and "leansearch" in out["error"]


# ---- on-disk TTL cache: serve the last good response when a service is down ---


def test_cache_serves_labeled_hits_on_network_failure(monkeypatch):
    # 1) a successful hit populates the cache ...
    _mock_http(monkeypatch, loogle=LOOGLE_PAYLOAD)
    live = fs.search_loogle("Real.sqrt", k=10)
    assert live["available"] is True and "cached" not in live
    # 2) ... then the service 502s (the real-world Loogle incident) ...
    _mock_http(
        monkeypatch,
        loogle_exc=urllib.error.HTTPError(
            fs.LOOGLE_URL, 502, "Bad Gateway", None, None
        ),
    )
    out = fs.search_loogle("Real.sqrt", k=10)
    # 3) ... and the cached response is served, CLEARLY labeled.
    assert out["available"] is False  # the live service really is down — no lie
    assert out["cached"] is True and out["cache_age_seconds"] >= 0
    assert [h["name"] for h in out["hits"]] == [h["name"] for h in live["hits"]]
    for h in out["hits"]:
        assert h["provenance"].startswith("external:loogle (cached, ")
    assert "502" in out["error"] and "cache" in out["error"]


def test_cache_is_per_query_and_per_k(monkeypatch):
    _mock_http(monkeypatch, loogle=LOOGLE_PAYLOAD)
    fs.search_loogle("Real.sqrt", k=10)
    _mock_http(monkeypatch, loogle_exc=urllib.error.URLError("down"))
    # different query (or k) => no cache entry => honest unavailable, no hits
    miss = fs.search_loogle("Nat.succ", k=10)
    assert miss["available"] is False and miss["hits"] == []
    assert "cached" not in miss
    miss = fs.search_loogle("Real.sqrt", k=3)
    assert miss["hits"] == [] and "cached" not in miss


def test_cache_ttl_expiry_honored(monkeypatch):
    _mock_http(monkeypatch, leansearch=LEANSEARCH_PAYLOAD)
    fs.search_leansearch("successor", k=5)
    _mock_http(monkeypatch, leansearch_exc=TimeoutError("timed out"))
    # fresh (age ~0) -> served from cache
    assert fs.search_leansearch("successor", k=5)["cached"] is True
    # beyond the 7-day TTL -> NOT served; honest unavailable again
    real_time = time.time
    monkeypatch.setattr(fs.time, "time", lambda: real_time() + fs.CACHE_TTL_S + 60)
    out = fs.search_leansearch("successor", k=5)
    assert out["available"] is False and out["hits"] == []
    assert "cached" not in out


def test_cache_refreshes_on_success(monkeypatch):
    _mock_http(monkeypatch, loogle=LOOGLE_PAYLOAD)
    fs.search_loogle("Real.sqrt", k=10)
    fresher = dict(LOOGLE_PAYLOAD, hits=[LOOGLE_PAYLOAD["hits"][0]], count=1)
    _mock_http(monkeypatch, loogle=fresher)
    fs.search_loogle("Real.sqrt", k=10)  # success again -> cache REFRESHED
    _mock_http(monkeypatch, loogle_exc=urllib.error.URLError("down"))
    out = fs.search_loogle("Real.sqrt", k=10)
    assert out["cached"] is True and len(out["hits"]) == 1  # the newer response


def test_cache_capped_and_corrupt_file_tolerated(monkeypatch):
    _mock_http(monkeypatch, loogle=LOOGLE_PAYLOAD)
    path = fs._cache_path()
    # corrupt cache file == empty cache, never an exception
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("{not json")
    assert fs.search_loogle("Real.sqrt", k=10)["available"] is True
    # entry cap: oldest entries are evicted, newest kept
    for i in range(fs.CACHE_MAX_ENTRIES + 25):
        fs.search_loogle(f"q{i}", k=10)
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    assert len(data) <= fs.CACHE_MAX_ENTRIES
    assert f"loogle|10|q{fs.CACHE_MAX_ENTRIES + 24}" in data  # newest survives


def test_cache_disabled_via_env(monkeypatch):
    monkeypatch.setenv("MATHLAS_NO_CACHE", "1")
    _mock_http(monkeypatch, loogle=LOOGLE_PAYLOAD)
    fs.search_loogle("Real.sqrt", k=10)
    _mock_http(monkeypatch, loogle_exc=urllib.error.URLError("down"))
    out = fs.search_loogle("Real.sqrt", k=10)
    assert out["available"] is False and out["hits"] == []  # nothing was stored


def test_merged_note_labels_cache_served_backend(monkeypatch):
    _mock_http(monkeypatch, loogle=LOOGLE_PAYLOAD, leansearch=LEANSEARCH_PAYLOAD)
    fs.search_formal_math("succ", k=10, backend="auto")  # primes both caches
    _mock_http(
        monkeypatch,
        loogle_exc=urllib.error.HTTPError(
            fs.LOOGLE_URL, 502, "Bad Gateway", None, None
        ),
        leansearch=LEANSEARCH_PAYLOAD,
    )
    out = fs.search_formal_math("succ", k=10, backend="auto")
    assert out["backends"]["loogle"]["cached"] is True
    assert "SERVED FROM CACHE: loogle" in out["note"]
    assert "UNAVAILABLE" not in out["note"]  # cached-served is labeled, not hidden
    # cached hits still flow into the merged list, label intact
    assert any("(cached," in h["provenance"] for h in out["hits"])


# ---- server wiring -----------------------------------------------------------


def test_server_tool_search_formal_math(monkeypatch):
    _mock_http(monkeypatch, loogle=LOOGLE_PAYLOAD, leansearch=LEANSEARCH_PAYLOAD)
    from mathlas.server import tool_search_formal_math

    out = tool_search_formal_math("sqrt nonneg", k=3, backend="auto")
    assert out["hits"] and all(
        h["provenance"].startswith("external:") for h in out["hits"]
    )


def test_server_funsearch_single_tool_and_aliases():
    from mathlas.server import _dispatch, tool_names

    names = tool_names()
    assert "funsearch" in names and "search_formal_math" in names
    for old in ("funsearch_evaluate", "funsearch_register", "funsearch_status"):
        assert old not in names  # collapsed, not listed
    # the single tool dispatches by action enum
    res = _dispatch(
        "tools/call",
        {
            "name": "funsearch",
            "arguments": {"action": "status", "problem_id": "cap_set"},
        },
    )
    assert res["isError"] is False
    assert res["structuredContent"]["action"] == "status"
    # ... and the old names still work as thin aliases (back-compat, unlisted)
    res = _dispatch(
        "tools/call",
        {"name": "funsearch_status", "arguments": {"problem_id": "cap_set"}},
    )
    assert res["isError"] is False
    # missing-arg errors are agent-actionable, in-band (isError result)
    res = _dispatch(
        "tools/call",
        {
            "name": "funsearch",
            "arguments": {"action": "evaluate", "problem_id": "cap_set"},
        },
    )
    assert res["structuredContent"]["ok"] is False
    assert "program_src" in res["structuredContent"]["error"]


def test_server_tools_list_has_title_and_output_schema():
    from mathlas.server import _dispatch

    listed = _dispatch("tools/list", {})["tools"]
    assert all(t.get("title") for t in listed)
    assert all(t.get("outputSchema", {}).get("type") == "object" for t in listed)
    fsm = next(t for t in listed if t["name"] == "search_formal_math")
    assert fsm["inputSchema"]["properties"]["backend"]["enum"] == [
        "auto",
        "loogle",
        "leansearch",
    ]


# ---- OPT-IN live test (MATHLAS_LIVE=1) ----------------------------------------


@pytest.mark.skipif(
    os.environ.get("MATHLAS_LIVE") != "1",
    reason="live network test; set MATHLAS_LIVE=1 to run",
)
def test_live_formal_search_real_services():
    """Real HTTP to the public services. Asserts the honest CONTRACT (shape +
    availability flags + no fabricated hits), not service uptime."""
    out = fs.search_formal_math("successor of a natural number", k=3, backend="auto")
    for b in ("loogle", "leansearch"):
        blk = out["backends"][b]
        assert isinstance(blk["available"], bool)
        if blk["available"]:
            assert isinstance(blk["hits"], list)
            for h in blk["hits"]:
                assert h["provenance"] == f"external:{b}"
        else:
            assert blk["hits"] == [] and blk["error"]
