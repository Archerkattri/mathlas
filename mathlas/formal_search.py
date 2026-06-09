"""mathlas.formal_search — search FORMAL math (mathlib declarations) via the
public Loogle and LeanSearch services. NO LLM, no API key.

This is the one mathlas tool that makes a web call ITSELF (every other tool is
fully local; web work is normally delegated to the calling AI via
``search_directive``). It is a thin, provenance-labeled proxy:

  * **Loogle** (``https://loogle.lean-lang.org/json``) — pattern/type search over
    mathlib (e.g. ``Real.sqrt``, ``?a * ?b = ?b * ?a``, ``|- tsum _ = _ * tsum _``).
  * **LeanSearch** (``https://leansearch.net/search``) — natural-language search
    over mathlib (e.g. "a bounded monotone sequence converges").

Every hit is returned with ``provenance`` ``external:loogle`` /
``external:leansearch`` — these are EXTERNAL indexes, not the mathlas corpus, and
a hit is a mathlib *declaration* (name + type), i.e. formally-stated existing
math. If a service is down, its block reports ``available: False`` with the real
error — an honest "service unavailable," never fabricated hits.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

LOOGLE_URL = "https://loogle.lean-lang.org/json"
LEANSEARCH_URL = "https://leansearch.net/search"
_USER_AGENT = "mathlas-formal-search/1.0 (+https://github.com/Archerkattri/mathlas)"
DEFAULT_TIMEOUT_S = 15.0

#: Known backends, in the order ``backend="auto"`` queries them.
BACKENDS = ("loogle", "leansearch")


def _http_json(
    url: str,
    payload: Optional[Dict[str, Any]] = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> Any:
    """GET (``payload=None``) or POST-JSON ``url`` and parse the JSON response.

    The single network choke point — tests mock THIS function. Raises
    ``urllib.error.URLError`` / ``HTTPError`` / ``json.JSONDecodeError`` /
    ``TimeoutError`` on failure; callers convert to an honest unavailable block."""
    data = None
    headers = {"User-Agent": _USER_AGENT, "Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _mathlib_docs_url(module: Optional[str], name: Optional[str]) -> Optional[str]:
    """The leanprover-community mathlib4 docs URL for a declaration, if derivable."""
    if not module or not name:
        return None
    return (
        "https://leanprover-community.github.io/mathlib4_docs/"
        f"{module.replace('.', '/')}.html#{urllib.parse.quote(name)}"
    )


def search_loogle(
    query: str, k: int = 10, timeout_s: float = DEFAULT_TIMEOUT_S
) -> Dict[str, Any]:
    """Query the public Loogle JSON API (pattern/type search over mathlib).

    Returns ``{"backend": "loogle", "available": bool, "hits": [...], ...}``.
    ``available: False`` + ``error`` when the service is unreachable (honest
    unavailable, no fabricated hits). A Loogle *query-syntax* error comes back
    with ``available: True`` and the parser message in ``error`` (plus Loogle's
    ``suggestions`` when it offers any)."""
    url = LOOGLE_URL + "?" + urllib.parse.urlencode({"q": query})
    try:
        raw = _http_json(url, timeout_s=timeout_s)
    except Exception as e:  # network/HTTP/parse — service unavailable, say so.
        return {
            "backend": "loogle",
            "available": False,
            "hits": [],
            "error": f"Loogle unreachable ({e.__class__.__name__}: {e}). "
            "Retry later or use backend='leansearch'.",
        }
    if not isinstance(raw, dict):
        return {
            "backend": "loogle",
            "available": False,
            "hits": [],
            "error": f"Loogle returned unexpected JSON of type "
            f"{type(raw).__name__}; service may be degraded.",
        }
    hits: List[Dict[str, Any]] = []
    for h in (raw.get("hits") or [])[: int(k)]:
        name = h.get("name")
        module = h.get("module")
        hits.append(
            {
                "name": name,
                "type": h.get("type"),
                "module": module,
                "doc": h.get("doc"),
                "url": _mathlib_docs_url(module, name),
                "source": "loogle",
                "provenance": "external:loogle",
            }
        )
    out: Dict[str, Any] = {
        "backend": "loogle",
        "available": True,
        "hits": hits,
        "count": raw.get("count", len(hits)),
        "error": raw.get("error"),
    }
    if raw.get("suggestions"):
        out["suggestions"] = raw["suggestions"]
    return out


def search_leansearch(
    query: str, k: int = 10, timeout_s: float = DEFAULT_TIMEOUT_S
) -> Dict[str, Any]:
    """Query the public LeanSearch API (natural-language search over mathlib).

    POSTs ``{"query": [query], "num_results": k}`` to ``leansearch.net/search``
    (endpoint verified live 2026-06-09). Same honest-unavailable contract as
    :func:`search_loogle`."""
    try:
        raw = _http_json(
            LEANSEARCH_URL,
            payload={"query": [query], "num_results": int(k)},
            timeout_s=timeout_s,
        )
    except Exception as e:
        return {
            "backend": "leansearch",
            "available": False,
            "hits": [],
            "error": f"LeanSearch unreachable ({e.__class__.__name__}: {e}). "
            "Retry later or use backend='loogle'.",
        }
    # Response shape: one list per query, each entry {"result": {...}, "distance": d}.
    rows: List[Any] = []
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        rows = raw[0]
    elif isinstance(raw, list):
        rows = raw
    hits: List[Dict[str, Any]] = []
    for row in rows[: int(k)]:
        r = row.get("result", row) if isinstance(row, dict) else {}
        name_parts = r.get("name") or []
        mod_parts = r.get("module_name") or []
        name = ".".join(name_parts) if isinstance(name_parts, list) else str(name_parts)
        module = ".".join(mod_parts) if isinstance(mod_parts, list) else str(mod_parts)
        hits.append(
            {
                "name": name or None,
                "type": r.get("type"),
                "module": module or None,
                "kind": r.get("kind"),
                "signature": r.get("signature"),
                "doc": r.get("docstring") or r.get("informal_description"),
                "informal_name": r.get("informal_name"),
                "distance": row.get("distance") if isinstance(row, dict) else None,
                "url": _mathlib_docs_url(module or None, name or None),
                "source": "leansearch",
                "provenance": "external:leansearch",
            }
        )
    return {
        "backend": "leansearch",
        "available": True,
        "hits": hits,
        "count": len(hits),
        "error": None,
    }


def search_formal_math(
    query: str, k: int = 10, backend: str = "auto", timeout_s: float = DEFAULT_TIMEOUT_S
) -> Dict[str, Any]:
    """Search mathlib declarations through Loogle and/or LeanSearch.

    ``backend``: ``"auto"`` (both, interleaved), ``"loogle"`` (pattern/type
    queries), or ``"leansearch"`` (natural-language queries). Returns per-backend
    blocks plus a merged, source-labeled ``hits`` list (deduped by declaration
    name, interleaved across backends). All hits carry ``provenance``
    ``external:<service>`` — they come from external indexes, not the mathlas
    corpus."""
    backend = (backend or "auto").strip().lower()
    if backend in ("auto", "both", "all"):
        wanted = list(BACKENDS)
    elif backend in BACKENDS:
        wanted = [backend]
    else:
        return {
            "query": query,
            "backend": backend,
            "backends": {},
            "hits": [],
            "error": (
                f"unknown backend {backend!r} — use 'auto', 'loogle' "
                "(pattern/type queries) or 'leansearch' "
                "(natural-language queries)."
            ),
        }
    blocks: Dict[str, Dict[str, Any]] = {}
    for b in wanted:
        fn = search_loogle if b == "loogle" else search_leansearch
        blocks[b] = fn(query, k=k, timeout_s=timeout_s)

    # Interleave hits across backends (rank 1 of each, rank 2 of each, ...),
    # dedupe by declaration name so the same decl found twice appears once.
    merged: List[Dict[str, Any]] = []
    seen: set = set()
    lists = [blocks[b]["hits"] for b in wanted]
    for i in range(max((len(l) for l in lists), default=0)):
        for l in lists:
            if i < len(l):
                h = l[i]
                key = h.get("name") or id(h)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(h)
    merged = merged[: int(k)]

    unavailable = [b for b in wanted if not blocks[b]["available"]]
    note = (
        "Hits are mathlib DECLARATIONS (name + type) from external public "
        "indexes (Loogle = pattern/type, LeanSearch = natural language), "
        "provenance-labeled external:<service>. They are formally-stated "
        "existing math — compose with applicability_checklist, and use "
        "verify_formal to kernel-check any Lean snippet you write."
    )
    if unavailable:
        note += (
            " UNAVAILABLE: "
            + ", ".join(unavailable)
            + " (see backends[<name>].error — honest unavailability, "
            "no fabricated hits)."
        )
    return {
        "query": query,
        "backend": backend,
        "backends": blocks,
        "hits": merged,
        "note": note,
    }
