"""HF corpus upload assembly (scripts/hf_upload_corpus.py). Pinned contracts:

  * routing: every served-meta record lands in the right config by source_key
    (theoremsearch <- arxiv/proofwiki/stacks/other, dolma <- dolma);
  * schema: the shipped parquet carries exactly the documented columns,
    and findings NEVER ship embedded vectors (dense/dense_vec stripped);
  * licensing exclusion: --exclude-source removes the rows AND the config,
    and the generated card reflects the shipped counts;
  * the audited-count strict gate refuses a drifted build;
  * the dataset card contains no em/en dashes (public-facing text rule);
  * the built dataset loads with ``datasets`` and validate() agrees with
    the build stats.

These tests run on a tiny synthetic corpus; the real 3.68M dry run is
exercised by ``python3 scripts/hf_upload_corpus.py --dry-run``.
"""
import importlib.util
import json
import os
import sys

import pytest

_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "scripts", "hf_upload_corpus.py")
_spec = importlib.util.spec_from_file_location("hf_upload_corpus", _SCRIPT)
up = importlib.util.module_from_spec(_spec)
sys.modules["hf_upload_corpus"] = up  # dataclasses needs the module registered
_spec.loader.exec_module(up)

pytest.importorskip("pyarrow")


META_ROWS = [
    # TheoremSearch base: arxiv / stacks / proofwiki / other
    {"doc_id": "1", "name": "Theorem A", "slogan": "an arxiv slogan",
     "statement": "$x$", "source": "http://arxiv.org/abs/2511.12345v1",
     "title": "T", "label": "thm:A", "citations": 3.0, "category": "math.DS"},
    {"doc_id": "2", "name": "Lemma 1.2.", "slogan": "a stacks slogan",
     "statement": "$y$", "source": "https://stacks.math.columbia.edu/tag/06UU",
     "title": "Adequate", "label": "lemma-x", "citations": "None",
     "category": None},
    {"doc_id": "3", "name": "PW", "slogan": "a proofwiki slogan",
     "statement": "$z$", "source": "https://proofwiki.org/wiki/Union",
     "title": "ProofWiki", "label": None, "citations": None, "category": None},
    {"doc_id": "4", "name": "Prop 3.14.", "slogan": "a textbook slogan",
     "statement": "$w$", "source": "https://math.uchicago.edu/~amathew/CRing.pdf",
     "title": "Categories", "label": None, "citations": None, "category": None},
    # Dolma rows (doc_id prefix is the strongest routing signal)
    {"doc_id": "dolma:abc#0", "name": None, "slogan": "a dolma slogan",
     "statement": "$d_0$", "source": "arXiv (dolma-v1_7)", "title": None,
     "label": None, "citations": None, "category": None},
    {"doc_id": "dolma:abc#1", "name": None, "slogan": "another dolma slogan",
     "statement": "$d_1$", "source": "arXiv (dolma-v1_7)", "title": None,
     "label": None, "citations": None, "category": None},
    {"doc_id": "dolma:def#2", "name": None, "slogan": "a third dolma slogan",
     "statement": "$d_2$", "source": "arXiv (dolma-v1_7)", "title": None,
     "label": None, "citations": None, "category": None},
]

FINDING_ROWS = [
    {"doc_id": "web::1::2", "name": "Theorem 3.1", "slogan": "a finding",
     "statement": "$f$", "source": "https://arxiv.org/abs/2310.15076",
     "provenance": "web_added", "added_ts": 1780839161.5,
     "dense": True, "dense_vec": [0.1, -0.2, 0.3]},
    {"doc_id": "web::3::4", "name": None, "slogan": "another finding",
     "statement": "$g$", "source": "https://arxiv.org/abs/1234.5678",
     "provenance": "web_added", "added_ts": 1780839200.0},
]


@pytest.fixture()
def corpus(tmp_path):
    meta = tmp_path / "meta.jsonl"
    meta.write_text("\n".join(json.dumps(r) for r in META_ROWS) + "\n")
    findings = tmp_path / "findings.jsonl"
    findings.write_text("\n".join(json.dumps(r) for r in FINDING_ROWS) + "\n")
    return str(meta), str(findings), str(tmp_path / "out")


def _build(corpus, **kw):
    meta, findings, out = corpus
    kw.setdefault("strict", False)
    kw.setdefault("progress", False)
    kw.setdefault("rows_per_shard", 2)  # force multi-shard on the tiny corpus
    kw.setdefault("batch_rows", 2)      # rotation happens at batch granularity
    return up.build(out, meta_path=meta, findings_path=findings, **kw)


def _read_parquet_rows(out, config):
    import pyarrow.parquet as pq
    d = os.path.join(out, "data", config)
    rows = []
    for p in sorted(os.listdir(d)):
        rows.extend(pq.read_table(os.path.join(d, p)).to_pylist())
    return rows


def test_routing_and_counts(corpus):
    stats = _build(corpus)
    assert stats["configs"]["theoremsearch"]["rows"] == 4
    assert stats["configs"]["dolma"]["rows"] == 3
    assert stats["configs"]["findings"]["rows"] == 2
    assert stats["per_source"] == {"arxiv": 1, "dolma": 3, "other": 1,
                                   "proofwiki": 1, "stacks": 1}
    # rows_per_shard=2 -> theoremsearch(4)=2 shards, dolma(3)=2 shards
    assert stats["configs"]["theoremsearch"]["shards"] == 2
    assert stats["configs"]["dolma"]["shards"] == 2
    rows = _read_parquet_rows(corpus[2], "theoremsearch")
    assert sorted(r["source_key"] for r in rows) == [
        "arxiv", "other", "proofwiki", "stacks"]
    assert all(r["source_key"] == "dolma"
               for r in _read_parquet_rows(corpus[2], "dolma"))


def test_main_schema_and_coercions(corpus):
    _build(corpus)
    rows = _read_parquet_rows(corpus[2], "theoremsearch")
    want_cols = ["doc_id", "name", "slogan", "statement", "source", "title",
                 "label", "citations", "category", "source_key"]
    assert sorted(rows[0]) == sorted(want_cols)
    by_id = {r["doc_id"]: r for r in rows}
    assert by_id["1"]["citations"] == 3          # float 3.0 -> int
    assert by_id["2"]["citations"] is None       # the string 'None' -> null
    assert by_id["2"]["category"] is None


def test_findings_vectors_never_ship(corpus):
    _build(corpus)
    rows = _read_parquet_rows(corpus[2], "findings")
    assert len(rows) == 2
    for r in rows:
        assert "dense" not in r and "dense_vec" not in r
    assert sorted(rows[0]) == sorted(
        ["doc_id", "name", "slogan", "statement", "source", "provenance",
         "added_ts"])
    assert rows[0]["provenance"] == "web_added"


def test_licensing_exclusion_honored(corpus):
    stats = _build(corpus, exclude_sources=["dolma"])
    assert "dolma" not in stats["configs"]
    assert stats["excluded_rows"] == 3
    assert "dolma" not in stats["per_source"]
    assert not os.path.exists(os.path.join(corpus[2], "data", "dolma"))
    card = open(os.path.join(corpus[2], "README.md")).read()
    assert "EXCLUDED from this build for licensing reasons: dolma" in card
    assert "config_name: dolma" not in card
    # unknown keys are an error, not a silent no-op
    with pytest.raises(ValueError):
        _build(corpus, exclude_sources=["mathlib"])


def test_strict_gate_refuses_count_drift(corpus):
    with pytest.raises(RuntimeError, match="audited"):
        _build(corpus, strict=True)


def test_card_is_em_dash_free_and_cites(corpus):
    _build(corpus)
    card = open(os.path.join(corpus[2], "README.md")).read()
    assert "—" not in card and "–" not in card
    assert up.ZENODO_DOI in card
    assert "cc-by-sa-4.0" in card and "odc-by" in card
    with pytest.raises(ValueError, match="em dash"):
        up._assert_no_em_dash("bad — text", "x")


def test_validate_loads_with_datasets(corpus):
    pytest.importorskip("datasets")
    _build(corpus)
    report = up.validate(corpus[2])
    assert set(report) == {"theoremsearch", "dolma", "findings"}
    for c, entry in report.items():
        assert entry["match"], c
    assert report["theoremsearch"]["rows_loaded"] == 4
