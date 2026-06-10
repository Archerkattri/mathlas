"""Tests for verify_formal PROOF CHECKING (statement + AI-supplied proof).

mathlas never GENERATES a proof — it only kernel-checks one the calling AI
wrote (the generator/verifier split is absolute). These tests cover the whole
contract:

  (a) a correct proof (term `rfl` AND a multi-line tactic block) VERIFIES;
  (b) a WRONG proof is REFUTED with the kernel's error message verbatim
      (the repair-loop payload);
  (c) `sorry`/`admit` holes are REJECTED — crucially Lean exits 0 on a sorried
      proof (warning only), so a naive exit-code check would call it verified;
  (d) toolchain-absent => honest UNDETERMINED (find_lean monkeypatched away);
  plus elaboration (declaration building, import hoisting), timeout, and the
  MCP tool surface (schema, repair-loop fields, 12 tools unchanged).

Kernel tests run against the real Lean toolchain when one is found (the repo's
elan install under reference/downloads/elan) and skip honestly otherwise.
"""

import pytest

import mathlas.verify_apply as va
from mathlas.verify_apply import (
    PROOF_REFUTED,
    PROOF_UNDETERMINED,
    PROOF_VERIFIED,
    build_proof_declaration,
    check_lean_proof,
    find_lean,
    verify_formal,
)

LEAN = find_lean()
needs_lean = pytest.mark.skipif(
    LEAN is None, reason="no Lean toolchain found (honest skip — see docs/methods.md)"
)


# ---- elaboration: building the checkable declaration (no Lean needed) --------


def test_declaration_form_term_proof():
    decl = build_proof_declaration("2 + 2 = 4", "rfl")
    assert "theorem _mathlas_check : 2 + 2 = 4 :=" in decl
    assert decl.rstrip().endswith("rfl")


def test_declaration_strips_leading_assign_and_indents_tactics():
    # tolerate an AI passing ':= by ...'; relative tactic indentation preserved,
    # no proof line left at column 0 (would start a new Lean command).
    decl = build_proof_declaration("∀ n : Nat, n + 0 = n", ":= by\n  intro n\n  rfl")
    body = decl.split(":=\n", 1)[1]
    lines = body.rstrip().splitlines()
    assert lines[0] == "  by"
    assert lines[1] == "    intro n"
    assert all(ln.startswith("  ") for ln in lines)


def test_declaration_hoists_imports_to_file_top():
    # mirrors the plain-snippet path, where the snippet IS the file and carries
    # its own imports: import lines in either field move above the theorem.
    decl = build_proof_declaration("import Mathlib\n2 + 2 = 4", "import Foo\nrfl")
    head, theorem = decl.split("theorem", 1)
    assert "import Mathlib" in head and "import Foo" in head
    assert "import" not in theorem


# ---- sorry/admit rejection (pre-kernel: works with or without Lean) ----------


def test_sorry_is_rejected():
    status, detail, _ = check_lean_proof("2 + 2 = 4", "sorry")
    assert status == PROOF_REFUTED and "sorry" in detail


def test_admit_is_rejected_even_inside_tactic_block():
    status, detail, _ = check_lean_proof("2 + 2 = 4", "by\n  admit")
    assert status == PROOF_REFUTED and "admit" in detail


@needs_lean
def test_sorry_in_a_comment_does_not_trip_the_scan():
    status, _, _ = check_lean_proof("2 + 2 = 4", "by\n  -- no sorry here\n  decide")
    assert status == PROOF_VERIFIED


# ---- empty inputs: honest UNDETERMINED with an actionable message ------------


def test_empty_statement_or_proof_is_undetermined():
    status, detail, _ = check_lean_proof("", "rfl")
    assert status == PROOF_UNDETERMINED and "statement" in detail
    status, detail, _ = check_lean_proof("2 + 2 = 4", "   ")
    assert status == PROOF_UNDETERMINED and "proof" in detail


# ---- (a) correct proofs VERIFY against the real kernel ------------------------


@needs_lean
def test_correct_term_proof_verifies():
    status, _, decl = check_lean_proof("2 + 2 = 4", "rfl")
    assert status == PROOF_VERIFIED
    assert "theorem _mathlas_check" in decl


@needs_lean
def test_correct_tactic_block_proof_verifies():
    status, _, _ = check_lean_proof("∀ n : Nat, n + 0 = n", "by\n  intro n\n  rfl")
    assert status == PROOF_VERIFIED


# ---- (b) wrong proofs are REFUTED with the kernel's error (repair payload) ----


@needs_lean
def test_wrong_proof_refuted_with_kernel_error():
    status, detail, _ = check_lean_proof("2 + 2 = 5", "rfl")
    assert status == PROOF_REFUTED
    assert "error" in detail  # the kernel's own diagnosis, verbatim


@needs_lean
def test_kernel_error_is_specific_enough_for_repair():
    # the message must say WHY (here: a definitional-equality mismatch), not
    # just "failed" — that specificity is what makes the repair loop work.
    _, detail, _ = check_lean_proof("2 + 2 = 5", "rfl")
    assert "2 + 2" in detail and "5" in detail


# ---- (d) toolchain absent / timeout => honest UNDETERMINED --------------------


def test_no_toolchain_is_honest_undetermined(monkeypatch):
    monkeypatch.setattr(va, "find_lean", lambda: None)
    status, detail, _ = check_lean_proof("2 + 2 = 4", "rfl")
    assert status == PROOF_UNDETERMINED
    assert "toolchain" in detail


@needs_lean
def test_timeout_is_honest_undetermined():
    status, detail, _ = check_lean_proof("2 + 2 = 4", "rfl", timeout_s=0.001)
    assert status == PROOF_UNDETERMINED
    assert "timed out" in detail


@needs_lean
def test_unresolvable_import_is_undetermined_not_refuted():
    # a missing module on this bare toolchain is an ENVIRONMENT limitation,
    # never presented as evidence the proof is wrong. (If a full mathlib
    # toolchain is installed, the same call legitimately verifies instead.)
    status, detail, _ = check_lean_proof("import Mathlib\n2 + 2 = 4", "rfl")
    assert status in (PROOF_UNDETERMINED, PROOF_VERIFIED)
    if status == PROOF_UNDETERMINED:
        assert "import" in detail


# ---- library-level verify_formal(proof=...) ----------------------------------


@needs_lean
def test_verify_formal_proof_mode_verdicts():
    good = verify_formal(None, statement="2 + 2 = 4", proof="rfl")
    assert good.applies is True and good.confidence == 1.0
    assert good.conditions[0].satisfied is True
    bad = verify_formal(None, statement="2 + 2 = 5", proof="rfl")
    assert bad.applies is False and bad.failure
    assert "error" in bad.conditions[0].evidence


def test_verify_formal_proof_mode_undetermined(monkeypatch):
    monkeypatch.setattr(va, "find_lean", lambda: None)
    v = verify_formal(None, statement="2 + 2 = 4", proof="rfl")
    assert v.applies is False and v.conditions[0].satisfied is None
    assert "UNDETERMINED" in v.note


# ---- MCP tool surface ----------------------------------------------------------


@needs_lean
def test_tool_verify_formal_proof_check_roundtrip():
    from mathlas.server import tool_verify_formal

    out = tool_verify_formal("2 + 2 = 4", proof="rfl")
    assert out["mode"] == "proof_check"
    assert out["proof_status"] == PROOF_VERIFIED
    assert out["checked"] is True and out["stub"] is False
    assert out["typechecks"] is True and out["kernel_error"] is None
    assert "theorem _mathlas_check" in out["declaration"]

    out = tool_verify_formal("2 + 2 = 5", proof="rfl")
    assert out["proof_status"] == PROOF_REFUTED
    assert out["checked"] is True and out["typechecks"] is False
    assert out["kernel_error"] and "error" in out["kernel_error"]
    assert "repair" in out["note"]  # the repair-loop story, agent-facing


def test_tool_verify_formal_proof_check_no_toolchain(monkeypatch):
    monkeypatch.setattr(va, "find_lean", lambda: None)
    from mathlas.server import tool_verify_formal

    out = tool_verify_formal("2 + 2 = 4", proof="rfl")
    assert out["proof_status"] == PROOF_UNDETERMINED
    assert out["checked"] is False and out["stub"] is True
    assert out["typechecks"] is None
    assert "remediation" in out  # exactly what unblocks a real check


def test_tool_verify_formal_plain_snippet_path_unchanged():
    # no `proof` => the original statement-snippet contract, byte-for-byte
    # fields (no proof-mode keys leak in).
    from mathlas.server import tool_verify_formal

    out = tool_verify_formal("claim", lean=None)
    assert out["lean_provided"] is False and out["checked"] is False
    assert "proof_status" not in out and "mode" not in out


def test_schema_has_proof_param_and_still_12_tools():
    from mathlas.server import _dispatch, tool_names

    assert len(tool_names()) == 12  # extends verify_formal, adds no tool
    listed = _dispatch("tools/list", {})["tools"]
    vf = next(t for t in listed if t["name"] == "verify_formal")
    props = vf["inputSchema"]["properties"]
    assert "proof" in props and vf["inputSchema"]["required"] == ["statement"]
    assert vf["outputSchema"]["properties"]["proof_status"]["enum"] == [
        "VERIFIED_PROOF",
        "REFUTED",
        "UNDETERMINED",
    ]
    assert "REFUTED" in vf["description"]  # agent-actionable: repair loop named
