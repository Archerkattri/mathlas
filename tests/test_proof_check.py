"""Integration checks against Lean's kernel and the public proof-check API."""

import pytest

from mathlas.verify_apply import (
    PROOF_REFUTED,
    PROOF_UNDETERMINED,
    PROOF_VERIFIED,
    check_lean_proof,
    find_lean,
    verify_formal,
)

LEAN = find_lean()
needs_lean = pytest.mark.skipif(
    LEAN is None, reason="Lean is unavailable; kernel integration check not run"
)


@needs_lean
def test_real_lean_accepts_term_and_tactic_proofs():
    term_status, _, _ = check_lean_proof("2 + 2 = 4", "rfl")
    tactic_status, _, _ = check_lean_proof(
        "∀ n : Nat, n + 0 = n", "by\n  intro n\n  rfl"
    )

    assert term_status == PROOF_VERIFIED
    assert tactic_status == PROOF_VERIFIED


@needs_lean
def test_real_lean_refutes_wrong_proof_with_kernel_diagnostic():
    status, detail, _ = check_lean_proof("2 + 2 = 5", "rfl")

    assert status == PROOF_REFUTED
    assert "error" in detail
    assert "2 + 2" in detail and "5" in detail


@needs_lean
def test_real_lean_rejects_sorry_and_admit_holes():
    sorry_status, sorry_detail, _ = check_lean_proof("2 + 2 = 4", "sorry")
    admit_status, admit_detail, _ = check_lean_proof("2 + 2 = 4", "by\n  admit")

    assert sorry_status == PROOF_REFUTED and "sorry" in sorry_detail
    assert admit_status == PROOF_REFUTED and "admit" in admit_detail


@needs_lean
def test_real_lean_missing_import_is_not_reported_as_a_refutation():
    status, detail, _ = check_lean_proof("import Mathlib\n2 + 2 = 4", "rfl")

    assert status in (PROOF_VERIFIED, PROOF_UNDETERMINED)
    if status == PROOF_UNDETERMINED:
        assert "import" in detail


@needs_lean
def test_public_proof_check_api_returns_the_kernel_verdict():
    verified = verify_formal(None, statement="2 + 2 = 4", proof="rfl")
    refuted = verify_formal(None, statement="2 + 2 = 5", proof="rfl")

    assert verified.applies is True and verified.confidence == 1.0
    assert verified.conditions[0].satisfied is True
    assert refuted.applies is False and refuted.failure
    assert refuted.conditions[0].evidence
