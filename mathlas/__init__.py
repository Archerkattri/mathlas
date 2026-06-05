"""math_engine — an open math-application engine.

Map a problem to the EXISTING math that solves it, then VERIFY that it applies,
with an honest provenance label (never "novel"). Two domains share one spine
(retrieve/identify -> MAP -> VERIFY -> PROVENANCE):

  NUMERIC   a real value/constant -> a known closed form, airtight independent
            re-evaluation. ``identify`` / ``engine.py``. No LLM, no network.

  PROBLEM   a described problem -> existing theorems that apply, via OUR OWN
            hybrid (dense+BM25+RRF) retrieval, two-stage needs<->guarantees
            mapping, and structured-adversarial applicability verification.
            ``solve`` / ``solve.py``. Pluggable LLM brain + embedder.

The engine builds its own retrieval index (over the open theorem dataset, used
as raw data) and contacts no third-party running system. See docs/.

    >>> import mpmath
    >>> from mathlas import identify
    >>> print(identify(mpmath.zeta(2)))      # doctest: +SKIP
    1.64493406684823 -> pi**2/6  [known_form, verified 48 digits]
"""
# Numeric domain (no heavy deps).
from .engine import identify, Result, Candidate
from .provenance import Provenance, Novelty
from .verify import verify_closed_form, VerifyResult

# Retrieval/informal domain. ``solve`` pulls in numpy/scipy (declared deps);
# the heavier retrieval corpus/embedder imports stay lazy inside their modules.
from .solve import solve, Solution, AppliedResult
from .map import map_candidates, extract_signature, Mapping, Signature
from .verify_apply import (verify_informal, verify_numeric_claim, verify_formal,
                           ApplyVerdict, Tier)
from .llm import LLM, EchoLLM, AnthropicLLM
from .retrieve import Retriever, Candidate as RetrievedCandidate

__version__ = "0.0.1"
__all__ = [
    # numeric
    "identify", "Result", "Candidate",
    "verify_closed_form", "VerifyResult",
    # provenance
    "Provenance", "Novelty",
    # problem/informal
    "solve", "Solution", "AppliedResult",
    "map_candidates", "extract_signature", "Mapping", "Signature",
    "verify_informal", "verify_numeric_claim", "verify_formal",
    "ApplyVerdict", "Tier",
    "LLM", "EchoLLM", "AnthropicLLM",
    "Retriever", "RetrievedCandidate",
]
