"""Cross-encoder reranking of retrieval candidates (Qwen3-Reranker).

The hybrid retriever's two first-stage channels (dense bi-encoder + BM25) score
query and document INDEPENDENTLY -- neither ever reads the query and a candidate
together. A cross-encoder does exactly that: it feeds (query, document) jointly
through the model and asks "does this document answer this query?", which is the
standard second-stage precision lift (rerank the top-N, leave recall to stage 1).

Production model: ``Qwen/Qwen3-Reranker-0.6B`` -- same model family as the
Qwen3-Embedding index, instruction-aware, scored as a binary yes/no judgement:
the model card's documented format is a chat template whose final assistant
token distribution is read out at the "yes"/"no" token ids;
``P(yes)/(P(yes)+P(no))`` (computed in float32) is the relevance score. Note
the deliberate saturation behaviour: very confident pairs round to P(yes)==1.0
and tie; ``HybridRetriever._maybe_rerank`` breaks ties by the FIRST-STAGE fused
order (stable sort), so among candidates the reranker is equally sure about,
the dense+BM25 prior decides. Measured on the 3.68M-corpus self-recall proxy
(docs/RETRIEVAL_UPGRADE_NOTES.md, 2026-06-09) this tie-break is load-bearing:
fully de-tied scores (logit differences) reorder confident candidates and DROP
R@1, because a cross-encoder ranks by relevance, not by exact row identity, and
this corpus contains many same-content theorem variants. We use the card's
exact prefix/suffix tokens and a task instruction tuned to this corpus (a
documented +1-5% over the generic instruction, mirroring ``Qwen3Embedder``).

Design mirrors ``embed.py``: an abstract interface (so tests can inject a cheap
fake), a lazy-loaded production backend (importing mathlas never pulls torch),
and honest errors -- if torch/transformers/weights are unavailable the backend
raises ``ImportError``/``OSError`` at first use; ``HybridRetriever`` catches
that, says so on stderr, and serves the un-reranked fusion (never a silent
pretend-rerank).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

import numpy as np


class Reranker(ABC):
    """Scores (query, document-text) pairs jointly; higher = more relevant."""

    @abstractmethod
    def score(self, query: str, docs: Sequence[str]) -> np.ndarray:
        """Return a ``(len(docs),)`` float32 array of relevance scores for the
        given query. Scores are comparable only within one call."""
        ...


class Qwen3Reranker(Reranker):
    """``Qwen/Qwen3-Reranker-0.6B`` cross-encoder (lazy-loaded).

    Implements the model card's documented scoring protocol: chat-templated
    ``<Instruct>/<Query>/<Document>`` prompt, last-token logits restricted to
    the "yes"/"no" token ids, ``softmax`` -> ``P(yes)`` as the score.

    ``max_length`` bounds the TOKENIZED pair length (prefix + instruct + query
    + document + suffix); long LaTeX statements are truncated from the document
    side, which is safe for reranking (the head of a theorem statement carries
    its identity).
    """

    #: Task instruction for this corpus (queries are math problems/statements;
    #: documents are theorem records). Documented +1-5% over the generic
    #: web-search instruction, same mechanism as the embedder's instruct.
    DEFAULT_INSTRUCT = (
        "Given a math problem or statement, judge whether the theorem answers it"
    )

    # The card's exact prompt frame (tokenized once and re-used).
    _PREFIX = (
        "<|im_start|>system\nJudge whether the Document meets the requirements "
        "based on the Query and the Instruct provided. Note that the answer can "
        "only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    )
    _SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, model: str = "Qwen/Qwen3-Reranker-0.6B",
                 device: Optional[str] = None, max_length: int = 2048,
                 batch_size: int = 32,
                 instruct: Optional[str] = DEFAULT_INSTRUCT) -> None:
        self.model_name = model
        self.device = device
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.instruct = instruct or self.DEFAULT_INSTRUCT
        self._tok = None      # lazy: nothing heavy happens until first .score()
        self._model = None

    # ------------------------------------------------------------------ #
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:  # pragma: no cover - optional heavy dep
            raise ImportError(
                "Qwen3Reranker needs torch + transformers "
                "(pip install 'mathlas[embed]'); without them reranking is "
                "unavailable and HybridRetriever serves the un-reranked fusion."
            ) from e
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tok = AutoTokenizer.from_pretrained(self.model_name,
                                                  padding_side="left")
        dtype = torch.float16 if dev.startswith("cuda") else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, dtype=dtype).to(dev).eval()
        self.device = dev
        self._yes = self._tok.convert_tokens_to_ids("yes")
        self._no = self._tok.convert_tokens_to_ids("no")
        self._prefix_ids = self._tok.encode(self._PREFIX, add_special_tokens=False)
        self._suffix_ids = self._tok.encode(self._SUFFIX, add_special_tokens=False)

    def _pair_ids(self, query: str, doc: str) -> List[int]:
        body = (f"<Instruct>: {self.instruct}\n"
                f"<Query>: {query}\n<Document>: {doc}")
        budget = self.max_length - len(self._prefix_ids) - len(self._suffix_ids)
        ids = self._tok.encode(body, add_special_tokens=False,
                               truncation=True, max_length=max(budget, 16))
        return self._prefix_ids + ids + self._suffix_ids

    def score(self, query: str, docs: Sequence[str]) -> np.ndarray:
        if not docs:
            return np.zeros(0, dtype=np.float32)
        self._ensure_loaded()
        import torch
        out = np.zeros(len(docs), dtype=np.float32)
        pad = self._tok.pad_token_id
        if pad is None:  # Qwen tokenizers define one, but stay defensive
            pad = self._tok.eos_token_id
        with torch.inference_mode():
            for b in range(0, len(docs), self.batch_size):
                chunk = docs[b:b + self.batch_size]
                seqs = [self._pair_ids(query, d) for d in chunk]
                L = max(len(s) for s in seqs)
                input_ids = torch.full((len(seqs), L), pad, dtype=torch.long)
                attn = torch.zeros((len(seqs), L), dtype=torch.long)
                for i, s in enumerate(seqs):     # left-pad (padding_side="left")
                    input_ids[i, L - len(s):] = torch.tensor(s, dtype=torch.long)
                    attn[i, L - len(s):] = 1
                input_ids = input_ids.to(self._model.device)
                attn = attn.to(self._model.device)
                # Only the LAST position's logits are read; materialising the
                # full (batch, seq, 151k-vocab) logits tensor OOMs a 32GB GPU
                # at batch 32 x 2048 tokens. ``logits_to_keep=1`` restricts the
                # lm_head to the final position (supported by Qwen3 in
                # transformers >= 4.49); fall back to full logits if absent.
                try:
                    logits = self._model(input_ids=input_ids, attention_mask=attn,
                                         logits_to_keep=1).logits[:, -1, :]
                except TypeError:  # older transformers: no logits_to_keep
                    logits = self._model(input_ids=input_ids,
                                         attention_mask=attn).logits[:, -1, :]
                # P(yes) over {yes,no}, float32. Saturation to exactly 1.0 for
                # very confident pairs is INTENTIONAL (see module docstring):
                # ties at the top are broken downstream by the first-stage
                # order, which measurably beats fully de-tied logit scores on
                # a corpus with many same-content theorem variants.
                pair = torch.stack([logits[:, self._no].float(),
                                    logits[:, self._yes].float()], dim=1)
                p_yes = torch.softmax(pair, dim=1)[:, 1]
                out[b:b + len(chunk)] = p_yes.cpu().numpy()
        return out


def doc_rerank_text(name: Optional[str], slogan: Optional[str],
                    statement: Optional[str], max_chars: int = 4000) -> str:
    """The document text a cross-encoder judges: name + NL slogan + the REAL
    statement (unlike the dense channel, a cross-encoder handles LaTeX fine and
    the statement carries the discriminative detail), bounded for token budget."""
    parts = [p for p in (name, slogan, statement) if p]
    return " -- ".join(parts)[:max_chars]


__all__ = ["Reranker", "Qwen3Reranker", "doc_rerank_text"]
