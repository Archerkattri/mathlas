"""Quantized dense tier — laptop-class serving of the flagship index.

The flagship dense channel is a 3.68M x 4096 fp16 matrix (~30 GB) that
``HybridRetriever.from_index`` loads to fp32 RAM (~60 GB) — fine on the build
box, unusable on a laptop. This module ships the proven fix (TheoremSearch
runs binary embeddings + full-precision rescore at 9.2M scale): quantize the
SAME index once, then serve it from disk-memmapped quantized artifacts:

  * **int8**  — per-dimension symmetric quantization (one fp32 scale per dim,
    ``q = round(x/scale)`` clipped to [-127, 127]). ~15 GB on disk, served by
    memmap (resident memory = whatever the OS caches). Search is an exact
    chunked dequantized dot product over all rows.
  * **binary** — sign bits packed 8/byte (4096 dims -> 512 bytes/doc, ~1.9 GB).
    Search is two-stage: Hamming distance over the packed bits selects a
    ``shortlist`` (default 1000), which is RESCORED with exact dot products
    against int8-dequantized (or fp16 original) rows — only ``shortlist`` rows
    are ever gathered, so the rescore is a few MB of I/O per query.

Both are opt-in backends of ``HybridRetriever.from_index(..., quantized=...)``;
default behaviour (fp32-in-RAM) is unchanged.

HONESTY NOTE (query encoder): quantization shrinks the *document* side only.
Queries must still be embedded by the model the index was built with
(Qwen3-Embedding-8B) — a smaller encoder (e.g. the 0.6B) lives in a DIFFERENT
embedding space and would need its own document index built from scratch. The
laptop tier is therefore "quantized index + 8B query embeddings from a GPU box
or a remote encode endpoint" until a 0.6B-built index exists. Measured
quality/footprint numbers: docs/QUANTIZED_TIER.md.

Hamming kernel: numpy 1.x has no ``bitwise_count``; we XOR the packed rows
with the packed query and look the per-uint16 popcount up in a 65536-entry
table — one fused pass per chunk, no Python-level per-row loop.
"""
from __future__ import annotations

import json
import os
import struct
import zipfile
from typing import List, Optional, Sequence, Tuple

import numpy as np

#: popcount of every uint16 value; XOR-ed packed rows are viewed as uint16 and
#: looked up here (half the lookups of a uint8 table).
_POP16 = np.array([bin(v).count("1") for v in range(1 << 16)], dtype=np.uint8)


# --------------------------------------------------------------------- #
# memmap a .npy member stored INSIDE an .npz (np.savez writes ZIP_STORED,
# i.e. uncompressed, so the raw array bytes sit contiguously in the file).
# This is how we stream the 30 GB fp16 matrix without ever materializing it.
# --------------------------------------------------------------------- #
def npz_member_memmap(npz_path: str, member: str = "matrix.npy") -> np.memmap:
    """Return a read-only ``np.memmap`` view of an uncompressed npz member."""
    with zipfile.ZipFile(npz_path) as z:
        try:
            info = z.getinfo(member)
        except KeyError:
            raise KeyError(f"{member!r} not found in {npz_path}") from None
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(
                f"{member!r} in {npz_path} is compressed; only np.savez "
                f"(uncompressed) npz files can be memmapped")
    with open(npz_path, "rb") as f:
        # local file header: 30 fixed bytes, then name + extra fields.
        f.seek(info.header_offset)
        hdr = f.read(30)
        if hdr[:4] != b"PK\x03\x04":
            raise ValueError(f"bad zip local header for {member!r} in {npz_path}")
        n_name, n_extra = struct.unpack("<HH", hdr[26:30])
        f.seek(info.header_offset + 30 + n_name + n_extra)
        version = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format._read_array_header(f, version)
        data_offset = f.tell()
    if fortran:
        raise ValueError(f"{member!r} is Fortran-ordered; expected C order")
    return np.memmap(npz_path, dtype=dtype, mode="r", offset=data_offset,
                     shape=shape)


# --------------------------------------------------------------------- #
# artifact paths + one-time build
# --------------------------------------------------------------------- #
def quant_paths(index_path: str) -> dict:
    """Sidecar artifact paths for a given dense ``index.npz``."""
    stem = os.path.splitext(os.fspath(index_path))[0]
    return {
        "int8": stem + ".q8.npy",
        "int8_scale": stem + ".q8.scale.npy",
        "binary": stem + ".qbin.npy",
        "meta": stem + ".quant.json",
    }


def quantize_matrix_int8(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-dim symmetric int8 quantization of (already unit-norm) fp32 rows.

    Returns ``(q8, scale)`` with ``q8 = clip(round(M/scale), -127, 127)`` and
    ``scale[d] = absmax_d / 127`` (zero-variance dims get scale 1 -> all-zero
    column, exact)."""
    M = np.asarray(M, dtype=np.float32)
    absmax = np.abs(M).max(axis=0) if M.size else np.zeros(M.shape[1])
    scale = (absmax / 127.0).astype(np.float32)
    scale[scale == 0] = 1.0
    q8 = np.clip(np.rint(M / scale), -127, 127).astype(np.int8)
    return q8, scale


def dequantize_int8(q8: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Inverse of :func:`quantize_matrix_int8` (up to rounding error)."""
    return q8.astype(np.float32) * scale.astype(np.float32)


def pack_signs(M: np.ndarray) -> np.ndarray:
    """Sign-bit packing: (n, dim) floats -> (n, dim/8) uint8 (bit = x > 0)."""
    return np.packbits(np.asarray(M) > 0, axis=1)


def _unit_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def build_quantized_artifacts(index_path: str, kinds: Sequence[str] = ("int8", "binary"),
                              chunk_rows: int = 100_000, force: bool = False,
                              log=print) -> dict:
    """Stream the fp16 npz matrix in row slabs and write the quantized sidecars.

    Two passes for int8 (per-dim absmax, then quantize+write), one for binary.
    Peak extra memory is one fp32 slab (~1.6 GB at the default 100k x 4096);
    the source matrix is memmapped, never loaded whole. Existing artifacts are
    kept unless ``force``. Rows are L2-normalized in fp32 before quantization
    (the serving dot product is cosine)."""
    paths = quant_paths(index_path)
    M = npz_member_memmap(index_path)
    n, dim = M.shape
    if dim % 8:
        raise ValueError(f"dim {dim} not a multiple of 8; cannot bit-pack")
    todo = [k for k in kinds
            if force or not (os.path.exists(paths[k]) and
                             (k != "int8" or os.path.exists(paths["int8_scale"])))]
    if "int8" in todo:
        absmax = np.zeros(dim, dtype=np.float32)
        for lo in range(0, n, chunk_rows):
            slab = _unit_rows(M[lo:lo + chunk_rows])
            np.maximum(absmax, np.abs(slab).max(axis=0), out=absmax)
            log(f"  int8 pass1 absmax {min(lo + chunk_rows, n)}/{n}")
        scale = (absmax / 127.0).astype(np.float32)
        scale[scale == 0] = 1.0
        np.save(paths["int8_scale"], scale)
        q8 = np.lib.format.open_memmap(paths["int8"], mode="w+",
                                       dtype=np.int8, shape=(n, dim))
        for lo in range(0, n, chunk_rows):
            slab = _unit_rows(M[lo:lo + chunk_rows])
            q8[lo:lo + chunk_rows] = np.clip(
                np.rint(slab / scale), -127, 127).astype(np.int8)
            log(f"  int8 pass2 write {min(lo + chunk_rows, n)}/{n}")
        q8.flush()
        del q8
    if "binary" in todo:
        qb = np.lib.format.open_memmap(paths["binary"], mode="w+",
                                       dtype=np.uint8, shape=(n, dim // 8))
        for lo in range(0, n, chunk_rows):
            qb[lo:lo + chunk_rows] = pack_signs(M[lo:lo + chunk_rows])
            log(f"  binary write {min(lo + chunk_rows, n)}/{n}")
        qb.flush()
        del qb
    st = os.stat(index_path)
    meta = {"n": int(n), "dim": int(dim), "source": os.path.abspath(index_path),
            "source_size": int(st.st_size), "source_mtime_ns": int(st.st_mtime_ns),
            "kinds": sorted(set(list(kinds) + (
                json.load(open(paths["meta"])).get("kinds", [])
                if os.path.exists(paths["meta"]) else [])))}
    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return paths


# --------------------------------------------------------------------- #
# serving
# --------------------------------------------------------------------- #
def hamming_distances(packed_rows: np.ndarray, packed_q: np.ndarray,
                      chunk_rows: int = 500_000) -> np.ndarray:
    """Hamming distance of every packed row to one packed query (uint8 rows).

    XOR + uint16 popcount table, chunked so the temporary stays small."""
    n = packed_rows.shape[0]
    out = np.empty(n, dtype=np.int32)
    q16 = packed_q.reshape(1, -1)
    for lo in range(0, n, chunk_rows):
        x = np.bitwise_xor(packed_rows[lo:lo + chunk_rows], q16)
        out[lo:lo + chunk_rows] = _POP16[x.view(np.uint16)].sum(
            axis=1, dtype=np.int32)
    return out


class QuantizedDenseIndex:
    """Memmap-served quantized dense index (the laptop tier).

    ``kind="int8"``  : exact dequantized dot product over all rows (one pass
                       over the ~15 GB memmap, chunked).
    ``kind="binary"``: packed-bit Hamming over ~1.9 GB selects ``shortlist``
                       candidates, then those rows only are rescored with an
                       exact dot product against the rescore source:
                       ``rescore="auto"`` prefers int8 sidecar, falls back to
                       the original fp16 npz matrix, else raw Hamming order.
    """

    def __init__(self, index_path: str, kind: str = "int8",
                 shortlist: int = 1000, rescore: str = "auto",
                 chunk_rows: int = 250_000) -> None:
        if kind not in ("int8", "binary"):
            raise ValueError(f"kind must be 'int8' or 'binary', got {kind!r}")
        self.kind = kind
        self.shortlist = int(shortlist)
        self.chunk_rows = int(chunk_rows)
        p = quant_paths(index_path)
        self.index_path = os.fspath(index_path)
        self._q8 = self._scale = self._bits = self._fp16 = None
        if kind == "int8" or rescore in ("auto", "int8"):
            if os.path.exists(p["int8"]) and os.path.exists(p["int8_scale"]):
                self._q8 = np.load(p["int8"], mmap_mode="r")
                self._scale = np.load(p["int8_scale"]).astype(np.float32)
            elif kind == "int8" or rescore == "int8":
                raise FileNotFoundError(
                    f"int8 artifacts missing beside {index_path} "
                    f"(expected {p['int8']}); build them once with "
                    f"mathlas.retrieve.quantized.build_quantized_artifacts()")
        if kind == "binary":
            if not os.path.exists(p["binary"]):
                raise FileNotFoundError(
                    f"binary artifact missing beside {index_path} "
                    f"(expected {p['binary']}); build it once with "
                    f"mathlas.retrieve.quantized.build_quantized_artifacts()")
            self._bits = np.load(p["binary"], mmap_mode="r")
            if rescore in ("auto", "fp16") and self._q8 is None:
                if os.path.exists(index_path):
                    self._fp16 = npz_member_memmap(index_path)
                elif rescore == "fp16":
                    raise FileNotFoundError(f"fp16 source {index_path} missing")
            if rescore == "none":
                self._q8 = self._fp16 = None
        base = self._q8 if self._q8 is not None else self._bits
        self.n = int(base.shape[0])
        self.dim = int(self._q8.shape[1] if self._q8 is not None
                       else self._bits.shape[1] * 8)

    # -- internals ------------------------------------------------------ #
    def _int8_sims(self, q: np.ndarray) -> np.ndarray:
        """Exact dequantized cosine of every row vs a unit query (chunked)."""
        qs = (np.asarray(q, dtype=np.float32) * self._scale)
        sims = np.empty(self.n, dtype=np.float32)
        for lo in range(0, self.n, self.chunk_rows):
            sims[lo:lo + self.chunk_rows] = \
                self._q8[lo:lo + self.chunk_rows].astype(np.float32) @ qs
        return sims

    def _rescore(self, q: np.ndarray, cand: np.ndarray) -> np.ndarray:
        """Exact scores for candidate rows (gather only those rows)."""
        q = np.asarray(q, dtype=np.float32)
        if self._q8 is not None:
            rows = self._q8[cand].astype(np.float32)
            return rows @ (q * self._scale)
        if self._fp16 is not None:
            rows = _unit_rows(self._fp16[cand])
            return rows @ q
        return None  # rescore disabled

    # -- API ------------------------------------------------------------ #
    def search(self, q: np.ndarray, k: int = 10) -> List[int]:
        """Top-``k`` doc rows for a unit-norm fp32 query vector, best-first."""
        q = np.asarray(q, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            raise ValueError(f"query dim {q.shape[0]} != index dim {self.dim}")
        k = min(int(k), self.n)
        if self.kind == "int8":
            sims = self._int8_sims(q)
            top = np.argpartition(-sims, k - 1)[:k]
            return [int(i) for i in top[np.argsort(-sims[top])]]
        # binary: Hamming shortlist -> exact rescore
        qbits = pack_signs(q.reshape(1, -1))[0]
        dist = hamming_distances(self._bits, qbits, self.chunk_rows)
        s = min(max(self.shortlist, k), self.n)
        cand = np.argpartition(dist, s - 1)[:s]
        scores = self._rescore(q, cand)
        if scores is None:          # rescore disabled: raw Hamming order
            order = np.argsort(dist[cand], kind="stable")[:k]
            return [int(cand[i]) for i in order]
        order = np.argsort(-scores)[:k]
        return [int(cand[i]) for i in order]


__all__ = ["QuantizedDenseIndex", "build_quantized_artifacts", "quant_paths",
           "quantize_matrix_int8", "dequantize_int8", "pack_signs",
           "hamming_distances", "npz_member_memmap"]
