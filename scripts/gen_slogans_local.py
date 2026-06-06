#!/usr/bin/env python3
"""Generate NL slogans for the dolma corpus with a LOCAL Qwen3.6-35B-A3B (GGUF) on the GPU.

The free + fast alternative to the Claude-agent path: a local generative LLM batch-generates
the ~9.76M slogans (compute only, no per-token cost). It drives llama.cpp's `llama-server`
(true continuous batching) over its OpenAI-compatible API.

ONE server per GPU, so you control placement exactly:
  * `--gpus 0`      -> Qwen3.6 on GPU 0 only; GPU 1 stays free for the mathlas embed.
  * `--gpus 0,1`    -> a server on each GPU, work round-robined -> ~2x throughput (when neither
                       GPU is needed for mathlas).

Output: `slogan_out/slog_<shard>.jsonl` ({doc_id, slogan}) in --workdir — the exact format
`apply_slogans.py` folds back into the docs shards. Resumable: skips a shard whose slog file
exists OR whose docs already carry slogans (e.g. the shards 0-3 the Claude agents did).

Setup (one-time, when a GPU is free): build the server with `scripts/setup_llama_server.sh`,
then pass its path via --server-bin. (Qwen3.6 has a recent architecture — use an up-to-date
llama.cpp; the Unsloth GGUF in benbi confirms it's supported.)

Run (slogans on GPU 0, leaving GPU 1 for mathlas):
  benbi/benbi_env_311/bin/python3.11 scripts/gen_slogans_local.py \
      --server-bin ~/llama.cpp/build/bin/llama-server \
      --gguf benbi/models/qwen36_gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
      --workdir reference/downloads/dolma_index_build --gpus 0
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

SYS_PROMPT = (
    "You write ONE concise natural-language slogan for a mathematics theorem. Given its LaTeX "
    "statement, output a single plain-English sentence, at most 26 words, stating the theorem's "
    "main result. No LaTeX, no math symbols, no proof details, no citations or label names. "
    "Paraphrase faithfully. Output only the slogan."
)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def start_server(server_bin: str, gguf: str, gpu: int, port: int, slots: int, ctx: int):
    """Launch a llama-server pinned to one GPU with `slots` continuous-batching sequences."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    proc = subprocess.Popen(
        [server_bin, "-m", gguf, "-ngl", "999", "--parallel", str(slots),
         "--ctx-size", str(ctx * slots), "--cont-batching", "--flash-attn", "on", "--jinja",
         "--host", "127.0.0.1", "--port", str(port), "--no-webui"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    health = f"http://127.0.0.1:{port}/health"
    for _ in range(900):                       # up to ~30 min for first model load
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server on GPU {gpu} exited (code {proc.returncode})")
        try:
            with urllib.request.urlopen(health, timeout=2) as r:
                if r.status == 200:
                    return proc
        except Exception:
            pass
        time.sleep(2)
    proc.terminate()
    raise RuntimeError(f"llama-server on GPU {gpu} did not become healthy")


def gen_one(port: int, statement: str) -> str | None:
    """One slogan via the server's OpenAI-compatible chat endpoint (Qwen3.6 thinking off)."""
    body = json.dumps({
        "messages": [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": statement[:1200]},
        ],
        # Qwen3.6 IGNORES the Qwen3-style "/no_think" suffix: it reasons until it exhausts
        # max_tokens and returns EMPTY content. Disable reasoning via the chat-template switch.
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0.3, "max_tokens": 80, "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                out = json.loads(r.read())
            sl = out["choices"][0]["message"]["content"]
            sl = _THINK.sub("", sl).strip().splitlines()
            sl = (sl[-1] if sl else "").strip().strip('"').strip()
            return sl or None
        except Exception:
            time.sleep(1 + attempt)
    return None


def _already_sloganned(shard: str) -> bool:
    with open(shard) as f:
        first = f.readline()
    return bool(first) and bool(json.loads(first).get("slogan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-bin", required=True, help="path to llama.cpp llama-server")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--workdir", required=True, help="dolma_index_build (has docs_*.jsonl)")
    ap.add_argument("--gpus", default="0", help='GPU ids, e.g. "0" or "0,1"')
    ap.add_argument("--slots", type=int, default=16, help="parallel sequences per server")
    ap.add_argument("--ctx", type=int, default=2048, help="context per slot (tokens)")
    ap.add_argument("--port-base", type=int, default=8080)
    ap.add_argument("--shard-start", type=int, default=0)
    ap.add_argument("--shard-end", type=int, default=None)
    args = ap.parse_args()

    gpus = [int(g) for g in args.gpus.split(",") if g.strip() != ""]
    outdir = os.path.join(args.workdir, "slogan_out")
    os.makedirs(outdir, exist_ok=True)
    shards = sorted(glob.glob(os.path.join(args.workdir, "docs_*.jsonl")))[args.shard_start:args.shard_end]

    servers = []                                # (proc, port)
    for i, gpu in enumerate(gpus):
        port = args.port_base + i
        print(f"[server] starting Qwen3.6 on GPU {gpu} (port {port}, {args.slots} slots) ...", flush=True)
        servers.append((start_server(args.server_bin, args.gguf, gpu, port, args.slots, args.ctx), port))
    ports = [p for _, p in servers]
    total_slots = args.slots * len(servers)
    print(f"[server] {len(servers)} server(s) ready; {total_slots} concurrent slots", flush=True)

    def shutdown():
        for proc, _ in servers:
            try:
                proc.send_signal(signal.SIGTERM)
            except Exception:
                pass

    grand = 0
    t_all = time.time()
    try:
        for shard in shards:
            base = os.path.basename(shard).replace(".jsonl", "")
            out = os.path.join(outdir, f"slog_{base}.jsonl")
            if os.path.exists(out) or _already_sloganned(shard):
                continue                        # resumable: skip done shards
            rows = [json.loads(l) for l in open(shard)]
            res = {}
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=total_slots) as ex:
                futs = {ex.submit(gen_one, ports[k % len(ports)], r["statement"]): r["doc_id"]
                        for k, r in enumerate(rows)}
                for fut in as_completed(futs):
                    sl = fut.result()
                    if sl:
                        res[futs[fut]] = sl
            with open(out + ".tmp", "w") as f:
                for r in rows:                  # write in shard order
                    if r["doc_id"] in res:
                        f.write(json.dumps({"doc_id": r["doc_id"], "slogan": res[r["doc_id"]]},
                                           ensure_ascii=False) + "\n")
            os.replace(out + ".tmp", out)
            grand += len(res)
            dt = time.time() - t0
            print(f"[gen] {base}: {len(res)}/{len(rows)} in {dt:.0f}s "
                  f"({len(res)/max(dt,1):.1f}/s) | total {grand:,} "
                  f"({grand/max(time.time()-t_all,1):.1f}/s avg)", flush=True)
    finally:
        shutdown()
        print(f"[done] {grand:,} slogans this run. servers shut down.", flush=True)


if __name__ == "__main__":
    main()
