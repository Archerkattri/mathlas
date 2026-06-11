# Changelog — math_engine

All notable changes, per version. Auto-generated from git tags by
`third_party/launch_materials/gen_changelogs.sh`; do not edit by hand.

## Unreleased

- docs: add per-version CHANGELOG (2bfc52e)
- Bump server.json to 1.4.0 for MCP registry re-publish (49b328d)

## v1.4.0 — 2026-06-11

- Release v1.4.0 (3afb888)
- Declare MCP ToolAnnotations on every tool (trust/safety hints) (2ae72bb)
- security(funsearch): replace socket monkey-patch with real OS-level sandbox (caafe47)
- fix(doi): update Zenodo DOI to correct concept record (3571a8f)
- Ignore paper/ (manuscripts never live in the repo) (4a205ab)
- Release v1.3.0 (bc7995b)
- 0.6B end-to-end laptop tier: built, measured, served (MATHLAS_ENCODER=0.6b) (2bad71d)

## v1.3.0 — 2026-06-10

- Release v1.3.0 (ac1d8cb)
- 0.6B end-to-end laptop tier: built, measured, served (MATHLAS_ENCODER=0.6b) (5f23a4e)
- Paper: first full draft of the mathlas workshop paper (RQ1 agent bench, RQ2 self-augmenting loop, RQ3 serving tiers) (c4d701f)

## v1.2.0 — 2026-06-10

- Release v1.2.0 (400d80a)
- Docs: v1.2 README/RESULTS refresh with dual-channel numbers + honest caveats (79ca131)
- Statement channel shipped: full-corpus dual-channel retrieval, measured + served opt-in (d1c5e5b)
- Read __version__ from package metadata (kill version drift) (79954bd)
- Determinism + input-validation fixes for the dev retrieval path (3b3a997)
- Agent bench hard set: 8 verification-bottleneck tasks discriminate again (18/18 vs 15/18) (4b3be24)
- HF corpus release: audited licensing matrix + one-command upload script (1397283)
- Source-aware retrieval: opt-in source_filter / source_weights on search_existing_math (f6e162e)
- Re-measure every headline on the served 3.68M index + Claude Fable 5 agent loop (954f0f7)
- Quantized laptop tier: int8/binary memmap serving of the 3.68M dense index — recall-lossless, 2.4 s/query CPU (27bcb64)
- docs: proof checking (the repair loop) + formal-search cache (6741fd5)
- search_formal_math: on-disk TTL cache — serve the last good response when Loogle/LeanSearch are down (aefcd53)
- verify_formal: kernel-check AI-supplied Lean 4 proofs (repair-loop ready) (84ac5f4)
- Add Glama score badge (251a247)
- DOI badge: static shields.io (Zenodo badge endpoint 302s through GitHub's proxy) (68ea8cf)
- glama.json: claim server listing (a479cee)

## v1.1.2 — 2026-06-10

- Relicense code to Apache-2.0 (v1.1.2) (b92643c)
- Dockerfile for registry checks (Glama / Docker MCP Catalog) (81f050c)
- v1.1.1: registry namespace case fix, Zenodo DOI, CITATION.cff (e5f0ebd)

## v1.1.0 — 2026-06-09

- v1.1.0 retrieval: RRF k=10, reranker blend, dual-channel statement indexing (1f15579)
- v1.1.0 product: positioning truth-pass, uvx, 12-tool schema pass, search_formal_math, registry prep (77c2d1c)
- chore: scrub local model path + sync version to 1.0.1 (701969a)
- docs: restructure README — hook first, split core/full tools, drop solve() confusion, add logs/ to gitignore (70ffe60)
- chore: remove unused square logo (banner is the only displayed image) (efc45ec)
- docs: badge row — add PyPI version + real links, drop vanity badges (1a1139e)

## v1.0.1 — 2026-06-07

- release 1.0.1: show banner on PyPI (absolute-URL README header) + version bump (bc95de7)
- docs: landscape banner in README header + transparent (bg-removed) square logo for PyPI/icon (f7301a5)
- docs: add logo to README header + social banner asset (23f1aae)
- docs: add icon brief for the mathlas logo (38fbc99)
- release: publish to PyPI as mathlas-mcp (import name stays 'mathlas') (10b4ddb)
- license: Apache-2.0 -> PolyForm Noncommercial 1.0.0 (noncommercial use only) (6529362)
- docs(README): move Results + self-augmenting-loop sections to the top (f5b3510)
- docs: polish remaining docs to production grade (d69b827)
- chore: production-clean repo + product README (280->170) (451f39d)
- mathlas 1.0.0 — production-ready: real 1.635M index, self-augmenting dense loop, MCP, beats TheoremSearch+LLMs on the augmented 110 (0eab09d)
- feat(mathlas): merge dolma into 1.63M EXACT dense index + train/val/test splits + serving (d28e54e)
- fix(slogans): Qwen3.6 needs enable_thinking=false + --jinja (the Qwen3 /no_think is ignored -> empty slogans); truncate prompt to [:1200]. Wire HybridRetriever.from_faiss for serving the 11.3M dolma index. (2af910b)
- mathlas: discovery + web-augmentation layer (13 MCP tools) (a73de58)
- mathlas: add OEIS integer-sequence domain + real Lean formal-verify tier (7 MCP tools) (93291bf)
- mathlas v0.1.0: re-architect as a tool an AI USES (MCP server, no LLM dependency) (f499a55)
- mathlas v0: open math-application engine (b00f9db)

