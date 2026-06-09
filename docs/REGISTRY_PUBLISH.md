# Publishing mathlas to the official MCP registry

Everything below is run **locally by hand** — there is deliberately **no CI**.
The artifacts are already prepared in the repo:

- [`server.json`](../server.json) — the registry manifest
  (`io.github.archerkattri/mathlas`, PyPI package `mathlas-mcp`, stdio transport,
  `uvx` runtime hint).
- The `mcp-name: io.github.archerkattri/mathlas` marker is in `README.md`, which
  is the PyPI long description (`readme = "README.md"` in `pyproject.toml`) — the
  registry verifies PyPI package ownership by finding this marker in the PyPI
  page's long description.

## 0. Prerequisites (one-time)

- The GitHub account **Archerkattri** (the `io.github.archerkattri/*` namespace
  is proven by logging into that account).
- A PyPI account that can publish `mathlas-mcp`.

## 1. Publish to PyPI FIRST

The registry validates that the PyPI package exists and contains the `mcp-name:`
marker, so PyPI must be published before `mcp-publisher publish`. Keep
`server.json`'s `version` + `packages[0].version` in sync with
`pyproject.toml`'s `version` (currently **1.1.0**).

```bash
cd third_party/math_engine
python3 -m pip install --upgrade build twine
python3 -m build                      # builds dist/mathlas_mcp-1.1.0*
python3 -m twine upload dist/*        # PyPI credentials / token
```

Sanity-check after upload: <https://pypi.org/project/mathlas-mcp/> must render
the README **including the literal line** `mcp-name: io.github.archerkattri/mathlas`,
and the one-liner must work:

```bash
uvx mathlas-mcp   # starts the MCP server over stdio (Ctrl-C to quit)
```

## 2. Install the publisher CLI (no CI — local binary)

```bash
brew install mcp-publisher
```

or grab the prebuilt binary:

```bash
curl -L "https://github.com/modelcontextprotocol/registry/releases/latest/download/mcp-publisher_$(uname -s | tr '[:upper:]' '[:lower:]')_$(uname -m).tar.gz" \
  | tar xz mcp-publisher
sudo mv mcp-publisher /usr/local/bin/   # or keep it in the repo dir and use ./mcp-publisher
```

## 3. Login (GitHub device flow, as Archerkattri)

```bash
cd third_party/math_engine
mcp-publisher login github
```

A device-flow URL + code is printed — authorize it **logged in as Archerkattri**
(this is what grants the `io.github.archerkattri/*` namespace). The token lands
in a local `.mcpregistry_*` credential file — it is gitignored; never commit it.

## 4. Publish

```bash
cd third_party/math_engine
mcp-publisher publish      # reads ./server.json
```

## 5. Verify

```bash
curl -s "https://registry.modelcontextprotocol.io/v0/servers?search=io.github.archerkattri/mathlas" | python3 -m json.tool
```

The entry should show `"status": "active"` with version `1.1.0`. Aggregators
(PulseMCP, Smithery, mcp.so, Glama) scrape the official registry, so they pick
the server up from here — no separate submissions needed.

## 6. Releasing a new version later

1. Bump `version` in `pyproject.toml` AND both `version` fields in `server.json`.
2. Repeat step 1 (build + twine upload).
3. `mcp-publisher login github` (if the token expired) + `mcp-publisher publish`.
