FROM python:3.12-slim
RUN pip install --no-cache-dir mathlas-mcp
# stdio MCP server; starts dependency-free (official SDK if present, fallback otherwise)
ENTRYPOINT ["mathlas-mcp"]
