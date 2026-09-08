# linescout-api

FastAPI runtime for LineScout. One Uvicorn worker; GPU inference (Milestone 4)
is serialised behind a single-request semaphore.

```bash
cd services/api
uv sync --frozen --extra dev --python 3.11   # lockfile-pinned; add --extra gpu for torch later

# Run against the synthetic gallery (no models, deterministic fixture results)
.venv/bin/python -m linescout_api.main      # synthetic gallery by default

# Health, docs, OpenAPI
curl -s http://127.0.0.1:8000/api/v1/health | python -m json.tool
open http://127.0.0.1:8000/api/v1/docs

# Checks
.venv/bin/ruff check . && .venv/bin/mypy linescout_api && .venv/bin/pytest
```

## Configuration

Every setting has a local, free default. Override with `LINESCOUT_*` variables
or a `.env` file here. Relative paths resolve from the repository root.

| Variable | Default | Notes |
|---|---|---|
| `LINESCOUT_HOST` / `LINESCOUT_PORT` | `127.0.0.1` / `8000` | Loopback only; never `0.0.0.0` |
| `LINESCOUT_DEVICE` | `auto` | `auto`, `cuda` (fail if missing), `cpu` |
| `LINESCOUT_FIXTURE_MODE` | `true` | Deterministic ranker until models exist |
| `LINESCOUT_GALLERY_MANIFEST` | unset | Path to a validated `manifest.json` |
| `LINESCOUT_DB_PATH` | `data/linescout.sqlite3` | SQLite in WAL mode |
| `LINESCOUT_CURATION_MODE` | `false` | Mounts `/api/v1/curation/*` |
| `LINESCOUT_CORS_ORIGINS` | Vite dev origins | Comma-separated; each must be an exact `http(s)://host[:port]` origin (no wildcard, path, query, or credentials) |
| `LINESCOUT_MAX_IMAGE_BYTES` | `4194304` | 4 MiB snapshot PNG, as uploaded |
| `LINESCOUT_MAX_STROKES_BYTES` | `262144` | 256 KiB gzipped strokes, as uploaded |
| `LINESCOUT_MAX_STROKES_DECOMPRESSED_BYTES` | `1048576` | 1 MiB stroke JSON after gunzip |
| `LINESCOUT_ADDITIONAL_ALLOWED_HOSTS` | empty | Test-only comma-separated bare hostnames accepted in the Host header (e.g. `testserver`); keep empty in production |

## Request size limits

Three distinct layers, smallest-first:

1. **Per upload** (router, while spooling): `image` ≤ 4 MiB; `strokes` ≤ 256 KiB
   *compressed* and ≤ 1 MiB *after* gzip decompression. Byte budgets are read
   with `read(limit + 1)` so an oversized part never fully buffers; the gzip
   member is expanded with a bounded `zlib.decompressobj` (remaining-output
   bound + one overflow-detection byte) that must end exactly at the input's
   end — truncated streams, bad CRCs, trailing bytes, and concatenated
   members are all rejected.
2. **Total request envelope** (`linescout_api.limits`): `max_image_bytes +
   max_strokes_bytes + 64 KiB` of multipart overhead, counted on the ASGI
   receive channel *before* any parsing. `Content-Length` is only an
   early-rejection optimization — chunked bodies are counted as they arrive,
   and a multipart body is also limited to 16 parts. This envelope is
   deliberately larger than the 4 MiB image limit; the slack covers the
   remaining form fields.
3. **Image raster** (`preprocessing.decode_snapshot`): the PNG header is
   inspected for format and dimensions (16–4096 px edge, PIL's own
   decompression-bomb guard included) *before* any pixel decoding, so a
   hostile header cannot materialize a giant raster.

All early rejections use the standard structured error envelope
(`schema_version`, `request_id`, `retryable`, `error{code,message,field}`):
`413 request_too_large` / `413 too_many_parts` / `400 invalid_content_length`
from the middleware, `400 http_400` from the multipart parser (including its
1 MiB per-field cap), and `413 image_too_large` / `413 strokes_too_large` /
`400 strokes_malformed` / `422 image_dimensions` from the search router.

## Endpoints

| Method | Path | Milestone 1 status |
|---|---|---|
| `GET` | `/api/v1/health` | Complete |
| `POST` | `/api/v1/search` | Full multipart validation (400/413/422), insufficient rule, fixture ranking |
| `POST` | `/api/v1/events` | Complete |
| `GET`/`PUT` | `/api/v1/preferences` | Complete (Laplace + 30-day half-life) |
| `GET` | `/api/v1/assets/{id}/thumbnail` · `/line-art` | Complete; enabled + SFW assets only |
| `*` | `/api/v1/curation/*` | Gated by `CURATION_MODE`; progress works, rest 501 until M2 |
