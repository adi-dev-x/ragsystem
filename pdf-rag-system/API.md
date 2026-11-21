# API Reference

Base URL: `http://localhost:8000`

Interactive docs (try every call in the browser): <http://localhost:8000/docs>
OpenAPI schema: <http://localhost:8000/openapi.json>

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | API and PostgreSQL status |
| GET | `/api/v1/documents` | List indexed PDF documents |
| POST | `/api/v1/rag/query` | Ask a question, get a grounded answer with citations |

All responses are JSON. Every response includes an `X-Request-ID` header. You can send your own `X-Request-ID` (letters, digits, `.`, `_`, `-`, up to 64 characters) to correlate logs.

Start the server from the `pdf-rag-system` folder with the virtual environment active:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## 1. `GET /health`

Reports whether the API is up and PostgreSQL (with pgvector and the schema) is reachable.

**Request:** no body, no parameters.

```bash
curl -s http://localhost:8000/health
```

**Response `200`**: database reachable:

```json
{
  "status": "ok",
  "api": "ok",
  "database": {
    "status": "ok",
    "latency_ms": 1.19,
    "pgvector_version": "0.8.7",
    "schema_ready": true
  },
  "llm_provider": "ollama",
  "embedding_model": "nomic-embed-text"
}
```

**Response `503`**: database unreachable:

```json
{
  "status": "degraded",
  "api": "ok",
  "database": {
    "status": "error",
    "schema_ready": false,
    "error": "Timed out connecting to PostgreSQL"
  },
  "llm_provider": "ollama",
  "embedding_model": "nomic-embed-text"
}
```

| Field | Meaning |
|---|---|
| `status` | `ok`, or `degraded` if the database is down or `schema_ready` is false |
| `database.schema_ready` | `false` means migrations haven't run: `python -m scripts.migrate` |
| `database.pgvector_version` | installed pgvector extension version |
| `llm_provider` / `embedding_model` | what the server is configured to use |

---

## 2. `GET /api/v1/documents`

Lists the PDFs that have been ingested.

**Query parameters**

| Parameter | Type | Default | Rules |
|---|---|---|---|
| `limit` | integer | 100 | 1 to 1000 |
| `offset` | integer | 0 | 0 or more |

```bash
curl -s "http://localhost:8000/api/v1/documents"
```

```bash
curl -s "http://localhost:8000/api/v1/documents?limit=20&offset=0"
```

**Response `200`**

```json
{
  "success": true,
  "total": 1,
  "limit": 20,
  "offset": 0,
  "documents": [
    {
      "id": "ccc298ee-0c11-46b0-9019-4bf764066591",
      "filename": "BambooLoops_AI_Consolidated_Project_and_Visual_Proposal_SoLVE.pdf",
      "file_hash": "96176b74a43d5f217dbe6cfc92f9ccc93e5f7a9cb51fcefd663583659f207566",
      "file_size": 2605,
      "total_pages": 16,
      "total_chunks": 10,
      "embedding_model": "nomic-embed-text",
      "metadata": {
        "title": "BambooLoops AI",
        "empty_pages": []
      },
      "created_at": "2026-11-15T09:21:50.869703+05:30",
      "updated_at": "2026-11-15T09:21:50.869703+05:30"
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `total` | number of indexed documents (not just this page) |
| `id` | document UUID, usable in the query `filters.document_ids` |
| `filename` | path relative to `dataloads/` |
| `file_hash` | SHA-256 of the PDF; a changed file gets a new hash and is re-indexed |
| `total_chunks` | `0` means the PDF had no extractable text |
| `metadata.empty_pages` | page numbers with no extractable text |

**Errors:** `422` for an invalid `limit` or `offset`, `503` if the database is unreachable.

---

## 3. `POST /api/v1/rag/query`

Embeds the question, finds the most similar chunks in PostgreSQL, and asks the configured LLM (`LLM_PROVIDER`) for an answer based only on those chunks.

**Headers:** `Content-Type: application/json`

### Request body

| Field | Type | Required | Rules |
|---|---|---|---|
| `question` | string | **yes** | 1 to 4000 characters; surrounding whitespace is trimmed |
| `top_k` | integer | no | 1 to 50; default is `TOP_K` (5) |
| `similarity_threshold` | number | no | 0 to 1; minimum cosine similarity; default is `SIMILARITY_THRESHOLD` |
| `filters` | object | no | restricts which documents are searched (below) |
| `filters.filenames` | string[] | no | exact filenames, up to 100 |
| `filters.document_ids` | UUID[] | no | document IDs from `/api/v1/documents`, up to 100 |
| `filters.page_from` | integer | no | 1 or more; matches chunks that end on or after this page |
| `filters.page_to` | integer | no | 1 or more; matches chunks that start on or before this page; must be at least `page_from` |

Unknown fields are rejected with `422`. All filters given must match.

**Minimal body**

```json
{
  "question": "What does the BambooLoops AI assistant do?"
}
```

```bash
curl -s -X POST http://localhost:8000/api/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does the BambooLoops AI assistant do?"}'
```

**With options**

```json
{
  "question": "What does the BambooLoops AI assistant do?",
  "top_k": 5,
  "similarity_threshold": 0.3
}
```

**Filter by filename and pages**

```json
{
  "question": "What does the BambooLoops AI assistant do?",
  "top_k": 3,
  "filters": {
    "filenames": ["BambooLoops_AI_Consolidated_Project_and_Visual_Proposal_SoLVE.pdf"],
    "page_from": 1,
    "page_to": 10
  }
}
```

**Filter by document ID**

```json
{
  "question": "Summarize the proposal",
  "filters": {
    "document_ids": ["ccc298ee-0c11-46b0-9019-4bf764066591"]
  }
}
```

```bash
curl -s -X POST http://localhost:8000/api/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does the BambooLoops AI assistant do?", "top_k": 3, "filters": {"filenames": ["BambooLoops_AI_Consolidated_Project_and_Visual_Proposal_SoLVE.pdf"], "page_from": 1, "page_to": 10}}'
```

### Response `200`

The shape is the same for every provider (`claude`, `openai`, `ollama`). The `answer` text below is illustrative; it depends on your documents and model.

```json
{
  "success": true,
  "question": "What does the BambooLoops AI assistant do?",
  "answer": "The BambooLoops AI assistant is a multilingual, voice-enabled help layer that works across the whole platform rather than as a separate module [1].\n\nWhat it does:\n- Helps users discover services, communicate in Indian languages, complete listings, understand processes and access relevant knowledge [2].\n- Accepts typed or spoken Malayalam and English at launch, with more Indian languages added in phases [1].\n\nTrust controls:\n- Visible transcripts, confirmation before consequential actions, answers grounded in approved knowledge, source references, confidence checks and escalation to a human expert [1].",
  "sources": [
    {
      "id": 1,
      "document": "BambooLoops_AI_Consolidated_Project_and_Visual_Proposal_SoLVE.pdf",
      "document_id": "ccc298ee-0c11-46b0-9019-4bf764066591",
      "page": 10,
      "page_end": 16,
      "chunk_id": "c2301bcf-3b4e-4c75-847a-b02001d821e0",
      "chunk_index": 8,
      "relevance_score": 0.4248,
      "content": "BambooLoops AI | SoLVE | © 2026 SoLVE • Page 10\n\n1. The application..."
    }
  ],
  "model": "llama3.1:8b",
  "provider": "ollama",
  "usage": {
    "input_tokens": 1198,
    "output_tokens": 15
  }
}
```

| Field | Meaning |
|---|---|
| `answer` | generated from the retrieved sources only. Plain text by default (no Markdown); line breaks arrive as `\n` in the JSON. Citations like `[1]` refer to `sources[].id`. |
| `sources[].id` | citation number used in the answer: `[1]` is the source with `"id": 1` |
| `sources[].document` | PDF filename |
| `sources[].page` / `page_end` | first and last page of the chunk; equal for single-page chunks |
| `sources[].chunk_id` | chunk UUID |
| `sources[].relevance_score` | cosine similarity, 0 to 1 (higher is closer) |
| `sources[].content` | the text sent to the model |
| `model` | model ID reported by the provider |
| `provider` | `claude`, `openai` or `ollama` |
| `usage` | token counts reported by the provider |

### Answer format

Two settings in `.env` control how `answer` is written (restart the server after changing them):

| Setting | Values | Effect |
|---|---|---|
| `ANSWER_FORMAT` | `plain` (default) | Plain sentences and `- ` lists, no Markdown symbols. Safe to show to users as is. Any Markdown the model adds anyway is removed. |
| | `markdown` | Headings, bold and bullets, for apps that render Markdown. |
| | `spoken` | Natural conversational speech for a text-to-speech engine: direct answer, connected sentences, no citations, headings, lists, Markdown, emojis or symbols. `CITATION_STYLE` is ignored. See [Spoken answers](#spoken-answers-for-text-to-speech). |
| `CITATION_STYLE` | `numbered` (default) | Short `[1]`, `[2]` markers matching `sources[].id`. Numbers that don't match a source are removed. |
| | `document` | Full citations in the text, e.g. `[file.pdf, pp. 10-16]`. |

To show the answer with its sources, display `answer`, then list `sources` by `id`, for example `[1] BambooLoops_....pdf, pages 10–16`.

### Spoken answers (for text-to-speech)

With `ANSWER_FORMAT=spoken`, `answer` is written as a person would say it and can be passed straight to a TTS engine. `sources` is still returned for logging or display; it's just not mentioned in the spoken text.

```json
{
  "success": true,
  "question": "What does the BambooLoops AI assistant do?",
  "answer": "The BambooLoops AI assistant is designed to make the whole platform easier to use through voice and several languages. You can type or simply speak to it in Malayalam or English, and it will help you discover services, complete listings, understand different processes and find the information you need.\n\nThere are also safety measures built in. For example, you can see a transcript of the conversation, and the assistant asks for your confirmation before doing anything important. Its answers come from approved knowledge, and when it isn't sure, it can pass you to a human expert.",
  "sources": [ { "id": 1, "document": "BambooLoops_AI_Consolidated_Project_and_Visual_Proposal_SoLVE.pdf", "page": 10, "page_end": 16, "...": "..." } ],
  "model": "claude-sonnet-5-5",
  "provider": "claude",
  "usage": { "input_tokens": 6629, "output_tokens": 160 }
}
```

The answer text is illustrative. When nothing relevant is found, the spoken reply is `"I'm sorry, I don't have information about that right now."`

### When nothing relevant is found

If no chunk reaches the similarity threshold, the model is not called. The response is `200`:

```json
{
  "success": true,
  "question": "What is the capital of Peru?",
  "answer": "The requested information was not found in the provided documents.",
  "sources": [],
  "model": "llama3.1:8b",
  "provider": "ollama",
  "usage": {
    "input_tokens": 0,
    "output_tokens": 0
  }
}
```

If a question you expect to work returns this, lower the threshold (for example `"similarity_threshold": 0.1`) to check, then adjust `SIMILARITY_THRESHOLD` in `.env`.

---

## Errors

Every error uses the same shape:

```json
{
  "success": false,
  "error": {
    "code": "validation_error",
    "message": "Request validation failed.",
    "details": [
      { "field": "top_k", "message": "Input should be less than or equal to 50", "type": "less_than_equal" }
    ]
  },
  "request_id": "cd7528cc15d942ef8ab513ca05294939"
}
```

`details` is present only when there is extra information. `request_id` matches the `X-Request-ID` response header and the server logs.

| HTTP | `error.code` | When |
|---|---|---|
| 404 | `not_found` | unknown URL |
| 405 | `method_not_allowed` | wrong HTTP method for the URL |
| 422 | `validation_error` | invalid body or parameters; `details` lists each field |
| 422 | `llm_refusal` | the model declined to answer |
| 502 | `embedding_error` | the embedding provider (OpenAI or Ollama) failed after retries, or Ollama isn't running or the model isn't pulled |
| 502 | `llm_error` | the answer provider (Claude, OpenAI or Ollama) failed after retries, or Ollama isn't running or the model isn't pulled |
| 503 | `database_unavailable` | PostgreSQL unreachable or no free connection |
| 500 | `configuration_error`, `database_error`, `internal_error` | server-side problem; details are only in the server logs |

---

## Test cases

| Test | Request | Expected |
|---|---|---|
| Health | `GET /health` | 200, `status: ok` |
| Documents indexed | `GET /api/v1/documents` | 200, `total` above 0 |
| Normal question | `POST /api/v1/rag/query` `{"question": "What does the assistant do?"}` | 200, non-empty `sources` |
| Question not in the documents | `{"question": "What is the capital of Peru?"}` | 200, "not found" answer, `sources: []` |
| Empty question | `{"question": ""}` | 422 `validation_error` |
| Missing question | `{}` | 422 `validation_error` |
| `top_k` too large | `{"question": "hi", "top_k": 99}` | 422 `validation_error` |
| Bad threshold | `{"question": "hi", "similarity_threshold": 1.5}` | 422 `validation_error` |
| Unknown field | `{"question": "hi", "foo": 1}` | 422 `validation_error` |
| Pages reversed | `{"question": "hi", "filters": {"page_from": 5, "page_to": 2}}` | 422 `validation_error` |
| Unknown file in a filter | `{"question": "hi", "filters": {"filenames": ["nope.pdf"]}}` | 200, "not found" answer |
| Wrong method | `GET /api/v1/rag/query` | 405 `method_not_allowed` |
| Unknown URL | `GET /api/v1/nope` | 404 `not_found` |
| Ollama stopped | any valid question | 502 `embedding_error` |
