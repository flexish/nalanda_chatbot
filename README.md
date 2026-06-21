# Ask Nalanda — Multimodal Agentic RAG

An AI-powered question-answering chatbot for the **Archaeological Site of Nalanda Mahavihara** (UNESCO World Heritage Site). Built on a multimodal Agentic RAG pipeline — it understands text, tables, and images from heritage documents, retrieves relevant context, and answers using a vision-capable LLM. When local documents lack information, it falls back to verified web sources.

---

## Features

| Capability | Details |
|---|---|
| **Multimodal RAG** | Parses PDFs for text, tables, and embedded images; retrieves all three |
| **Agentic pipeline** | 4-node LangGraph graph: retrieve → generate → verify → web search |
| **Domain restriction** | Only answers questions within the Nalanda Mahavihara domain |
| **Web search fallback** | DuckDuckGo text + image search when local docs are insufficient |
| **Image cross-check** | LLM verifies retrieved images match the query before showing them |
| **Self-RAG verification** | Scores answer grounding; re-queries web if answer is unsupported |
| **Query rewriting** | Multi-query + HyDE expansion + pronoun resolution from chat history |
| **Cross-encoder reranking** | `ms-marco-MiniLM-L-6-v2` reranks retrieved chunks before generation |
| **Chat history** | Follow-up questions resolved using conversation context |
| **Auth + roles** | JWT-like sessions, admin and regular user roles |
| **Admin panel** | Manage users, index PDFs, configure URLs from the UI |
| **Multiple LLM providers** | OpenAI · Anthropic · OpenRouter (switch via `.env`) |

---

## Why Docker?

Running this project locally requires installing:
- Python 3.10+, pip packages (`langchain`, `chromadb`, `sentence-transformers`, `unstructured`, etc.)
- **Tesseract OCR** (Windows-specific path setup needed for `hi_res` PDF parsing)
- System libraries for `unstructured` (`poppler`, `libmagic`, `lxml`, etc.)
- Proper virtual environment isolation

**Docker solves all of this** — it packages the exact Python version, all system dependencies, Tesseract, and the app into a single container image. On any machine (Windows, Linux, Mac, cloud VM), you just run `docker compose up` and the server is ready. No dependency conflicts, no OS-specific setup, and consistent behavior across dev/staging/production.

---

## Project Structure

```
MultiModel_Agentic_RAG/
├── web/
│   ├── api.py              # FastAPI server — REST + SSE streaming endpoints
│   ├── database.py         # SQLite user management (auth, roles)
│   └── static/
│       ├── index.html      # Single-page chat UI
│       ├── app.js          # Streaming chat, image display, admin panel
│       └── styles.css      # UI styling
│
├── utils/
│   ├── rag_graph.py        # Core pipeline: LangGraph, retrieval, generation, web search
│   ├── vectorstore.py      # Chroma + MultiVectorRetriever (text/image dual-store)
│   ├── ingest.py           # Unstructured PDF extraction (text, tables, images)
│   ├── summarizer.py       # LLM summarization of chunks for indexing
│   ├── config.py           # All env-var config in one place
│   ├── url_fetcher.py      # Admin-configured URL content fetcher
│   └── tesseract_setup.py  # Windows Tesseract PATH helper
│
├── index.py                # CLI: index a PDF or folder into the vector store
├── query.py                # CLI: one-shot question from terminal
├── app.py                  # (Legacy) Streamlit UI — use web/api.py instead
│
├── data/                   # Heritage PDFs (UNESCO docs, SOC reports, site images)
├── test_data/              # Small PDFs for quick testing
├── vectorstore/            # Chroma DB + docstore.pkl (auto-created on first index)
│
├── .env                    # Your secrets and config (never commit this)
├── .env.example            # Template for .env
├── Dockerfile              # Container image definition
├── docker-compose.yml      # Local development with Docker
├── railway.toml            # Railway deployment config (backend)
├── vercel.json             # Vercel deployment config (frontend only)
└── requirements.txt        # Python dependencies
```

---

## Deployment

### Architecture

This app has two parts that can be deployed independently:

```
Browser
  │
  ├── Static frontend (HTML/CSS/JS)
  │     → Vercel   (global CDN, free tier, instant deploys)
  │
  └── API calls → FastAPI backend (Python, Chroma DB, ML models)
                → Railway  (always-running server, Docker, persistent volumes)
```

**Why not Vercel for the backend?**
Vercel runs serverless functions with a 10-second timeout and no persistent storage. This project needs:
- Always-running process (Chroma DB lives in-memory + on disk)
- Heavy ML models (`sentence-transformers`, `unstructured`, Tesseract OCR)
- Long-running requests (RAG pipeline takes 5–20 seconds)
- Persistent disk for `vectorstore/` and `nalanda_users.db`

**Railway** solves all of this — it runs your Docker container 24/7 with mounted volumes.

---

### Option A — Railway only (simplest)

Deploy the full app (frontend + backend) on Railway. FastAPI serves the static files itself — no Vercel needed.

#### 1. Push to GitHub

```bash
git add .
git commit -m "initial deploy"
git push
```

#### 2. Create Railway project

1. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**
2. Select this repository
3. Railway auto-detects `Dockerfile` and `railway.toml`

#### 3. Add environment variables

In Railway dashboard → **Variables**, add:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_BACKEND=openai
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
ENABLE_WEB_SEARCH=true
ADMIN_USERNAME=admin
ADMIN_PASSWORD=your-strong-password
VECTORSTORE_PATH=/data/vectorstore
```

#### 4. Add a persistent volume

In Railway → **Volumes** → **New Volume**:
- Mount path: `/data`
- This persists `vectorstore/` and the SQLite database across deploys

#### 5. Index your documents

```bash
# Using Railway CLI
railway run python index.py --folder data
```

Or upload PDFs via the Admin panel in the web UI.

#### 6. Done

Railway gives you a public URL like `https://nalanda-rag.up.railway.app`.

---

### Option B — Vercel (frontend) + Railway (backend)

Use this when you want a faster global frontend via Vercel's CDN.

#### 1. Deploy backend on Railway

Follow all steps from Option A above. Note your Railway URL (e.g. `https://nalanda-rag.up.railway.app`).

#### 2. Set the backend URL in `index.html`

Edit `web/static/index.html` and set your Railway URL:

```html
<script>
  window.API_BASE = "https://nalanda-rag.up.railway.app";
</script>
```

Commit and push.

#### 3. Deploy frontend on Vercel

1. Go to [vercel.com](https://vercel.com) → **Add New Project** → import your GitHub repo
2. Vercel reads `vercel.json` automatically — no build settings needed
3. Deploy

Vercel serves `web/static/` as a static site. All `/api/...` calls go to Railway.

---

### Option C — Local development with Docker

```bash
cp .env.example .env
# Fill in your API key in .env
docker compose up --build
```

Server runs at **http://localhost:8000**. Index documents:

```bash
docker compose exec app python index.py --folder data
```

---

## Quick Start — Local (Without Docker)

### Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | 3.11 recommended |
| [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) | Windows: install to `C:\Program Files\Tesseract-OCR\` |
| OpenAI API key | Or Anthropic / OpenRouter |

### 1. Create virtual environment

```bash
cd MultiModel_Agentic_RAG
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / Mac
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 4. Index documents

```bash
python index.py --folder data
```

### 5. Start the server

```bash
python -m uvicorn web.api:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000**

---

## Configuration Reference

All settings live in `.env`. Copy `.env.example` to get started.

### LLM Provider

```env
# Choose one: openai | anthropic | openrouter
LLM_PROVIDER=openai
```

| Provider | Required keys | Best for |
|---|---|---|
| `openai` | `OPENAI_API_KEY` | Best quality, vision support built-in |
| `anthropic` | `ANTHROPIC_API_KEY` | Claude models |
| `openrouter` | `OPENROUTER_API_KEY` | Free/cheap models via OpenRouter |

### API Keys

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-opus-4-6

OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct:free
```

### Embeddings

```env
# openai (recommended) or huggingface (offline, no API key needed)
EMBEDDING_BACKEND=openai
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

### Storage

```env
VECTORSTORE_PATH=./vectorstore   # Where Chroma DB and docstore.pkl are saved
DATA_FOLDER=./data               # Default folder for index.py --folder
```

### Web Search

```env
ENABLE_WEB_SEARCH=true           # Enable DuckDuckGo text + image fallback
WEB_SEARCH_MAX_RESULTS=5         # Max web results per query
```

> Requires `ddgs` package (already in `requirements.txt`). No API key needed.

### Advanced RAG

```env
RETRIEVAL_TOP_K=4                # Chunks retrieved per query
ENABLE_RERANKING=true            # Cross-encoder reranking (recommended)
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
MAX_ITERATIVE_HOPS=1             # Self-RAG retry limit
```

### Admin Credentials

```env
ADMIN_USERNAME=admin
ADMIN_PASSWORD=nalanda@admin123   # Change this in production!
```

---

## How It Works

```
User question
      │
      ▼
Domain guard ─── not Nalanda-related? ──► "I can only answer about Nalanda"
      │
      ▼
Query rewriting (LLM)
  ├── QUERY1: pronoun-resolved semantic search query
  ├── QUERY2: alternative phrasing
  └── HYDE: hypothetical document passage
      │
      ▼
Retrieve (parallel)
  ├── Keyword text search
  ├── Semantic vector search (Chroma)
  └── Cross-encoder reranking
      │
      ├─ Image request? ──► image search (keyword + entity check)
      │                         │
      │                   no local match? ──► DuckDuckGo image search
      │                         │
      │                   cross-check: LLM verifies image matches query
      │
      ▼
CRAG: quality score < 0.3? ──► augment with DuckDuckGo web text
      │
      ▼
Generate (vision LLM)
  ├── receives text chunks + images in one prompt
  └── streams response to browser via SSE
      │
      ▼
Self-RAG verification
  ├── grounded? ──► show answer
  └── not grounded? ──► web search → regenerate
```

---

## Indexing Documents

### Index a folder

```bash
python index.py --folder data
```

### Index a single PDF

```bash
python index.py --pdf "data/Excavated Remains of Nalanda Mahavihara.pdf"
```

### What happens during indexing

1. **Extract** — Unstructured `hi_res` partitions the PDF into text chunks, tables, and images
2. **Summarize** — LLM generates a searchable summary for each chunk
3. **Embed** — Summaries are embedded and stored in Chroma
4. **Store** — Original content (full text, base64 images) saved in `docstore.pkl`

> Re-running on the same file appends duplicates. Delete `vectorstore/` for a clean rebuild.

### Test with small data first

```bash
python index.py --folder test_data   # 2 small PDFs, fast to index
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/login` | Login, returns session token |
| `POST` | `/api/logout` | Invalidate session |
| `POST` | `/api/chat/stream` | SSE streaming chat (recommended) |
| `POST` | `/api/chat` | Blocking chat (returns full response) |
| `GET` | `/api/store/info` | Vector store stats |
| `POST` | `/api/index` | Trigger indexing (admin) |
| `POST` | `/api/upload` | Upload and index a PDF (admin) |
| `GET/POST` | `/api/admin/users` | User management (admin) |
| `GET` | `/health` | Health check |

### Example streaming request

```bash
curl -X POST http://localhost:8000/api/chat/stream \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"question": "Who destroyed Nalanda University?", "top_k": 4, "history": []}'
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Tesseract not found` | Install Tesseract; Windows default path: `C:\Program Files\Tesseract-OCR\tesseract.exe` |
| `No answer / knowledge base empty` | Run `index.py` first; check `/api/store/info` for vector count |
| `429 rate limit` from LLM | Add billing credits or switch provider in `.env` |
| `ddgs` not found | Run `pip install ddgs` in the project venv |
| Slow indexing | Normal — `hi_res` runs OCR per page. Index one PDF at a time for testing |
| Wrong image shown | Server may need restart to pick up latest code changes |
| Out-of-domain questions | Chatbot intentionally refuses; ask something about Nalanda Mahavihara |
| Docker: port already in use | Change `8000:8000` to `8080:8000` in `docker-compose.yml` |

---

## Data

The `data/` folder includes UNESCO World Heritage documents for Nalanda Mahavihara:

- Nomination files and ICOMOS evaluation reports
- State of Conservation reports (2018, 2021)
- World Heritage Committee decisions
- Archaeological site plans and image PDFs

All documents are publicly available UNESCO heritage records.

---

## License

For research and educational use. Ensure compliance with:
- OpenAI / Anthropic / OpenRouter API terms of service
- UNESCO document usage terms
- Applicable copyright for any additional PDFs you add
