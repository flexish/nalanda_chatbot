# Multimodal RAG — Project Wiki

Detailed documentation for the Nalanda multimodal RAG application. For a short overview and quick start, see [README.md](README.md).

---

## Table of contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [The MultiVector RAG pattern](#the-multivector-rag-pattern)
4. [Indexing pipeline](#indexing-pipeline)
5. [Query pipeline (LangGraph)](#query-pipeline-langgraph)
6. [Module reference](#module-reference)
7. [Configuration](#configuration)
8. [Indexing multiple PDFs](#indexing-multiple-pdfs)
9. [Streamlit application](#streamlit-application)
10. [CLI reference](#cli-reference)
11. [Relationship to the notebook](#relationship-to-the-notebook)
12. [Tesseract OCR (Windows)](#tesseract-ocr-windows)
13. [Troubleshooting](#troubleshooting)
14. [Future improvements](#future-improvements)

---

## Overview

### What this project does

This system lets you ask natural-language questions about a corpus of PDF documents that contain **text**, **tables**, and **images** — typical of UNESCO heritage reports, conservation documents, and archaeological guides.

Unlike text-only RAG:

1. **Images are extracted** from PDFs (maps, site photos, diagrams).
2. **Images are summarized** by a vision model so they can be found via semantic search.
3. **At answer time**, the original images (not just summaries) are sent to the LLM alongside retrieved text, so the model can reason over visual content.

The domain focus is the **Archaeological Site of Nalanda Mahavihara**, but the pipeline works for any PDF collection if you adjust prompts and data paths.

### Technology stack

| Layer | Technology |
|-------|------------|
| PDF extraction | [Unstructured](https://docs.unstructured.io/) `partition_pdf`, `strategy="hi_res"` |
| OCR | Tesseract (via `unstructured_pytesseract`) |
| Text/table summaries | Groq `llama-3.1-8b-instant` (or OpenAI fallback) |
| Image summaries | OpenAI `gpt-4o-mini` (vision) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector database | [Chroma](https://www.trychroma.com/) |
| Retrieval pattern | LangChain `MultiVectorRetriever` |
| Query orchestration | [LangGraph](https://langchain-ai.github.io/langgraph/) `StateGraph` |
| Answer generation | OpenAI `gpt-4o-mini` (multimodal) |
| Web UI | [Streamlit](https://streamlit.io/) |

---

## Architecture

### High-level data flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INDEXING (offline)                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   PDF file(s)                                                           │
│       │                                                                 │
│       ▼                                                                 │
│   unstructured.partition_pdf (hi_res)                                   │
│       │                                                                 │
│       ├── CompositeElement  ──► text chunks                             │
│       ├── Table             ──► table chunks (incl. HTML)               │
│       └── Image (in CompositeElement) ──► base64 JPEG payloads        │
│       │                                                                 │
│       ▼                                                                 │
│   Summarization                                                         │
│       ├── text  ──► Groq / OpenAI                                       │
│       ├── table ──► Groq / OpenAI (from text_as_html when available)    │
│       └── image ──► OpenAI vision                                       │
│       │                                                                 │
│       ├──────────────────────────┬──────────────────────────────────  │
│       ▼                          ▼                                      │
│   Chroma (embed summaries)   docstore.pkl (store originals)             │
│   + doc_id metadata          ParentDocument | base64 string             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         QUERY (online)                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   User question                                                         │
│       │                                                                 │
│       ▼                                                                 │
│   LangGraph: retrieve node                                              │
│       │  MultiVectorRetriever.invoke(question)                          │
│       │  → similarity search on summaries in Chroma                     │
│       │  → lookup doc_id in docstore → parent documents                 │
│       ▼                                                                 │
│   parse_docs: split parents into texts vs base64 images                 │
│       │                                                                 │
│       ▼                                                                 │
│   LangGraph: generate node                                              │
│       │  Build HumanMessage with text context + image_url blocks        │
│       │  → ChatOpenAI (gpt-4o-mini)                                     │
│       ▼                                                                 │
│   Answer + optional source context (texts, images)                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### LangGraph state machine

```
    START
      │
      ▼
  ┌──────────┐
  │ retrieve │  retriever.invoke(question) → parse_docs()
  └──────────┘
      │
      ▼
  ┌──────────┐
  │ generate │  build_multimodal_messages() → ChatOpenAI
  └──────────┘
      │
      ▼
     END
```

**State schema** (`RAGState` in `utils/rag_graph.py`):

| Field | Type | Description |
|-------|------|-------------|
| `question` | `str` | User query |
| `retrieved_docs` | `list` | Raw parent documents from docstore |
| `context` | `dict` | `{"texts": [...], "images": [...]}` after parsing |
| `response` | `str` | Final LLM answer |

---

## The MultiVector RAG pattern

Standard RAG embeds document chunks directly. **MultiVector RAG** (parent–child / summary indexing) uses two stores:

| Store | Contents | Used for |
|-------|----------|----------|
| **Vector store (Chroma)** | Short *summaries* of each chunk | Semantic similarity search |
| **Docstore (`docstore.pkl`)** | Full *parent* content | What the LLM actually reads |

Each summary document in Chroma carries a `doc_id` UUID. The docstore maps that ID to either:

- A `ParentDocument` (text or table), or
- A raw base64 image string.

### Why summaries?

- **Images** cannot be embedded directly with text embedding models. A vision-generated *description* becomes searchable text.
- **Long text chunks** benefit from concise summaries that match query phrasing more closely.
- **Tables** are summarized from HTML representation for richer semantic capture.

### Retrieval flow

1. User asks: *"What is Sarai Mound?"*
2. Query is embedded and compared to summary vectors.
3. Top-k matching summaries return their `doc_id` values.
4. Docstore returns the **original** text chunks and/or image bytes.
5. The answer LLM receives full context — not the summaries.

This matches the LangChain tutorial pattern implemented in `langchain_multimodal.ipynb`.

---

## Indexing pipeline

### Step 1 — PDF partitioning

**Module:** `utils/ingest.py` → `partition_pdf_file()`

```python
partition_pdf(
    filename=...,
    infer_table_structure=True,
    strategy="hi_res",
    extract_image_block_types=["Image"],
    extract_image_block_to_payload=True,
    chunking_strategy="by_title",
    max_characters=10000,
    combine_text_under_n_chars=2000,
    new_after_n_chars=6000,
)
```

| Parameter | Purpose |
|-----------|---------|
| `strategy="hi_res"` | Layout-aware parsing; required for tables and image blocks |
| `infer_table_structure=True` | Extract structured tables |
| `extract_image_block_to_payload=True` | Store images as base64 in element metadata |
| `chunking_strategy="by_title"` | Group content under document headings |

**Output element types:**

- `CompositeElement` — grouped text (titles, narrative, lists)
- `Table` — tabular data
- `Image` — nested inside `CompositeElement.metadata.orig_elements`

> **Note:** Not every `CompositeElement` contains images. Image elements appear only in chunks where Unstructured detected figures.

### Step 2 — Element extraction

**Module:** `utils/ingest.py` → `extract_from_chunks()`

Produces an `IngestedPDF` dataclass:

```python
@dataclass
class IngestedPDF:
    source: str              # PDF file path
    texts: List[ParentDocument]
    tables: List[ParentDocument]
    images: List[str]        # base64 strings
```

`ParentDocument` stores:

- `text` — chunk body
- `metadata` — page number, source path, `kind`, etc.
- `kind` — `"text"` | `"table"` | `"image"`

### Step 3 — Summarization

**Module:** `utils/summarizer.py`

| Modality | Model | Input |
|----------|-------|-------|
| Text | Groq `llama-3.1-8b-instant` (if `GROQ_API_KEY` set) else OpenAI | `ParentDocument.text` |
| Table | Same as text | `metadata.text_as_html` or `.text` |
| Image | OpenAI `gpt-4o-mini` vision | `data:image/jpeg;base64,{image}` |

Concurrency is controlled by `SUMMARIZE_CONCURRENCY` (default `3` for text; images use `2`).

### Step 4 — Vector indexing

**Module:** `utils/vectorstore.py` → `MultimodalVectorStore.add_ingested()`

For each modality:

1. Generate UUID `doc_id` per parent.
2. Add `Document(page_content=summary, metadata={doc_id: uuid})` to Chroma.
3. Add `(uuid, parent)` pairs to docstore.
4. Call `docstore.save()` → writes `docstore.pkl`.

**On disk after indexing:**

```
vectorstore/
├── chroma.sqlite3          # Chroma persistence
├── docstore.pkl            # Parent documents keyed by doc_id
└── ...                     # Chroma segment files
```

---

## Query pipeline (LangGraph)

**Module:** `utils/rag_graph.py`

### Entry point

```python
from utils.vectorstore import MultimodalVectorStore
from utils.rag_graph import query_with_sources

store = MultimodalVectorStore.load()
result = query_with_sources(store, "What is Sarai Mound?", top_k=4)
```

### Return shape

```python
{
    "question": "What is Sarai Mound?",
    "response": "The Sarai Mound is mentioned in the context of...",
    "context": {
        "texts": [ParentDocument, ...],   # retrieved text/table parents
        "images": ["base64...", ...]      # retrieved image parents
    }
}
```

This mirrors the notebook's `chain_with_sources.invoke()` output.

### Multimodal prompt construction

1. Concatenate all retrieved `text_element.text` values into `context_text`.
2. Build a template:

   ```
   Answer the question based only on the following context...
   Context: {context_text}
   Question: {question}
   ```

3. Append each image as:

   ```json
   {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
   ```

4. Send as a single `HumanMessage` to `ChatOpenAI`.

### Image vs text detection (`parse_docs`)

For each retrieved parent from the docstore:

- If it is a `str` that decodes as valid base64 → **image**
- Otherwise → **text** (including `ParentDocument` objects)

---

## Module reference

### `utils/config.py`

Loads `.env` via `python-dotenv`. Centralizes paths, API keys, prompts, and retrieval defaults.

### `utils/tesseract_setup.py`

Configures Tesseract on Windows:

- Adds `C:\Program Files\Tesseract-OCR` to `PATH`
- Sets `unstructured_pytesseract.pytesseract.tesseract_cmd`

Called automatically before `partition_pdf`.

### `utils/ingest.py`

| Function | Description |
|----------|-------------|
| `partition_pdf_file(path)` | Run Unstructured on one PDF |
| `extract_from_chunks(chunks, source)` | Split into texts/tables/images |
| `ingest_pdf(path)` | Full single-PDF ingest |
| `iter_pdfs(folder)` | Yield unique PDF paths in folder |
| `ingest_folder(folder)` | Process all PDFs with progress logs |

### `utils/summarizer.py`

| Function | Description |
|----------|-------------|
| `summarize_texts(docs)` | Batch text summaries |
| `summarize_tables(docs)` | Batch table summaries |
| `summarize_images(b64_list)` | Batch vision summaries |

### `utils/vectorstore.py`

| Class / method | Description |
|----------------|-------------|
| `PersistentDocstore` | Pickle-backed `InMemoryStore` |
| `MultimodalVectorStore` | Chroma + MultiVectorRetriever wrapper |
| `.add_ingested(ingested)` | Summarize + index one PDF |
| `.index_folder(folder)` | Index all PDFs in directory |
| `.stats()` | Vector count, docstore size |
| `.load(path)` | Load existing store |

### `utils/rag_graph.py`

| Function | Description |
|----------|-------------|
| `parse_docs(docs)` | Split retrieved parents |
| `build_multimodal_messages(context, question)` | Construct vision prompt |
| `create_rag_graph(store, top_k)` | Compile LangGraph |
| `query_with_sources(store, question)` | End-to-end query |

### `index.py`

CLI entry point for building the index.

### `query.py`

CLI entry point for questions.

### `app.py`

Streamlit UI: sidebar indexing, chat interface, source expander with text + images.

### `langchain_multimodal.ipynb`

Original exploratory notebook. **Not modified by the application.** Use it to understand the RAG pattern step-by-step.

---

## Configuration

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

### Required

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Embeddings (`text-embedding-3-small`), image summaries, final answers |

### Recommended

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | (empty) | Enables fast Groq summarization for text/tables |
| `OPENAI_MODEL` | `gpt-4o-mini` | Vision + answer model |
| `VECTORSTORE_PATH` | `./vectorstore` | Where Chroma and docstore are saved |
| `DATA_FOLDER` | `./data` | Default folder for `index.py` |

### Optional tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq summarization model |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Chroma embedding model |
| `CHROMA_COLLECTION` | `multi_modal_rag` | Chroma collection name |
| `RETRIEVAL_TOP_K` | `4` | Documents retrieved per query |
| `SUMMARIZE_CONCURRENCY` | `3` | Parallel summary batch size |
| `IMAGE_SUMMARY_PROMPT` | Nalanda-specific default | Override vision summary instructions |

### Streamlit

`.streamlit/config.toml` sets `fileWatcherType = "none"` to avoid Windows DLL issues with torch/transformers file watching.

---

## Indexing multiple PDFs

### Folder indexing

Place 2, 3, or more PDFs in any folder:

```
test_data/
├── doc_a.pdf
├── doc_b.pdf
└── doc_c.pdf
```

```bash
python index.py --folder test_data
```

The indexer:

1. Discovers all `*.pdf` files (including subfolders, deduplicated)
2. Processes each file sequentially
3. Appends all summaries and parents to the **same** Chroma collection

### Per-file indexing

```bash
python index.py --pdf "test_data/doc_a.pdf"
python index.py --pdf "test_data/doc_b.pdf"
```

Equivalent to folder indexing but gives explicit control over order and timing.

### Cross-document retrieval

A question like *"What is Sarai Mound?"* searches **all** indexed summaries. Retrieved chunks include `metadata.source` so you can identify which PDF supplied each passage.

### Re-indexing and duplicates

| Action | Result |
|--------|--------|
| Index same PDF twice | Duplicate vectors and docstore entries |
| Add new PDF to existing index | New content appended; old content kept |
| Clean rebuild | Delete `vectorstore/` directory, then re-run `index.py` |

There is no built-in deduplication yet. For production, add a `--reset` flag or track indexed file hashes.

### Performance expectations

| PDFs | Rough time (hi_res) | API calls |
|------|---------------------|-----------|
| 1 | 2–5 minutes | Summaries per text/table/image chunk |
| 3 | 6–15 minutes | Scales with chunk count |
| 40+ | Hours | Index in batches; use Groq for text |

---

## Streamlit application

### Launch

```bash
streamlit run app.py
```

### Sidebar

| Control | Function |
|---------|----------|
| Vector store path | Point to `vectorstore` or custom path |
| Data folder | PDF source directory (`data`, `test_data`, etc.) |
| Retrieval top-k | Slider 1–10 |
| Index scope | Single PDF or entire folder |
| Build / update index | Runs ingest + summarize + Chroma write |
| Metrics | Summary vector count, docstore entry count |

### Main panel

- Chat input for questions
- Assistant response (markdown)
- **Retrieved context** expander: text chunks with page numbers, image thumbnails

### Caching

`@st.cache_resource` on `get_store()` avoids reloading Chroma on every message. Cache is cleared after indexing via `get_store.clear()`.

---

## CLI reference

### `index.py`

```bash
# Index default DATA_FOLDER from .env
python index.py

# Index a specific folder
python index.py --folder test_data

# Index one PDF
python index.py --pdf "test_data/my_doc.pdf"

# Custom vectorstore location
python index.py --folder data --vectorstore ./my_index
```

### `query.py`

```bash
# Positional question
python query.py "What conservation work was done on Monastery 10?"

# Interactive prompt
python query.py

# Options
python query.py "..." --vectorstore ./vectorstore --top-k 6
```

---

## Relationship to the notebook

`langchain_multimodal.ipynb` is the **reference implementation**. The application reproduces its core ideas:

| Notebook step | Application equivalent |
|---------------|------------------------|
| Tesseract setup cell | `utils/tesseract_setup.py` |
| `partition_pdf(...)` | `utils/ingest.py` |
| Split texts / tables / images | `extract_from_chunks()` |
| Groq `summarize_chain` | `utils/summarizer.py` |
| OpenAI image summary chain | `summarize_images()` |
| `Chroma` + `InMemoryStore` + `MultiVectorRetriever` | `utils/vectorstore.py` (persistent docstore) |
| `parse_docs` / `build_prompt` | `utils/rag_graph.py` |
| `chain_with_sources.invoke()` | `query_with_sources()` |
| Manual cell execution | `index.py`, `app.py` sidebar |

**Differences:**

- Notebook uses **LCEL** chains; app uses **LangGraph** `StateGraph`.
- Notebook docstore is **in-memory**; app persists to **`docstore.pkl`**.
- Notebook indexes one PDF per session; app supports **folders** and **CLI**.

The notebook file is intentionally left unchanged so you can compare interactive exploration with the production pipeline.

---

## Tesseract OCR (Windows)

Unstructured `hi_res` requires Tesseract for OCR on scanned or complex layouts.

### Install

1. Download from [UB Mannheim Tesseract builds](https://github.com/UB-Mannheim/tesseract/wiki)
2. Install to default path: `C:\Program Files\Tesseract-OCR\`
3. Ensure `tesseract.exe` exists

### Verify

```bash
tesseract --version
```

### Application behavior

`configure_tesseract()` runs before every `partition_pdf` call. If Tesseract is missing, ingestion raises:

```
RuntimeError: Tesseract not found...
```

### Linux / macOS

Install via package manager and ensure `tesseract` is on `PATH`:

```bash
# Ubuntu
sudo apt-get install tesseract-ocr poppler-utils

# macOS
brew install tesseract poppler
```

---

## Troubleshooting

### `TesseractNotFoundError` / `Tesseract not found`

- Install Tesseract (see above)
- Restart terminal / Jupyter kernel after install
- On Windows, default path is auto-detected

### `IndexError: list index out of range` on `chunk_images[0]` (notebook)

`chunks[3]` may contain only text (Title, NarrativeText) with no `Image` elements. Images often appear in later chunks (e.g. chunk 6+). Scan all chunks for the first image instead of hardcoding index `3`.

### Vector store is empty

```
Vector store is empty. Run: python index.py
```

Run indexing before querying. Check Streamlit sidebar metrics (`Summary vectors` should be > 0).

### Slow indexing

- `hi_res` is computationally expensive — expected
- Index one PDF while developing
- Set `GROQ_API_KEY` for faster text/table summarization
- Large folders (40+ PDFs): index overnight or in batches

### Duplicate results after re-indexing

Re-running `index.py` on the same PDFs **appends** without removing old entries. Delete the `vectorstore/` folder for a clean slate.

### OpenAI rate limits / costs

Each PDF generates many summary API calls (one per text chunk, table, and image). Monitor usage in the OpenAI dashboard. Reduce corpus size or batch indexing during development.

### `ModuleNotFoundError: unstructured_inference`

```bash
pip install unstructured-inference
```

### Streamlit / torch DLL errors on Windows

`.streamlit/config.toml` disables file watching. If issues persist, use `EMBEDDING_BACKEND=openai` (already default) to avoid loading local embedding models.

### Images not retrieved for a question

- Image may not be semantically related to the query summary
- Increase `RETRIEVAL_TOP_K`
- Verify images were indexed (`images` count in index output)
- Image summaries must be descriptive enough for embedding match

---

## Future improvements

Potential extensions not yet implemented:

- [ ] `--reset` flag to wipe vectorstore before indexing
- [ ] File-hash deduplication (skip already-indexed PDFs)
- [ ] DuckDuckGo web search node in LangGraph (`ENABLE_WEB_SEARCH` in `.env`)
- [ ] Anthropic Claude as answer provider
- [ ] Conversation memory / multi-turn chat in Streamlit
- [ ] Export retrieved sources as downloadable report
- [ ] URL ingestion from `data/url.txt`
- [ ] Docker compose for reproducible deployment

---

## Glossary

| Term | Meaning |
|------|---------|
| **RAG** | Retrieval-Augmented Generation — fetch relevant context before LLM answer |
| **MultiVector RAG** | Embed summaries, retrieve full parent documents |
| **Parent document** | Original chunk (text, table, or image) stored in docstore |
| **Child / summary** | Short LLM-generated description embedded in Chroma |
| **CompositeElement** | Unstructured chunk grouping related text elements |
| **hi_res** | Unstructured strategy using layout detection + OCR |
| **top-k** | Number of nearest summary vectors to retrieve |

---

*Last updated to match the application structure in `multimodal_rag/` (LangGraph + Streamlit + persistent MultiVector RAG).*
