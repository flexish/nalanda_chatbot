# Multimodal RAG — Nalanda Heritage Documents

A production-style **multimodal Retrieval-Augmented Generation (RAG)** system for PDF collections about the **Archaeological Site of Nalanda Mahavihara**. It extracts text, tables, and images from PDFs, indexes them with a **MultiVector** pattern, and answers questions using a **vision-capable LLM**.

Built from the exploratory workflow in `langchain_multimodal.ipynb`, extended with **LangGraph** orchestration, **persistent storage**, and a **Streamlit** UI.

> **Full documentation:** see [WIKI.md](WIKI.md) for architecture, configuration, troubleshooting, and module reference.

---

## Features

- **Rich PDF parsing** via [Unstructured](https://unstructured.io/) `hi_res` strategy (text, tables, embedded images)
- **MultiVector RAG** — embed *summaries*, retrieve *original* content (text + base64 images)
- **Multimodal answers** — GPT-4o-mini receives retrieved text and images in one prompt
- **LangGraph pipeline** — `retrieve` → `generate` with explicit state
- **Persistent index** — Chroma vectors + `docstore.pkl` survive restarts
- **Multiple PDFs** — index a folder (`data/`, `test_data/`, etc.) into one shared knowledge base
- **CLI + Streamlit** — index and query from terminal or browser

---

## Quick start

### 1. Prerequisites

| Requirement | Notes |
|-------------|--------|
| Python 3.10+ | Conda env `tf` works well |
| [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) | Required for `hi_res` PDF parsing on Windows |
| OpenAI API key | Embeddings, image summaries, and final answers |
| Groq API key (optional) | Faster text/table summaries (`llama-3.1-8b-instant`) |

### 2. Install

```bash
cd multimodal_rag
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set OPENAI_API_KEY (and optionally GROQ_API_KEY)
```

### 3. Add PDFs

Place files in `data/` or `test_data/`:

```
test_data/
├── 7B - India - Nalanda 20171130 public.pdf
├── another_doc.pdf
└── third_doc.pdf
```

### 4. Index

**One PDF (recommended for first run):**

```bash
python index.py --pdf "test_data/7B - India - Nalanda 20171130 public.pdf"
```

**Entire folder (2–3 PDFs or more):**

```bash
python index.py --folder test_data
```

Indexing runs Unstructured `hi_res` per file and calls LLMs to summarize content. Expect **several minutes per PDF**.

### 5. Query

**CLI:**

```bash
python query.py "What is Sarai Mound?"
```

**Streamlit UI:**

```bash
streamlit run app.py
```

Use the sidebar to index PDFs, adjust `top-k`, then chat in the main panel.

---

## Project layout

```
multimodal_rag/
├── app.py                      # Streamlit web UI
├── index.py                    # CLI: build / extend the index
├── query.py                    # CLI: ask questions
├── langchain_multimodal.ipynb  # Original notebook (reference only)
├── requirements.txt
├── .env.example
├── WIKI.md                     # Detailed project wiki
├── data/                       # Default PDF folder
├── test_data/                  # Example / test PDFs
├── vectorstore/                # Chroma DB + docstore.pkl (created on index)
└── utils/
    ├── config.py               # Environment configuration
    ├── ingest.py               # Unstructured PDF extraction
    ├── summarizer.py           # Text/table/image summaries
    ├── vectorstore.py          # Chroma + MultiVectorRetriever
    ├── rag_graph.py            # LangGraph query pipeline
    └── tesseract_setup.py      # Windows Tesseract PATH setup
```

---

## How it works (short)

```
PDFs  →  Unstructured (text / tables / images)
      →  LLM summaries  →  Chroma (vectors)
      →  Originals       →  docstore.pkl

Question  →  similarity search on summaries
         →  fetch parent text + images
         →  vision LLM answer
```

See [WIKI.md § Architecture](WIKI.md#architecture) for diagrams and deep dive.

---

## Configuration

Key variables in `.env` (see `.env.example`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | — | **Required** — embeddings, vision, answers |
| `OPENAI_MODEL` | `gpt-4o-mini` | Answer + image summary model |
| `GROQ_API_KEY` | — | Optional — text/table summaries |
| `DATA_FOLDER` | `./data` | Default PDF folder for indexing |
| `VECTORSTORE_PATH` | `./vectorstore` | Chroma + docstore location |
| `RETRIEVAL_TOP_K` | `4` | Chunks retrieved per question |

---

## Multiple PDFs

All PDFs in a folder are indexed into **one** vector store. Queries search across every indexed file. Each chunk stores a `source` metadata field with the originating PDF path.

```bash
# Index 3 PDFs in test_data
python index.py --folder test_data
```

Re-running index on the same files **appends** duplicates. Delete `vectorstore/` for a clean rebuild. Details: [WIKI.md § Indexing multiple PDFs](WIKI.md#indexing-multiple-pdfs).

---

## Notebook vs application

| | `langchain_multimodal.ipynb` | This project |
|--|------------------------------|--------------|
| Orchestration | LangChain LCEL | **LangGraph** |
| Docstore | In-memory (lost on restart) | **Persistent** `docstore.pkl` |
| UI | Jupyter cells | **Streamlit** + CLI |
| Indexing | Manual cell execution | `index.py` / sidebar button |

The notebook remains the reference implementation; the app automates and persists the same RAG pattern.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Tesseract not found` | Install Tesseract; see [WIKI.md § Tesseract](WIKI.md#tesseract-ocr-windows) |
| Empty index / no answers | Run `index.py` first; check sidebar metrics |
| Slow indexing | Normal for `hi_res`; index one PDF at a time while testing |
| `IndexError` on `chunk_images[0]` in notebook | That chunk has no images; scan all chunks (see notebook cell fix) |

More: [WIKI.md § Troubleshooting](WIKI.md#troubleshooting).

---

## Example question

After indexing the Nalanda state-of-conservation PDF:

```
What is Sarai Mound?
```

The system retrieves landscape-development passages (e.g. garden work around Temple sites 12–14) and synthesizes an answer grounded in the document.

---

## License

Use and adapt for research and education. Ensure compliance with API provider terms and document copyright for heritage PDFs.
