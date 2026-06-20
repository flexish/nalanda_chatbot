# Production Web App (Non-Streamlit)

This adds a production-style web layer while keeping your tested RAG pipeline unchanged.

## What is included

- `web/api.py`: FastAPI backend exposing:
  - `GET /api/health`
  - `POST /api/chat`
- `web/static/index.html`: chat UI shell
- `web/static/styles.css`: responsive visual design
- `web/static/app.js`: frontend chat client integration
- `requirements-web.txt`: backend web dependencies

## Run

```powershell
pip install -r requirements-web.txt
uvicorn web.api:app --host 0.0.0.0 --port 8000
```

Open:

- `http://localhost:8000/`

## Notes

- Backend reuses existing pipeline: `utils.rag_graph.query_with_sources`.
- It uses same vector store as your current app by default (`VECTORSTORE_PATH` from `.env`).
- No existing tested files were modified for frontend integration.
