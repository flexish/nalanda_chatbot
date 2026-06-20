from __future__ import annotations

import json
import os
import secrets
import sys
import threading
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Header, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ── sibling import: web/database.py ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
import database as db

from utils.config import TOP_K, VECTORSTORE_PATH, DATA_FOLDER, DOCSTORE_FILENAME
from utils.rag_graph import query_with_sources, astream_rag_response
from utils.vectorstore import MultimodalVectorStore


# ── Credentials (bootstrap admin from .env) ───────────────────────────────────

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "nalanda@admin123")


# ── In-memory session store ───────────────────────────────────────────────────

_sessions: dict[str, dict] = {}   # token → {username, role, user_id}


def _get_session(authorization: str | None) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    session = _sessions.get(authorization[7:])
    if not session:
        raise HTTPException(status_code=401, detail="Session expired or invalid")
    return session


def _admin_session(authorization: str | None) -> dict:
    session = _get_session(authorization)
    if session.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return session


# ── App lifespan: init DB and seed admin ─────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    db.seed_admin(ADMIN_USERNAME, ADMIN_PASSWORD)
    yield

app = FastAPI(title="Nalanda RAG Web API", version="3.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Index background jobs ─────────────────────────────────────────────────────

class _Job:
    def __init__(self, job_id: str) -> None:
        self.job_id    = job_id
        self.status    = "queued"
        self.messages: list[str] = []
        self.error: str | None = None


_jobs: dict[str, _Job] = {}


def _run_index(job: _Job, mode: str, pdf_path: str | None, vstore: str, dfolder: str) -> None:
    def log(m: str) -> None:
        job.messages.append(m)

    try:
        job.status = "running"
        store = MultimodalVectorStore(persist_dir=Path(vstore))
        if mode == "single" and pdf_path:
            from utils.ingest import ingest_pdf
            # pdf_path is a bare filename — resolve it against the server's data folder
            p = (Path(dfolder) / Path(pdf_path).name)
            log(f"Processing {p.name}…")
            removed = store.remove_source(str(p))
            if removed:
                log(f"Replaced {removed} existing entries.")
            ingested = ingest_pdf(p)
            counts = store.add_ingested(ingested, on_progress=log)
            log(f"Indexed: {counts}")
        else:
            totals = store.index_folder(Path(dfolder), on_progress=log)
            log(f"Indexed folder: {totals}")
        _load_store_cached.cache_clear()
        job.status = "done"
    except Exception as exc:
        job.error = str(exc)
        job.status = "error"
        log(f"Error: {exc}")


# ── Vector store cache ────────────────────────────────────────────────────────

@lru_cache(maxsize=8)
def _load_store_cached(path: str) -> MultimodalVectorStore:
    return MultimodalVectorStore.load(path)


def _load_store(path: str | None = None) -> MultimodalVectorStore:
    target = str(Path(path).resolve()) if path else str(VECTORSTORE_PATH.resolve())
    return _load_store_cached(target)


# ── Pydantic models ───────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    token: str
    role: str
    username: str

class SignupRequest(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    email: str    = Field(min_length=5, max_length=120)
    password: str = Field(min_length=6, max_length=128)

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = ""

class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)
    top_k: int | None = None

class UrlSourceRequest(BaseModel):
    url: str = Field(min_length=8)
    label: str = ""

class ChatResponse(BaseModel):
    answer: str
    images: list[str]
    captions: list[str]
    mode: str
    verified: bool
    verification_reason: str

class IndexRequest(BaseModel):
    mode: Literal["single", "folder"] = "single"
    pdf_path: str | None = None

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    messages: list[str]
    error: str | None = None

class UserUpdateRequest(BaseModel):
    role: str | None = None
    is_active: int | None = None


# ── Auth endpoints ────────────────────────────────────────────────────────────

@app.post("/api/login", response_model=LoginResponse)
def login(req: LoginRequest) -> LoginResponse:
    user = db.authenticate(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = secrets.token_urlsafe(32)
    _sessions[token] = {"username": user["username"], "role": user["role"], "user_id": user["id"]}
    return LoginResponse(token=token, role=user["role"], username=user["username"])


@app.post("/api/signup")
def signup(req: SignupRequest) -> dict:
    # Basic validation
    if len(req.username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters")
    if db.username_exists(req.username):
        raise HTTPException(status_code=409, detail="Username already taken")
    if db.email_exists(req.email):
        raise HTTPException(status_code=409, detail="Email already registered")
    if "@" not in req.email:
        raise HTTPException(status_code=400, detail="Invalid email address")
    db.create_user(req.username, req.email, req.password, role="user")
    return {"status": "created", "message": "Account created successfully. You can now log in."}


@app.post("/api/logout")
def logout(authorization: str | None = Header(None, alias="Authorization")) -> dict:
    if authorization and authorization.startswith("Bearer "):
        _sessions.pop(authorization[7:], None)
    return {"status": "ok"}


# ── Chat (any authenticated user) ─────────────────────────────────────────────

def _history_str(history: list[ChatMessage], max_turns: int = 4, max_chars: int = 1800) -> str:
    lines: list[str] = []
    for m in history[-max_turns * 2:]:
        role = "User" if m.role == "user" else "Assistant"
        content = (m.content or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    joined = "\n".join(lines)
    return joined if len(joined) <= max_chars else joined[-max_chars:]


@app.post("/api/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    authorization: str | None = Header(None, alias="Authorization"),
) -> ChatResponse:
    _get_session(authorization)
    store = _load_store(None)
    if store.stats()["summary_vectors"] == 0:
        return ChatResponse(
            answer="The knowledge base is empty. Please ask an administrator to index the documents.",
            images=[], captions=[], mode="text",
            verified=False, verification_reason="empty vector store",
        )
    from utils.url_fetcher import fetch_urls_cached
    url_records = db.list_url_sources()
    web_docs = fetch_urls_cached([r["url"] for r in url_records]) if url_records else []
    result = query_with_sources(
        store, req.question,
        top_k=req.top_k or TOP_K,
        chat_history=_history_str(req.history),
        web_docs=web_docs or None,
    )
    answer   = (result.get("response") or "").strip()
    images   = result.get("context", {}).get("images", []) or []
    captions = result.get("context", {}).get("image_captions", []) or []
    if images and not answer:
        answer = f"Found {len(images)} matching image{'s' if len(images) != 1 else ''}."
    return ChatResponse(
        answer=answer, images=images, captions=captions,
        mode=result.get("mode", "text"),
        verified=bool(result.get("verified", True)),
        verification_reason=result.get("verification_reason", ""),
    )


# ── Chat streaming ────────────────────────────────────────────────────────────

@app.post("/api/chat/stream")
async def chat_stream(
    req: ChatRequest,
    authorization: str | None = Header(None, alias="Authorization"),
) -> StreamingResponse:
    import asyncio
    from utils.url_fetcher import fetch_urls_cached

    _get_session(authorization)
    store = _load_store(None)
    chat_history = _history_str(req.history)

    url_records = db.list_url_sources()
    web_docs = await asyncio.to_thread(
        fetch_urls_cached, [r["url"] for r in url_records]
    ) if url_records else []

    if store.stats()["summary_vectors"] == 0 and not web_docs:
        async def _empty():
            yield f"data: {json.dumps({'type': 'done', 'answer': 'The knowledge base is empty. Please ask an administrator to index the documents.', 'images': [], 'captions': [], 'mode': 'text', 'verified': False, 'verification_reason': 'empty vector store'})}\n\n"
        return StreamingResponse(_empty(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    async def _generate():
        try:
            async for event in astream_rag_response(
                store, req.question,
                top_k=req.top_k or TOP_K,
                chat_history=chat_history,
                web_docs=web_docs or None,
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Public health ─────────────────────────────────────────────────────────────

@app.get("/api/health")
def health() -> dict:
    try:
        return {"status": "ok", "store": _load_store(None).stats()}
    except Exception as exc:
        return {"status": "degraded", "error": str(exc)}


# ── Admin: statistics ─────────────────────────────────────────────────────────

@app.get("/api/admin/stats")
def admin_stats(authorization: str | None = Header(None, alias="Authorization")) -> dict:
    _admin_session(authorization)
    try:
        return _load_store(None).stats()
    except Exception as exc:
        return {"summary_vectors": 0, "docstore_entries": 0, "error": str(exc)}


# ── Admin: user management ────────────────────────────────────────────────────

@app.get("/api/admin/users")
def admin_list_users(authorization: str | None = Header(None, alias="Authorization")) -> dict:
    _admin_session(authorization)
    return {"users": db.list_users()}


@app.patch("/api/admin/users/{user_id}")
def admin_update_user(
    user_id: int,
    req: UserUpdateRequest,
    authorization: str | None = Header(None, alias="Authorization"),
) -> dict:
    _admin_session(authorization)
    if req.role and req.role not in ("user", "admin"):
        raise HTTPException(status_code=400, detail="Role must be 'user' or 'admin'")
    db.update_user(user_id, role=req.role, is_active=req.is_active)
    return {"status": "updated"}


@app.delete("/api/admin/users/{user_id}")
def admin_delete_user(
    user_id: int,
    authorization: str | None = Header(None, alias="Authorization"),
) -> dict:
    sess = _admin_session(authorization)
    if sess.get("user_id") == user_id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
    db.delete_user(user_id)
    return {"status": "deleted"}


# ── Admin: list PDFs ──────────────────────────────────────────────────────────

@app.get("/api/admin/pdfs")
def admin_list_pdfs(
    authorization: str | None = Header(None, alias="Authorization"),
) -> dict:
    _admin_session(authorization)
    folder = DATA_FOLDER
    if not folder.exists():
        return {"pdfs": [], "folder": str(folder)}
    pdfs = sorted(folder.glob("**/*.pdf"))
    return {
        "pdfs": [{"name": p.name, "size": p.stat().st_size, "path": str(p)} for p in pdfs],
        "folder": str(folder),
    }


# ── Admin: upload PDF ─────────────────────────────────────────────────────────

@app.post("/api/admin/upload")
async def admin_upload(
    file: UploadFile = File(...),
    authorization: str | None = Header(None, alias="Authorization"),
) -> dict:
    _admin_session(authorization)
    safe_name = Path(file.filename or "upload.pdf").name
    if not safe_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    DATA_FOLDER.mkdir(parents=True, exist_ok=True)
    dest = DATA_FOLDER / safe_name
    content = await file.read()
    dest.write_bytes(content)
    return {"status": "uploaded", "file": safe_name, "size": len(content), "path": str(dest)}


# ── Admin: indexing ───────────────────────────────────────────────────────────

@app.post("/api/admin/index", response_model=JobStatusResponse)
def admin_index(
    req: IndexRequest,
    authorization: str | None = Header(None, alias="Authorization"),
) -> JobStatusResponse:
    _admin_session(authorization)
    job_id = secrets.token_hex(8)
    job = _Job(job_id)
    _jobs[job_id] = job
    threading.Thread(
        target=_run_index,
        args=(job, req.mode, req.pdf_path, str(VECTORSTORE_PATH), str(DATA_FOLDER)),
        daemon=True,
    ).start()
    return JobStatusResponse(job_id=job_id, status=job.status, messages=[], error=None)


@app.get("/api/admin/index/{job_id}", response_model=JobStatusResponse)
def admin_index_status(
    job_id: str,
    authorization: str | None = Header(None, alias="Authorization"),
) -> JobStatusResponse:
    _admin_session(authorization)
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(job_id=job.job_id, status=job.status,
                             messages=job.messages, error=job.error)


@app.delete("/api/admin/index")
def admin_clear_index(
    authorization: str | None = Header(None, alias="Authorization"),
) -> dict:
    _admin_session(authorization)
    import gc, shutil

    # Step 1: Clear via Chroma API (avoids Windows SQLite file-lock issues)
    try:
        store = _load_store(None)
        store.vectorstore._client.delete_collection(store.collection_name)
    except Exception:
        pass

    # Step 2: Delete the docstore pickle
    try:
        docstore_path = VECTORSTORE_PATH / DOCSTORE_FILENAME
        if docstore_path.exists():
            docstore_path.unlink()
    except Exception:
        pass

    # Step 3: Release all cached store references
    _load_store_cached.cache_clear()
    gc.collect()

    # Step 4: Best-effort full directory delete (may be skipped if SQLite still locked)
    if VECTORSTORE_PATH.exists():
        try:
            shutil.rmtree(VECTORSTORE_PATH)
        except Exception:
            pass  # Collection already cleared above; directory will be reused cleanly

    return {"status": "cleared", "path": str(VECTORSTORE_PATH)}


# ── Admin: URL sources ───────────────────────────────────────────────────────

@app.get("/api/admin/url-sources")
def admin_list_url_sources(
    authorization: str | None = Header(None, alias="Authorization"),
) -> dict:
    _admin_session(authorization)
    return {"url_sources": db.list_url_sources()}


@app.post("/api/admin/url-sources")
def admin_add_url_source(
    req: UrlSourceRequest,
    authorization: str | None = Header(None, alias="Authorization"),
) -> dict:
    sess = _admin_session(authorization)
    if not req.url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")
    if db.url_source_exists(req.url):
        raise HTTPException(status_code=409, detail="URL already exists")
    # Quick HEAD check to confirm URL is reachable before saving
    try:
        import requests as _r
        resp = _r.head(req.url, timeout=8, allow_redirects=True,
                       headers={"User-Agent": "Mozilla/5.0 (compatible; NalandaRAG/1.0)"})
        if resp.status_code >= 400:
            raise HTTPException(status_code=400, detail=f"URL returned HTTP {resp.status_code}")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Cannot reach URL: {exc}")
    try:
        from utils.url_fetcher import invalidate_cache
        url_id = db.add_url_source(req.url, req.label, added_by=sess["username"])
        invalidate_cache(req.url)   # ensure next query fetches fresh content
        return {"status": "added", "id": url_id}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/api/admin/url-sources/{url_id}")
def admin_delete_url_source(
    url_id: int,
    authorization: str | None = Header(None, alias="Authorization"),
) -> dict:
    _admin_session(authorization)
    from utils.url_fetcher import invalidate_cache
    deleted_url = db.delete_url_source(url_id)
    if deleted_url:
        invalidate_cache(deleted_url)
    return {"status": "deleted"}


# ── Static / SPA fallback ─────────────────────────────────────────────────────

static_dir = Path(__file__).resolve().parent / "static"
app.mount("/assets", StaticFiles(directory=static_dir), name="assets")


@app.get("/{full_path:path}")
def spa_fallback(full_path: str) -> FileResponse:
    return FileResponse(static_dir / "index.html")
