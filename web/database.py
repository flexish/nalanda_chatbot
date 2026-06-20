"""
Nalanda User Database — SQLite with PBKDF2-SHA256 password hashing.
No external dependencies; uses Python stdlib only.
"""
from __future__ import annotations

import hashlib
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "nalanda_users.db"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260_000).hex()


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ── Schema ────────────────────────────────────────────────────────────────────

def init_db() -> None:
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT    UNIQUE NOT NULL COLLATE NOCASE,
                email         TEXT    UNIQUE NOT NULL COLLATE NOCASE,
                password_hash TEXT    NOT NULL,
                salt          TEXT    NOT NULL,
                role          TEXT    NOT NULL DEFAULT 'user',
                is_active     INTEGER NOT NULL DEFAULT 1,
                created_at    TEXT    NOT NULL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS url_sources (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                url        TEXT    UNIQUE NOT NULL,
                label      TEXT    NOT NULL DEFAULT '',
                added_by   TEXT    NOT NULL DEFAULT 'admin',
                added_at   TEXT    NOT NULL
            )
        """)
        c.commit()


# ── Seed ──────────────────────────────────────────────────────────────────────

def seed_admin(username: str, password: str, email: str = "admin@nalanda.local") -> None:
    """Insert the admin account on first run; silently skips if already exists."""
    with _conn() as c:
        if not c.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone():
            salt = secrets.token_hex(16)
            c.execute(
                "INSERT INTO users (username,email,password_hash,salt,role,created_at) "
                "VALUES (?,?,?,?,'admin',?)",
                (username, email, _hash(password, salt), salt, _now()),
            )
            c.commit()


# ── Auth ──────────────────────────────────────────────────────────────────────

def authenticate(username: str, password: str) -> dict | None:
    """Return the user row as a dict if credentials are valid, else None."""
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM users WHERE username=? AND is_active=1", (username,)
        ).fetchone()
    if row and _hash(password, row["salt"]) == row["password_hash"]:
        return dict(row)
    return None


# ── Registration ──────────────────────────────────────────────────────────────

def create_user(username: str, email: str, password: str, role: str = "user") -> None:
    salt = secrets.token_hex(16)
    with _conn() as c:
        c.execute(
            "INSERT INTO users (username,email,password_hash,salt,role,created_at) "
            "VALUES (?,?,?,?,?,?)",
            (username, email, _hash(password, salt), salt, role, _now()),
        )
        c.commit()


def username_exists(username: str) -> bool:
    with _conn() as c:
        return bool(c.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone())


def email_exists(email: str) -> bool:
    with _conn() as c:
        return bool(c.execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone())


# ── Admin CRUD ────────────────────────────────────────────────────────────────

def list_users() -> list[dict]:
    with _conn() as c:
        return [
            dict(r)
            for r in c.execute(
                "SELECT id,username,email,role,is_active,created_at FROM users ORDER BY id"
            ).fetchall()
        ]


def update_user(user_id: int, *, role: str | None = None, is_active: int | None = None) -> None:
    parts, params = [], []
    if role is not None:
        parts.append("role=?"); params.append(role)
    if is_active is not None:
        parts.append("is_active=?"); params.append(is_active)
    if not parts:
        return
    params.append(user_id)
    with _conn() as c:
        c.execute(f"UPDATE users SET {','.join(parts)} WHERE id=?", params)
        c.commit()


def delete_user(user_id: int) -> None:
    with _conn() as c:
        c.execute("DELETE FROM users WHERE id=?", (user_id,))
        c.commit()


# ── URL Sources ───────────────────────────────────────────────────────────────

def list_url_sources() -> list[dict]:
    with _conn() as c:
        return [
            dict(r) for r in c.execute(
                "SELECT id, url, label, added_by, added_at FROM url_sources ORDER BY id"
            ).fetchall()
        ]


def url_source_exists(url: str) -> bool:
    with _conn() as c:
        return bool(c.execute("SELECT 1 FROM url_sources WHERE url=?", (url,)).fetchone())


def add_url_source(url: str, label: str = "", added_by: str = "admin") -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO url_sources (url, label, added_by, added_at) VALUES (?,?,?,?)",
            (url, label, added_by, _now()),
        )
        c.commit()
        return cur.lastrowid


def delete_url_source(url_id: int) -> str | None:
    """Delete a URL source by id. Returns the URL string if found, else None."""
    with _conn() as c:
        row = c.execute("SELECT url FROM url_sources WHERE id=?", (url_id,)).fetchone()
        url = row["url"] if row else None
        c.execute("DELETE FROM url_sources WHERE id=?", (url_id,))
        c.commit()
    return url
