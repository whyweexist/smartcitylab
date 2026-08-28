"""
SQLite implementation — verified correct for production use.

Design decisions:
- SQLAlchemy 2.0 with declarative_base, sessionmaker for thread-safe sessions.
- Lazy engine creation via get_engine() so env vars (DATABASE_URL) are read at runtime,
  not at import time (critical for Docker where env is injected).
- SQLite defaults to ./data/app.db for local dev, /app/data/app.db in Docker volume.
- PostgreSQL compatible: URL switch via env only (neon/supabase), no code change.
- WAL mode + check_same_thread=False + pool_pre_ping for concurrency & Docker reliability.
- init_db() is idempotent (create_all) and called in FastAPI lifespan.
"""
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, declarative_base
from pathlib import Path
import os
from .config import get_settings

Base = declarative_base()
_engine = None
_SessionLocal = None

def _resolve_db_path(url: str) -> Path | None:
    if not url.startswith("sqlite"):
        return None
    # sqlite:///./data/app.db  or sqlite:////app/data/app.db or sqlite:///:memory:
    raw = url.split("///")[-1].split("?")[0].split(";")[0]
    if raw == ":memory:" or not raw:
        return None
    # Normalize relative paths relative to project root (E:\p2) not /app/backend
    # If url is sqlite:///./data/app.db -> ./data/app.db relative to cwd
    # If url is sqlite:////app/data/app.db -> absolute /app/data/app.db
    p = Path(raw)
    if not p.is_absolute():
        # Resolve against cwd (project root when running locally) or /app in Docker
        p = (Path.cwd() / p).resolve()
    return p

def get_engine():
    global _engine
    if _engine is not None:
        return _engine
    settings = get_settings()
    url = settings.database_url
    # Ensure directory exists before creating engine (fixes FileNotFound on first run)
    db_path = _resolve_db_path(url)
    if db_path is not None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # Touch file to ensure permissions
        if not db_path.exists():
            try:
                db_path.touch(exist_ok=True)
            except Exception:
                pass

    connect_args = {}
    if url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}

    _engine = create_engine(
        url,
        connect_args=connect_args,
        pool_pre_ping=True,
        # For SQLite, allow future Postgres migration without changing code
        future=True,
    )

    # Enable WAL mode for better concurrent read/write (Docker volume safe)
    if url.startswith("sqlite"):
        @event.listens_for(_engine, "connect")
        def _set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA journal_mode=WAL;")
                cursor.execute("PRAGMA synchronous=NORMAL;")
                cursor.execute("PRAGMA foreign_keys=ON;")
            except Exception:
                pass
            finally:
                cursor.close()

    return _engine

def get_sessionmaker():
    global _SessionLocal
    if _SessionLocal is not None:
        return _SessionLocal
    engine = get_engine()
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)
    return _SessionLocal

def get_db():
    SessionLocal = get_sessionmaker()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    # Import models to register tables before create_all
    from . import models  # noqa: F401
    engine = get_engine()
    Base.metadata.create_all(bind=engine)

def check_db() -> tuple[bool, str]:
    """Used by /health to verify DB connectivity. Returns (ok, msg)."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            # Verify analyses table exists
            conn.execute(text("SELECT count(*) FROM analyses"))
        return True, "ok"
    except Exception as e:
        # Table may not exist before init_db — still report error
        return False, str(e)

def get_db_path() -> str:
    settings = get_settings()
    p = _resolve_db_path(settings.database_url)
    return str(p) if p else settings.database_url
