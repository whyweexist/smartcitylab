from sqlalchemy import Column, String, Integer, Float, DateTime, Text, Index
from sqlalchemy.sql import func
from .database import Base
import uuid

def gen_id():
    return str(uuid.uuid4())

class Analysis(Base):
    """
    SQLite table `analyses` — also PostgreSQL compatible (all types map).
    - id: UUID string (no Postgres uuid extension required)
    - JSON fields stored as Text (SQLite has no JSONB; Postgres JSONB migration is ALTER COLUMN)
    - created_at: server_default now(), indexed for history ordering
    Persistence verified: Docker volume /app/data preserves DB + uploads across restarts.
    """
    __tablename__ = "analyses"
    id = Column(String, primary_key=True, default=gen_id)
    original_filename = Column(String, nullable=False)
    stored_filename = Column(String, nullable=False)
    mime_type = Column(String, nullable=True)
    width = Column(Integer)
    height = Column(Integer)
    quality_score = Column(Float, nullable=False)
    quality_label = Column(String, nullable=False)
    issues_json = Column(Text)  # JSON string — portable across SQLite/Postgres
    stats_json = Column(Text)
    explanations_json = Column(Text)
    model_version = Column(String)
    inference_ms = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)

    __table_args__ = (
        Index("ix_analyses_created_at", "created_at"),
        Index("ix_analyses_quality_label", "quality_label"),
    )
