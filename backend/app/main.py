from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path

from .config import get_settings
from .database import init_db
from .routers import health, analysis
from .inference import load_bundle

settings = get_settings()
logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing database...")
    init_db()
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    try:
        bundle = load_bundle()
        logger.info(f"Model loaded version={bundle.model_version} features={len(bundle.feature_names)}")
    except Exception as e:
        logger.warning(f"Model not loaded at startup: {e} . Training required.")
    yield
    logger.info("Shutting down")

app = FastAPI(title="AI Image Quality & Defect Detection", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.allowed_origins == "*" else [o.strip() for o in settings.allowed_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(analysis.router, prefix="/api/v1", tags=["analysis"])

@app.get("/")
def root():
    return {"service": "AI Image Quality & Defect Detection", "version": "1.0.0", "docs": "/docs"}

@app.get("/api")
def api_root():
    return {"message": "Use /api/v1/* endpoints", "health": "/api/v1/health"}
