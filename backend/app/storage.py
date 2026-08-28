import os
import uuid
from pathlib import Path
from .config import get_settings

def _settings():
    return get_settings()

def ensure_upload_dir():
    Path(_settings().upload_dir).mkdir(parents=True, exist_ok=True)

def save_bytes(data: bytes, original_filename: str) -> str:
    ensure_upload_dir()
    s = _settings()
    ext = Path(original_filename).suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"]:
        ext = ".jpg"
    fname = f"{uuid.uuid4().hex}{ext}"
    fpath = Path(s.upload_dir) / fname
    with open(fpath, "wb") as f:
        f.write(data)
    return fname

def get_file_path(stored_filename: str) -> Path:
    return Path(_settings().upload_dir) / stored_filename
