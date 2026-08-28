from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ..database import get_db, check_db, get_db_path
from ..config import get_settings
from ..inference import is_loaded

router = APIRouter()
settings = get_settings()

@router.get("/health")
def health(db: Session = Depends(get_db)):
    ok, msg = check_db()
    db_status = "ok" if ok else f"error: {msg}"
    return {
        "status": "ok" if ok else "degraded",
        "database": db_status,
        "db_path": get_db_path(),
        "model_loaded": is_loaded(),
        "model_version": settings.model_version
    }
