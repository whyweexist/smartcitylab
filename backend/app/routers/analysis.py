from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import Optional
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from ..database import get_db
from ..models import Analysis
from ..schemas import AnalysisResponse, HistoryItem, IssueResult
from ..validation import validate_upload
from ..service import analyze_image
from ..storage import get_file_path
from ..config import get_settings

router = APIRouter()
settings = get_settings()

def _to_response(row: Analysis):
    issues = json.loads(row.issues_json) if row.issues_json else []
    stats = json.loads(row.stats_json) if row.stats_json else {}
    expl = json.loads(row.explanations_json) if row.explanations_json else []
    # map issues to IssueResult shape handled outside
    return {
        "id": row.id,
        "timestamp": row.created_at,
        "quality_score": row.quality_score,
        "quality_label": row.quality_label,
        "confidence": 0.8,
        "issues": issues,
        "image_stats": stats,
        "image_metadata": {"width": row.width, "height": row.height, "original_filename": row.original_filename},
        "model_version": row.model_version,
        "inference_ms": row.inference_ms,
        "image_url": f"/api/v1/image/{row.stored_filename}",
        "raw": row
    }

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...), db: Session = Depends(get_db)):
    data = await file.read()
    size = len(data)
    validate_upload(file.filename or "upload.jpg", file.content_type or "", size, data)
    # service does safe decode + ML
    try:
        result = analyze_image(data, file.filename or "upload.jpg", file.content_type or "image/jpeg")
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    # persist
    row = Analysis(
        original_filename=file.filename or "upload.jpg",
        stored_filename=result["stored_filename"],
        mime_type=file.content_type,
        width=result["width"],
        height=result["height"],
        quality_score=result["quality_score"],
        quality_label=result["quality_label"],
        issues_json=json.dumps(result["issues"]),
        stats_json=json.dumps(result["image_stats"]),
        explanations_json=json.dumps(result["explanations"]),
        model_version=result["model_version"],
        inference_ms=result["inference_ms"],
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    issues = [IssueResult(**i) for i in result["issues"]]
    return AnalysisResponse(
        id=row.id,
        timestamp=row.created_at,
        quality_score=row.quality_score,
        quality_label=row.quality_label,
        confidence=result["confidence"],
        issues=issues,
        image_stats=result["image_stats"],
        image_metadata={"width": row.width, "height": row.height, "original_filename": row.original_filename, "mime_type": row.mime_type},
        model_version=row.model_version,
        inference_ms=row.inference_ms,
        image_url=f"/api/v1/image/{row.stored_filename}"
    )

@router.get("/history")
def history(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0), db: Session = Depends(get_db)):
    rows = db.query(Analysis).order_by(Analysis.created_at.desc()).offset(offset).limit(limit).all()
    items = []
    for r in rows:
        issues = json.loads(r.issues_json) if r.issues_json else []
        items.append({
            "id": r.id,
            "timestamp": r.created_at.isoformat() if r.created_at else "",
            "quality_score": r.quality_score,
            "quality_label": r.quality_label,
            "original_filename": r.original_filename,
            "width": r.width,
            "height": r.height,
            "issues": issues,
            "image_url": f"/api/v1/image/{r.stored_filename}"
        })
    total = db.query(Analysis).count()
    return {"items": items, "total": total, "limit": limit, "offset": offset}

@router.get("/analysis/{analysis_id}", response_model=AnalysisResponse)
def get_analysis(analysis_id: str, db: Session = Depends(get_db)):
    row = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Analysis not found")
    issues = json.loads(row.issues_json) if row.issues_json else []
    stats = json.loads(row.stats_json) if row.stats_json else {}
    return AnalysisResponse(
        id=row.id,
        timestamp=row.created_at,
        quality_score=row.quality_score,
        quality_label=row.quality_label,
        confidence=0.8,
        issues=[IssueResult(**i) for i in issues],
        image_stats=stats,
        image_metadata={"width": row.width, "height": row.height, "original_filename": row.original_filename},
        model_version=row.model_version,
        inference_ms=row.inference_ms,
        image_url=f"/api/v1/image/{row.stored_filename}"
    )

@router.get("/image/{filename}")
def get_image(filename: str):
    # prevent traversal
    p = Path(filename)
    if p.name != filename or ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    fpath = get_file_path(filename)
    if not fpath.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(str(fpath))
