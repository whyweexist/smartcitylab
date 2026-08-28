from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class IssueResult(BaseModel):
    type: str
    severity: str  # low|medium|high
    confidence: float
    explanation: str
    evidence: Optional[Dict[str, Any]] = None

class AnalysisResponse(BaseModel):
    id: str
    timestamp: datetime
    quality_score: float
    quality_label: str
    confidence: float
    issues: List[IssueResult]
    image_stats: Dict[str, Any]
    image_metadata: Dict[str, Any]
    model_version: str
    inference_ms: float
    image_url: Optional[str] = None

class HistoryItem(BaseModel):
    id: str
    timestamp: datetime
    quality_score: float
    quality_label: str
    original_filename: str
    width: int
    height: int
    issues: List[IssueResult]
    image_url: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    database: str
    model_loaded: bool
    model_version: str
