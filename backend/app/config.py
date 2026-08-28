from functools import lru_cache
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    database_url: str = "sqlite:///./data/app.db"
    upload_dir: str = "./data/uploads"
    max_upload_bytes: int = 10 * 1024 * 1024
    allowed_origins: str = "*"
    model_path: str = "./backend/artifacts/model.joblib"
    log_level: str = "INFO"
    max_image_dim: int = 2048
    api_version: str = "v1"
    model_version: str = "1.0.0"

    # Thresholds centralized
    sharpness_blur_thresh: float = 80.0
    dark_ratio_thresh: float = 0.35
    bright_ratio_thresh: float = 0.35
    contrast_low_thresh: float = 25.0
    noise_high_thresh: float = 12.0
    anomaly_thresh: float = -0.1  # isolation forest decision offset

    # Scoring weights
    weight_blur: float = 18.0
    weight_underexposed: float = 15.0
    weight_overexposed: float = 15.0
    weight_noise: float = 14.0
    weight_severe: float = 22.0
    weight_defect: float = 20.0

    class Config:
        env_file = ".env"
        extra = "ignore"

@lru_cache
def get_settings() -> Settings:
    return Settings()

# Quality labels
ACCEPTABLE = "ACCEPTABLE"
DEGRADED = "DEGRADED"
POTENTIALLY_DEFECTIVE = "POTENTIALLY_DEFECTIVE"

ISSUE_TYPES = ["blur", "underexposure", "overexposure", "noise", "severe_degradation", "potential_defect"]
