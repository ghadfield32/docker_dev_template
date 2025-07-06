"""Core configuration for the FastAPI application."""

import os
from typing import List


class Settings:
    """Application settings."""

    # Basic app settings
    PROJECT_NAME: str = "ML Full Stack API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # CORS settings
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server
        "http://localhost:8080",  # Alternative frontend
        "http://localhost:5000",  # Another common port
        "https://*.netlify.app",  # Netlify deployments
        "*",  # Allow all origins for development/testing
    ]

    # MLflow settings
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns_local")
    MLFLOW_EXPERIMENT_NAME: str = "ml_fullstack_models"

    # Model settings
    DEV_AUTOTRAIN: bool = os.getenv("DEV_AUTOTRAIN", "true").lower() == "true"

    # Database settings (if needed later)
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./app.db")


# Global settings instance
settings = Settings()
