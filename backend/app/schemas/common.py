"""Common schema definitions."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Model information schema."""
    name: str
    version: Optional[str] = None
    status: str
    accuracy: Optional[float] = None
    created_at: Optional[str] = None
    run_id: Optional[str] = None

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "name": "iris_random_forest",
                "version": "1",
                "status": "production",
                "accuracy": 0.95,
                "created_at": "2024-01-01T00:00:00Z",
                "run_id": "abc123"
            }
        }


class PredictionResponse(BaseModel):
    """Base prediction response schema."""
    predictions: List[float] = Field(..., description="Model predictions")
    model_used: str = Field(..., description="Name of the model used")
    model_version: Optional[str] = Field(None, description="Version of the model")
    uncertainty: Optional[List[Dict[str, float]]] = Field(None, description="Uncertainty estimates")

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "predictions": [0.95, 0.89, 0.76],
                "model_used": "iris_random_forest",
                "model_version": "1",
                "uncertainty": [
                    {"lower": 0.90, "upper": 0.98},
                    {"lower": 0.82, "upper": 0.94}
                ]
            }
        }


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(..., description="Overall health status")
    version: str = Field(..., description="API version")
    models: Dict[str, ModelInfo] = Field(..., description="Model status information")
    mlflow_uri: Optional[str] = Field(None, description="MLflow tracking URI")

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "models": {
                    "iris_rf": {
                        "name": "iris_random_forest",
                        "status": "loaded",
                        "accuracy": 0.95
                    }
                },
                "mlflow_uri": "http://mlflow:5000"
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "error": "Model not found",
                "detail": "The requested model is not available"
            }
        }
