"""Pydantic schemas for API request/response models."""

from .iris import IrisFeatures, IrisPredictRequest, IrisPredictResponse
from .cancer import CancerFeatures, CancerPredictRequest, CancerPredictResponse
from .common import HealthResponse, ModelInfo, PredictionResponse

__all__ = [
    "IrisFeatures",
    "IrisPredictRequest",
    "IrisPredictResponse",
    "CancerFeatures",
    "CancerPredictRequest",
    "CancerPredictResponse",
    "HealthResponse",
    "ModelInfo",
    "PredictionResponse",
]
