"""Model management endpoints."""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from ....schemas.common import HealthResponse
from ....services.ml.model_service import model_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=Dict[str, Any])
async def get_health():
    """
    Get the health status of all models.

    Returns information about:
    - Overall service status
    - Individual model status
    - MLflow connection status
    - Model metrics
    """
    try:
        return await model_service.get_health_status()
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@router.get("/metrics")
async def get_model_metrics():
    """
    Get performance metrics for all loaded models.

    Returns metrics like accuracy, F1 score, precision, and recall
    for each available model.
    """
    try:
        return await model_service.get_model_metrics()
    except Exception as e:
        logger.error(f"Metrics retrieval error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model metrics")


@router.get("/list")
async def list_models():
    """
    List all available models with their basic information.

    Returns:
    - Model names and types
    - Load status
    - Basic model information
    """
    try:
        health_status = await model_service.get_health_status()

        models_info = {}
        for model_name, info in health_status["models"].items():
            models_info[model_name] = {
                "name": info["name"],
                "status": info["status"],
                "version": info.get("version"),
                "accuracy": info.get("accuracy"),
                "dataset": "iris" if "iris" in model_name else "breast_cancer",
                "algorithm": _get_algorithm_type(model_name)
            }

        return {
            "models": models_info,
            "total_models": len(models_info),
            "loaded_models": len([m for m in models_info.values() if m["status"] == "loaded"]),
            "available_datasets": ["iris", "breast_cancer"],
            "available_algorithms": ["random_forest", "logistic_regression", "bayesian_logistic_regression"]
        }

    except Exception as e:
        logger.error(f"Model listing error: {e}")
        raise HTTPException(status_code=500, detail="Failed to list models")


@router.post("/reload")
async def reload_models():
    """
    Reload all models from MLflow registry.

    This endpoint can be used to refresh the model cache after
    new models have been trained and registered in MLflow.
    """
    try:
        # Cleanup and reinitialize
        await model_service.cleanup()
        await model_service.initialize()

        health_status = await model_service.get_health_status()

        return {
            "message": "Models reloaded successfully",
            "status": health_status["status"],
            "loaded_models": health_status.get("loaded_models", "0/0")
        }

    except Exception as e:
        logger.error(f"Model reload error: {e}")
        raise HTTPException(status_code=500, detail="Failed to reload models")


def _get_algorithm_type(model_name: str) -> str:
    """Get the algorithm type from model name."""
    if "rf" in model_name or "random_forest" in model_name:
        return "random_forest"
    elif "logreg" in model_name:
        return "logistic_regression"
    elif "bayes" in model_name:
        return "bayesian_logistic_regression"
    else:
        return "unknown"
