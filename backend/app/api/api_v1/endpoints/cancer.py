"""Cancer prediction endpoints."""

import logging
from typing import List
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from ....schemas.cancer import (
    CancerPredictRequest,
    CancerPredictResponse,
    CancerTrainingRequest
)
from ....services.ml.model_service import model_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/predict", response_model=CancerPredictResponse)
async def predict_cancer(request: CancerPredictRequest):
    """
    Make predictions on breast cancer data.

    - **model_type**: Choose between 'bayes' (Bayesian), 'logreg' (Logistic Regression), or 'rf' (Random Forest)
    - **samples**: List of cancer feature vectors (30 features each)
    - **posterior_samples**: Number of posterior samples for uncertainty estimation (Bayesian only)
    """
    try:
        # Convert samples to list format
        samples_list = [sample.values for sample in request.samples]

        # Make prediction
        result = await model_service.predict_cancer(
            samples=samples_list,
            model_type=request.model_type,
            posterior_samples=request.posterior_samples
        )

        return CancerPredictResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Cancer prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@router.get("/models")
async def get_cancer_models():
    """Get available cancer models and their information."""
    try:
        health_status = await model_service.get_health_status()
        cancer_models = {
            k: v for k, v in health_status["models"].items()
            if "cancer" in k or "breast" in k
        }

        return {
            "available_models": cancer_models,
            "model_types": ["bayes", "logreg", "rf"],
            "description": {
                "bayes": "Bayesian Logistic Regression - Provides uncertainty estimates",
                "logreg": "Logistic Regression - Fast linear model",
                "rf": "Random Forest - Ensemble method with high accuracy"
            }
        }

    except Exception as e:
        logger.error(f"Error getting cancer models: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model information")


@router.get("/sample-data")
async def get_sample_data():
    """Get sample cancer data for testing predictions."""
    # Sample from breast cancer dataset - malignant case
    malignant_sample = [
        17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
        1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
        25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
    ]

    # Sample from breast cancer dataset - benign case
    benign_sample = [
        13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766,
        0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023,
        15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259
    ]

    sample_data = [
        {
            "values": malignant_sample,
            "expected_class": "malignant",
            "description": "Sample features from malignant tumor"
        },
        {
            "values": benign_sample,
            "expected_class": "benign",
            "description": "Sample features from benign tumor"
        }
    ]

    return {
        "samples": sample_data,
        "description": "Sample breast cancer data for testing predictions",
        "feature_info": "30 features: mean, SE, and worst values for 10 measurements",
        "usage": "Use the 'values' array (30 floats) for predictions"
    }


@router.post("/retrain")
async def retrain_cancer(
    req: CancerTrainingRequest,
    background: BackgroundTasks,
):
    """
    Trigger Bayesian retrain with custom MCMC parameters.
    Returns immediately; client can poll /models/metrics for progress.
    """
    try:
        params = req.hyperparameters or {}
        draws = params.get("draws", 800)
        tune = params.get("tune", 400)
        target_accept = params.get("target_accept", 0.9)

        def _background_job():
            """Background task to run the retraining."""
            import asyncio
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(
                    model_service.retrain_cancer_bayes(
                        draws=draws,
                        tune=tune,
                        target_accept=target_accept
                    )
                )
            except Exception as e:
                logger.error(f"Background cancer retrain failed: {e}")
            finally:
                loop.close()

        background.add_task(_background_job)

        return {
            "status": "started",
            "detail": f"Cancer Bayesian retrain running in background",
            "parameters": {
                "draws": draws,
                "tune": tune,
                "target_accept": target_accept
            },
            "estimated_time": f"{draws + tune} seconds"
        }

    except Exception as e:
        logger.error(f"Failed to start cancer retrain: {e}")
        raise HTTPException(status_code=500, detail=f"Retrain failed: {str(e)}")
