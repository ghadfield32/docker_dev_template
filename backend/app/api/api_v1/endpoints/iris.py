"""Iris prediction endpoints."""

import logging
from typing import List
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import ValidationError
from ....schemas.iris import (
    IrisPredictRequest,
    IrisPredictResponse,
    IrisTrainingRequest
)
from ....services.ml.model_service import model_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/predict", response_model=IrisPredictResponse)
async def predict_iris(request: IrisPredictRequest):
    """
    Make predictions on iris flowers.

    - **model_type**: Choose between 'rf' (Random Forest) or 'logreg' (Logistic Regression)
    - **samples**: List of iris measurements (sepal/petal length and width)
    """
    try:
        # Convert samples to dict format
        samples_dict = [sample.dict() for sample in request.samples]

        # Make prediction
        result = await model_service.predict_iris(
            samples=samples_dict,
            model_type=request.model_type
        )

        return IrisPredictResponse(**result)

    except ValidationError as e:
        # Log validation errors with details
        logger.warning(f"Validation error in iris predict: {e.errors()}")
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Invalid input data",
                "errors": e.errors(),
                "valid_ranges": {
                    "sepal_length": "4.0-8.0 cm",
                    "sepal_width": "2.0-4.5 cm",
                    "petal_length": "1.0-7.0 cm",
                    "petal_width": "0.1-2.5 cm"
                }
            }
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Iris prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@router.get("/models")
async def get_iris_models():
    """Get available iris models and their information."""
    try:
        health_status = await model_service.get_health_status()
        iris_models = {
            k: v for k, v in health_status["models"].items()
            if "iris" in k
        }

        return {
            "available_models": iris_models,
            "model_types": ["rf", "logreg"],
            "description": {
                "rf": "Random Forest - Ensemble method with high accuracy",
                "logreg": "Logistic Regression - Fast linear model"
            }
        }

    except Exception as e:
        logger.error(f"Error getting iris models: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model information")


@router.get("/sample-data")
async def get_sample_data():
    """Get sample iris data for testing predictions."""
    sample_data = [
        {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
            "expected_class": "setosa"
        },
        {
            "sepal_length": 7.0,
            "sepal_width": 3.2,
            "petal_length": 4.7,
            "petal_width": 1.4,
            "expected_class": "versicolor"
        },
        {
            "sepal_length": 6.3,
            "sepal_width": 3.3,
            "petal_length": 6.0,
            "petal_width": 2.5,
            "expected_class": "virginica"
        }
    ]

    return {
        "samples": sample_data,
        "description": "Sample iris data for testing predictions",
        "usage": "Use the first 4 fields (sepal_length, sepal_width, petal_length, petal_width) for predictions"
    }


@router.post("/retrain")
async def retrain_iris(
    req: IrisTrainingRequest,
    background: BackgroundTasks,
):
    """
    Trigger asynchronous Optuna retrain for Iris models.
    Returns immediately; client can poll /models/metrics for progress.
    """
    try:
        params = req.hyperparameters or {}
        n_trials = params.get("n_trials", 50)

        def _background_job():
            """Background task to run the retraining."""
            import asyncio
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(model_service.retrain_iris(n_trials=n_trials))
            except RuntimeError as e:
                if "run_all_trainings not available" in str(e):
                    logger.info("Training module not available in this deployment")
                else:
                    logger.error(f"Background iris retrain failed: {e}")
            except Exception as e:
                logger.error(f"Background iris retrain failed: {e}")
            finally:
                loop.close()

        background.add_task(_background_job)

        return {
            "status": "started",
            "detail": f"Iris retrain running in background with {n_trials} trials",
            "model_types": ["rf", "logreg"],
            "estimated_time": f"{n_trials * 2} seconds"
        }

    except Exception as e:
        logger.error(f"Failed to start iris retrain: {e}")
        raise HTTPException(status_code=500, detail=f"Retrain failed: {str(e)}")
