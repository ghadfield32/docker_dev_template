"""Simple FastAPI test server for testing integration."""

from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import socket
import time
import uvicorn
import random
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Test API",
    description="Simple test API for ML predictions",
    version="1.0.0"
)

# Create API router for v1 endpoints
api_router = APIRouter(prefix="/api/v1")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class IrisPredictRequest(BaseModel):
    rows: List[IrisFeatures]

class CancerFeatures(BaseModel):
    values: List[float]

class CancerPredictRequest(BaseModel):
    rows: List[CancerFeatures]
    posterior_samples: Optional[int] = None

class PredictResponse(BaseModel):
    predictions: List[float]
    model_used: str
    uncertainty: Optional[List[Dict[str, float]]] = None

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "ML Test API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/health"
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "ml-test-api",
        "models": {
            "iris_rf": {"loaded": True, "status": "mock"},
            "cancer_bayes": {"loaded": True, "status": "mock"}
        }
    }

@app.head("/health", include_in_schema=False)
async def health_head():
    """HEAD variant for wait-on and health probes."""
    # HEAD responses must not include a body - just return 200 OK
    from fastapi.responses import Response
    return Response(status_code=200)

@api_router.post("/iris/predict", response_model=PredictResponse)
async def predict_iris(request: IrisPredictRequest):
    """Make iris predictions."""
    logger.info(f"Received iris prediction request for {len(request.rows)} samples")

    # Mock prediction logic
    predictions = []
    for row in request.rows:
        # Simple mock logic: if petal_length < 2, predict setosa (0), else non-setosa (1)
        pred = 0.0 if row.petal_length < 2.0 else 1.0
        predictions.append(pred)

    return PredictResponse(
        predictions=predictions,
        model_used="iris_random_forest_mock"
    )

@api_router.post("/cancer/predict", response_model=PredictResponse)
async def predict_cancer(request: CancerPredictRequest):
    """Make cancer predictions."""
    logger.info(f"Received cancer prediction request for {len(request.rows)} samples")

    # Mock prediction logic
    predictions = []
    uncertainty = []

    for row in request.rows:
        # Mock logic: use mean of first 5 features as probability
        values = row.values[:5] if len(row.values) >= 5 else row.values
        prob = min(max(sum(values) / len(values) / 50.0, 0.0), 1.0)  # Normalize to 0-1
        predictions.append(prob)

        if request.posterior_samples:
            uncertainty.append({
                "lower": max(prob - 0.1, 0.0),
                "upper": min(prob + 0.1, 1.0)
            })

    return PredictResponse(
        predictions=predictions,
        model_used="cancer_bayes_mock",
        uncertainty=uncertainty if request.posterior_samples else None
    )

@api_router.get("/models/info")
async def model_info():
    """Get model information."""
    return {
        "iris_random_forest": {
            "version": "1.0",
            "status": "loaded",
            "accuracy": 0.95,
            "type": "classification"
        },
        "cancer_bayes": {
            "version": "1.0",
            "status": "loaded",
            "accuracy": 0.93,
            "type": "bayesian_classification"
        }
    }

@api_router.get("/iris/sample")
async def iris_sample():
    """Get sample iris data."""
    return {
        "sample": {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        },
        "description": "Sample Setosa flower measurements"
    }

@api_router.get("/cancer/sample")
async def cancer_sample():
    """Get sample cancer data."""
    return {
        "sample": {
            "values": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
                      1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
                      25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
        },
        "description": "Sample breast cancer features (30 dimensions)"
    }

# Add health endpoint at /api/v1/health to match frontend expectations
@api_router.get("/health")
async def api_v1_health():
    """Health check endpoint for /api/v1/health."""
    return {
        "status": "healthy",
        "service": "ml-test-api",
        "models": {
            "iris_rf": {"loaded": True, "status": "mock"},
            "cancer_bayes": {"loaded": True, "status": "mock"}
        }
    }

@api_router.head("/health", include_in_schema=False)
async def api_v1_health_head():
    """HEAD variant for /api/v1/health probes."""
    from fastapi.responses import Response
    return Response(status_code=200)

# Add missing endpoints that frontend expects
@api_router.get("/models/health")
async def models_health():
    """Health check endpoint for /api/v1/models/health."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "models": {
            "iris_random_forest": {
                "name": "iris_random_forest",
                "version": "1.0",
                "status": "loaded",
                "accuracy": 0.95,
                "created_at": "2024-01-01T00:00:00Z",
                "run_id": "mock_run_123"
            },
            "breast_cancer_bayes": {
                "name": "breast_cancer_bayes",
                "version": "1.0",
                "status": "loaded",
                "accuracy": 0.93,
                "created_at": "2024-01-01T00:00:00Z",
                "run_id": "mock_run_456"
            }
        },
        "mlflow_uri": "file:./mlruns_local",
        "loaded_models": "2/2"
    }



@api_router.get("/models/list")
async def models_list():
    """List all available models."""
    return {
        "models": {
            "iris_random_forest": {
                "name": "iris_random_forest",
                "status": "loaded",
                "version": "1.0",
                "accuracy": 0.95,
                "dataset": "iris",
                "algorithm": "random_forest"
            },
            "breast_cancer_bayes": {
                "name": "breast_cancer_bayes",
                "status": "loaded",
                "version": "1.0",
                "accuracy": 0.93,
                "dataset": "breast_cancer",
                "algorithm": "bayesian_logistic_regression"
            }
        },
        "total_models": 2,
        "loaded_models": 2,
        "available_datasets": ["iris", "breast_cancer"],
        "available_algorithms": ["random_forest", "logistic_regression", "bayesian_logistic_regression"]
    }

# Add retraining endpoints with mock implementations
# Global metrics store for retraining
_METRICS = {
    "iris_random_forest": {
        "accuracy": 0.95,
        "f1_macro": 0.94,
        "precision_macro": 0.93,
        "recall_macro": 0.95,
        "version": "1.0",
        "run_id": "mock_run_123"
    },
    "breast_cancer_bayes": {
        "accuracy": 0.93,
        "f1_macro": 0.92,
        "precision_macro": 0.91,
        "recall_macro": 0.93,
        "version": "1.0",
        "run_id": "mock_run_456"
    }
}

@api_router.post("/iris/retrain")
async def iris_retrain(body: dict):
    """Mock retrain for Iris models."""
    n_trials = body.get("hyperparameters", {}).get("n_trials", 50)

    # Simulate training time
    await asyncio.sleep(3)

    # Generate new mock metrics
    new_accuracy = round(random.uniform(0.92, 0.97), 3)
    _METRICS["iris_random_forest"] = {
        "accuracy": new_accuracy,
        "f1_macro": round(new_accuracy - 0.01, 3),
        "precision_macro": round(new_accuracy - 0.02, 3),
        "recall_macro": round(new_accuracy - 0.01, 3),
        "version": "2.0",
        "run_id": f"mock_run_{random.randint(1000, 9999)}"
    }

    return {
        "status": "started",
        "detail": f"Iris retrain running in background with {n_trials} trials",
        "model_types": ["rf", "logreg"],
        "estimated_time": f"{n_trials * 2} seconds"
    }

@api_router.post("/cancer/retrain")
async def cancer_retrain(body: dict):
    """Mock retrain for Cancer models."""
    params = body.get("hyperparameters", {})
    draws = params.get("draws", 800)
    tune = params.get("tune", 400)
    target_accept = params.get("target_accept", 0.9)

    # Simulate training time
    await asyncio.sleep(5)

    # Generate new mock metrics
    new_accuracy = round(random.uniform(0.89, 0.94), 3)
    _METRICS["breast_cancer_bayes"] = {
        "accuracy": new_accuracy,
        "f1_macro": round(new_accuracy - 0.01, 3),
        "precision_macro": round(new_accuracy - 0.02, 3),
        "recall_macro": round(new_accuracy - 0.01, 3),
        "version": "2.0",
        "run_id": f"mock_run_{random.randint(1000, 9999)}"
    }

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

# Update the metrics endpoint to use the global store
@api_router.get("/models/metrics")
async def models_metrics():
    """Get model metrics endpoint."""
    return _METRICS

# Include the API router
app.include_router(api_router)

# Add an additional health endpoint at /api/health for frontend proxy
@app.get("/api/health")
async def api_health():
    """Health check endpoint for API proxy."""
    return {
        "status": "healthy",
        "service": "ml-test-api",
        "models": {
            "iris_rf": {"loaded": True, "status": "mock"},
            "cancer_bayes": {"loaded": True, "status": "mock"}
        }
    }

@app.head("/api/health", include_in_schema=False)
async def api_health_head():
    """HEAD variant for API health probes."""
    from fastapi.responses import Response
    return Response(status_code=200)

if __name__ == "__main__":
    def is_port_in_use(port):
        """Check if a port is in use."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('0.0.0.0', port))
                return False
            except OSError:
                return True

    # Check if port is in use and wait if necessary
    port = 8000
    max_retries = 5
    retry_delay = 2

    for attempt in range(max_retries):
        if is_port_in_use(port):
            if attempt < max_retries - 1:
                print(f"⚠️  Port {port} is in use, waiting {retry_delay}s before retry {attempt + 1}/{max_retries}")
                time.sleep(retry_delay)
                continue
            else:
                print(f"❌ Port {port} is still in use after {max_retries} attempts. Please check for other processes.")
                exit(1)
        else:
            break

    print(f"🚀 Starting FastAPI server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
