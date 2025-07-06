"""ML Model Service for loading and managing trained models."""

import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

from ...core.config import settings
from ...schemas.common import HealthResponse, ModelInfo

# Import existing ML utilities
# 🔧 Note: These backend.ML imports are for optional ML training modules
#     that may not be present in all deployments. They're kept as-is since
#     they're already wrapped in try-except and are optional.
try:
    from backend.ML.mlops.experiment_utils import setup_mlflow_experiment
    from backend.ML.mlops.training import run_all_trainings
    from backend.ML.mlops.training_bayes import train_bayes_logreg
except ImportError as e:
    logging.warning(f"Could not import existing ML modules: {e}")
    setup_mlflow_experiment = None
    run_all_trainings = None
    train_bayes_logreg = None

logger = logging.getLogger(__name__)


class ModelService:
    """Service for managing ML models with MLflow integration."""

    def __init__(self):
        """Initialize the model service."""
        self.models: Dict[str, mlflow.pyfunc.PyFuncModel] = {}
        self.model_info: Dict[str, Dict[str, Any]] = {}
        self.mlflow_client: Optional[MlflowClient] = None
        self.initialized = False

        # Model configurations
        self.model_configs = {
            "iris_random_forest": {
                "type": "classification",
                "dataset": "iris",
                "class_names": ["setosa", "versicolor", "virginica"]
            },
            "iris_logreg": {
                "type": "classification",
                "dataset": "iris",
                "class_names": ["setosa", "versicolor", "virginica"]
            },
            "breast_cancer_bayes": {
                "type": "classification",
                "dataset": "breast_cancer",
                "class_names": ["malignant", "benign"]
            }
        }

    async def initialize(self) -> None:
        """Initialize the model service."""
        if self.initialized:
            return

        logger.info("Initializing ModelService...")

        try:
            # Set up MLflow
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            self.mlflow_client = MlflowClient(settings.MLFLOW_TRACKING_URI)
            logger.info(f"Connected to MLflow at {settings.MLFLOW_TRACKING_URI}")

            # Setup MLflow experiment
            if setup_mlflow_experiment:
                setup_mlflow_experiment(settings.MLFLOW_EXPERIMENT_NAME)

            # Load models
            await self._load_models()

            self.initialized = True
            logger.info("✅ ModelService initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize ModelService: {e}")
            raise

    async def cleanup(self) -> None:
        """Cleanup the model service."""
        logger.info("Cleaning up ModelService...")
        self.models.clear()
        self.model_info.clear()
        self.initialized = False

    async def _load_models(self) -> None:
        """Load all models from MLflow registry."""
        logger.info("Loading models from MLflow registry...")

        for model_name in self.model_configs.keys():
            try:
                model = await self._load_production_model(model_name)
                if model:
                    self.models[model_name] = model
                    logger.info(f"✅ Loaded model: {model_name}")
                else:
                    logger.warning(f"⚠️ Model {model_name} not loaded")

            except Exception as e:
                logger.info(f"⚠️ Model {model_name} not loaded - will auto-train if enabled: {e}")

                # Auto-train if enabled
                if settings.DEV_AUTOTRAIN:
                    await self._auto_train_model(model_name)

    async def _load_production_model(self, model_name: str) -> Optional[mlflow.pyfunc.PyFuncModel]:
        """Load a model from the Production stage in MLflow registry."""
        try:
            # Try to get the latest Production version
            versions = self.mlflow_client.get_latest_versions(
                model_name, stages=["Production"]
            )

            if not versions:
                logger.info(f"No Production version found for model {model_name} - will auto-train if enabled")
                return None

            version = versions[0]
            model_uri = f"models:/{model_name}/Production"

            # Load the model
            model = mlflow.pyfunc.load_model(model_uri)

            # Store model info
            run = self.mlflow_client.get_run(version.run_id)
            self.model_info[model_name] = {
                "name": model_name,
                "version": version.version,
                "status": "loaded",
                "stage": version.current_stage,
                "run_id": version.run_id,
                "accuracy": run.data.metrics.get("accuracy"),
                "created_at": version.creation_timestamp,
                "metrics": run.data.metrics
            }

            return model

        except Exception as e:
            logger.info(f"Model {model_name} not yet in registry - skipping: {e}")
            return None

    async def _auto_train_model(self, model_name: str) -> None:
        """Auto-train a model if auto-training is enabled."""
        if not settings.DEV_AUTOTRAIN:
            return

        logger.info(f"Auto-training model {model_name}...")

        try:
            if model_name in ["iris_random_forest", "iris_logreg"] and run_all_trainings:
                # Train both iris models
                run_all_trainings(n_trials=10)  # Quick training for demo

            elif model_name == "breast_cancer_bayes" and train_bayes_logreg:
                # Train bayesian model
                train_bayes_logreg(draws=100, tune=50, register=True)

            # Promote to production
            await self._promote_latest_to_production(model_name)

            # Try to load again
            model = await self._load_production_model(model_name)
            if model:
                self.models[model_name] = model
                logger.info(f"✅ Auto-trained and loaded model: {model_name}")

        except Exception as e:
            logger.error(f"Auto-training failed for {model_name}: {e}")

    async def _promote_latest_to_production(self, model_name: str) -> None:
        """Promote the latest model version to Production."""
        try:
            # Get latest version from None stage
            latest_versions = self.mlflow_client.get_latest_versions(
                model_name, stages=["None"]
            )

            if latest_versions:
                version = latest_versions[0]
                self.mlflow_client.transition_model_version_stage(
                    name=model_name,
                    version=version.version,
                    stage="Production",
                    archive_existing_versions=True
                )
                logger.info(f"Promoted {model_name} v{version.version} to Production")

        except Exception as e:
            logger.error(f"Failed to promote {model_name}: {e}")

    async def predict_iris(
        self,
        samples: List[Dict[str, float]],
        model_type: str = "rf"
    ) -> Dict[str, Any]:
        """Make iris predictions."""
        model_name = f"iris_{model_type}"

        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")

        model = self.models[model_name]

        # Convert to DataFrame
        df = pd.DataFrame(samples)

        # Make predictions
        predictions = model.predict(df)

        # Convert to class names if predictions are numeric
        class_names = self.model_configs[model_name]["class_names"]
        if isinstance(predictions[0], (int, np.integer)):
            predicted_classes = [class_names[int(p)] for p in predictions]
        else:
            predicted_classes = [class_names[int(np.argmax(p))] for p in predictions]

        return {
            "predictions": predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions),
            "predicted_classes": predicted_classes,
            "class_names": class_names,
            "model_used": model_name,
            "model_version": self.model_info.get(model_name, {}).get("version")
        }

    async def predict_cancer(
        self,
        samples: List[List[float]],
        model_type: str = "bayes",
        posterior_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """Make cancer predictions."""
        model_name = f"breast_cancer_{model_type}" if model_type == "bayes" else f"cancer_{model_type}"

        # Handle different model naming conventions
        available_models = [k for k in self.models.keys() if "cancer" in k or "breast" in k]
        if model_name not in self.models and available_models:
            model_name = available_models[0]  # Use first available cancer model

        if model_name not in self.models:
            raise ValueError(f"Cancer model not available. Available: {list(self.models.keys())}")

        model = self.models[model_name]

        # Convert to DataFrame with proper feature names
        df = pd.DataFrame(samples)

        # Make predictions
        predictions = model.predict(df)

        # Handle uncertainty for Bayesian models
        uncertainty = None
        if posterior_samples and "bayes" in model_name:
            # Simple uncertainty estimation (could be improved)
            uncertainty = [
                {"lower": max(0, p - 0.1), "upper": min(1, p + 0.1)}
                for p in predictions
            ]

        # Convert to class names
        class_names = ["malignant", "benign"]
        predicted_classes = []

        for p in predictions:
            if isinstance(p, (int, np.integer)):
                predicted_classes.append(class_names[int(p)])
            else:
                # For probability predictions, use threshold
                predicted_classes.append(class_names[0] if p > 0.5 else class_names[1])

        return {
            "predictions": predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions),
            "predicted_classes": predicted_classes,
            "class_names": class_names,
            "model_used": model_name,
            "model_version": self.model_info.get(model_name, {}).get("version"),
            "uncertainty": uncertainty,
            "posterior_samples": posterior_samples if posterior_samples else None
        }

    async def get_health_status(self) -> Dict[str, Any]:
        """Get the health status of the service."""
        if not self.initialized:
            return {
                "status": "initializing",
                "version": settings.VERSION,
                "models": {},
                "mlflow_uri": settings.MLFLOW_TRACKING_URI
            }

        # Check model status
        model_status = {}
        for model_name, config in self.model_configs.items():
            is_loaded = model_name in self.models
            info = self.model_info.get(model_name, {})

            model_status[model_name] = ModelInfo(
                name=model_name,
                version=info.get("version"),
                status="loaded" if is_loaded else "not_loaded",
                accuracy=info.get("accuracy"),
                created_at=info.get("created_at"),
                run_id=info.get("run_id")
            )

        # Overall status
        loaded_models = sum(1 for name in self.model_configs.keys() if name in self.models)
        total_models = len(self.model_configs)

        if loaded_models == total_models:
            status = "healthy"
        elif loaded_models > 0:
            status = "degraded"
        else:
            status = "unhealthy"

        return {
            "status": status,
            "version": settings.VERSION,
            "models": {k: v.dict() for k, v in model_status.items()},
            "mlflow_uri": settings.MLFLOW_TRACKING_URI,
            "loaded_models": f"{loaded_models}/{total_models}"
        }

    async def get_model_metrics(self) -> Dict[str, Any]:
        """Get metrics for all loaded models."""
        metrics = {}

        for model_name in self.models.keys():
            info = self.model_info.get(model_name, {})
            model_metrics = info.get("metrics", {})

            metrics[model_name] = {
                "accuracy": model_metrics.get("accuracy"),
                "f1_macro": model_metrics.get("f1_macro"),
                "precision_macro": model_metrics.get("precision_macro"),
                "recall_macro": model_metrics.get("recall_macro"),
                "version": info.get("version"),
                "run_id": info.get("run_id")
            }

        return metrics

    # ---------- Retraining Methods -----------------------------------------
    async def retrain_iris(self, n_trials: int = 50) -> Dict[str, float]:
        """
        Fire-and-forget Optuna retrain for Iris RF + LR.
        Returns best metric dict for immediate response.
        """
        logger.info(f"Starting Iris retrain with {n_trials} trials")

        try:
            if run_all_trainings:
                run_all_trainings(n_trials=n_trials)
                await self._promote_latest_to_production("iris_random_forest")
                await self._promote_latest_to_production("iris_logreg")
                await self._load_models()  # hot-reload cache

                # Surface new metrics
                metrics = await self.get_model_metrics()
                iris_metrics = metrics.get("iris_random_forest", {})
                logger.info(f"Iris retrain completed. New accuracy: {iris_metrics.get('accuracy')}")
                return iris_metrics
            else:
                raise RuntimeError("run_all_trainings not available")

        except Exception as e:
            logger.error(f"Iris retrain failed: {e}")
            raise

    async def retrain_cancer_bayes(
        self,
        draws: int = 800,
        tune: int = 400,
        target_accept: float = 0.9,
    ) -> Dict[str, float]:
        """
        Launch PyMC retrain with custom MCMC params.
        """
        logger.info(f"Starting Cancer Bayesian retrain: draws={draws}, tune={tune}, target_accept={target_accept}")

        try:
            if train_bayes_logreg:
                train_bayes_logreg(
                    draws=draws,
                    tune=tune,
                    register=True,
                    target_accept=target_accept
                )
                await self._promote_latest_to_production("breast_cancer_bayes")
                await self._load_models()

                metrics = await self.get_model_metrics()
                cancer_metrics = metrics.get("breast_cancer_bayes", {})
                logger.info(f"Cancer retrain completed. New accuracy: {cancer_metrics.get('accuracy')}")
                return cancer_metrics
            else:
                raise RuntimeError("train_bayes_logreg not available")

        except Exception as e:
            logger.error(f"Cancer retrain failed: {e}")
            raise


# Global model service instance
model_service = ModelService()





