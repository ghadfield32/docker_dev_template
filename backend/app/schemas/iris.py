"""Iris dataset schema definitions."""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field

from .common import PredictionResponse


class IrisFeatures(BaseModel):
    """Single Iris flower measurement."""
    sepal_length: float = Field(..., ge=4.0, le=8.0, description="Sepal length in cm")
    sepal_width: float = Field(..., ge=2.0, le=4.5, description="Sepal width in cm")
    petal_length: float = Field(..., ge=1.0, le=7.0, description="Petal length in cm")
    petal_width: float = Field(..., ge=0.1, le=2.5, description="Petal width in cm")

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }


class IrisPredictRequest(BaseModel):
    """Iris prediction request schema."""
    model_type: str = Field("rf", description="Model type: 'rf' or 'logreg'")
    samples: List[IrisFeatures] = Field(
        ...,
        alias="rows",  # Allow 'rows' as an alias for backward compatibility
        description="List of iris measurements"
    )

    class Config:
        """Pydantic config."""
        populate_by_name = True  # Enable alias support
        json_schema_extra = {
            "example": {
                "model_type": "rf",
                "samples": [
                    {
                        "sepal_length": 5.1,
                        "sepal_width": 3.5,
                        "petal_length": 1.4,
                        "petal_width": 0.2
                    }
                ]
            }
        }


class IrisPredictResponse(PredictionResponse):
    """Iris prediction response schema."""
    class_names: List[str] = Field(
        default=["setosa", "versicolor", "virginica"],
        description="Iris class names"
    )
    predicted_classes: Optional[List[str]] = Field(
        None,
        description="Predicted class names"
    )

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "predictions": [0, 2, 1],
                "predicted_classes": ["setosa", "virginica", "versicolor"],
                "class_names": ["setosa", "versicolor", "virginica"],
                "model_used": "iris_random_forest",
                "model_version": "1"
            }
        }


class IrisTrainingRequest(BaseModel):
    """Iris model training request schema."""
    model_type: str = Field("rf", description="Model type: 'rf' or 'logreg'")
    hyperparameters: Optional[Dict] = Field(None, description="Model hyperparameters")

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "model_type": "rf",
                "hyperparameters": {
                    "n_estimators": 100,
                    "max_depth": 5,
                    "random_state": 42
                }
            }
        }
