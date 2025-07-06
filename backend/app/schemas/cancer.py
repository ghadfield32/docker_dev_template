"""Breast cancer dataset schema definitions."""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field, field_validator

from .common import PredictionResponse


class CancerFeatures(BaseModel):
    """Breast cancer features in the correct order for sklearn breast cancer dataset."""
    # Using a list to match the sklearn breast cancer dataset exactly
    values: List[float] = Field(
        ...,
        min_items=30,
        max_items=30,
        description="30 features from breast cancer dataset"
    )

    @field_validator("values")
    @classmethod
    def validate_feature_count(cls, v: List[float]) -> List[float]:
        """Validate exactly 30 features."""
        if len(v) != 30:
            raise ValueError("Must provide exactly 30 features")
        return v

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "values": [
                    17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
                    1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
                    25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
                ]
            }
        }


class CancerPredictRequest(BaseModel):
    """Cancer prediction request schema."""
    model_type: str = Field("bayes", description="Model type: 'bayes', 'logreg', or 'rf'")
    samples: List[CancerFeatures] = Field(..., description="List of cancer feature vectors")
    posterior_samples: Optional[int] = Field(
        None,
        ge=10,
        le=10000,
        description="Number of posterior samples for Bayesian model"
    )

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "model_type": "bayes",
                "samples": [
                    {
                        "values": [
                            17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
                            1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
                            25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
                        ]
                    }
                ],
                "posterior_samples": 100
            }
        }


class CancerPredictResponse(PredictionResponse):
    """Cancer prediction response schema."""
    class_names: List[str] = Field(
        default=["malignant", "benign"],
        description="Cancer class names"
    )
    predicted_classes: Optional[List[str]] = Field(
        None,
        description="Predicted class names"
    )
    posterior_samples: Optional[int] = Field(
        None,
        description="Number of posterior samples used"
    )

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "predictions": [0.85, 0.23],
                "predicted_classes": ["malignant", "benign"],
                "class_names": ["malignant", "benign"],
                "model_used": "breast_cancer_bayes",
                "model_version": "1",
                "uncertainty": [
                    {"lower": 0.78, "upper": 0.91},
                    {"lower": 0.18, "upper": 0.29}
                ],
                "posterior_samples": 100
            }
        }


class CancerTrainingRequest(BaseModel):
    """Cancer model training request schema."""
    model_type: str = Field("bayes", description="Model type: 'bayes', 'logreg', or 'rf'")
    hyperparameters: Optional[Dict] = Field(None, description="Model hyperparameters")

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "model_type": "bayes",
                "hyperparameters": {
                    "draws": 1000,
                    "tune": 500,
                    "target_accept": 0.9
                }
            }
        }
