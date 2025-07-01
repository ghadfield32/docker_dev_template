
"""Training utilities with MLflow integration."""
import mlflow
import optuna
from typing import Optional
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from .config import RANDOM_STATE, TEST_SIZE
from .experiment_utils import setup_mlflow_experiment

# Re-export for convenience
__all__ = ['setup_mlflow_experiment', 'load_and_prepare_iris_data',
           'train_logistic_regression', 'train_random_forest_with_optimization']
from .logging import (
    log_model_metrics,
    log_confusion_matrix,
    log_feature_importance,
    log_dataset_info,
    log_parameters
)


def load_and_prepare_iris_data(test_size: float = TEST_SIZE,
                               random_state: int = RANDOM_STATE):
    """
    Load and prepare the Iris dataset.
    
    Args:
        test_size: Fraction of data to use for testing
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (X_train_scaled, X_test_scaled, y_train, y_test, 
                 feature_names, target_names, scaler)
    """
    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return (X_train_scaled, X_test_scaled, y_train, y_test,
            iris.feature_names, iris.target_names, scaler)

