"""MLflow logging utilities for metrics, parameters, and artifacts."""
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional
from matplotlib.figure import Figure
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def log_model_metrics(y_true, y_pred, prefix: str = "") -> Dict[str, float]:
    """
    Calculate and log model evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        prefix: Optional prefix for metric names
        
    Returns:
        Dictionary of calculated metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average='macro'),
        "recall_macro": recall_score(y_true, y_pred, average='macro'),
        "f1_macro": f1_score(y_true, y_pred, average='macro')
    }
    
    # Add prefix if provided
    if prefix:
        metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
    
    # Log metrics to MLflow
    mlflow.log_metrics(metrics)
    
    return metrics


def _log_fig(fig: Figure, artifact_name: str) -> None:
    """Log a Matplotlib figure directly without temp files."""
    mlflow.log_figure(fig, artifact_file=artifact_name)
    plt.close(fig)


def log_confusion_matrix(y_true, y_pred, *, class_names=None,
                         artifact_name: str = "confusion_matrix.png"):
    """Create + log confusion matrix using mlflow.log_figure."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set(xlabel="Predicted", ylabel="Actual", title="Confusion Matrix")
    _log_fig(fig, artifact_name)


def log_feature_importance(feature_names: list, importances: list,
                           artifact_name: str = "feature_importance.png"):
    """Bar plot logged via mlflow.log_figure (no disk I/O)."""
    imp_df = (pd.DataFrame({"feature": feature_names,
                            "importance": importances})
              .sort_values("importance"))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=imp_df, x="importance", y="feature", ax=ax)
    ax.set_title("Feature Importances")
    _log_fig(fig, artifact_name)


def log_parameters(params: Dict[str, Any]) -> None:
    """
    Log parameters to MLflow.
    
    Args:
        params: Dictionary of parameter names and values
    """
    mlflow.log_params(params)


def log_dataset_info(X_train, X_test, y_train, y_test) -> None:
    """
    Log dataset information as parameters.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
    """
    dataset_params = {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "n_features": (X_train.shape[1] if hasattr(X_train, 'shape') 
                       else len(X_train[0])),
        "n_classes": (len(set(y_train)) if hasattr(y_train, '__iter__') 
                      else 1)
    }
    
    log_parameters(dataset_params) 
