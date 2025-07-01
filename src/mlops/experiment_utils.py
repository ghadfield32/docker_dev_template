"""MLflow experiment utilities."""
import os
import pathlib
import mlflow
import mlflow.tracking
from typing import Optional, Dict, Any
import requests

from .config import EXPERIMENT_NAME, TRACKING_URI

_HEALTH_ENDPOINTS = ("/health", "/version")


def _ping_tracking_server(uri: str, timeout: float = 2.0) -> bool:
    """Return True iff an HTTP MLflow server is reachable at *uri*."""
    if not uri.startswith("http"):
        return False                        # file store – nothing to ping
    try:
        # Use new health endpoints
        for ep in _HEALTH_ENDPOINTS:
            response = requests.get(uri.rstrip("/") + ep, timeout=timeout)
            response.raise_for_status()
        return True
    except Exception:
        return False


def _fallback_uri() -> str:
    """Return a local file-based URI relative to the repo root."""
    root = pathlib.Path.cwd()
    return f"file:{root}/mlruns"


def setup_mlflow_experiment(experiment_name: Optional[str] = None) -> None:
    """
    Idempotently configure MLflow **and** guarantee the experiment exists.
    • If TRACKING_URI points at an HTTP server that is unreachable, we fall
      back to a local file store so the script never crashes.
    
    Args:
        experiment_name: Name of the experiment. If None, uses default.
    """
    # Use provided name or fall back to config default
    exp_name = experiment_name or EXPERIMENT_NAME
    
    uri = TRACKING_URI
    if not _ping_tracking_server(uri):
        local_uri = _fallback_uri()
        print(
            f"⚠️  MLflow server unreachable at {uri} – "
            f"falling back to local store {local_uri}"
        )
        uri = local_uri

    mlflow.set_tracking_uri(uri)

    if mlflow.get_experiment_by_name(exp_name) is None:
        mlflow.create_experiment(
            exp_name,
            artifact_location=os.getenv("MLFLOW_ARTIFACT_ROOT",
                                        _fallback_uri())
        )
    mlflow.set_experiment(exp_name)
    print(f"🗂  Using MLflow experiment '{exp_name}' @ {uri}")


def get_best_run(
    experiment_name: Optional[str] = None,
    metric_key: str = "accuracy",
    maximize: bool = True,
) -> Dict[str, Any]:
    """
    Return a *shallow* dict with run_id, metrics.*, and params.* keys
    so downstream code can use predictable dotted paths.
    """
    exp_name = experiment_name or EXPERIMENT_NAME
    setup_mlflow_experiment(exp_name)

    client = mlflow.tracking.MlflowClient()
    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        raise ValueError(f"Experiment '{exp_name}' not found")

    order = "DESC" if maximize else "ASC"
    run = client.search_runs(
        [exp.experiment_id],
        order_by=[f"metrics.{metric_key} {order}"],
        max_results=1,
    )[0]

    # Build a *flat* mapping -------------------------------------------------
    flat: Dict[str, Any] = {"run_id": run.info.run_id}

    # Metrics
    for k, v in run.data.metrics.items():
        flat[f"metrics.{k}"] = v

    # Params
    for k, v in run.data.params.items():
        flat[f"params.{k}"] = v

    # Tags (optional but handy)
    for k, v in run.data.tags.items():
        flat[f"tags.{k}"] = v

    return flat

