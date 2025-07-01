"""Training utilities with MLflow integration."""
import mlflow
from mlflow import sklearn  # type: ignore
from mlflow import models  # type: ignore
import optuna
from optuna.integration.mlflow import MLflowCallback
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Callable, cast, Any, Dict, NoReturn, TypeAlias
from numpy.typing import NDArray
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import Bunch

from .config import RANDOM_STATE, TEST_SIZE
from .experiment_utils import setup_mlflow_experiment
from .logging import (
    log_model_metrics,
    log_confusion_matrix,
    log_feature_importance,
    log_dataset_info,
    log_parameters
)
from .explainer import build_and_log_dashboard

# Type aliases for complex types
FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
DatasetTuple: TypeAlias = Tuple[FloatArray, FloatArray, IntArray, IntArray, List[str], List[str], StandardScaler]

def load_and_prepare_iris_data(
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
) -> DatasetTuple:
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
    iris: Any = load_iris()
    X: NDArray[np.float64] = cast(NDArray[np.float64], iris.data)
    y: NDArray[np.int64] = cast(NDArray[np.int64], iris.target)
    feature_names: List[str] = list(iris.feature_names)
    target_names: List[str] = list(iris.target_names)
    
    # Split data
    X_train: NDArray[np.float64]
    X_test: NDArray[np.float64]
    y_train: NDArray[np.int64]
    y_test: NDArray[np.int64]
    X_train, X_test, y_train, y_test = cast(
        Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]],
        train_test_split(X, y, test_size=test_size, random_state=random_state)
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled: NDArray[np.float64] = cast(NDArray[np.float64], scaler.fit_transform(X_train))
    X_test_scaled: NDArray[np.float64] = cast(NDArray[np.float64], scaler.transform(X_test))
    
    return (X_train_scaled, X_test_scaled, y_train, y_test,
            feature_names, target_names, scaler)


# === (A) Logistic-Regression baseline =====================================
def train_logistic_regression_autolog(
    X_train: FloatArray,
    y_train: IntArray,
    X_test: FloatArray,
    y_test: IntArray,
    feature_names: list[str],
    target_names: list[str],
    run_name: str = "lr_autolog",
    register: bool = True,
    dashboard: bool = False,
    dashboard_port: int | None = None,
) -> str:
    """
    Fit + evaluate a Logistic-Regression baseline.

    Key improvements
    ----------------
    • Manually logs `accuracy` (and friends) via `log_model_metrics`
      so downstream code can rely on the key.  
    • Keeps the robust signature / input_example logic.
    """
    setup_mlflow_experiment()
    mlflow.sklearn.autolog(log_models=True)

    with mlflow.start_run(run_name=run_name) as run:
        model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1_000).fit(
            X_train, y_train
        )

        # ── 1) manual metric logging ──────────────────────────────────────
        y_pred_test = model.predict(X_test)
        log_model_metrics(y_test, y_pred_test)              # <-- new
        log_confusion_matrix(y_test, y_pred_test, class_names=target_names)

        # ── 2) model artefact with signature ──────────────────────────────
        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
        sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="iris_logistic_regression" if register else None,
            signature=signature,
            input_example=X_test[:5],
        )

        mlflow.evaluate(
            model=f"runs:/{run.info.run_id}/model",
            data=X_test,
            targets=y_test,
            model_type="classifier",
            evaluator_config={"label_list": list(range(len(target_names)))},
        )

        # ── 3) optional dashboard ─────────────────────────────────────────
        if dashboard:
            X_test_df = pd.DataFrame(X_test, columns=feature_names)
            build_and_log_dashboard(
                model, X_test_df, y_test, labels=target_names,
                run=run, port=dashboard_port, serve=True
            )
        return run.info.run_id



def create_rf_objective(
    X_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    X_valid: NDArray[np.float64],
    y_valid: NDArray[np.int64],
) -> Callable[[optuna.trial.Trial], float]:
    """
    Optuna objective that returns validation accuracy – *no* MLflow calls here.
    All logging is delegated to Optuna's MLflowCallback.
    """
    def objective(trial: optuna.trial.Trial) -> float:
        params: Dict[str, Any] = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 200),
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "random_state": RANDOM_STATE,
        }
        clf = RandomForestClassifier(**params)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_valid)
        return accuracy_score(y_valid, preds)

    return objective


# === (B) Random-Forest with Optuna ========================================
def train_random_forest_with_optimization(
    X_train: FloatArray,
    y_train: IntArray,
    X_test: FloatArray,
    y_test: IntArray,
    feature_names: list[str],
    target_names: list[str],
    *,
    n_trials: int = 50,
    run_name: str = "rf_optimization",
    register: bool = True,
    dashboard: bool = False,
    dashboard_port: int | None = None,
) -> str:
    """
    Optuna hyper-parameter search + robust metric logging (now includes `accuracy`).
    """
    setup_mlflow_experiment()
    disable_flag = mlflow.sklearn.autolog(disable=True)

    try:
        with mlflow.start_run(run_name=run_name) as parent:
            log_dataset_info(X_train, X_test, y_train, y_test)

            study = optuna.create_study(direction="maximize")
            study.optimize(
                create_rf_objective(X_train, y_train, X_test, y_test),
                n_trials=n_trials,
                callbacks=[
                    MLflowCallback(
                        tracking_uri=mlflow.get_tracking_uri(),
                        metric_name="accuracy",
                        mlflow_kwargs={"nested": True},
                    )
                ],
            )

            best_model = RandomForestClassifier(**study.best_params).fit(X_train, y_train)

            # ── 1) manual metric logging ──────────────────────────────────
            y_pred_test = best_model.predict(X_test)
            log_model_metrics(y_test, y_pred_test)           # <-- new
            log_confusion_matrix(y_test, y_pred_test, class_names=target_names)
            log_feature_importance(feature_names, best_model.feature_importances_)

            mlflow.log_metric("best_accuracy", study.best_value)

            # ── 2) model artefact with signature ──────────────────────────
            signature = mlflow.models.infer_signature(X_train, best_model.predict(X_train))
            sklearn.log_model(
                best_model,
                artifact_path="model",
                registered_model_name="iris_random_forest" if register else None,
                signature=signature,
                input_example=X_test[:5],
            )

            mlflow.evaluate(
                model=f"runs:/{parent.info.run_id}/model",
                data=X_test,
                targets=y_test,
                model_type="classifier",
                evaluator_config={"label_list": list(range(len(target_names)))},
            )

            if dashboard:
                X_test_df = pd.DataFrame(X_test, columns=feature_names)
                build_and_log_dashboard(
                    best_model, X_test_df, y_test, labels=target_names,
                    run=parent, port=dashboard_port, serve=True
                )
            return parent.info.run_id
    finally:
        if not disable_flag:
            mlflow.sklearn.autolog(disable=False)





# === (C) Robust comparator ===============================================
def compare_models(
    experiment_name: Optional[str] = None,
    metric_key: str = "accuracy",
    maximize: bool = True,
) -> None:
    """
    Print the best run according to *metric_key* while gracefully
    falling-back to common alternates when the preferred key is missing.
    """
    from .experiment_utils import get_best_run

    fallback_keys = ["accuracy_score", "best_accuracy"]
    try:
        best = get_best_run(experiment_name, metric_key, maximize)
        rid = best["run_id"]

        # choose first key that exists
        score = best.get(f"metrics.{metric_key}")
        if score is None:
            for alt in fallback_keys:
                score = best.get(f"metrics.{alt}")
                if score is not None:
                    metric_key = alt
                    break

        model_type = best.get("params.model_type", "unknown")
        print(f"🏆 Best run: {rid}")
        print(f"📈 {metric_key}: {score if score is not None else 'N/A'}")
        print(f"🔖 Model type: {model_type}")
    except Exception as err:
        print(f"❌ Error comparing models: {err}")
