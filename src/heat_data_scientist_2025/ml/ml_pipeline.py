"""
ML pipeline 
"""

from __future__ import annotations

# Standard library imports
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

# Local imports
from src.heat_data_scientist_2025.utils.config import CFG, ML_CONFIG
from src.heat_data_scientist_2025.data.feature_engineering import engineer_features

# Ordinal feature hierarchies
ORDINAL_ORDERS = {
    "minutes_tier": ["Bench", "Role Player", "Starter", "Star"],
}


# Convert fitted LabelEncoder to stable dictionaries
def _freeze_label_encoder(le: LabelEncoder) -> dict:
    classes = list(le.classes_)
    mapping = {cls: idx for idx, cls in enumerate(classes)}
    reverse = {idx: cls for idx, cls in enumerate(classes)}
    return {"forward": mapping, "reverse": reverse}


# Validate data contains only numeric columns for model training
def ensure_numeric_matrix(df_like: pd.DataFrame, context: str = "X") -> None:
    obj_cols = [c for c in df_like.columns if df_like[c].dtype == "object"]
    if obj_cols:
        examples = {c: df_like[c].dropna().astype(str).unique()[:5].tolist() for c in obj_cols}
        msg = [
            f"[ensure_numeric_matrix] {context} contains object/string columns:",
            f"  Columns: {obj_cols}",
            f"  Sample values: {examples}",
            "  -> Encode these before model training.",
        ]
        raise ValueError("\n".join(msg))


# Check for potential target leakage in feature names
def validate_target_feature_separation(target: str, feature_names: List[str], verbose: bool = True) -> bool:
    bad = [f for f in feature_names if f == target or (f.startswith(target + "_") and not f.endswith("_lag1"))]
    ok = len(bad) == 0
    if verbose and not ok:
        print(f"[leakage-check] Potential leakage for target '{target}': {bad}")
    return ok


# Build feature list excluding contemporaneous target
def create_target_specific_features(
    target_name: str,
    base_numerical_features: List[str],
    nominal_categoricals: List[str],
    ordinal_categoricals: List[str],
) -> List[str]:
    target_name = str(target_name).strip()
    exclusions = {target_name}
    safe_numerical = [f for f in base_numerical_features if f not in exclusions]
    return safe_numerical + nominal_categoricals + ordinal_categoricals


# Container for categorical encoders
@dataclass
class EncodersBundle:
    nominal_maps: dict
    ordinal_maps: dict
    raw_label_encoders: dict


# Encode categorical features with stable mappings
def encode_categoricals(
    df: pd.DataFrame,
    nominal_categoricals: List[str],
    ordinal_categoricals: List[str],
    strict: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, EncodersBundle]:
    out = df.copy()
    raw_label_encoders: dict[str, LabelEncoder] = {}
    nominal_maps: dict[str, dict] = {}
    ordinal_maps: dict[str, dict] = {}

    # Handle nominal categories
    for col in nominal_categoricals:
        if col not in out.columns:
            continue
        ser = out[col].astype("string").fillna("Unknown")
        le = LabelEncoder()
        le.fit(ser.to_numpy())
        raw_label_encoders[col] = le
        maps = _freeze_label_encoder(le)
        nominal_maps[col] = maps
        out[col] = ser.map(maps["forward"]).astype("Int32")
        if strict and out[col].isna().any():
            unseen = sorted(ser[out[col].isna()].dropna().unique().tolist())
            raise ValueError(f"Unknown nominal categories in '{col}': {unseen}")
        out[col] = out[col].fillna(-1).astype("int32")

    # Handle ordinal categories
    for col in ordinal_categoricals:
        if col not in out.columns:
            continue
        ser = out[col].astype("string").fillna("Unknown")
        order = ORDINAL_ORDERS.get(col)
        if order is None:
            raise ValueError(f"No explicit order for ordinal '{col}'. Add to ORDINAL_ORDERS.")
        if "Unknown" not in order:
            order = order + ["Unknown"]
        forward = {lvl: i for i, lvl in enumerate(order)}
        reverse = {i: lvl for lvl, i in forward.items()}
        ordinal_maps[col] = {"forward": forward, "reverse": reverse}

        out[col] = ser.map(forward).astype("Int16")
        if strict and out[col].isna().any():
            unseen = sorted(ser[out[col].isna()].dropna().unique().tolist())
            raise ValueError(f"Unknown ordinal categories in '{col}': {unseen}. Allowed={order}")
        out[col] = out[col].fillna(forward["Unknown"]).astype("int16")

    if verbose:
        print(f"Encoded: {len(nominal_maps)} nominal, {len(ordinal_maps)} ordinal columns")

    return out, EncodersBundle(nominal_maps, ordinal_maps, raw_label_encoders)


# Apply saved encoders to new data
def apply_encoders_to_frame(
    df: pd.DataFrame, encoders: EncodersBundle, strict: bool = True, verbose: bool = True
) -> pd.DataFrame:
    out = df.copy()

    for col, maps in encoders.nominal_maps.items():
        if col not in out.columns:
            continue
        ser = out[col].astype("string").fillna("Unknown")
        out[col] = ser.map(maps["forward"]).astype("Int32")
        if strict and out[col].isna().any():
            unseen = sorted(ser[out[col].isna()].dropna().unique().tolist())
            raise ValueError(f"Unknown nominal categories in '{col}': {unseen}")
        out[col] = out[col].fillna(-1).astype("int32")

    for col, maps in encoders.ordinal_maps.items():
        if col not in out.columns:
            continue
        ser = out[col].astype("string").fillna("Unknown")
        out[col] = ser.map(maps["forward"]).astype("Int16")
        if strict and out[col].isna().any():
            unseen = sorted(ser[out[col].isna()].dropna().unique().tolist())
            raise ValueError(f"Unknown ordinal categories in '{col}': {unseen}")
        out[col] = out[col].fillna(maps["forward"]["Unknown"]).astype("int16")

    return out


# Validate lag feature integrity across seasons
def audit_lag_feature_integrity(
    df: pd.DataFrame,
    person_col: str = "personId",
    season_col: str = "season_start_year",
    lag_pairs: List[Tuple[str, str]] = [("season_pie_lag1", "season_pie")],
    verbose: bool = True,
) -> None:
    if not {person_col, season_col}.issubset(df.columns):
        if verbose:
            print("Skipping lag audit: missing id/season columns")
        return

    g = df.sort_values([person_col, season_col]).groupby(person_col, group_keys=False)
    prev_year = g[season_col].shift(1)
    year_gap = df[season_col] - prev_year

    valid_prev = (g.cumcount() > 0) & (year_gap == 1)
    if verbose:
        total = int((g.cumcount() > 0).sum())
        good = int(valid_prev.sum())
        pct = (good / total * 100) if total else 100.0
        print(f"Consecutive season pairs: {good}/{total} ({pct:.1f}%)")

    for lag_col, base_col in lag_pairs:
        if lag_col not in df.columns or base_col not in df.columns:
            continue
        expected = g[base_col].shift(1)
        mism = (df[lag_col] != expected) & valid_prev
        mism_ct = int(mism.sum())
        if verbose:
            print(f"{lag_col} vs {base_col}: {mism_ct} mismatches")


# Comprehensive feature availability analysis
def validate_and_evaluate_features(
    df: pd.DataFrame,
    numerical_features: List[str],
    nominal_categoricals: List[str],
    ordinal_categoricals: List[str],
    y_variables: List[str],
    verbose: bool = True,
) -> Dict[str, Any]:
    results = {
        "numerical_features": {},
        "nominal_categoricals": {},
        "ordinal_categoricals": {},
        "y_variables": {},
        "missing_features": [],
        "available_features": [],
        "feature_completeness": {},
    }

    groups = [
        ("numerical_features", numerical_features),
        ("nominal_categoricals", nominal_categoricals),
        ("ordinal_categoricals", ordinal_categoricals),
        ("y_variables", y_variables),
    ]
    
    for feature_type, features in groups:
        available = [f for f in features if f in df.columns]
        missing = [f for f in features if f not in df.columns]

        results[feature_type]["available"] = available
        results[feature_type]["missing"] = missing
        results[feature_type]["availability_pct"] = (len(available) / len(features) * 100) if features else 100.0

        for feature in available:
            non_null_count = int(df[feature].notna().sum())
            total_count = int(len(df))
            completeness_pct = (non_null_count / total_count * 100.0) if total_count else 0.0
            results["feature_completeness"][feature] = {
                "non_null_count": non_null_count,
                "total_count": total_count,
                "completeness_pct": float(completeness_pct),
            }

        if verbose and features:
            pct = (len(available) / len(features) * 100.0)
            print(f"{feature_type.replace('_', ' ').title()}: {len(available)}/{len(features)} ({pct:.1f}%)")

    all_specified = numerical_features + nominal_categoricals + ordinal_categoricals + y_variables
    total_spec = len(all_specified)
    total_avail = len([f for f in all_specified if f in df.columns])
    
    results["overall"] = {
        "total_specified": int(total_spec),
        "total_available": int(total_avail),
        "availability_pct": float((total_avail / total_spec * 100.0) if total_spec else 100.0),
    }
    results["missing_features"] = [f for f in all_specified if f not in df.columns]
    results["available_features"] = [f for f in all_specified if f in df.columns]

    return results


# Calculate permutation importance for numerical features
def calculate_permutation_importance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_name: str,
    numerical_features: List[str],
    n_repeats: int = 10,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    available = [f for f in numerical_features if f in X_train.columns]
    if not available:
        return pd.DataFrame(columns=[
            "feature", "importance_mean", "importance_std", "target", "importance_lower", "importance_upper"
        ])

    Xtr = X_train[available].copy()
    Xte = X_test[available].copy()

    if Xtr.isna().any().any() or Xte.isna().any().any() or y_train.isna().any() or y_test.isna().any():
        raise ValueError("Unexpected nulls found in importance calculation data")

    base_model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
    base_model.fit(Xtr, y_train)

    perm = permutation_importance(
        base_model, Xte, y_test, n_repeats=n_repeats, random_state=random_state, scoring="r2"
    )

    df_imp = (
        pd.DataFrame(
            {
                "feature": available,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
                "target": target_name,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    df_imp["importance_lower"] = df_imp["importance_mean"] - 1.96 * df_imp["importance_std"]
    df_imp["importance_upper"] = df_imp["importance_mean"] + 1.96 * df_imp["importance_std"]

    if verbose and not df_imp.empty:
        print(f"Top 5 important features for {target_name}:")
        for i, (_, row) in enumerate(df_imp.head(5).iterrows(), start=1):
            print(f" {i:2d}. {row['feature']:<25} {row['importance_mean']:7.4f}")

    return df_imp


# Filter features by importance threshold
def filter_features_by_importance(
    importance_df: pd.DataFrame,
    min_importance: float = 0.001,
    max_features: Optional[int] = None,
    target_name: str = "unknown",
    verbose: bool = True,
) -> List[str]:
    if importance_df.empty:
        return []

    kept = importance_df[importance_df["importance_mean"] > min_importance].copy()
    if max_features and len(kept) > max_features:
        kept = kept.head(max_features)

    feature_names = kept["feature"].tolist()

    if verbose:
        total = len(importance_df)
        k = len(feature_names)
        print(f"Kept {k}/{total} features for {target_name} (threshold={min_importance})")

    return feature_names


# Create game score per 36 minutes feature
def create_game_score_per36_feature(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    
    req = [
        "total_points", "total_fgm", "total_fga", "total_fta", "total_ftm",
        "total_reb_off", "total_reb_def", "total_steals", "total_assists", 
        "total_blocks", "total_pf", "total_tov", "total_minutes",
    ]
    
    missing = [c for c in req if c not in out.columns]
    if missing:
        print(f"Missing game score columns {missing}; using season_pie proxy")
        out["game_score_per36"] = out.get("season_pie", 0.1) * 36.0
        return out, ["game_score_per36"]

    out["game_score_total"] = (
        out["total_points"]
        + 0.4 * out["total_fgm"]
        - 0.7 * out["total_fga"]
        - 0.4 * (out["total_fta"] - out["total_ftm"])
        + 0.7 * out["total_reb_off"]
        + 0.3 * out["total_reb_def"]
        + out["total_steals"]
        + 0.7 * out["total_assists"]
        + 0.7 * out["total_blocks"]
        - 0.4 * out["total_pf"]
        - out["total_tov"]
    )
    out["game_score_per36"] = np.where(
        out["total_minutes"] > 0, out["game_score_total"] * 36.0 / out["total_minutes"], 0.0
    )

    return out, ["game_score_total", "game_score_per36"]


# Build train/test datasets for multiple targets
def create_multi_target_datasets(
    df_engineered: pd.DataFrame,
    numerical_features: List[str],
    nominal_categoricals: List[str],
    ordinal_categoricals: List[str],
    y_variables: List[str],
    strategy: str = "filter_complete",
    test_seasons: Optional[List[int]] = None,
    season_col: str = "season_start_year",
    verbose: bool = True,
) -> Dict[str, Any]:
    if verbose:
        print(f"Creating datasets for {len(y_variables)} targets using {strategy} strategy")

    # Audit lag features
    try:
        audit_lag_feature_integrity(df_engineered, verbose=verbose)
    except Exception as e:
        print(f"Lag audit error: {e}")

    # Encode categoricals once for all targets
    df_processed, enc_bundle = encode_categoricals(
        df_engineered, nominal_categoricals, ordinal_categoricals, verbose=verbose
    )

    if test_seasons is None:
        test_seasons = ML_CONFIG.TEST_YEARS

    train_mask = ~df_processed[season_col].isin(test_seasons)
    test_mask = df_processed[season_col].isin(test_seasons)

    results = {
        "datasets": {},
        "encoders": enc_bundle,
        "label_encoders": enc_bundle.raw_label_encoders,
        "train_seasons": sorted(df_processed[train_mask][season_col].unique()),
        "test_seasons": sorted(df_processed[test_mask][season_col].unique()),
    }

    for target in y_variables:
        if target not in df_processed.columns:
            if verbose:
                print(f"Target '{target}' not found; skipping")
            continue

        target_features = create_target_specific_features(
            target, numerical_features, nominal_categoricals, ordinal_categoricals
        )
        available_features = [f for f in target_features if f in df_processed.columns]

        if not validate_target_feature_separation(target, available_features, verbose):
            continue

        target_mask = df_processed[target].notna()
        if strategy == "filter_complete":
            feat_mask = df_processed[available_features].notna().all(axis=1)
            full_mask = target_mask & feat_mask
        else:
            full_mask = target_mask

        train_data = df_processed[train_mask & full_mask]
        test_data = df_processed[test_mask & full_mask]

        if len(train_data) == 0 or len(test_data) == 0:
            if verbose:
                print(f"Insufficient data for {target} (train={len(train_data)}, test={len(test_data)})")
            continue

        X_train = train_data[available_features].copy()
        y_train = train_data[target].copy()
        X_test = test_data[available_features].copy()
        y_test = test_data[target].copy()

        results["datasets"][target] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "feature_names": available_features,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "target_name": target,
        }

        if verbose:
            print(f"{target}: train={len(X_train)}, test={len(X_test)} samples")

    return results


# Train models with importance-based feature selection
def train_multi_target_models(
    datasets: Dict[str, Any],
    numerical_features: List[str],
    importance_threshold: float = 0.001,
    max_features_per_target: Optional[int] = None,
    n_importance_repeats: int = 10,
    verbose: bool = True,
) -> Dict[str, Any]:
    results = {
        "models": {},
        "importance_scores": {},
        "filtered_features": {},
        "evaluation_metrics": {},
    }

    for target_name, data in datasets["datasets"].items():
        if verbose:
            print(f"\nTraining model for {target_name}")

        X_train, y_train = data["X_train"], data["y_train"]
        X_test, y_test = data["X_test"], data["y_test"]
        feature_names = data["feature_names"]

        target_numerical = [f for f in numerical_features if f in feature_names]

        # Train initial model
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        # Calculate importance
        importance_df = calculate_permutation_importance(
            X_train, y_train, X_test, y_test, target_name, target_numerical, 
            n_repeats=n_importance_repeats, verbose=verbose
        )
        results["importance_scores"][target_name] = importance_df

        if importance_df.empty:
            results["models"][target_name] = model
            results["filtered_features"][target_name] = feature_names
            y_pred = model.predict(X_test)
            results["evaluation_metrics"][target_name] = {
                "r2": r2_score(y_test, y_pred),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "mae": mean_absolute_error(y_test, y_pred),
            }
            continue

        # Filter by importance
        important_features = filter_features_by_importance(
            importance_df, importance_threshold, max_features_per_target, target_name, verbose
        )
        categorical_features = [f for f in feature_names if f not in numerical_features 
                              and f in X_train.columns and f != "prediction_season"]
        final_features = important_features + categorical_features

        if not final_features:
            if verbose:
                print(f"No features passed threshold for {target_name}")
            continue

        # Train final model
        X_train_f = X_train[final_features]
        X_test_f = X_test[final_features]

        final_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        final_model.fit(X_train_f, y_train)

        y_pred = final_model.predict(X_test_f)
        metrics = {
            "r2": r2_score(y_test, y_pred),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae": mean_absolute_error(y_test, y_pred),
        }

        results["models"][target_name] = final_model
        results["filtered_features"][target_name] = final_features
        results["evaluation_metrics"][target_name] = metrics

        if verbose:
            print(f"Final R²: {metrics['r2']:.3f}, Features: {len(final_features)}")

    return results


# Save importance results and model metrics
def save_feature_importance_results(results: Dict[str, Any], output_dir: Path, verbose: bool = True) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for target_name, importance_df in results["importance_scores"].items():
        if not importance_df.empty:
            p = out / f"{target_name}_permutation_importance.csv"
            importance_df.to_csv(p, index=False)
            if verbose:
                print(f"Saved {target_name} importance to {p}")

    filtered_summary = {
        t: {"features": feats, "count": len(feats)} for t, feats in results["filtered_features"].items()
    }
    (out / "filtered_features_summary.json").write_text(json.dumps(filtered_summary, indent=2))

    (out / "model_evaluation_metrics.json").write_text(
        json.dumps(results["evaluation_metrics"], indent=2, default=str)
    )
    
    if verbose:
        print(f"Saved summaries to {out}")


# Print final model performance summary
def print_final_results(results: Dict[str, Any], verbose: bool = True) -> None:
    if not verbose:
        return

    print("\nFINAL MODEL RESULTS")
    print("-" * 30)

    for target_name in results["models"].keys():
        print(f"\n{target_name.upper()}")
        
        if target_name in results["evaluation_metrics"]:
            m = results["evaluation_metrics"][target_name]
            print(f"  R²: {m['r2']:.4f}, RMSE: {m['rmse']:.4f}, MAE: {m['mae']:.4f}")

        if target_name in results["filtered_features"]:
            feats = results["filtered_features"][target_name]
            print(f"  Features: {len(feats)}")


# Generate predictions for future seasons
def generate_and_save_predictions(
    df_engineered: pd.DataFrame,
    datasets: Dict[str, Any],
    model_results: Dict[str, Any],
    season_col: str = "season_start_year",
    id_cols: List[str] = ["personId", "player_name"],
    verbose: bool = True,
) -> Dict[str, Path]:
    enc_bundle: EncodersBundle = datasets.get("encoders")
    if enc_bundle is None:
        raise RuntimeError("Missing encoders; run create_multi_target_datasets() first")

    pred_year = ML_CONFIG.PREDICTION_YEAR
    source_year = ML_CONFIG.SOURCE_YEAR

    base = df_engineered.loc[df_engineered[season_col] == source_year].copy()
    if verbose:
        print(f"Generating predictions from {source_year} data: {len(base)} rows")

    saved_paths = {}

    for target, model in model_results.get("models", {}).items():
        final_feats = model_results["filtered_features"].get(target)
        if not final_feats:
            continue

        missing = [c for c in final_feats if c not in base.columns]
        if missing:
            raise KeyError(f"Missing features for {target}: {missing}")

        X_pred_raw = base[final_feats].copy()
        X_pred = apply_encoders_to_frame(X_pred_raw, enc_bundle, verbose=False)
        ensure_numeric_matrix(X_pred, context=f"X_pred ({target})")

        y_hat = model.predict(X_pred)

        pred_df = base[id_cols + [season_col]].copy()
        pred_df["prediction_season"] = int(pred_year)
        pred_df[f"{target}_pred"] = y_hat

        path = CFG.predictions_path(target, year=pred_year)
        path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_parquet(path, index=False)
        saved_paths[target] = path

        if verbose:
            print(f"Saved {target} predictions to {path}")

    return saved_paths


# Main pipeline orchestrator
class MLPipeline:
    """Multi-target ML pipeline with importance filtering and consistent encodings"""

    def __init__(
        self,
        numerical_features: List[str],
        nominal_categoricals: List[str],
        ordinal_categoricals: List[str],
        y_variables: List[str],
        importance_threshold: float = 0.001,
        max_features_per_target: Optional[int] = None,
        verbose: bool = True,
    ):
        self.numerical_features = numerical_features
        self.nominal_categoricals = nominal_categoricals
        self.ordinal_categoricals = ordinal_categoricals
        self.y_variables = y_variables
        self.importance_threshold = importance_threshold
        self.max_features_per_target = max_features_per_target
        self.verbose = verbose
        self.results = {}
        CFG.ensure_ml_dirs()

    # Run complete pipeline from feature validation to predictions
    def run_complete_pipeline(self, df_engineered: pd.DataFrame) -> Dict[str, Any]:
        if self.verbose:
            print("MULTI-TARGET ML PIPELINE")
            print(f"Targets: {len(self.y_variables)}")
            print(f"Features: {len(self.numerical_features)} numerical + {len(self.nominal_categoricals + self.ordinal_categoricals)} categorical")

        # Validate feature availability
        validation_results = validate_and_evaluate_features(
            df_engineered, self.numerical_features, self.nominal_categoricals,
            self.ordinal_categoricals, self.y_variables, self.verbose
        )

        # Create datasets
        datasets = create_multi_target_datasets(
            df_engineered, self.numerical_features, self.nominal_categoricals,
            self.ordinal_categoricals, self.y_variables, verbose=self.verbose
        )

        # Train models
        model_results = train_multi_target_models(
            datasets, self.numerical_features, self.importance_threshold,
            self.max_features_per_target, verbose=self.verbose
        )

        # Save results
        save_feature_importance_results(model_results, CFG.ml_evaluation_dir, self.verbose)
        saved_pred_paths = generate_and_save_predictions(
            df_engineered, datasets, model_results, verbose=self.verbose
        )
        print_final_results(model_results, self.verbose)

        self.results = {
            "feature_validation": validation_results,
            "datasets": datasets,
            "model_results": model_results,
            "saved_predictions": {k: str(v) for k, v in saved_pred_paths.items()},
            "config": {
                "numerical_features": self.numerical_features,
                "nominal_categoricals": self.nominal_categoricals,
                "ordinal_categoricals": self.ordinal_categoricals,
                "y_variables": self.y_variables,
                "importance_threshold": self.importance_threshold,
                "max_features_per_target": self.max_features_per_target,
            },
        }
        return self.results

# -----
# Example entrypoint (kept for parity; ok to remove in production scripts)
# -----
def run_enhanced_pipeline_example():
    """Example usage with your default feature lists; returns results dict."""

    numerical_features = [
        # lagged features
        "season_pie_lag1",
        "ts_pct_lag1",
        "efg_pct_lag1",
        "fg_pct_lag1",
        "fg3_pct_lag1",
        "ft_pct_lag1",
        "pts_per36_lag1",
        "ast_per36_lag1",
        "reb_per36_lag1",
        "defensive_per36_lag1",
        "production_per36_lag1",
        "stocks_per36_lag1",
        "three_point_rate_lag1",
        "ft_rate_lag1",
        "pts_per_shot_lag1",
        "ast_to_tov_lag1",
        "usage_events_per_min_lag1",
        "usage_per_min_lag1",
        "games_played_lag1",
        "total_minutes_lag1",
        "total_points_lag1",
        "total_assists_lag1",
        "total_rebounds_lag1",
        "total_steals_lag1",
        "total_blocks_lag1",
        "total_fga_lag1",
        "total_fta_lag1",
        "total_3pa_lag1",
        "total_3pm_lag1",
        "total_tov_lag1",
        "win_pct_lag1",
        "avg_plus_minus_lag1",
        "team_win_pct_final_lag1",
        "offensive_impact_lag1",
        "two_way_impact_lag1",
        "efficiency_volume_score_lag1",
        "versatility_score_lag1",
        "shooting_score_lag1",
    ]

    nominal_categoricals = ["prediction_season"]
    ordinal_categoricals = ["minutes_tier"]
    y_variables = ["season_pie", "game_score_per36"]

    from src.heat_data_scientist_2025.data.load_data_utils import load_data_optimized

    df = load_data_optimized(CFG.ml_dataset_path, drop_null_rows=True)
    df_engineered, _, _ = engineer_features(df, verbose=True)

    if "game_score_per36" in y_variables and "game_score_per36" not in df_engineered.columns:
        df_engineered, _ = create_game_score_per36_feature(df_engineered)

    pipeline = MLPipeline(
        numerical_features=numerical_features,
        nominal_categoricals=nominal_categoricals,
        ordinal_categoricals=ordinal_categoricals,
        y_variables=y_variables,
        importance_threshold=0.001,
        max_features_per_target=30,
        verbose=True,
    )
    results = pipeline.run_complete_pipeline(df_engineered)
    return results


if __name__ == "__main__":
    _results = run_enhanced_pipeline_example()
    print("\nEnhanced pipeline completed. Check output directories for artifacts.")
