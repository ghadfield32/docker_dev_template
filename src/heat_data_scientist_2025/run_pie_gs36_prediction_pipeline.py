#!/usr/bin/env python3
"""
PIE & GS36 Prediction Pipeline (function-first)
- Uses train_eval_predict_and_compare() per target
- Persists predictions parquet for downstream leaderboards
- Preserves permutation importance & leaderboard comparisons
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.heat_data_scientist_2025.utils.config import CFG
from src.heat_data_scientist_2025.data.load_data_utils import load_data_optimized
from src.heat_data_scientist_2025.data.feature_engineering import engineer_features
from src.heat_data_scientist_2025.ml.enhanced_ml_pipeline import (
    train_eval_predict_and_compare,
    save_artifacts,  # updated signature (accepts predictions_df)
)

# --------- small local helper (keeps original GS36 behavior available) ---------

def _ensure_game_score_per36(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute game_score_per36 if missing, matching kaggle_pull math.
    """
    if "game_score_per36" in df.columns:
        return df

    required = [
        "total_points","total_fgm","total_fga","total_fta","total_ftm",
        "total_reb_off","total_reb_def","total_steals","total_assists",
        "total_blocks","total_pf","total_tov","total_minutes"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        # Don’t hard-fail; just return as-is and let downstream warn
        print(f"[WARN] Cannot compute game_score_per36; missing: {missing}")
        return df

    s = (
        df["total_points"]
        + 0.4 * df["total_fgm"]
        - 0.7 * df["total_fga"]
        - 0.4 * (df["total_fta"] - df["total_ftm"])
        + 0.7 * df["total_reb_off"]
        + 0.3 * df["total_reb_def"]
        + df["total_steals"]
        + 0.7 * df["total_assists"]
        + 0.7 * df["total_blocks"]
        - 0.4 * df["total_pf"]
        - df["total_tov"]
    )
    df = df.copy()
    df["game_score_per36"] = np.where(df["total_minutes"] > 0, s * 36.0 / df["total_minutes"], np.nan)
    return df

# --------- summary writer ---------

def _save_summary(results_per_target: Dict[str, Dict], simple_lb: Dict, comp_lb: Dict]) -> None:
    out = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "prediction_year": 2025,
        "targets": list(results_per_target.keys()),
        "eval": {t: r.get("eval", {}) for t, r in results_per_target.items()},
        "feature_importance_top": {
            t: (
                None if r.get("permutation_importance") is None or r["permutation_importance"].empty
                else {
                    "feature": r["permutation_importance"].iloc[0]["feature"],
                    "importance_mean": float(r["permutation_importance"].iloc[0]["importance_mean"]),
                }
            )
            for t, r in results_per_target.items()
        },
        "simple_leaderboards_written": list(simple_lb.keys()),
        "comprehensive_leaderboards_written": list(comp_lb.keys()),
    }
    CFG.ml_evaluation_dir.mkdir(parents=True, exist_ok=True)
    p = CFG.ml_evaluation_dir / "pipeline_summary.json"
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[SAVE] Summary → {p}")

# --------- main ---------

def main():
    print("🚀 PIE & GS36 Prediction Pipeline (functions, no classes)")
    CFG.ensure_ml_dirs()

    # 1) load + engineer
    print("\n[1/4] Loading engineered dataset…")
    df = load_data_optimized(CFG.ml_dataset_path, drop_null_rows=False, debug=False)
    df = engineer_features(df, drop_null_lag_rows=True, verbose=True)
    df = _ensure_game_score_per36(df)

    # 2) train/eval/predict per target
    print("\n[2/4] Training, evaluating, and predicting…")
    results_per_target: Dict[str, Dict] = {}
    for target in ("season_pie", "game_score_per36"):
        print(f"\n→ Target: {target}")
        res = train_eval_predict_and_compare(
            df_eng=df,
            target=target,
            test_years=[2023, 2024],
            predict_year=2025,
        )
        results_per_target[target] = res

        # 3) persist artifacts (feature importance, predictions parquet, predicted leaderboards)
        print("[3/4] Saving artifacts…")
        save_artifacts(
            target=target,
            model=res["model"],
            importance_df=res["permutation_importance"],
            preds_leaderboards=res["predicted_leaderboards"],
            predict_year=2025,
            predictions_df=res["predictions_df"],   # <- NEW: write predictions parquet
        )

    # 4) leaderboards from saved predictions + combined comparison
    print("\n[4/4] Creating leaderboards from saved predictions…")
    from src.heat_data_scientist_2025.ml.leaderboard_compare import (
        create_simple_leaderboards_from_predictions,
        build_leaderboards_with_predictions,
    )

    simple_lb = create_simple_leaderboards_from_predictions(
        metrics=("game_score_per36", "season_pie"),
        prediction_year=2025,
        top_n=50,
        verbose=True,
    )

    comp_lb = build_leaderboards_with_predictions(
        metrics=("game_score_per36", "season_pie"),
        prediction_year=2025,
        save=True,
        verbose=True,
    )

    # summary
    _save_summary(results_per_target, simple_lb, comp_lb)
    print("\n✅ Pipeline complete.")
    return {
        "results_per_target": results_per_target,
        "simple_leaderboards": simple_lb,
        "comprehensive_leaderboards": comp_lb,
    }


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unhandled error: {e}")
        sys.exit(1)
