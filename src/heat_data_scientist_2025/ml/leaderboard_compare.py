"""
 Leaderboard comparison module for basketball statistics
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
import numpy as np
import pandas as pd

from src.heat_data_scientist_2025.utils.config import CFG, ML_CONFIG


def _season_str(start_year: int) -> str:
    """Convert year to season format like 2024 -> '2024-25'"""
    return f"{start_year}-{str((start_year + 1) % 100).zfill(2)}"


def _as_float(x):
    """Convert input to float(s) safely."""
    import numpy as _np
    import pandas as _pd

    if isinstance(x, _pd.Series):
        return _pd.to_numeric(x, errors="coerce").astype(float)

    if isinstance(x, (list, tuple, _np.ndarray)):
        return _pd.to_numeric(_pd.Series(x), errors="coerce").astype(float)

    if x is None or (isinstance(x, float) and _np.isnan(x)):
        return _np.nan
    try:
        return float(x)
    except Exception:
        return _np.nan


def _find_prediction_column(df: pd.DataFrame, metric: str, verbose: bool = True) -> str:
    """Find the prediction column in dataframe, trying common naming patterns"""
    candidates = [
        metric,                           
        f"{metric}_pred",                 
        f"predicted_{metric}",            
        f"pred_{metric}",                 
    ]

    for candidate in candidates:
        if candidate in df.columns:
            if verbose:
                print(f"Found prediction column: '{candidate}' for {metric}")
            return candidate

    # Fallback to any column with 'pred' in the name
    pred_cols = [c for c in df.columns if 'pred' in c.lower() and pd.api.types.is_numeric_dtype(df[c])]
    if len(pred_cols) == 1:
        if verbose:
            print(f"Using fallback column: '{pred_cols[0]}' for {metric}")
        return pred_cols[0]
    elif len(pred_cols) > 1:
        raise KeyError(f"Multiple prediction columns found for {metric}: {pred_cols}")

    raise KeyError(f"No prediction column found for {metric}. Available: {list(df.columns)}")


def _load_hist_minimal(metric: str, minutes_gate: int = 200, verbose: bool = True) -> pd.DataFrame:
    """
    FIXED: Load historical data with more lenient filtering
    Changed minutes_gate from 500 to 200 to include more players
    """
    need_cols = ["player_name", "season", "games_played", "total_minutes", metric]
    df = pd.read_parquet(CFG.ml_dataset_path)

    if verbose:
        print(f"[_load_hist_minimal] Initial data: {len(df)} rows")

    # Keep only seasons up to 2024
    start_year = df["season"].astype(str).str.extract(r"^(\d{4})")[0].astype(int)
    df = df.loc[start_year <= 2024].copy()

    if verbose:
        print(f"[_load_hist_minimal] After year filter (<=2024): {len(df)} rows")

    # Calculate game_score_per36 if missing
    if metric == "game_score_per36" and "game_score_per36" not in df.columns:
        required_for_gs = {
            "total_points", "total_fgm", "total_fga", "total_fta", "total_ftm",
            "total_reb_off", "total_reb_def", "total_steals", "total_assists",
            "total_blocks", "total_pf", "total_tov", "total_minutes"
        }
        if required_for_gs.issubset(df.columns):
            if verbose:
                print(f"Computing {metric} from component stats")
            game_score_total = (
                df["total_points"] + 0.4 * df["total_fgm"] - 0.7 * df["total_fga"]
                - 0.4 * (df["total_fta"] - df["total_ftm"]) + 0.7 * df["total_reb_off"]
                + 0.3 * df["total_reb_def"] + df["total_steals"] + 0.7 * df["total_assists"]
                + 0.7 * df["total_blocks"] - 0.4 * df["total_pf"] - df["total_tov"]
            )
            df["game_score_per36"] = np.where(
                df["total_minutes"] > 0, 
                game_score_total * 36.0 / df["total_minutes"], 
                np.nan
            )
        else:
            if verbose:
                print(f"Cannot compute {metric} - missing required columns")
            df[metric] = np.nan

    # Make sure numeric columns are actually numeric
    for c in ["games_played", "total_minutes", metric]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = _as_float(df[c])

    # FIXED: More lenient filtering
    df = df.loc[df["total_minutes"] >= minutes_gate].copy()
    result = df[need_cols].dropna(subset=[metric])

    if verbose:
        print(f"[_load_hist_minimal] After minutes filter (>={minutes_gate}): {len(df)} rows")
        print(f"[_load_hist_minimal] After dropna on {metric}: {len(result)} rows")

    return result


def _load_predictions(metric: str, prediction_year: int, verbose: bool = True) -> pd.DataFrame:
    """FIXED: Load prediction data with better handling of missing columns"""
    pth = CFG.predictions_path(metric, prediction_year)
    if verbose:
        print(f"Loading predictions from {pth}")

    if not pth.exists():
        raise FileNotFoundError(f"Predictions file not found: {pth}")

    df = pd.read_parquet(pth)
    pred_col = _find_prediction_column(df, metric, verbose=verbose)

    if "player_name" not in df.columns:
        raise KeyError("Predictions missing player_name column")

    # FIXED: Handle missing columns more gracefully
    gp_col = _as_float(df["games_played"]) if "games_played" in df.columns else 65.0  # Reasonable default
    tm_col = _as_float(df["total_minutes"]) if "total_minutes" in df.columns else 2000.0  # Reasonable default

    # Build output with standard schema
    out = pd.DataFrame({
        "player_name": df["player_name"].astype(str),
        "season": _season_str(prediction_year),
        metric: _as_float(df[pred_col]),
        "games_played": gp_col,
        "total_minutes": tm_col,
        "source": "pred"
    })

    # Remove null predictions
    before_len = len(out)
    out = out.dropna(subset=[metric])
    if verbose and len(out) != before_len:
        print(f"Removed {before_len - len(out)} null predictions")

    if verbose:
        print(f"[_load_predictions] Final predictions: {len(out)} rows")

    return out


def _rank_three_buckets(df: pd.DataFrame, metric: str, top_n: int = 10, 
                        middle_n: int = 10, bottom_n: int = 10, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """FIXED: Create top/middle/bottom rankings with better debugging"""

    if verbose:
        print(f"[_rank_three_buckets] Input data: {len(df)} rows")
        print(f"[_rank_three_buckets] Columns: {list(df.columns)}")

    use = df.loc[df[metric].notna()].copy()

    if verbose:
        print(f"[_rank_three_buckets] After notna filter: {len(use)} rows")
        if len(use) > 0:
            print(f"[_rank_three_buckets] {metric} range: {use[metric].min():.6f} to {use[metric].max():.6f}")

    if len(use) == 0:
        if verbose:
            print(f"[_rank_three_buckets] WARNING: No valid data after filtering!")
        return {"top": pd.DataFrame(), "middle": pd.DataFrame(), "bottom": pd.DataFrame()}

    # Fill missing values for tie-breaking
    for col in ["total_minutes", "games_played"]:
        if col not in use.columns:
            use[col] = 0.0
        else:
            use[col] = _as_float(use[col]).fillna(0.0)

    # Top players (highest values)
    top = (
        use.sort_values([metric, "total_minutes", "games_played", "player_name"],
                        ascending=[False, False, False, True], kind="stable")
           .head(top_n).copy()
    )

    # Bottom players (lowest values)
    bottom = (
        use.sort_values([metric, "total_minutes", "games_played", "player_name"],
                        ascending=[True, False, False, True], kind="stable")
           .head(bottom_n).copy()
    )

    # Middle players (closest to median)
    median_val = use[metric].median(skipna=True)
    use_middle = use.copy()
    use_middle["__dist_to_median"] = (use_middle[metric] - median_val).abs()
    middle = (
        use_middle.sort_values(["__dist_to_median", metric, "total_minutes", "games_played", "player_name"],
                               ascending=[True, False, False, False, True], kind="stable")
                  .head(middle_n)
                  .drop(columns=["__dist_to_median"]).copy()
    )

    if verbose:
        print(f"[_rank_three_buckets] Results - Top: {len(top)}, Middle: {len(middle)}, Bottom: {len(bottom)}")

    return {"top": top, "middle": middle, "bottom": bottom}


def _generate_closest_predictions(combined_df: pd.DataFrame, boards: Dict[str, pd.DataFrame], 
                                  metric: str, prediction_year: int, k: int = 10) -> Dict[str, pd.DataFrame]:
    """Generate lists of predictions that just missed making each leaderboard"""
    season_tag = _season_str(prediction_year)
    predictions_only = combined_df.loc[combined_df["source"] == "pred"].copy()

    closest = {}

    for bucket_name, board_df in boards.items():
        if board_df.empty:
            closest[f"closest_{bucket_name}"] = pd.DataFrame()
            continue

        # Find predictions that didn't make this board
        board_players = set(zip(board_df["player_name"], board_df["season"]))
        missed_preds = predictions_only[
            ~predictions_only.apply(lambda r: (r["player_name"], r["season"]) in board_players, axis=1)
        ].copy()

        if bucket_name == "top":
            cutoff_value = board_df[metric].min() if not board_df.empty else float('-inf')
            missed_preds["gap_to_cutoff"] = cutoff_value - missed_preds[metric] 
            closest_missed = (
                missed_preds.sort_values(["gap_to_cutoff", metric], ascending=[True, False])
                           .head(k).copy()
            )

        elif bucket_name == "bottom":
            cutoff_value = board_df[metric].max() if not board_df.empty else float('inf')
            missed_preds["gap_to_cutoff"] = missed_preds[metric] - cutoff_value
            closest_missed = (
                missed_preds.sort_values(["gap_to_cutoff", metric], ascending=[True, True])
                           .head(k).copy()
            )

        else:  # middle
            median_val = combined_df[metric].median(skipna=True)
            missed_preds["gap_to_median"] = (missed_preds[metric] - median_val).abs()
            closest_missed = (
                missed_preds.sort_values(["gap_to_median", metric], ascending=[True, False])
                           .head(k).copy()
            )

        closest_missed["Notes"] = f"Closest {season_tag} prediction to {bucket_name}"
        closest[f"closest_{bucket_name}"] = closest_missed.reset_index(drop=True)

    return closest


def _build_new_boards(hist_df: pd.DataFrame, preds_df: pd.DataFrame,
                      metric: str, prediction_year: int, verbose: bool = True) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """FIXED: Combine historical data with predictions to create leaderboards"""

    if verbose:
        print(f"[_build_new_boards] Historical: {len(hist_df)} rows")
        print(f"[_build_new_boards] Predictions: {len(preds_df)} rows")

    # Make sure both dataframes have same structure
    hist_df = hist_df.copy()
    hist_df["source"] = "historical"

    preds_df = preds_df.copy()
    for col in ["games_played", "total_minutes"]:
        if col not in hist_df.columns:
            hist_df[col] = np.nan
        if col not in preds_df.columns: 
            preds_df[col] = np.nan

    # Combine datasets
    combined = pd.concat([hist_df, preds_df], ignore_index=True, sort=False)

    if verbose:
        print(f"[_build_new_boards] Combined: {len(combined)} rows")
        print(f"[_build_new_boards] Predictions in combined: {(combined['source'] == 'pred').sum()}")

    # Create rankings
    boards = _rank_three_buckets(combined, metric, verbose=verbose)

    # Add rank numbers and notes for predictions
    season_tag = _season_str(prediction_year)
    for bucket_name, board_df in boards.items():
        if board_df.empty:
            continue

        board_df = board_df.copy()
        board_df.insert(0, "Rank", range(1, len(board_df) + 1))

        is_prediction = (board_df.get("source", "historical") == "pred") | (board_df["season"] == season_tag)
        board_df["Notes"] = np.where(is_prediction, f"NEW {season_tag} prediction", "")

        boards[bucket_name] = board_df

        if verbose:
            pred_count = is_prediction.sum()
            print(f"[_build_new_boards] {bucket_name}: {pred_count}/10 are 2025 predictions")

    # Find predictions that just missed each board
    closest = _generate_closest_predictions(combined, boards, metric, prediction_year)

    return boards, closest


def build_leaderboards_with_predictions(metrics: Tuple[str, ...] = ("game_score_per36", "season_pie"),
                                        prediction_year: int = 2025,
                                        save: bool = True, 
                                        verbose: bool = True) -> Dict[str, Dict[str, pd.DataFrame]]:
    """FIXED: Main function to build comprehensive leaderboards with predictions"""
    if verbose:
        print(f"\nBuilding leaderboards for {metrics} with {prediction_year} predictions")

    results = {}
    output_dir = CFG.ml_predictions_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric in metrics:
        if verbose:
            print(f"\nProcessing {metric}")

        try:
            # Load data with more lenient filtering
            hist_df = _load_hist_minimal(metric, minutes_gate=200, verbose=verbose)  # FIXED: Lower threshold
            pred_df = _load_predictions(metric, prediction_year, verbose=verbose)

            if verbose:
                print(f"Historical data: {len(hist_df):,} player-seasons")
                print(f"Predictions: {len(pred_df):,} players")

            # Build leaderboards
            boards, closest = _build_new_boards(hist_df, pred_df, metric, prediction_year, verbose=verbose)
            results[metric] = {"boards": boards, "closest": closest}

            if save:
                # Save main leaderboards
                for bucket_name, board_df in boards.items():
                    board_path = output_dir / f"{metric}_{bucket_name}_leaderboard_{prediction_year}_with_predictions.csv"
                    board_df.to_csv(board_path, index=False)
                    if verbose:
                        print(f"Saved {bucket_name} leaderboard: {board_path.name} ({len(board_df)} rows)")

                # Save closest miss lists  
                closest_path = output_dir / f"{metric}_closest_misses_{prediction_year}.csv"
                if closest:
                    closest_combined = pd.concat(
                        closest.values(), 
                        keys=list(closest.keys())
                    ).reset_index(level=0).rename(columns={"level_0": "category"})
                    closest_combined.to_csv(closest_path, index=False)
                    if verbose:
                        print(f"Saved closest misses: {closest_path.name}")

        except Exception as e:
            print(f"Error processing {metric}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    if verbose:
        print(f"\nCompleted leaderboard generation for {len(results)} metrics")

    return results


def create_simple_leaderboards_from_predictions(metrics: Tuple[str, ...] = ("game_score_per36", "season_pie"),
                                                prediction_year: int = 2025,
                                                top_n: int = 50,
                                                verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """Create simple leaderboards from prediction files (unchanged - this works)"""
    if verbose:
        print(f"\nCreating simple leaderboards for {prediction_year}")

    CFG.ensure_ml_dirs()
    results = {}

    for metric in metrics:
        if verbose:
            print(f"\nProcessing {metric}")

        try:
            pred_path = CFG.predictions_path(metric, prediction_year)
            if not pred_path.exists():
                print(f"Predictions not found: {pred_path}")
                continue

            preds_df = pd.read_parquet(pred_path)
            if verbose:
                print(f"Loaded {len(preds_df):,} predictions")

            pred_col = _find_prediction_column(preds_df, metric, verbose=verbose)

            leaderboard_cols = ["player_name", pred_col]
            if any(c not in preds_df.columns for c in leaderboard_cols):
                print(f"Missing required columns for {metric}")
                continue

            leaderboard_df = preds_df[leaderboard_cols].copy()
            leaderboard_df = leaderboard_df.dropna(subset=[pred_col])

            if "season" not in leaderboard_df.columns:
                leaderboard_df["season"] = _season_str(prediction_year)

            leaderboard_df = leaderboard_df.sort_values(
                [pred_col, "player_name"], 
                ascending=[False, True]
            ).head(top_n).reset_index(drop=True)

            leaderboard_df = leaderboard_df.rename(columns={pred_col: metric})
            leaderboard_df.insert(0, "rank", range(1, len(leaderboard_df) + 1))
            leaderboard_df[metric] = leaderboard_df[metric].round(6)

            lb_path = CFG.leaderboard_path(metric, prediction_year)
            leaderboard_df.to_csv(lb_path, index=False)

            if verbose:
                print(f"Saved: {lb_path.name}")
                print(f"Top 3: {leaderboard_df.head(3)['player_name'].tolist()}")

            results[metric] = leaderboard_df

        except Exception as e:
            print(f"Error with {metric}: {str(e)}")
            continue

    if verbose:
        print(f"\nCreated {len(results)} simple leaderboards")

    return results
