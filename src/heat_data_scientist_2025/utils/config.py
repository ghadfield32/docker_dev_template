# src/heat_data_scientist_2025/utils/config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple, List, Dict, Optional
import difflib
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # allow overriding via .env or real env; keep your robust repo_root default
    repo_root: Path = Path(__file__).resolve().parents[3]
    heat_data_root: Path = Field(default=Path("data"), alias="HEAT_DATA_ROOT")

    model_config = SettingsConfigDict(
        env_file=".env",          # loads .env if present
        env_prefix="",            # we already use explicit aliases
        case_sensitive=False,     # friendlier on Windows
        extra="ignore",           # ignore unknown envs
    )

    @property
    def data_root(self) -> Path:
        return (self.repo_root / self.heat_data_root
                if not self.heat_data_root.is_absolute()
                else self.heat_data_root)


S = Settings()

_REPO_ROOT = S.repo_root
DATA_ROOT  = S.data_root

RAW_DIR =       DATA_ROOT / "raw" / "heat_data_scientist_2025"
PROCESSED_DIR = DATA_ROOT / "processed" / "heat_data_scientist_2025"
QUALITY_DIR =   DATA_ROOT / "quality" / "quality_reports"
ML_DATASET_PATH = PROCESSED_DIR / "nba_ml_dataset.parquet"
NBA_CATALOG_PATH = PROCESSED_DIR / "nba_data_catalog.md"
SQLITE_PATH = PROCESSED_DIR / "nba.sqlite"
RANKINGS_PATH = PROCESSED_DIR / "nba_rankings_results.txt"

# EDA needs
EDA_OUT_DIR = PROCESSED_DIR / "eda"

# ML Pipeline paths
ML_MODELS_DIR = PROCESSED_DIR / "models"
ML_PREDICTIONS_DIR = PROCESSED_DIR / "predictions"
ML_EVALUATION_DIR = PROCESSED_DIR / "evaluation"

# production yaml
PROJECT_ROOT: Path = Path("src")
COLUMN_SCHEMA_PATH: Path = PROJECT_ROOT / "heat_data_scientist_2025" / "data" / "column_schema.yaml"

# Project Settings:
start_season = 2009
end_season = 2024
final_top_data_amt = 10
season_type = 'Regular Season'
minutes_total_minimum_per_season = 500

# --- ML Pipeline Configuration ---
class MLPipelineConfig:
    """Configuration for ML Pipeline automation"""
    
    # Target and prediction settings
    TARGET_COLUMN = "season_pie"
    PREDICTION_YEAR = 2025
    SOURCE_YEAR = 2024  # Use 2024 data to predict 2025
    
    # Training settings
    TRAIN_START_YEAR = 2011  # First year with reliable lag features
    TEST_YEARS = [2023, 2024]  # Hold out for validation
    
    # Model settings
    DEFAULT_STRATEGY = "filter_complete"  # "filter_complete", "two_stage", "auto"
    RANDOM_STATE = 42
    
    # Required lag features for complete cases
    REQUIRED_LAG_FEATURES = [
        "season_pie",
        "pts_per36", "ast_per36", "reb_per36",
        "ts_pct", "efg_pct",
        "usage_events_per_min",
        "games_played", "total_minutes",
        "defensive_per36", "production_per36",
        "win_pct", "team_win_pct_final"
    ]
    
    # Feature engineering settings
    NULL_STRATEGY = "diagnose_only"
    CREATE_LAG_YEARS = [1]  # Create lag1 features
    
    # Model parameters
    MODEL_PARAMS = {
        'random_forest': {
            'n_estimators': 200,
            'max_depth': 12,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': RANDOM_STATE,
            'n_jobs': -1
        },
        'xgboost': {
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'random_state': RANDOM_STATE,
            'n_jobs': -1
        }
    }
    
    # Evaluation settings
    EVALUATION_METRICS = ['r2', 'rmse', 'mae', 'mape']
    TOP_N_PREDICTIONS = 50  # Top N players for leaderboard
    
    # Feature importance settings
    MIN_FEATURE_IMPORTANCE = 0.001
    TOP_FEATURES_COUNT = 20
    
    # Output settings
    SAVE_ENGINEERED_DATA = True
    SAVE_MODELS = True
    SAVE_PREDICTIONS = True
    SAVE_EVALUATION = True
    
    # Automation settings
    AUTO_FEATURE_SELECTION = True
    AUTO_HYPERPARAMETER_TUNING = False  # Set to True for automated tuning
    CROSS_VALIDATION_FOLDS = 5

# Initialize ML config
ML_CONFIG = MLPipelineConfig()

# --- Kaggle dataset handle ---
KAGGLE_DATASET = "eoinamoore/historical-nba-data-and-player-box-scores"

# --- Tables we actually care about right now ---
IMPORTANT_TABLES = ["PlayerStatistics", "TeamStatistics"]

# --- Table -> CSV filename mapping (simple & explicit) ---
KAGGLE_TABLE_TO_CSV = {
    "PlayerStatistics": "PlayerStatistics.csv",
    "TeamStatistics": "TeamStatistics.csv",
}
PARQUET_DIR = Path(DATA_ROOT) / "parquet_cache"

# Canonical ML export list (final parquet column order)
ML_EXPORT_COLUMNS = [
    # IDs & core season
    "personId", "player_name", "season",

    # playing time & outcomes
    "games_played", "total_minutes",
    "win_pct", "home_games_pct", "avg_plus_minus", "total_plus_minus",

    # efficiency & per-36
    "season_pie", "ts_pct", "fg_pct", "fg3_pct", "ft_pct",
    "pts_per36", "ast_per36", "reb_per36",
    "usage_per_min", "efficiency_per_game",

    # raw season totals
    "total_points", "total_assists", "total_rebounds",
    "total_steals", "total_blocks", "total_turnovers",
    "total_fgm", "total_fga", "total_ftm", "total_fta", "total_3pm", "total_3pa",

    # player bio & role (expanded to include everything you listed)
    "height", "bodyWeight", "draftYear", "draftRound", "draftNumber",
    "birthdate", "country", "position",

    # share-of-team season metrics
    "share_pts", "share_ast", "share_reb",
    "share_stl", "share_blk",
    "share_fga", "share_fgm",
    "share_3pa", "share_3pm",
    "share_fta", "share_ftm",
    "share_tov", "share_reb_off", "share_reb_def", "share_pf",
    "season_game_score_total", "game_score_per36",
]

def _first_existing(candidates: Iterable[Path]) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "None of the candidate files exist:\n" + "\n".join(str(c) for c in candidates)
    )


# training features
numerical_features = [
    "season_start_year",

    "season_pie_lag1", "ts_pct_lag1", "efg_pct_lag1", "fg_pct_lag1", "fg3_pct_lag1", "ft_pct_lag1",
    "pts_per36_lag1", "ast_per36_lag1", "reb_per36_lag1", "defensive_per36_lag1",
    "production_per36_lag1", "stocks_per36_lag1", "three_point_rate_lag1", "ft_rate_lag1",
    "pts_per_shot_lag1", "ast_to_tov_lag1", "usage_events_per_min_lag1", "usage_per_min_lag1",
    "games_played_lag1", "total_minutes_lag1", "total_points_lag1", "total_assists_lag1",
    "total_rebounds_lag1", "total_steals_lag1", "total_blocks_lag1", "total_fga_lag1",
    "total_fta_lag1", "total_3pa_lag1", "total_3pm_lag1", "total_tov_lag1", "win_pct_lag1",
    "avg_plus_minus_lag1", "team_win_pct_final_lag1",
    "offensive_impact_lag1", "two_way_impact_lag1", "efficiency_volume_score_lag1",
    "versatility_score_lag1", "shooting_score_lag1",


    "total_reb_off_lag1", "total_reb_def_lag1",
    "total_fgm_lag1", "total_ftm_lag1",
    "total_plus_minus_lag1",
    "wins_lag1", "home_games_lag1",
    "season_game_score_total_lag1", "game_score_per36_lag1",
    "season_uPER_lag1", "season_aPER_lag1", "season_PER_lag1",
    "season_BPM_lag1", "season_max_games_lag1", "season_VORP_lag1", "season_EWA_lag1",
    "ts_attempts_lag1",
    "usage_events_total_lag1", "total_usage_lag1",
    "shot_creation_lag1", "shot_creation_per36_lag1",
    "minutes_per_game_lag1",
    "games_pct_lag1",
    "home_games_pct_lag1",
    "fg_vs_league_lag1", "fg3_vs_league_lag1", "ft_vs_league_lag1",
    "portability_index_lag1", "pi_scoring_eff_lag1", "pi_shooting_lag1",
    "pi_defense_lag1", "pi_versatility_lag1", "pi_passing_lag1", "pi_usage_term_lag1",

    
    "fgm_per36_lag1", "fga_per36_lag1", "ftm_per36_lag1", "fta_per36_lag1",
    "oreb_per36_lag1", "dreb_per36_lag1", "stl_per36_lag1", "blk_per36_lag1",
    "pf_per36_lag1", "tov_per36_lag1",
]

nominal_categoricals = []
ordinal_categoricals = ["minutes_tier"] 
y_variables = ["season_pie", "game_score_per36", "season_PER", "season_EWA", "season_VORP"]



@dataclass(frozen=True)
class Paths:
    repo_root: Path = _REPO_ROOT
    data_root: Path = DATA_ROOT
    raw_dir: Path = RAW_DIR
    processed_dir: Path = PROCESSED_DIR
    quality_dir: Path = QUALITY_DIR
    ml_dataset_path: Path = ML_DATASET_PATH
    nba_catalog_path: Path = NBA_CATALOG_PATH
    sqlite_path: Path = SQLITE_PATH
    rankings_path: Path = RANKINGS_PATH
    eda_out_dir: Path = EDA_OUT_DIR
    column_schema_path: Path = COLUMN_SCHEMA_PATH
    
    # ML Pipeline paths
    ml_models_dir: Path = ML_MODELS_DIR
    ml_predictions_dir: Path = ML_PREDICTIONS_DIR
    ml_evaluation_dir: Path = ML_EVALUATION_DIR

    def ensure_ml_dirs(self) -> None:
        """Create ML pipeline directories if they don't exist."""
        self.ml_models_dir.mkdir(parents=True, exist_ok=True)
        self.ml_predictions_dir.mkdir(parents=True, exist_ok=True)
        self.ml_evaluation_dir.mkdir(parents=True, exist_ok=True)

    def csv(self, name: str) -> Path:
        """Return a path under RAW_DIR for an exact filename, with rich diagnostics on failure."""
        p = (self.raw_dir / name).resolve()
        if p.exists():
            return p

        raw_exists = self.raw_dir.exists()
        all_csvs = []
        if raw_exists:
            try:
                all_csvs = sorted([q.name for q in self.raw_dir.glob("*.csv")])
            except Exception:
                all_csvs = []

        suggestions = difflib.get_close_matches(name, all_csvs, n=5, cutoff=0.5) if all_csvs else []

        lines = [
            f"Expected CSV not found: {p}",
            f"RAW_DIR: {self.raw_dir}  (exists={raw_exists})",
            f"DATA_ROOT (override with HEAT_DATA_ROOT): {self.data_root}",
            f"CSV files found in RAW_DIR ({len(all_csvs)}): {all_csvs[:25]}{' ...' if len(all_csvs) > 25 else ''}",
        ]
        if suggestions:
            lines.append(f"Closest names: {suggestions}")
        lines.append("If your data lives elsewhere, set the environment variable HEAT_DATA_ROOT to that base folder.")
        raise FileNotFoundError("\n".join(lines))

    def csv_any(self, *names: str) -> Path:
        return _first_existing([(self.raw_dir / n).resolve() for n in names])

    # canonical asset getters (7 kaggle tables + optional league schedule)
    def playerstats_csv(self) -> Path:       return self.csv("PlayerStatistics.csv")
    def teamstats_csv(self) -> Path:         return self.csv("TeamStatistics.csv")
    def games_csv(self) -> Path:             return self.csv("Games.csv")
    def teams_csv(self) -> Path:             return self.csv("Teams.csv")
    def team_histories_csv(self) -> Path:    return self.csv_any("CoachHistory.csv", "Coaches.csv")
    def league_schedule_csv(self) -> Path:   return (self.raw_dir / "LeagueSchedule24_25.csv")

    # ML Pipeline file getters
    def model_path(self, model_name: str, year: int) -> Path:
        """Get path for saving/loading trained models."""
        return self.ml_models_dir / f"{model_name}_{year}_model.pkl"
    
    def predictions_path(self, target: str, year: int | None = None) -> Path:
        """
        Flexible getter for per-target predictions path.

        Accepts either:
        - (target, year) e.g., ("season_pie", 2025)
        - single string with trailing year, e.g., "season_pie_2025"
        - single string that already includes a *_predictions_YYYY tag, we'll normalize it

        Produces: .../predictions/{target}_predictions_{year}.parquet
        """
        self.ensure_ml_dirs()

        safe_target = str(target).strip().lower()

        # If year isn't separately provided, try to parse it from the 'target' string
        if year is None:
            # tolerate patterns like "season_pie_2025" or "season_pie_predictions_2025"
            import re
            m = re.search(r'(\d{4})$', safe_target)
            if m:
                year = int(m.group(1))
                # strip known suffixes to get the target core
                safe_target = re.sub(r'(_predictions)?_\d{4}$', '', safe_target)
            else:
                raise TypeError("predictions_path() requires 'year' or a target string ending with a 4-digit year.")

        return self.ml_predictions_dir / f"{safe_target}_predictions_{int(year)}.parquet"


    def leaderboard_path(self, target: str, year: int | None = None) -> Path:
        """
        Flexible getter for per-target leaderboard path.

        Accepts either:
        - (target, year) e.g., ("game_score_per36", 2025)
        - single string with trailing year, e.g., "game_score_per36_2025"
        - single string that already includes a *_leaderboard_YYYY tag, we'll normalize it

        Produces: .../predictions/{target}_leaderboard_{year}.csv
        """
        self.ensure_ml_dirs()

        safe_target = str(target).strip().lower()

        if year is None:
            import re
            m = re.search(r'(\d{4})$', safe_target)
            if m:
                year = int(m.group(1))
                safe_target = re.sub(r'(_leaderboard)?_\d{4}$', '', safe_target)
            else:
                raise TypeError("leaderboard_path() requires 'year' or a target string ending with a 4-digit year.")

        return self.ml_predictions_dir / f"{safe_target}_leaderboard_{int(year)}.csv"

    
    def evaluation_path(self, model_name: str, year: int) -> Path:
        """Get path for saving/loading evaluation results."""
        return self.ml_evaluation_dir / f"{model_name}_{year}_evaluation.json"
    
    def feature_importance_path(self, model_name: str, year: int) -> Path:
        """Get path for saving/loading feature importance."""
        return self.ml_evaluation_dir / f"{model_name}_{year}_feature_importance.csv"

CFG = Paths()

if __name__ == "__main__":
    print("Config paths:")
    print(f"Raw dir: {CFG.raw_dir}")
    print(f"Processed dir: {CFG.processed_dir}")
    print(f"ML models dir: {CFG.ml_models_dir}")
    print(f"ML predictions dir: {CFG.ml_predictions_dir}")
    print(f"ML evaluation dir: {CFG.ml_evaluation_dir}")
    
    print("\nML Config:")
    print(f"Target column: {ML_CONFIG.TARGET_COLUMN}")
    print(f"Prediction year: {ML_CONFIG.PREDICTION_YEAR}")
    print(f"Required lag features: {len(ML_CONFIG.REQUIRED_LAG_FEATURES)}")
