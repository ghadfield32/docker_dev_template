"""
Configuration module for NFL Kicker Analysis package.
Contains all constants, paths, and configuration parameters.
"""
from pathlib import Path
from typing import Dict, List, Tuple
import os

class Config:
    """Main configuration class for the NFL Kicker Analysis package."""
    
    # Data paths
    DATA_DIR = Path("data")
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    OUTPUT_DIR = Path("output")
    
    # Raw data files
    KICKERS_FILE = RAW_DATA_DIR / "kickers.csv"
    ATTEMPTS_FILE = RAW_DATA_DIR / "field_goal_attempts.csv"
    
    # Processed data files
    MODELING_DATA_FILE = PROCESSED_DATA_DIR / "field_goal_modeling_data.csv"
    LEADERBOARD_FILE = PROCESSED_DATA_DIR / "leaderboard.csv"
    
    # Analysis parameters
    MIN_DISTANCE = 18
    MAX_DISTANCE = 63
    MIN_KICKER_ATTEMPTS = 5
    
    # Distance profile for EPA calculation
    DISTANCE_PROFILE = [20, 25, 30, 35, 40, 45, 50, 55, 60]
    DISTANCE_WEIGHTS = [0.05, 0.10, 0.20, 0.20, 0.20, 0.15, 0.08, 0.02, 0.01]
    
    # Distance ranges for analysis
    DISTANCE_RANGES = [
        (18, 29, "Short (18-29 yards)"),
        (30, 39, "Medium-Short (30-39 yards)"),
        (40, 49, "Medium (40-49 yards)"),
        (50, 59, "Long (50-59 yards)"),
        (60, 75, "Extreme (60+ yards)")
    ]
    
    # Model parameters
    BAYESIAN_MCMC_SAMPLES = 2000
    BAYESIAN_TUNE = 1000
    BAYESIAN_CHAINS = 2
    
    # Rating thresholds
    ELITE_THRESHOLD = 0.15
    STRUGGLING_THRESHOLD = -0.20
    
    # Visualization settings
    FIGURE_SIZE = (12, 8)
    DPI = 100
    
    # Season types to include
    SEASON_TYPES = ['Reg']  # Regular season only by default
    
    # ─── Feature flags ───────────────────────────────────────────
    FILTER_RETIRED_INJURED = False   # keep everyone by default
    
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist."""
        for dir_path in [cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR, cls.OUTPUT_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)

# Create global config instance
config = Config()

# ───────────────────────── Feature catalogue ─────────────────────────
# Single source of truth for column roles – centralized from all modules
FEATURE_LISTS: Dict[str, List[str]] = {
    "numerical": [
        "attempt_yards", "age_at_attempt", "distance_squared",
        "career_length_years", "season_progress", "exp_100",   # 👈 NEW
        "rolling_success_rate", "current_streak",
        "distance_zscore", "distance_percentile",
        "seasons_of_experience", "career_year",
        "age_c", "age_c2",  # 👈 NEW: centered age variables
        # "age_spline_1", "age_spline_2", "age_spline_3",
        "importance", "days_since_last_kick",  # Days since player's last field goal
        "age_dist_interact", "exp_dist_interact",  # 👈 NEW: interaction terms
    ],
    "ordinal": ["season", "week", "month", "day_of_year"],
    "nominal": [
        "kicker_id", "kicker_idx",
        "is_long_attempt", "is_very_long_attempt",
        "is_rookie_attempt", "distance_category", "experience_category",
        "is_early_season", "is_late_season", "is_playoffs",
        # "player_status",  # ✅ Removed: Status should never be a predictor
    ],
    "y_variable": ["success"],
}

if __name__ == "__main__":
    # Test the configuration
    print("NFL Kicker Analysis Configuration")
    print("=" * 40)
    print(f"Data directory: {config.DATA_DIR}")
    print(f"Min distance: {config.MIN_DISTANCE}")
    print(f"Max distance: {config.MAX_DISTANCE}")
    print(f"Distance profile: {config.DISTANCE_PROFILE}")
    print(f"Elite threshold: {config.ELITE_THRESHOLD}")
    
    # Test directory creation
    config.ensure_directories()
    print("******* Configuration loaded and directories created!")



