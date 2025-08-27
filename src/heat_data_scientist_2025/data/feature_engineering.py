"""
Current columns:
Index(['personId', 'player_name', 'season', 'games_played', 'total_minutes',
       'total_points', 'total_assists', 'total_rebounds', 'total_reb_off',
       'total_reb_def', 'total_blocks', 'total_steals', 'total_pf',
       'total_tov', 'total_fga', 'total_fgm', 'total_fta', 'total_ftm',
       'total_3pa', 'total_3pm', 'total_plus_minus', 'avg_plus_minus', 'wins',
       'home_games', 'season_pie_num', 'season_pie_den', 'fg_pct', 'fg3_pct',
       'ft_pct', 'ts_pct', 'season_pie', 'pts_per36', 'ast_per36', 'reb_per36',
       'usage_per_min', 'efficiency_per_game', 'win_pct', 'home_games_pct',
       'teamCity', 'teamName', 'team_win_pct_final', "season_PER", "season_VORP", "season_EWA"],
      dtype='object')


Feature engineering for NBA player-season data.

"""


from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd


# quick check: makes sure we have the columns we need
def require_columns(df: pd.DataFrame, cols: List[str], context: str) -> None:
    """Check if required columns exist, throw error if missing."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for {context}: {missing}")


# helper: finds first column that actually exists in the data  
def _first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return first column name that exists in df, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None





# parse season string like '2023-24' into year 2023
def add_season_start_year(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Extract numeric start year from season string like 'YYYY-YY'."""
    out = df.copy()
    if "season" not in out.columns:
        raise ValueError("Need 'season' column")
    
    out["season_start_year"] = (
        out["season"].astype(str).str.extract(r"(\d{4})")[0].astype(int)
    )
    return out, ["season_start_year"]


# experience stuff - years in league and rough groupings
def add_experience_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Add experience features from draft year if available."""
    out = df.copy()
    created: List[str] = []

    # cumulative games played (simple running total)
    if "personId" in out.columns and "games_played" in out.columns:
        out = out.sort_values(["personId", "season_start_year"])
        out["games_played_total"] = out.groupby("personId")["games_played"].cumsum()
        created.append("games_played_total")

    # years since draft (if we have draft year)
    if "draftYear" in out.columns and "season_start_year" in out.columns:
        out["years_experience"] = (out["season_start_year"] - out["draftYear"]).clip(lower=0)
        
        # rough experience buckets
        def exp_bucket(exp):
            if pd.isna(exp): return "Unknown"
            if exp <= 2: return "Rookie/Sophomore"  
            if exp <= 5: return "Young Player"
            if exp <= 9: return "Prime Years"
            if exp <= 15: return "Veteran"
            return "Elder Statesman"
        
        out["experience_bucket"] = out["years_experience"].apply(exp_bucket)
        created.extend(["years_experience", "experience_bucket"])
        
    return out, created


# advanced box score metrics - efficiency stuff mostly
def add_advanced_metrics(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Create advanced metrics from basic box score stats."""
    out = df.copy()
    created: List[str] = []

    # find the columns we need (handles different naming)
    fga = _first_present(out, ["total_fga"])
    fta = _first_present(out, ["total_fta"])
    fgm = _first_present(out, ["total_fgm"])
    tpa = _first_present(out, ["total_3pa"])
    tpm = _first_present(out, ["total_3pm"])
    ast = _first_present(out, ["total_assists"])
    blk = _first_present(out, ["total_blocks"])
    stl = _first_present(out, ["total_steals"])
    pts = _first_present(out, ["total_points"])
    mins = _first_present(out, ["total_minutes"])
    reb = _first_present(out, ["total_rebounds"])
    dreb = _first_present(out, ["total_reb_def"])
    oreb = _first_present(out, ["total_reb_off"])
    tov = _first_present(out, ["total_tov", "total_turnovers"])

    # true shooting attempts estimate
    if fga and fta:
        out["ts_attempts"] = out[fga] + 0.44 * out[fta]
        created.append("ts_attempts")
        
    # shooting rates and efficiency
    if fga and tpa:
        out["three_point_rate"] = np.where(out[fga] > 0, out[tpa] / out[fga], 0.0)
        created.append("three_point_rate")
    if fga and fta:
        out["ft_rate"] = np.where(out[fga] > 0, out[fta] / out[fga], 0.0)
        created.append("ft_rate")
    if fga and fgm and tpm:
        out["efg_pct"] = np.where(out[fga] > 0, (out[fgm] + 0.5 * out[tpm]) / out[fga], 0.0)
        created.append("efg_pct")
    if fga and pts:
        out["pts_per_shot"] = np.where(out[fga] > 0, out[pts] / out[fga], 0.0)
        created.append("pts_per_shot")

    # defensive and overall production per 36
    if mins and blk and stl:
        out["defensive_per36"] = np.where(out[mins] > 0, (out[blk] + out[stl]) * 36 / out[mins], 0.0)
        out["stocks_per36"] = out["defensive_per36"].copy()  # same thing
        created.extend(["defensive_per36", "stocks_per36"])
    if mins and pts and ast and reb:
        out["production_per36"] = np.where(out[mins] > 0, (out[pts] + out[ast] + out[reb]) * 36 / out[mins], 0.0)
        created.append("production_per36")
    if mins and tov:
        out["tov_per36"] = np.where(out[mins] > 0, out[tov] * 36 / out[mins], 0.0)
        created.append("tov_per36")

    # rebounding shares
    if reb and dreb and oreb:
        total_reb_safe = out[reb].replace(0, np.nan)
        out["dreb_share"] = (out[dreb] / total_reb_safe).fillna(0.0)
        out["oreb_share"] = (out[oreb] / total_reb_safe).fillna(0.0)
        created.extend(["dreb_share", "oreb_share"])

    # usage events (shots, fts, turnovers)
    if fga and fta and tov and mins:
        out["usage_events_total"] = out[fga] + 0.44 * out[fta] + out[tov]
        out["usage_events_per_min"] = np.where(out[mins] > 0, out["usage_events_total"] / out[mins], 0.0)
        created.extend(["usage_events_total", "usage_events_per_min"])

    # assist to turnover ratio
    if ast and tov:
        out["ast_to_tov"] = np.where(out[tov] > 0, out[ast] / out[tov], out[ast])
        created.append("ast_to_tov")

    return out, created


# usage and shot creation features
def add_usage_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Usage and shot creation metrics."""
    out = df.copy()
    created: List[str] = []
    
    # total usage from per-minute usage  
    if "usage_per_min" in out.columns:
        min_col = _first_present(out, ["total_minutes"])
        if min_col:
            out["total_usage"] = out["usage_per_min"] * out[min_col]
            created.append("total_usage")

    # shot creation (shots + assists)
    fga_col = _first_present(out, ["total_fga"])
    fta_col = _first_present(out, ["total_fta"])
    ast_col = _first_present(out, ["total_assists"])
    min_col = _first_present(out, ["total_minutes"])
    
    if fga_col and fta_col and ast_col:
        out["shot_creation"] = out[fga_col] + out[fta_col] + out[ast_col]
        if min_col:
            out["shot_creation_per36"] = np.where(out[min_col] > 0, out["shot_creation"] * 36 / out[min_col], 0.0)
        else:
            out["shot_creation_per36"] = 0.0
        created.extend(["shot_creation", "shot_creation_per36"])
        
    return out, created





# figures out which numeric columns to lag automatically
def _build_lag_stat_list_auto(df: pd.DataFrame, season_col: str = "season_start_year") -> List[str]:
    """Auto-select numeric columns to lag, excluding obvious problem ones."""
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # stuff we definitely don't want to lag
    exclude_exact = {
        "personId", season_col, "games_played_total", "forecast_season", 
        "source_season", "season_pie_num", "season_pie_den"
    }
    
    # filter out excluded columns and existing lag columns
    base = [c for c in numeric if c not in exclude_exact 
            and not c.endswith("_lag1")]
    
    # skip columns with id/index type names
    bad_substrings = ("_id", "_idx", "_code")
    base = [c for c in base if not any(s in c.lower() for s in bad_substrings)]
    
    return base


# creates lagged features by player
def add_lag_features(df: pd.DataFrame, stats: Optional[List[str]] = None, 
                    lags: List[int] = [1], season_col: str = "season_start_year") -> Tuple[pd.DataFrame, List[str]]:
    """Add lag features by player-season. Nulls will be in first seasons only."""
    out = df.copy()
    created: List[str] = []
    require_columns(out, ["personId", season_col], "add_lag_features")
    
    # auto-detect stats to lag if not provided
    if stats is None:
        stats = _build_lag_stat_list_auto(out, season_col=season_col)
    else:
        # filter to numeric columns only
        num_cols = set(out.select_dtypes(include=[np.number]).columns)
        stats = [s for s in stats if s in num_cols]
    
    out = out.sort_values(["personId", season_col])
    gp = out.groupby("personId", group_keys=False)
    
    # create lag columns
    for col in stats:
        for k in lags:
            name = f"{col}_lag{k}"
            out[name] = gp[col].shift(k)
            created.append(name)
    
    # add helper column to identify first seasons (used for lag validation)
    out["has_prior_season"] = gp.cumcount() > 0
    created.append("has_prior_season")
    
    return out, created


# minutes and availability features  
def add_minutes_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Playing time and availability metrics."""
    out = df.copy()
    created: List[str] = []
    
    if "games_played" in out.columns and "total_minutes" in out.columns:
        out["minutes_per_game"] = np.where(out["games_played"] > 0, out["total_minutes"] / out["games_played"], 0.0)
        out["games_pct"] = out["games_played"] / 82.0
        created.extend(["minutes_per_game", "games_pct"])
        
        # playing time tiers
        out["minutes_tier"] = pd.cut(out["minutes_per_game"], bins=[0, 15, 25, 35, 48], 
                                   labels=["Bench", "Role Player", "Starter", "Star"], include_lowest=True)
        created.append("minutes_tier")
        
        # total minutes tiers (handles duplicate edges)
        try:
            out["total_minutes_tier"] = pd.qcut(out["total_minutes"], q=5, 
                                              labels=["Very Low", "Low", "Medium", "High", "Very High"])
        except ValueError:
            ranks = out["total_minutes"].rank(method="average")
            out["total_minutes_tier"] = pd.qcut(ranks, q=5,
                                              labels=["Very Low", "Low", "Medium", "High", "Very High"])
        created.append("total_minutes_tier")
    
    return out, created


# shooting performance relative to league average by season  
def add_performance_consistency(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Shooting performance vs league medians and composite score."""
    out = df.copy()
    created: List[str] = []
    
    need = ["season_start_year", "fg_pct", "fg3_pct", "ft_pct"]
    missing = [c for c in need if c not in out.columns]
    if missing:
        return out, created
        
    # season-level medians for comparison
    grp = out.groupby("season_start_year", group_keys=False)
    league_medians = ["fg_league_med", "fg3_league_med", "ft_league_med"]
    out["fg_league_med"] = grp["fg_pct"].transform("median")
    out["fg3_league_med"] = grp["fg3_pct"].transform("median") 
    out["ft_league_med"] = grp["ft_pct"].transform("median")
    
    # differences from league median
    out["fg_vs_league"] = out["fg_pct"] - out["fg_league_med"]
    out["fg3_vs_league"] = out["fg3_pct"] - out["fg3_league_med"]
    out["ft_vs_league"] = out["ft_pct"] - out["ft_league_med"]
    created.extend(["fg_vs_league", "fg3_vs_league", "ft_vs_league"])
    
    # composite shooting score
    out["shooting_score"] = out["fg_pct"] * 0.4 + out["fg3_pct"] * 0.3 + out["ft_pct"] * 0.3
    created.append("shooting_score")
    
    # clean up temp columns
    out.drop(columns=league_medians, inplace=True)
    return out, created


# composite features combining multiple stats
def create_composite_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Create composite impact metrics."""
    out = df.copy()
    created: List[str] = []
    
    # make sure we have the base columns (fill with nan if missing)
    for need in ["pts_per36", "ast_per36", "reb_per36", "defensive_per36"]:
        if need not in out.columns:
            out[need] = np.nan
    
    # offensive impact score
    out["offensive_impact"] = out["pts_per36"] * 0.4 + out["ast_per36"] * 0.3 + out["ts_pct"] * 100 * 0.3
    created.append("offensive_impact")
    
    # two-way impact (offense + defense)
    out["two_way_impact"] = out["offensive_impact"] + out["defensive_per36"] * 10
    created.append("two_way_impact")
    
    # efficiency x volume
    if "efficiency_per_game" in out.columns and "total_usage" in out.columns:
        out["efficiency_volume_score"] = out["efficiency_per_game"] * out["total_usage"]  
        created.append("efficiency_volume_score")
    
    # versatility score (above median in multiple areas)
    scoring_contrib = (out["pts_per36"] > out["pts_per36"].median()).astype(int)
    assist_contrib = (out["ast_per36"] > out["ast_per36"].median()).astype(int)
    rebound_contrib = (out["reb_per36"] > out["reb_per36"].median()).astype(int) 
    defense_contrib = (out["defensive_per36"] > out["defensive_per36"].median()).astype(int)
    out["versatility_score"] = scoring_contrib + assist_contrib + rebound_contrib + defense_contrib
    created.append("versatility_score")
    
    return out, created


# helper for season-normalized z-scores
def _zscore_by_season(df: pd.DataFrame, col: str, season_col: str) -> pd.Series:
    """Z-score within each season, clipped to avoid extreme outliers."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    g = df.groupby(season_col)[col]
    z = (df[col] - g.transform("mean")) / (g.transform("std").replace(0, np.nan))
    return z.clip(-3, 3).fillna(0.0)


# portability index - how well skills transfer between situations  
def build_portability_index(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Portability index based on transferable skills."""
    out = df.copy()
    created: List[str] = []
    season_col = "season_start_year"
    require_columns(out, [season_col], "build_portability_index")
    
    # season-normalized z-scores for key components
    z = {}
    for col in ["ts_pct", "efg_pct", "pts_per_shot", "fg3_pct", "three_point_rate", "ft_pct",
                "stocks_per36", "dreb_share", "oreb_share", "ast_per36", "ast_to_tov", "usage_per_min"]:
        if col in out.columns:
            z[col] = _zscore_by_season(out, col, season_col)
        else:
            z[col] = pd.Series(0.0, index=out.index)
    
    # component scores
    score_eff = (z["ts_pct"] + z["efg_pct"] + z["pts_per_shot"]) / 3.0
    shoot_abil = (z["fg3_pct"] + z["three_point_rate"] + z["ft_pct"]) / 3.0  
    def_abil = z["stocks_per36"]
    
    # rebounding versatility (good at both, penalty for imbalance)
    reb_mean = (z["dreb_share"] + z["oreb_share"]) / 2.0
    reb_gap = (z["dreb_share"] - z["oreb_share"]).abs()
    def_vers = reb_mean - 0.25 * reb_gap
    
    pass_abil = (z["ast_per36"] + z["ast_to_tov"]) / 2.0
    
    # usage with diminishing returns (too much usage can hurt portability)
    usage_term = z["usage_per_min"] - 0.15 * (z["usage_per_min"] ** 2)
    
    # weighted combination (weights sum to 1.0)
    out["portability_index"] = (0.16 * score_eff + 0.40 * shoot_abil + 0.08 * def_abil + 
                               0.05 * def_vers + 0.25 * pass_abil + 0.06 * usage_term)
    created.append("portability_index")
    
    # save component scores too
    out["pi_scoring_eff"] = score_eff
    out["pi_shooting"] = shoot_abil  
    out["pi_defense"] = def_abil
    out["pi_versatility"] = def_vers
    out["pi_passing"] = pass_abil
    out["pi_usage_term"] = usage_term
    created.extend(["pi_scoring_eff", "pi_shooting", "pi_defense", "pi_versatility", "pi_passing", "pi_usage_term"])
    
    return out, created


# main feature engineering function
def engineer_features(df: pd.DataFrame, drop_null_lag_rows: bool = True, verbose: bool = False) -> pd.DataFrame:
    """
    Build all features and optionally drop first-season rows with null lags.
    
    Args:
        df: Input dataframe with player-season data
        drop_null_lag_rows: If True, drop rows where lag features are null (default True)
        verbose: Print progress info (default False)
    
    Returns:
        Processed dataframe with all engineered features
    """
    if verbose:
        print("Starting feature engineering...")
        
    original_shape = df.shape
    out = df.copy()
    
    # check we have required base columns
    required_base = ["personId", "season", "games_played", "total_minutes", "season_pie"]
    missing_base = [c for c in required_base if c not in out.columns]
    if missing_base:
        raise ValueError(f"Missing required columns: {missing_base}")
    
    # 1. parse season to get numeric year
    if verbose:
        print("Parsing seasons...")
    out, _ = add_season_start_year(out)
    
    # 2. sort by player and season for all subsequent operations
    out = out.sort_values(["personId", "season_start_year"])
    
    # 3. build features step by step
    if verbose:
        print("Adding experience features...")
    out, _ = add_experience_features(out)
    
    if verbose:
        print("Adding advanced metrics...")
    out, _ = add_advanced_metrics(out)
    
    if verbose:
        print("Adding usage features...")
    out, _ = add_usage_features(out)
    
    if verbose:
        print("Adding minutes features...")  
    out, _ = add_minutes_features(out)
    
    if verbose:
        print("Adding performance consistency...")
    out, _ = add_performance_consistency(out)
    
    if verbose:
        print("Creating composite features...")
    out, _ = create_composite_features(out)
    
    if verbose:
        print("Building portability index...")
    out, _ = build_portability_index(out)
    
    # 4. add lag features (creates nulls in first seasons)
    if verbose:
        print("Creating lag features...")
    out, _ = add_lag_features(out, lags=[1])
    
    # 5. clean up any infinite values
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    inf_cols = []
    for col in numeric_cols:
        if np.isinf(out[col]).any():
            inf_cols.append(col)
            out[col] = out[col].replace([np.inf, -np.inf], np.nan)
    
    if verbose and inf_cols:
        print(f"Cleaned infinite values in {len(inf_cols)} columns")
    
    # 6. handle lag nulls using has_prior_season instead of separate mask
    lag_cols = [c for c in out.columns if c.endswith("_lag1")]
    if lag_cols:
        lag_nulls_mask = out[lag_cols].isnull().any(axis=1)
        
        # use has_prior_season to identify first seasons (created in add_lag_features)
        if "has_prior_season" in out.columns:
            first_season_mask = ~out["has_prior_season"]
        else:
            # defensive fallback: treat missing as not prior season
            first_season_mask = pd.Series(False, index=out.index)
        
        # validate expectation: lag nulls should be subset of first seasons
        # this is slightly safer against occasional upstream data quirks
        unexpected_nulls = lag_nulls_mask & (~first_season_mask)
        if unexpected_nulls.any():
            print("⚠️ Warning: Some lag nulls are not from first seasons - check data quality")
        
        if drop_null_lag_rows:
            rows_before = len(out)
            out = out[~lag_nulls_mask].copy()
            rows_dropped = rows_before - len(out)
            if verbose:
                print(f"Dropped {rows_dropped} first-season rows with null lags")
    
    if verbose:
        print(f"Feature engineering complete: {original_shape[0]} → {len(out)} rows, {original_shape[1]} → {len(out.columns)} columns")
    
    return out


if __name__ == "__main__":
    from src.heat_data_scientist_2025.data.load_data_utils import load_data_optimized
    from src.heat_data_scientist_2025.utils.config import (CFG, numerical_features, nominal_categoricals, ordinal_categoricals, y_variables)

    df = load_data_optimized(
        CFG.ml_dataset_path,
        debug=True,
        drop_null_rows=True,
    )

    
    # run feature engineering
    try:
        df_eng = engineer_features(df, drop_null_lag_rows=True, verbose=True)
        print(f"✓ Success! Result shape: {df_eng.shape}")
        print("lag columns created:", [c for c in df_eng.columns if c.endswith('_lag1')])
        print(f"First season rows dropped: {len(df) - len(df_eng)}")
    except Exception as e:
        print(f"✗ Error: {e}")
        
    #print columns
    print(df_eng.columns)

    # check that there are these lists in the dataset    
    #check that the features are in the df_eng
    for feature in numerical_features:
        assert feature in df_eng.columns, f"{feature} is not in the dataset"
    for feature in nominal_categoricals:
        assert feature in df_eng.columns, f"{feature} is not in the dataset"
    for feature in ordinal_categoricals:
        assert feature in df_eng.columns, f"{feature} is not in the dataset"
    for feature in y_variables:
        assert feature in df_eng.columns, f"{feature} is not in the dataset"
