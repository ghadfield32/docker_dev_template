"""
NBA statistics analysis - PIE, Game Score per 36, and PER calculations
Load kaggle data, compute metrics, rank players, export results
"""

from __future__ import annotations
from typing import Dict, Iterable
from pathlib import Path
import pandas as pd
import numpy as np
import kagglehub as kh
from kagglehub import KaggleDatasetAdapter as KDA

from src.heat_data_scientist_2025.utils.config import (
    CFG,
    KAGGLE_DATASET,
    IMPORTANT_TABLES,
    KAGGLE_TABLE_TO_CSV,
    start_season as CFG_START_SEASON,
    season_type as CFG_SEASON_TYPE,
    minutes_total_minimum_per_season as CFG_MIN_SEASON_MINUTES,
)

# Core utilities
def _season_from_timestamp(ts: pd.Series) -> pd.Series:
    """Convert timestamps to season strings like '2010-11'"""
    dt = pd.to_datetime(ts, errors="coerce", utc=False)
    start_year = np.where(dt.dt.month >= 8, dt.dt.year, dt.dt.year - 1)
    end_year = start_year + 1
    start_s = pd.Series(start_year, index=ts.index).astype(str)
    end_s = (pd.Series(end_year, index=ts.index) % 100).astype(str).str.zfill(2)
    return start_s + "-" + end_s

def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Safe division - returns 0 when denominator <= 0"""
    if not hasattr(den, "fillna") or not hasattr(num, "index"):
        raise TypeError(f"Expected pandas Series, got {type(num)}, {type(den)}")
    
    den = den.fillna(0)
    out = pd.Series(np.zeros(len(num)), index=num.index, dtype="float64")
    mask = den > 0
    out.loc[mask] = (num[mask] / den[mask]).astype("float64")
    return out

def _safe_div_scalar(num: float, den: float) -> float:
    """Safe division for scalars"""
    return 0.0 if pd.isna(den) or den <= 0 else num / den

def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Convert columns to numeric, keeping NaN for errors"""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _resolve_player_team_cols(df: pd.DataFrame) -> tuple[str, str]:
    """Find player team columns across different naming conventions"""
    options = [
        ("playerteamCity", "playerteamName"),
        ("playerTeamCity", "playerTeamName"),
        ("teamCity", "teamName"),
    ]
    for city_col, name_col in options:
        if city_col in df.columns and name_col in df.columns:
            return city_col, name_col
    raise KeyError("Could not find player team columns")

# Data loading and filtering
def enforce_criteria_python(
    players_df: pd.DataFrame | None,
    player_stats_df: pd.DataFrame,
    team_stats_df: pd.DataFrame,
    start_season: int = CFG_START_SEASON,
    season_type: str = CFG_SEASON_TYPE,
    minutes_total_minimum_per_season: int = CFG_MIN_SEASON_MINUTES,
    defer_minutes_gate: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply season filters and compute player minutes totals"""
    
    ps = player_stats_df.copy()
    ps["season"] = _season_from_timestamp(ps["gameDate"])
    ts = team_stats_df.copy()
    ts["season"] = _season_from_timestamp(ts["gameDate"])

    # Convert numeric columns
    numeric_cols = [
        "numMinutes","points","assists","blocks","steals",
        "fieldGoalsAttempted","fieldGoalsMade","freeThrowsAttempted","freeThrowsMade",
        "threePointersAttempted","threePointersMade",
        "reboundsDefensive","reboundsOffensive","reboundsTotal",
        "foulsPersonal","turnovers","plusMinusPoints","home","win"
    ]
    ps = _to_numeric(ps, numeric_cols)
    
    for col in ["home","win"]:
        if col in ps.columns:
            ps[col] = ps[col].fillna(0).astype(int)

    # Filter by season and game type
    ps = ps.loc[
        (ps["gameType"] == season_type) &
        (ps["season"].str.slice(0, 4).astype(int) >= start_season)
    ].copy()

    # Create player names
    fn = ps.get("firstName", "").fillna("")
    ln = ps.get("lastName", "").fillna("")
    ps["player_name"] = (fn + " " + ln).str.strip()

    # Calculate season minutes
    season_minutes = (
        ps.groupby(["personId","season"], as_index=False)["numMinutes"]
          .sum().rename(columns={"numMinutes": "minutes_total"})
    )
    ps = ps.merge(season_minutes, on=["personId","season"], how="left")

    if not defer_minutes_gate:
        ps = ps.loc[ps["minutes_total"] >= minutes_total_minimum_per_season].copy()

    players_out = players_df.copy() if players_df is not None else None
    return players_out, ps, ts

# PIE calculation
def compute_player_game_pie(
    filtered_player_stats: pd.DataFrame,
    drop_zero_minute_games: bool = True,
    validate_game_sums: bool = True,
    atol: float = 1e-9,
    rtol: float = 1e-6,
) -> pd.DataFrame:
    """Compute per-game PIE values"""
    
    required_cols = [
        "points","fieldGoalsMade","freeThrowsMade","fieldGoalsAttempted","freeThrowsAttempted",
        "reboundsDefensive","reboundsOffensive","assists","steals","blocks","foulsPersonal","turnovers",
        "gameId","numMinutes"
    ]
    missing = [c for c in required_cols if c not in filtered_player_stats.columns]
    if missing:
        raise KeyError(f"Missing PIE columns: {missing}")

    df = filtered_player_stats.copy()
    df = _to_numeric(df, required_cols)

    if drop_zero_minute_games:
        before = len(df)
        df = df.loc[df["numMinutes"] > 0].copy()
        print(f"Dropped {before - len(df):,} zero-minute games")

    # Calculate PIE numerator
    df["pie_numerator"] = (
        df["points"] + df["fieldGoalsMade"] + df["freeThrowsMade"]
        - df["fieldGoalsAttempted"] - df["freeThrowsAttempted"]
        + df["reboundsDefensive"] + 0.5 * df["reboundsOffensive"]
        + df["assists"] + df["steals"] + 0.5 * df["blocks"]
        - df["foulsPersonal"] - df["turnovers"]
    )

    # Calculate game denominators
    game_totals = (
        df.groupby("gameId", as_index=False)["pie_numerator"]
          .sum().rename(columns={"pie_numerator": "pie_denominator"})
    )

    # Merge and calculate PIE
    merged = df.merge(game_totals, on="gameId", how="left", validate="many_to_one")
    merged["pie_denominator"] = pd.to_numeric(merged["pie_denominator"], errors="coerce")
    merged = merged.loc[merged["pie_denominator"] > 0].copy()
    merged["game_pie"] = _safe_div(merged["pie_numerator"], merged["pie_denominator"])

    # Validate game sums
    if validate_game_sums:
        sums = merged.groupby("gameId", as_index=False)["game_pie"].sum()
        bad_games = sums.loc[~np.isclose(sums["game_pie"], 1.0, rtol=rtol, atol=atol)]
        if not bad_games.empty:
            print(f"Warning: {len(bad_games):,} games don't sum to 1.0")

    return merged

# Team game calculations for PER
def compute_team_game_totals_and_pace(filtered_player_stats: pd.DataFrame) -> pd.DataFrame:
    """Calculate team totals and pace for PER computation"""
    
    df = filtered_player_stats.copy()
    if "season" not in df.columns:
        df["season"] = _season_from_timestamp(df["gameDate"])

    numeric_cols = [
        "numMinutes","points","assists","turnovers","reboundsOffensive","reboundsDefensive",
        "reboundsTotal","foulsPersonal","fieldGoalsAttempted","fieldGoalsMade",
        "freeThrowsAttempted","freeThrowsMade","threePointersMade"
    ]
    df = _to_numeric(df, [c for c in numeric_cols if c in df.columns])

    team_city_col, team_name_col = _resolve_player_team_cols(df)

    # Aggregate team totals per game
    team_totals = df.groupby(["gameId", "season", team_city_col, team_name_col], as_index=False).agg(
        team_min=("numMinutes","sum"),
        team_pts=("points","sum"),
        team_ast=("assists","sum"),
        team_tov=("turnovers","sum"),
        team_orb=("reboundsOffensive","sum"),
        team_drb=("reboundsDefensive","sum"),
        team_trb=("reboundsTotal","sum"),
        team_pf=("foulsPersonal","sum"),
        team_fga=("fieldGoalsAttempted","sum"),
        team_fgm=("fieldGoalsMade","sum"),
        team_fta=("freeThrowsAttempted","sum"),
        team_ftm=("freeThrowsMade","sum"),
        team_3pm=("threePointersMade","sum"),
    ).rename(columns={team_city_col: "teamCity", team_name_col: "teamName"})

    # Calculate possessions
    team_totals["team_poss"] = (
        team_totals["team_fga"] - team_totals["team_orb"] + 
        team_totals["team_tov"] + 0.44 * team_totals["team_fta"]
    )

    # Get opponent possessions
    opponent_data = team_totals.rename(columns={
        "teamCity": "oppCity", "teamName": "oppName", "team_poss": "opp_poss", "team_min": "opp_min"
    })
    
    merged = team_totals.merge(
        opponent_data[["gameId","oppCity","oppName","opp_poss","opp_min"]], 
        on="gameId", how="left"
    )

    # Handle self-joins by excluding same team
    same_team = (merged["teamCity"] == merged["oppCity"]) & (merged["teamName"] == merged["oppName"])
    if same_team.any():
        # Find proper opponents
        other_teams = team_totals[["gameId","teamCity","teamName","team_poss","team_min"]].copy()
        other_teams = other_teams.merge(team_totals, on="gameId", suffixes=("","_opp"))
        other_teams = other_teams[
            (other_teams["teamCity"] != other_teams["teamCity_opp"]) | 
            (other_teams["teamName"] != other_teams["teamName_opp"])
        ]
        other_teams = other_teams.rename(columns={
            "team_poss_opp": "opp_poss", "team_min_opp": "opp_min"
        })[["gameId","teamCity","teamName","opp_poss","opp_min"]].drop_duplicates()
        
        merged = team_totals.merge(other_teams, on=["gameId","teamCity","teamName"], how="left")

    # Calculate pace and assist ratio
    merged["pace"] = 48.0 * _safe_div((merged["team_poss"] + merged["opp_poss"]) / 2.0, merged["team_min"] / 5.0)
    merged["tm_ast_over_fg"] = _safe_div(merged["team_ast"], merged["team_fgm"])

    return merged

# League constants for PER
def compute_league_constants_per_season(team_game_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate league-wide constants for PER"""
    
    tg = team_game_df.copy()
    tg["pace_x_min"] = tg["pace"] * tg["team_min"]

    league_stats = tg.groupby("season", as_index=False).agg(
        lgFG=("team_fgm","sum"),
        lgFGA=("team_fga","sum"),
        lgFT=("team_ftm","sum"),
        lgFTA=("team_fta","sum"),
        lgAST=("team_ast","sum"),
        lgORB=("team_orb","sum"),
        lgDRB=("team_drb","sum"),
        lgTRB=("team_trb","sum"),
        lgTOV=("team_tov","sum"),
        lgPF=("team_pf","sum"),
        lgPTS=("team_pts","sum"),
        tot_min=("team_min","sum"),
        pace_x_min=("pace_x_min","sum"),
    )

    # Calculate derived constants
    league_stats["lgPace"] = _safe_div(league_stats["pace_x_min"], league_stats["tot_min"])
    league_stats["VOP"] = _safe_div(
        league_stats["lgPTS"],
        league_stats["lgFGA"] - league_stats["lgORB"] + league_stats["lgTOV"] + 0.44 * league_stats["lgFTA"]
    )
    league_stats["DRB_PCT"] = _safe_div(
        league_stats["lgTRB"] - league_stats["lgORB"], 
        league_stats["lgTRB"]
    )

    # Basketball-Reference factor calculation
    ast_over_fg = _safe_div(league_stats["lgAST"], league_stats["lgFG"])
    fg_over_ft = _safe_div(league_stats["lgFG"], league_stats["lgFT"])
    league_stats["factor"] = (2.0/3.0) - _safe_div(0.5 * ast_over_fg, 2.0 * fg_over_ft)
    
    league_stats["ft_per_pf"] = _safe_div(league_stats["lgFT"], league_stats["lgPF"])
    league_stats["fta_per_pf"] = _safe_div(league_stats["lgFTA"], league_stats["lgPF"])

    return league_stats[["season","VOP","DRB_PCT","factor","ft_per_pf","fta_per_pf","lgPace"]]

# PER calculation
def compute_player_game_per(
    filtered_player_stats: pd.DataFrame,
    team_game_df: pd.DataFrame,
    league_constants_df: pd.DataFrame,
    drop_zero_minute_games: bool = True,
) -> pd.DataFrame:
    """Calculate per-game PER values"""
    
    ps = filtered_player_stats.copy()
    if drop_zero_minute_games:
        ps = ps.loc[pd.to_numeric(ps["numMinutes"], errors="coerce") > 0].copy()

    required_cols = [
        "gameId","season","personId","numMinutes","threePointersMade","assists",
        "fieldGoalsMade","fieldGoalsAttempted","freeThrowsMade","freeThrowsAttempted",
        "turnovers","reboundsOffensive","reboundsTotal","steals","blocks","foulsPersonal","points"
    ]
    
    numeric_cols = [c for c in required_cols if c in ps.columns and c != "season"]
    ps = _to_numeric(ps, numeric_cols)
    
    missing = [c for c in required_cols if c not in ps.columns]
    if missing:
        raise KeyError(f"Missing PER columns: {missing}")

    # Join with team data
    t_city_col, t_name_col = _resolve_player_team_cols(ps)
    ps_with_team = ps.merge(
        team_game_df[["gameId","teamCity","teamName","pace","tm_ast_over_fg"]],
        left_on=["gameId", t_city_col, t_name_col],
        right_on=["gameId","teamCity","teamName"],
        how="left"
    )

    # Join with league constants
    ps_final = ps_with_team.merge(league_constants_df, on="season", how="left")

    # Calculate PER components
    MP = pd.to_numeric(ps_final["numMinutes"], errors="coerce")
    FG, FGA = ps_final["fieldGoalsMade"], ps_final["fieldGoalsAttempted"]
    FT, FTA = ps_final["freeThrowsMade"], ps_final["freeThrowsAttempted"]
    AST, TOV = ps_final["assists"], ps_final["turnovers"]
    ORB, TRB = ps_final["reboundsOffensive"], ps_final["reboundsTotal"]
    STL, BLK, PF = ps_final["steals"], ps_final["blocks"], ps_final["foulsPersonal"]
    TPM = ps_final["threePointersMade"]

    # League constants
    tm_ast_fg = ps_final["tm_ast_over_fg"].fillna(0.0)
    VOP = ps_final["VOP"].fillna(0.0)
    DRBP = ps_final["DRB_PCT"].fillna(0.0)
    factor = ps_final["factor"].fillna(0.0)
    ft_per_pf = ps_final["ft_per_pf"].fillna(0.0)
    fta_per_pf = ps_final["fta_per_pf"].fillna(0.0)
    lgPace = ps_final["lgPace"].fillna(0.0)
    tmPace = ps_final["pace"].fillna(0.0)

    # uPER calculation
    positive_box = (
        TPM + (2.0/3.0) * AST + (2.0 - factor * tm_ast_fg) * FG
        + 0.5 * FT * (2.0 - tm_ast_fg / 3.0) + VOP * (1.0 - DRBP) * (TRB - ORB)
        + VOP * DRBP * ORB + VOP * STL + VOP * DRBP * BLK
    )
    
    negative_box = (
        VOP * TOV + VOP * DRBP * (FGA - FG)
        + VOP * 0.44 * (0.44 + 0.56 * DRBP) * (FTA - FT)
        + PF * (ft_per_pf - 0.44 * fta_per_pf * VOP)
    )
    
    uPER = _safe_div(positive_box - negative_box, MP)
    
    # Pace adjustment
    pace_adj = _safe_div(lgPace, tmPace)
    aPER = uPER * pace_adj

    result = ps_final[["personId","gameId","season","numMinutes"]].copy()
    result["uPER"] = pd.to_numeric(uPER, errors="coerce")
    result["aPER"] = pd.to_numeric(aPER, errors="coerce")
    
    return result

# NEW: BPM calculation for VORP
def compute_player_game_bpm(
    filtered_player_stats: pd.DataFrame,
    team_game_df: pd.DataFrame,
    league_constants_df: pd.DataFrame,
    drop_zero_minute_games: bool = True,
) -> pd.DataFrame:
    """Calculate simplified Box Plus/Minus (BPM) for VORP computation"""
    
    ps = filtered_player_stats.copy()
    if drop_zero_minute_games:
        ps = ps.loc[pd.to_numeric(ps["numMinutes"], errors="coerce") > 0].copy()

    required_cols = [
        "gameId","season","personId","numMinutes","points","assists","reboundsTotal",
        "steals","blocks","turnovers","foulsPersonal","fieldGoalsMade","fieldGoalsAttempted",
        "freeThrowsMade","freeThrowsAttempted","threePointersMade"
    ]
    
    # Handle column name variations
    if "reboundsTotal" in ps.columns:
        ps["rebounds Total"] = ps["reboundsTotal"]
    
    numeric_cols = [c for c in required_cols if c in ps.columns and c != "season"]
    ps = _to_numeric(ps, numeric_cols)
    
    missing = [c for c in required_cols if c not in ps.columns]
    if missing:
        raise KeyError(f"Missing BPM columns: {missing}")
    
    # Join with team data to get pace
    t_city_col, t_name_col = _resolve_player_team_cols(ps)
    ps_with_team = ps.merge(
        team_game_df[["gameId","teamCity","teamName","pace","team_poss"]],
        left_on=["gameId", t_city_col, t_name_col],
        right_on=["gameId","teamCity","teamName"],
        how="left"
    )

    # Join with league constants for league averages
    ps_final = ps_with_team.merge(league_constants_df, on="season", how="left")

    # Calculate player possessions used (estimate)
    MP = pd.to_numeric(ps_final["numMinutes"], errors="coerce")
    team_pace = ps_final["pace"].fillna(100.0)
    
    # Player possessions per 100 team possessions
    player_poss_per100 = 100.0 * MP / 48.0  # Approximate possessions per 100 team possessions
    
    # Simplified BPM calculation based on box score stats per 100 possessions
    # This is a simplified version - full BPM requires more complex calculations
    pts_per100 = _safe_div(ps_final["points"] * 100.0, player_poss_per100)
    reb_per100 = _safe_div(ps_final.get("reboundsTotal", ps_final.get("rebounds Total", 0)) * 100.0, player_poss_per100)
    ast_per100 = _safe_div(ps_final["assists"] * 100.0, player_poss_per100)
    stl_per100 = _safe_div(ps_final["steals"] * 100.0, player_poss_per100)
    blk_per100 = _safe_div(ps_final["blocks"] * 100.0, player_poss_per100)
    tov_per100 = _safe_div(ps_final["turnovers"] * 100.0, player_poss_per100)
    
    # Calculate True Shooting Percentage
    tsa = ps_final["fieldGoalsAttempted"] + 0.44 * ps_final["freeThrowsAttempted"]
    ts_pct = _safe_div(ps_final["points"], 2.0 * tsa)
    
    # Simplified BPM formula (coefficients based on statistical impact)
    # These are approximated coefficients - full BPM uses regression analysis
    # Scale to approximate BPM range (-10 to +10)
    bpm = (
        0.15 * pts_per100 +
        0.12 * reb_per100 +
        0.18 * ast_per100 +
        0.25 * stl_per100 +
        0.15 * blk_per100 -
        0.20 * tov_per100 +
        5.0 * (ts_pct - 0.53)  # TS% adjustment (league average ~53%)
        - 5.0  # Baseline adjustment to center around 0
    )

    result = ps_final[["personId","gameId","season","numMinutes"]].copy()
    result["game_bpm"] = pd.to_numeric(bpm, errors="coerce").fillna(0.0)
    
    return result

# Season aggregation
def build_player_season_table_python(
    player_game_with_pie: pd.DataFrame,
    team_stats_df: pd.DataFrame,
    minutes_total_minimum_per_season: int = CFG_MIN_SEASON_MINUTES,
    player_game_per: pd.DataFrame | None = None,
    player_game_bpm: pd.DataFrame | None = None,  # NEW parameter for VORP/EWA
) -> pd.DataFrame:
    """Aggregate game-level stats to season level"""
    
    pg = player_game_with_pie.copy()
    
    # Convert numeric columns
    numeric_cols = [
        "numMinutes","points","assists","blocks","steals","reboundsDefensive","reboundsOffensive",
        "reboundsTotal","foulsPersonal","turnovers","fieldGoalsAttempted","fieldGoalsMade",
        "freeThrowsAttempted","freeThrowsMade","threePointersAttempted","threePointersMade",
        "plusMinusPoints","game_pie","home","win"
    ]
    pg = _to_numeric(pg, numeric_cols)

    # Season aggregations
    season_stats = pg.groupby(["personId","player_name","season"], as_index=False).agg(
        games_played=("gameId","nunique"),
        total_minutes=("numMinutes","sum"),
        total_points=("points","sum"),
        total_assists=("assists","sum"),
        total_rebounds=("reboundsTotal","sum"),
        total_oreb=("reboundsOffensive","sum"),
        total_dreb=("reboundsDefensive","sum"),
        total_blocks=("blocks","sum"),
        total_stls=("steals","sum"),
        total_pf=("foulsPersonal","sum"),
        total_tov=("turnovers","sum"),
        total_fga=("fieldGoalsAttempted","sum"),
        total_fgm=("fieldGoalsMade","sum"),
        total_fta=("freeThrowsAttempted","sum"),
        total_ftm=("freeThrowsMade","sum"),
        total_3pa=("threePointersAttempted","sum"),
        total_3pm=("threePointersMade","sum"),
        total_plus_minus=("plusMinusPoints","sum"),
        avg_plus_minus=("plusMinusPoints","mean"),
        wins=("win","sum"),
        home_games=("home","sum"),
        season_pie_num=("pie_numerator","sum"),
        season_pie_den=("pie_denominator","sum"),
    )

    # Calculate shooting percentages
    season_stats["fg_pct"] = _safe_div(season_stats["total_fgm"], season_stats["total_fga"])
    season_stats["fg3_pct"] = _safe_div(season_stats["total_3pm"], season_stats["total_3pa"])
    season_stats["ft_pct"] = _safe_div(season_stats["total_ftm"], season_stats["total_fta"])
    season_stats["ts_pct"] = _safe_div(
        season_stats["total_points"], 
        2.0 * (season_stats["total_fga"] + 0.44 * season_stats["total_fta"])
    )

    # PIE calculations
    season_stats["season_pie"] = _safe_div(season_stats["season_pie_num"], season_stats["season_pie_den"])
    season_stats["season_pie_pct"] = 100.0 * season_stats["season_pie"]

    # Per-36 minutes stats
    per36_cols = [
        ("pts_per36", "total_points"),
        ("ast_per36", "total_assists"), 
        ("reb_per36", "total_rebounds"),
        ("fgm_per36", "total_fgm"),
        ("fga_per36", "total_fga"),
        ("ftm_per36", "total_ftm"),
        ("fta_per36", "total_fta"),
        ("oreb_per36", "total_oreb"),
        ("dreb_per36", "total_dreb"),
        ("stl_per36", "total_stls"),
        ("blk_per36", "total_blocks"),
        ("pf_per36", "total_pf"),
        ("tov_per36", "total_tov"),
    ]
    
    for per36_col, total_col in per36_cols:
        season_stats[per36_col] = _safe_div(season_stats[total_col] * 36.0, season_stats["total_minutes"])

    # Additional efficiency metrics
    season_stats["usage_per_min"] = _safe_div(
        season_stats["total_fga"] + 0.44 * season_stats["total_fta"] + season_stats["total_tov"],
        season_stats["total_minutes"]
    )
    
    season_stats["efficiency_per_game"] = _safe_div(
        (season_stats["total_points"] + season_stats["total_rebounds"] + season_stats["total_assists"]
         + season_stats["total_stls"] + season_stats["total_blocks"]
         - (season_stats["total_fga"] - season_stats["total_fgm"])
         - (season_stats["total_fta"] - season_stats["total_ftm"])
         - season_stats["total_tov"]),
        season_stats["games_played"]
    )

    season_stats["win_pct"] = _safe_div(season_stats["wins"], season_stats["games_played"])
    season_stats["home_games_pct"] = _safe_div(season_stats["home_games"], season_stats["games_played"])

    # Get team information
    ts = team_stats_df.copy()
    if "season" not in ts.columns:
        ts["season"] = _season_from_timestamp(ts["gameDate"])
    ts = _to_numeric(ts, ["seasonWins","seasonLosses"])
    ts["seasonWins"] = ts["seasonWins"].fillna(0)
    ts["seasonLosses"] = ts["seasonLosses"].fillna(0)

    team_final_records = (
        ts.groupby(["teamCity","teamName","season"], as_index=False)
          .agg(final_wins=("seasonWins","max"), final_losses=("seasonLosses","max"))
    )
    team_final_records["team_win_pct_final"] = _safe_div(
        team_final_records["final_wins"], 
        team_final_records["final_wins"] + team_final_records["final_losses"]
    )

    # Find primary team for each player
    team_city_col = "playerteamCity" if "playerteamCity" in pg.columns else "playerTeamCity"
    team_name_col = "playerteamName" if "playerteamName" in pg.columns else "playerTeamName"
    
    player_team_minutes = (
        pg.groupby(["personId","season",team_city_col,team_name_col], as_index=False)
          .agg(minutes_on_team=("numMinutes","sum"))
    )
    
    primary_team_idx = player_team_minutes.groupby(["personId","season"])["minutes_on_team"].idxmax()
    primary_teams = player_team_minutes.loc[primary_team_idx, ["personId","season",team_city_col,team_name_col]].copy()
    primary_teams = primary_teams.rename(columns={team_city_col: "teamCity", team_name_col: "teamName"})

    # Merge team data
    season_with_team = season_stats.merge(primary_teams, on=["personId","season"], how="left")
    season_final = season_with_team.merge(
        team_final_records[["teamCity","teamName","season","team_win_pct_final"]],
        on=["teamCity","teamName","season"], how="left"
    )

    # Rename for consistency
    season_final = season_final.rename(columns={
        "total_stls": "total_steals",
        "total_oreb": "total_reb_off", 
        "total_dreb": "total_reb_def",
    })

    # Game Score calculation
    season_final["season_game_score_total"] = (
        season_final["total_points"]
        + 0.4 * season_final["total_fgm"]
        - 0.7 * season_final["total_fga"] 
        - 0.4 * (season_final["total_fta"] - season_final["total_ftm"])
        + 0.7 * season_final["total_reb_off"]
        + 0.3 * season_final["total_reb_def"]
        + season_final["total_steals"]
        + 0.7 * season_final["total_assists"]
        + 0.7 * season_final["total_blocks"]
        - 0.4 * season_final["total_pf"]
        - season_final["total_tov"]
    )
    season_final["game_score_per36"] = _safe_div(
        season_final["season_game_score_total"] * 36.0, 
        season_final["total_minutes"]
    )

    # Add PER if available
    if player_game_per is not None and not player_game_per.empty:
        per_df = player_game_per.copy()
        per_df = _to_numeric(per_df, ["numMinutes","uPER","aPER"])

        # Minutes-weighted PER aggregation
        def calculate_weighted_per(group):
            total_minutes = group["numMinutes"].sum()
            if total_minutes > 0:
                return pd.Series({
                    "minutes_per_season": total_minutes,
                    "season_aPER": (group["aPER"] * group["numMinutes"]).sum() / total_minutes,
                    "season_uPER": (group["uPER"] * group["numMinutes"]).sum() / total_minutes,
                })
            return pd.Series({
                "minutes_per_season": 0,
                "season_aPER": 0,
                "season_uPER": 0,
            })

        per_aggregated = (per_df.groupby(["personId","season"])
                         .apply(calculate_weighted_per, include_groups=False)
                         .reset_index())

        # League average aPER for scaling
        def league_aper_by_season(group):
            total_minutes = group["numMinutes"].sum()
            if total_minutes > 0:
                return (group["aPER"] * group["numMinutes"]).sum() / total_minutes
            return 0

        league_aper = (per_df.groupby("season")
                      .apply(league_aper_by_season, include_groups=False)
                      .rename("lg_aPER")
                      .reset_index())

        per_aggregated = per_aggregated.merge(league_aper, on="season", how="left")
        per_aggregated["per_scale"] = _safe_div(
            pd.Series(15.0, index=per_aggregated.index), 
            per_aggregated["lg_aPER"]
        )
        per_aggregated["season_PER"] = per_aggregated["season_aPER"] * per_aggregated["per_scale"]

        season_final = season_final.merge(
            per_aggregated[["personId","season","season_uPER","season_aPER","season_PER"]],
            on=["personId","season"], how="left"
        )

    # NEW: Add VORP and EWA if BPM available
    if player_game_bpm is not None and not player_game_bpm.empty:
        bpm_df = player_game_bpm.copy()
        bpm_df = _to_numeric(bpm_df, ["numMinutes","game_bpm"])

        # Minutes-weighted BPM aggregation
        def calculate_weighted_bpm(group):
            total_minutes = group["numMinutes"].sum()
            if total_minutes > 0:
                return pd.Series({
                    "season_BPM": (group["game_bpm"] * group["numMinutes"]).sum() / total_minutes,
                })
            return pd.Series({"season_BPM": 0})

        bpm_aggregated = (bpm_df.groupby(["personId","season"])
                         .apply(calculate_weighted_bpm, include_groups=False)
                         .reset_index())

        season_final = season_final.merge(
            bpm_aggregated[["personId","season","season_BPM"]],
            on=["personId","season"], how="left"
        )
        
        # Calculate team games per season (typically 82, but can vary)
        team_games_per_season = season_final.groupby("season")["games_played"].max().reset_index()
        team_games_per_season = team_games_per_season.rename(columns={"games_played": "season_max_games"})
        season_final = season_final.merge(team_games_per_season, on="season", how="left")
        
        # Calculate VORP
        # VORP = (BPM - (-2.0)) * (% of possessions played) * (team games / 82) / 2.77
        # Simplified: VORP = (BPM + 2.0) * minutes_pct * games_pct / 2.77
        replacement_level = -2.0
        minutes_in_season = season_final["total_minutes"]
        max_possible_minutes = season_final["season_max_games"] * 48  # 48 minutes per game max
        
        minutes_pct = _safe_div(minutes_in_season, max_possible_minutes)
        games_pct = _safe_div(season_final["season_max_games"], pd.Series(82, index=season_final.index))
        
        season_final["season_VORP"] = _safe_div(
            (season_final["season_BPM"] - replacement_level) * minutes_pct * games_pct,
            pd.Series(2.77, index=season_final.index)  # Points per win divisor
        )
        
        # Calculate EWA (Estimated Wins Added)
        # EWA = VORP (since VORP is already in wins above replacement)
        season_final["season_EWA"] = season_final["season_VORP"].copy()

    # Apply minutes filter
    before_filter = len(season_final)
    season_final = season_final.loc[season_final["total_minutes"] >= minutes_total_minimum_per_season].copy()
    after_filter = len(season_final)
    if before_filter != after_filter:
        print(f"Applied {minutes_total_minimum_per_season} minute filter: removed {before_filter - after_filter:,} players")

    return season_final

# Unified ranking system
def rank_seasons_by_metric(
    player_season_df: pd.DataFrame,
    metric: str,
    top_n: int = 10,
    middle_n: int = 10,
    bottom_n: int = 10,
    tie_breaker: str = "python",
    include_context: bool = False,
) -> dict[str, pd.DataFrame]:
    """Rank player seasons by specified metric"""
    
    df = player_season_df.copy()
    
    # Define metric configurations
    metric_config = {
        "pie": {
            "main_col": "season_pie",
            "display_cols": ["player_name","season","season_pie","season_pie_pct","games_played","total_minutes"],
            "context_cols": [
                "total_points", "total_fgm", "total_ftm", "total_fga", "total_fta",
                "total_reb_def", "total_reb_off", "total_assists", "total_steals", 
                "total_blocks", "total_pf", "total_tov", "total_minutes"
            ],
            "title": "PIE"
        },
        "gs36": {
            "main_col": "game_score_per36", 
            "display_cols": ["player_name","season","game_score_per36","games_played","total_minutes"],
            "context_cols": [
                "pts_per36", "fgm_per36", "fga_per36", "ftm_per36", "fta_per36",
                "oreb_per36", "dreb_per36", "stl_per36", "ast_per36", "blk_per36",
                "pf_per36", "tov_per36"
            ],
            "title": "Game Score per 36"
        },
        "per": {
            "main_col": "season_PER",
            "display_cols": ["player_name","season","season_PER","games_played","total_minutes"],
            "context_cols": [
                "season_uPER", "season_aPER", "total_points", "total_fgm", "total_fga",
                "total_ftm", "total_fta", "total_3pm", "total_assists", "total_steals",
                "total_blocks", "total_reb_off", "total_rebounds", "total_tov", "total_pf"
            ],
            "title": "PER"
        },
        "vorp": {  # NEW: VORP configuration
            "main_col": "season_VORP",
            "display_cols": ["player_name","season","season_VORP","games_played","total_minutes"],
            "context_cols": [
                "season_BPM", "total_points", "total_assists", "total_rebounds", 
                "total_steals", "total_blocks", "total_tov", "ts_pct"
            ],
            "title": "VORP"
        },
        "ewa": {  # NEW: EWA configuration
            "main_col": "season_EWA",
            "display_cols": ["player_name","season","season_EWA","games_played","total_minutes"],
            "context_cols": [
                "season_BPM", "season_VORP", "total_points", "total_assists", 
                "total_rebounds", "total_steals", "total_blocks", "total_tov", "ts_pct"
            ],
            "title": "EWA"
        }
    }
    
    if metric not in metric_config:
        raise ValueError(f"Metric must be one of: {list(metric_config.keys())}")
    
    config = metric_config[metric]
    main_col = config["main_col"]
    
    # Check required columns
    required = ["personId","player_name","season",main_col,"total_minutes","games_played"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for {metric} ranking: {missing}")

    # Set up sorting
    if tie_breaker == "duckdb" and "ts_pct" not in df.columns:
        df["ts_pct"] = np.nan
    
    sort_cols = [(main_col, False), ("total_minutes", False)]
    if tie_breaker == "duckdb":
        sort_cols.append(("ts_pct", False))
    else:
        sort_cols.append(("games_played", False))
    sort_cols.append(("player_name", True))

    # Get rankings
    sort_columns = [col for col, _ in sort_cols]
    sort_ascending = [asc for _, asc in sort_cols]
    
    top = df.sort_values(sort_columns, ascending=sort_ascending).head(top_n).copy()
    bottom = df.sort_values(
        [main_col] + sort_columns[1:],
        ascending=[True] + sort_ascending[1:]
    ).head(bottom_n).copy()

    # Middle (closest to median)
    median_val = df[main_col].median(skipna=True)
    df["dist_to_median"] = (df[main_col] - median_val).abs()
    middle = df.sort_values(
        ["dist_to_median"] + sort_columns,
        ascending=[True] + sort_ascending
    ).head(middle_n).drop(columns=["dist_to_median"]).copy()

    def format_basic_ranking(ranking_df: pd.DataFrame) -> pd.DataFrame:
        """Format basic ranking display"""
        result = ranking_df[config["display_cols"]].copy()
        
        # Round numeric columns appropriately
        if main_col in result.columns:
            if metric == "pie":
                result["season_pie"] = result["season_pie"].round(6)
                result["season_pie_pct"] = result["season_pie_pct"].round(2)
            elif metric == "gs36":
                result["game_score_per36"] = result["game_score_per36"].round(6)
            elif metric == "per":
                result["season_PER"] = result["season_PER"].round(3)
            elif metric in ["vorp", "ewa"]:  # NEW: VORP and EWA rounding
                result[main_col] = result[main_col].round(3)
        
        if "total_minutes" in result.columns:
            result["total_minutes"] = result["total_minutes"].round(1)
            
        return result.reset_index(drop=True)

    def format_context_ranking(ranking_df: pd.DataFrame) -> pd.DataFrame:
        """Format ranking with context columns"""
        if not include_context:
            return format_basic_ranking(ranking_df)
            
        result = ranking_df.copy().reset_index(drop=True)
        result.insert(0, "Rank", range(1, len(result) + 1))
        
        # Select relevant columns
        base_cols = ["Rank", "player_name", "season"]
        metric_cols = [main_col]
        if metric == "pie":
            metric_cols.append("season_pie_pct")
        context_cols = [c for c in config["context_cols"] if c in result.columns]
        
        selected_cols = base_cols + metric_cols + context_cols
        result = result[[c for c in selected_cols if c in result.columns]]
        
        # Apply rounding
        if metric == "pie":
            if "season_pie" in result.columns:
                result["season_pie"] = result["season_pie"].round(6)
            if "season_pie_pct" in result.columns:
                result["season_pie_pct"] = result["season_pie_pct"].round(2)
        elif metric == "gs36":
            if "game_score_per36" in result.columns:
                result["game_score_per36"] = result["game_score_per36"].round(6)
            # Round per-36 stats to 3 decimals
            for col in context_cols:
                if col.endswith("_per36") and col in result.columns:
                    result[col] = result[col].round(3)
        elif metric == "per":
            for col in ["season_PER", "season_uPER", "season_aPER"]:
                if col in result.columns:
                    result[col] = result[col].round(3)
        elif metric in ["vorp", "ewa"]:  # NEW: VORP and EWA context formatting
            # Round VORP/EWA and related metrics
            for col in [main_col, "season_BPM", "season_VORP", "season_EWA"]:
                if col in result.columns:
                    result[col] = result[col].round(3)
            if "ts_pct" in result.columns:
                result["ts_pct"] = result["ts_pct"].round(3)
        
        return result

    return {
        "top": format_context_ranking(top) if include_context else format_basic_ranking(top),
        "middle": format_context_ranking(middle) if include_context else format_basic_ranking(middle), 
        "bottom": format_context_ranking(bottom) if include_context else format_basic_ranking(bottom)
    }

# Output formatting and export functions
def print_rankings(player_season_df: pd.DataFrame, metric: str, **kwargs) -> dict[str, pd.DataFrame]:
    """Print formatted rankings for a metric"""
    
    rankings = rank_seasons_by_metric(player_season_df, metric=metric, **kwargs)
    
    metric_titles = {"pie": "PIE", "gs36": "Game Score per 36", "per": "PER"}
    title = metric_titles.get(metric, metric.upper())
    
    context_suffix = " (with context)" if kwargs.get("include_context", False) else ""
    
    print(f"\n=== Top 10 seasons by {title}{context_suffix} ===")
    print(rankings["top"].to_string(index=False))
    print(f"\n=== Middle 10 seasons by {title}{context_suffix} ===")
    print(rankings["middle"].to_string(index=False))
    print(f"\n=== Bottom 10 seasons by {title}{context_suffix} ===") 
    print(rankings["bottom"].to_string(index=False))
    
    return rankings

def format_for_submission(rankings: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Convert rankings to submission format (Rank, Player, Season)"""
    
    def format_submission_df(df: pd.DataFrame) -> pd.DataFrame:
        result = df[["player_name","season"]].copy()
        result.insert(0, "Rank", range(1, len(result) + 1))
        result = result.rename(columns={"player_name":"Player","season":"Season"})
        return result
    
    return {
        "top": format_submission_df(rankings["top"]),
        "middle": format_submission_df(rankings["middle"]),
        "bottom": format_submission_df(rankings["bottom"])
    }

# Output formatting and export functions
# NOTE: Exports rankings to one or more formats. Backward-compatible with string input.
def export_rankings(
    rankings: dict[str, pd.DataFrame],
    output_dir: Path | str,
    filename_prefix: str,
    file_format: str | Iterable[str] = "csv"
) -> None:
    """Export rankings dict to one or more file formats.

    Supported formats:
      - "csv": writes <prefix>_<top|middle|bottom>.csv
      - "txt": writes monospace tables via DataFrame.to_string()

    You can pass a single string (e.g., "csv") or an iterable like ("csv", "txt").
    """

    # -- normalize file_format to an iterable without breaking old calls
    if isinstance(file_format, str):
        formats = [file_format.lower()]
    else:
        formats = [str(fmt).lower() for fmt in file_format]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for rank_type, df in rankings.items():
        for fmt in formats:
            if fmt == "csv":
                filepath = output_path / f"{filename_prefix}_{rank_type}.csv"
                df.to_csv(filepath, index=False)
            elif fmt == "txt":
                filepath = output_path / f"{filename_prefix}_{rank_type}.txt"
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(df.to_string(index=False) + "\n")
            else:
                raise ValueError(f"Unsupported file format: {fmt}")

            print(f"Exported {rank_type} rankings to: {filepath}")


# Data loading functions
def load_nba_data(table_names: Iterable[str] = IMPORTANT_TABLES) -> Dict[str, pd.DataFrame]:
    """Load NBA data tables from Kaggle"""
    
    result = {}
    failed = {}

    for table in table_names:
        csv_filename = KAGGLE_TABLE_TO_CSV.get(table)
        if not csv_filename:
            failed[table] = "No CSV mapping found"
            continue
            
        try:
            df = kh.dataset_load(
                KDA.PANDAS, KAGGLE_DATASET, csv_filename, 
                pandas_kwargs={"low_memory": False}
            )
            result[table] = df
            print(f"Loaded {csv_filename} -> '{table}' ({len(df):,} rows)")
        except Exception as e:
            failed[table] = str(e)

    if failed:
        error_msg = "\n".join([f"- {t}: {e}" for t, e in failed.items()])
        raise RuntimeError(f"Failed to load tables:\n{error_msg}")
        
    return result

def build_player_game_data(
    start_season: int = CFG_START_SEASON,
    season_type: str = CFG_SEASON_TYPE, 
    minutes_minimum: int = CFG_MIN_SEASON_MINUTES,
) -> pd.DataFrame:
    """Build player-game data with PIE calculations"""
    
    data = load_nba_data(["PlayerStatistics","TeamStatistics"])
    _, player_stats, _ = enforce_criteria_python(
        None, data["PlayerStatistics"], data["TeamStatistics"],
        start_season=start_season, season_type=season_type,
        minutes_total_minimum_per_season=minutes_minimum,
        defer_minutes_gate=True,
    )
    return compute_player_game_pie(player_stats)

def build_per_data(
    start_season: int = CFG_START_SEASON,
    season_type: str = CFG_SEASON_TYPE,
    minutes_minimum: int = CFG_MIN_SEASON_MINUTES
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build complete PER calculation data"""
    
    print("Loading data for PER calculations...")
    data = load_nba_data(["PlayerStatistics", "TeamStatistics"])

    _, player_stats, _ = enforce_criteria_python(
        None, data["PlayerStatistics"], data["TeamStatistics"],
        start_season=start_season, season_type=season_type,
        minutes_total_minimum_per_season=minutes_minimum,
        defer_minutes_gate=True
    )
    print(f"Filtered player stats: {player_stats.shape}")

    team_game_data = compute_team_game_totals_and_pace(player_stats)
    print(f"Team game data: {team_game_data.shape}")

    league_constants = compute_league_constants_per_season(team_game_data)
    print(f"League constants: {league_constants.shape}")

    player_per_data = compute_player_game_per(player_stats, team_game_data, league_constants)
    print(f"Player PER data: {player_per_data.shape}")

    return team_game_data, league_constants, player_per_data

# NEW: Build VORP/EWA data
def build_vorp_ewa_data(
    start_season: int = CFG_START_SEASON,
    season_type: str = CFG_SEASON_TYPE,
    minutes_minimum: int = CFG_MIN_SEASON_MINUTES
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build complete VORP and EWA calculation data"""
    
    print("Loading data for VORP/EWA calculations...")
    data = load_nba_data(["PlayerStatistics", "TeamStatistics"])

    _, player_stats, _ = enforce_criteria_python(
        None, data["PlayerStatistics"], data["TeamStatistics"],
        start_season=start_season, season_type=season_type,
        minutes_total_minimum_per_season=minutes_minimum,
        defer_minutes_gate=True
    )
    print(f"Filtered player stats: {player_stats.shape}")

    team_game_data = compute_team_game_totals_and_pace(player_stats)
    print(f"Team game data: {team_game_data.shape}")

    league_constants = compute_league_constants_per_season(team_game_data)
    print(f"League constants: {league_constants.shape}")

    player_bpm_data = compute_player_game_bpm(player_stats, team_game_data, league_constants)
    print(f"Player BPM data: {player_bpm_data.shape}")

    return team_game_data, league_constants, player_bpm_data, player_stats

# Main execution
if __name__ == "__main__":
    print("=== NBA Player Rankings Analysis ===")

    # Build player-game data with PIE
    print("\n1. Building player-game data...")
    try:
        player_game_data = build_player_game_data()
        print(f"Success: {player_game_data.shape} player-game records")
    except Exception as e:
        print(f"Error building player-game data: {e}")
        player_game_data = None

    # Build PER data
    print("\n2. Building PER data...")
    try:
        team_games, league_consts, player_per = build_per_data()
        print(f"Success: PER data complete")
    except Exception as e:
        print(f"Error building PER data: {e}")
        player_per = None

    # NEW: Build VORP/EWA data
    print("\n3. Building VORP/EWA data...")
    try:
        team_games_vorp, league_consts_vorp, player_bpm, player_stats_vorp = build_vorp_ewa_data()
        print(f"Success: VORP/EWA data complete")
    except Exception as e:
        print(f"Error building VORP/EWA data: {e}")
        player_bpm = None

    # Build season-level data
    if player_game_data is not None:
        print("\n4. Building season-level data...")
        try:
            team_data = load_nba_data(["TeamStatistics"])
            season_data = build_player_season_table_python(
                player_game_data, 
                team_data["TeamStatistics"],
                player_game_per=player_per,
                player_game_bpm=player_bpm  # NEW parameter
            )
            print(f"Success: {season_data.shape} player-season records")

            # Save dataset
            season_data.to_parquet(CFG.ml_dataset_path, index=False)

            # Generate and export rankings for each metric (UPDATED with VORP and EWA)
            for metric in ["pie", "gs36", "per", "vorp", "ewa"]:
                required_col = {
                    "pie": "season_pie",
                    "gs36": "game_score_per36", 
                    "per": "season_PER",
                    "vorp": "season_VORP",
                    "ewa": "season_EWA"
                }[metric]
                
                if required_col not in season_data.columns:
                    print(f"Skipping {metric.upper()} - data not available")
                    continue

                print(f"\n=== {metric.upper()} Rankings ===")
                
                # Basic rankings
                basic_rankings = rank_seasons_by_metric(season_data, metric, tie_breaker="duckdb")
                submission_format = format_for_submission(basic_rankings)
                
                # Print and export submission format
                title = {
                    "pie": "PIE", 
                    "gs36": "Game Score per 36", 
                    "per": "PER",
                    "vorp": "VORP",
                    "ewa": "EWA"
                }[metric]
                print(f"\n=== Top 10 seasons by {title} ===")
                print(submission_format["top"].to_string(index=False))
                print(f"\n=== Middle 10 seasons by {title} ===")
                print(submission_format["middle"].to_string(index=False))
                print(f"\n=== Bottom 10 seasons by {title} ===") 
                print(submission_format["bottom"].to_string(index=False))
                
                export_rankings(
                    submission_format, CFG.processed_dir,
                    f"nba_{metric}_rankings", "txt"
                )
                
                # Context rankings
                context_rankings = rank_seasons_by_metric(
                    season_data, metric, tie_breaker="duckdb", include_context=True
                )

                # NEW: print the contextual rankings as tables in the console
                print_rankings(
                    season_data,
                    metric,
                    tie_breaker="duckdb",
                    include_context=True
                )

                # Export contextual rankings to CSV (existing) AND readable TXT tables (new)
                export_rankings(
                    context_rankings,
                    CFG.processed_dir,
                    f"nba_{metric}_rankings_context",
                    file_format=("csv", "txt")   # was: "csv"
                )


        except Exception as e:
            print(f"Error in season analysis: {e}")

    print("\n=== Analysis Complete ===")
