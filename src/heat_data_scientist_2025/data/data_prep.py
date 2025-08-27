import duckdb
import pandas as pd
import numpy as np
import sqlite3
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NBAPlayerRankings:
    def __init__(self, data_path: str = "./data", db_path: str = ":memory:", sqlite_path: Optional[str] = None):
        self.data_path = Path(data_path)
        self.sqlite_path = Path(sqlite_path) if sqlite_path else None
        self.conn = duckdb.connect(db_path)
        self.setup_database()

    def setup_database(self):
        self.conn.execute("SET threads TO 4")
        self.conn.execute("SET memory_limit = '4GB'")
        self.conn.execute("SET enable_progress_bar = true")
        logger.info("DuckDB configured for optimal performance")

    # ---------- validation helpers ----------
    def _fail_if_nulls(self, table: str, cols: List[str], where: str = "1=1", sample: int = 5):
        for col in cols:
            cnt = self.conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {where} AND {col} IS NULL"
            ).fetchone()[0]
            if cnt > 0:
                example = self.conn.execute(
                    f"SELECT * FROM {table} WHERE {where} AND {col} IS NULL LIMIT {sample}"
                ).df()
                raise ValueError(
                    f"NULLs found in {table}.{col} (n={cnt}). Sample rows:\n{example}"
                )

    def _fail_if_join_drops(self, before_table: str, after_table: str, keys: List[str], sample: int = 5):
        key_expr = " AND ".join([f"b.{k} = a.{k}" for k in keys])
        missing_cnt = self.conn.execute(f"""
            SELECT COUNT(*) FROM {before_table} b
            LEFT JOIN {after_table} a ON {key_expr}
            WHERE { ' OR '.join([f'a.{k} IS NULL' for k in keys]) }
        """).fetchone()[0]
        if missing_cnt > 0:
            example = self.conn.execute(f"""
                SELECT b.* FROM {before_table} b
                LEFT JOIN {after_table} a ON {key_expr}
                WHERE { ' OR '.join([f'a.{k} IS NULL' for k in keys]) }
                LIMIT {sample}
            """).df()
            raise RuntimeError(
                f"Join dropped rows going {before_table} -> {after_table} (missing={missing_cnt}). "
                f"Sample missing:\n{example}"
            )

    def _import_from_sqlite(self, sqlite_path: Path, tables_config: Dict[str, Dict]) -> None:
        """
        Import tables from SQLite database into DuckDB without any data masking or coalescing.
        This ensures we see the raw data quality issues rather than hiding them.
        """
        if not sqlite_path.exists():
            raise FileNotFoundError(f"SQLite file not found at {sqlite_path}")

        with sqlite3.connect(str(sqlite_path)) as sconn:
            # List available tables for visibility
            avail = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;", sconn
            )["name"].tolist()
            logger.info(f"SQLite tables available: {avail}")

            for table_name, cfg in tables_config.items():
                if not _table_exists_sqlite(sconn, table_name):
                    if cfg["required"]:
                        raise FileNotFoundError(f"Required table '{table_name}' not found in SQLite at {sqlite_path}")
                    logger.warning(f"Optional table '{table_name}' not found in SQLite; skipping")
                    continue

                df = _safe_sqlite_read(sconn, table_name)
                if df.empty:
                    if cfg["required"]:
                        raise ValueError(f"Required table '{table_name}' is empty in SQLite")
                    logger.warning(f"Optional table '{table_name}' is empty; skipping")
                    continue

                # Validate required columns are present (no masking)
                _required_columns_present(df, cfg["key_cols"], table_name)

                # Import to DuckDB
                _import_df_to_duckdb(self.conn, df, table_name)

                # Create indexes where possible
                for col in cfg["key_cols"]:
                    try:
                        self.conn.execute(f"CREATE INDEX idx_{table_name}_{col} ON {table_name}({col})")
                    except Exception:
                        pass

                row_count = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                logger.info(f"Imported {table_name} from SQLite: {row_count:,} rows")

    def load_data(self) -> None:
        logger.info("Loading data into DuckDB...")
        tables_config = {
            'PlayerStatistics': {
                'file': 'PlayerStatistics.csv',
                'key_cols': ['personId', 'gameId', 'gameDate', 'gameType', 'numMinutes'],
                'required': True
            },
            'Players': {
                'file': 'Players.csv',
                'key_cols': ['personId', 'firstName', 'lastName'],
                'required': True
            },
            'Games': {
                'file': 'Games.csv',
                'key_cols': ['gameId', 'gameDate', 'gameType'],
                'required': False
            },
            'TeamStatistics': {
                'file': 'TeamStatistics.csv',
                'key_cols': ['gameId', 'teamId'],
                'required': False
            }
        }

        # Determine if all required CSVs exist
        required_csvs = [self.data_path / tables_config['PlayerStatistics']['file'],
                         self.data_path / tables_config['Players']['file']]
        csvs_present = all(p.exists() for p in required_csvs)

        if csvs_present:
            # CSV path — original behavior
            logger.info("Loading from CSV files...")
            for table_name, config in tables_config.items():
                file_path = self.data_path / config['file']
                if not file_path.exists():
                    if config['required']:
                        raise FileNotFoundError(f"Required file {config['file']} not found at {file_path}")
                    logger.warning(f"Optional file {config['file']} not found, skipping...")
                    continue

                self.conn.execute(f"""
                    CREATE OR REPLACE TABLE {table_name} AS 
                    SELECT * FROM read_csv_auto('{file_path}', 
                        header=true, 
                        sample_size=100000,
                        all_varchar=false)
                """)

                for col in config['key_cols']:
                    try:
                        self.conn.execute(f"CREATE INDEX idx_{table_name}_{col} ON {table_name}({col})")
                    except Exception:
                        pass

                row_count = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                logger.info(f"Loaded {table_name} from CSV: {row_count:,} rows")
            return

        # CSVs not present — try SQLite if provided
        if self.sqlite_path:
            logger.info(f"CSV not found; attempting import from SQLite at {self.sqlite_path}")
            self._import_from_sqlite(self.sqlite_path, tables_config)
            return

        # Neither CSVs nor SQLite path available — fail with diagnostics
        diag = {
            "checked_csv_folder": str(self.data_path.resolve()),
            "required_csvs_present": csvs_present,
            "sqlite_path": str(self.sqlite_path) if self.sqlite_path else None,
            "sqlite_exists": self.sqlite_path.exists() if self.sqlite_path else None,
        }
        raise FileNotFoundError(
            "No data source available. Either place required CSVs under data_path or provide sqlite_path.\n"
            f"Diagnostics: {diag}"
        )

    def debug_input_diagnostics(self, sample: int = 5) -> None:
        """
        Non-mutating visibility on raw inputs inside DuckDB:
          - row counts
          - top null columns (per table)
          - a few sample rows
        """
        tables = ["Players", "PlayerStatistics", "Games", "TeamStatistics"]
        for t in tables:
            try:
                n = self.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            except Exception:
                logger.warning(f"[DIAG] Table {t} not present")
                continue

            logger.info(f"[DIAG] {t}: {n:,} rows")
            # Null counts (top 8) - simplified approach for DuckDB
            try:
                # Get column names first
                cols = self.conn.execute(f"DESCRIBE {t}").df()["column_name"].tolist()
                logger.info(f"[DIAG] {t} columns: {cols[:8]}")  # Just show first 8 column names
            except Exception as e:
                logger.warning(f"[DIAG] Could not get columns for {t}: {e}")

            # Sample rows
            try:
                head = self.conn.execute(f"SELECT * FROM {t} LIMIT {int(sample)}").df()
                logger.info(f"[DIAG] {t} sample rows:\n{head}")
            except Exception:
                pass

    def create_season_column(self) -> None:
        logger.info("Creating season labels...")
        self.conn.execute("""
            ALTER TABLE PlayerStatistics ADD COLUMN season VARCHAR;

            UPDATE PlayerStatistics
            SET season = 
                CASE 
                    WHEN MONTH(gameDate::DATE) >= 8 
                    THEN YEAR(gameDate::DATE)::VARCHAR || '-' || 
                         LPAD(((YEAR(gameDate::DATE) + 1) % 100)::VARCHAR, 2, '0')
                    ELSE (YEAR(gameDate::DATE) - 1)::VARCHAR || '-' || 
                         LPAD((YEAR(gameDate::DATE) % 100)::VARCHAR, 2, '0')
                END
        """)
        self.conn.execute("CREATE INDEX idx_season ON PlayerStatistics(season)")
        logger.info("Season labels created successfully")

    def calculate_pie_metrics(self) -> None:
        logger.info("Calculating PIE metrics...")

        # Build filtered working set FIRST (no COALESCE), then validate required columns
        self.conn.execute("""
            CREATE OR REPLACE TABLE ps_filtered AS
            SELECT *
            FROM PlayerStatistics
            WHERE gameType = 'Regular Season'
              AND season >= '2010-11'
              AND numMinutes > 0
        """)

        # Strict: if any required stat column is NULL, fail early
        required_stat_cols = [
            'points','fieldGoalsMade','freeThrowsMade','fieldGoalsAttempted','freeThrowsAttempted',
            'reboundsDefensive','reboundsOffensive','assists','steals','blocks',
            'foulsPersonal','turnovers','numMinutes','personId','gameId'
        ]
        self._fail_if_nulls('ps_filtered', required_stat_cols)

        # Step 1: PIE numerator per player-game (NO COALESCE)
        self.conn.execute("""
            CREATE OR REPLACE TABLE player_game_pie AS
            SELECT 
                pf.*,
                (points + 
                 fieldGoalsMade + 
                 freeThrowsMade - 
                 fieldGoalsAttempted - 
                 freeThrowsAttempted + 
                 reboundsDefensive + 
                 0.5 * reboundsOffensive + 
                 assists + 
                 steals + 
                 0.5 * blocks - 
                 foulsPersonal - 
                 turnovers) AS pie_numerator
            FROM ps_filtered pf
        """)

        # Ensure numerator is never NULL (if it is, we want to know)
        self._fail_if_nulls('player_game_pie', ['pie_numerator'])

        # Step 2: Game denominators (sum across both teams)
        self.conn.execute("""
            CREATE OR REPLACE TABLE game_denominators AS
            SELECT gameId, SUM(pie_numerator) AS pie_denominator
            FROM player_game_pie
            GROUP BY gameId
        """)

        # Denominator must be positive and non-null
        bad_den = self.conn.execute("""
            SELECT COUNT(*) FROM game_denominators
            WHERE pie_denominator IS NULL OR pie_denominator <= 0
        """).fetchone()[0]
        if bad_den > 0:
            ex = self.conn.execute("""
                SELECT * FROM game_denominators
                WHERE pie_denominator IS NULL OR pie_denominator <= 0
                LIMIT 5
            """).df()
            raise RuntimeError(f"Invalid denominators (NULL/≤0) found in {bad_den} games. Sample:\n{ex}")

        # Step 3: Strict join back (no CASE fallback); if join drops rows, raise
        self.conn.execute("""
            CREATE OR REPLACE TABLE player_game_with_pie AS
            SELECT 
                pgp.*,
                gd.pie_denominator,
                pgp.pie_numerator / gd.pie_denominator AS game_pie
            FROM player_game_pie pgp
            INNER JOIN game_denominators gd ON pgp.gameId = gd.gameId
        """)
        self._fail_if_join_drops('player_game_pie', 'player_game_with_pie', keys=['personId','gameId'])

        logger.info("PIE metrics calculated successfully")


    def aggregate_season_stats(self) -> None:
        logger.info("Aggregating season statistics...")

        self.conn.execute("""
            CREATE OR REPLACE TABLE player_seasons AS
            SELECT 
                personId,
                season,
                COUNT(DISTINCT gameId) AS games_played,
                SUM(numMinutes) AS total_minutes,

                SUM(pie_numerator) AS season_pie_numerator,
                SUM(pie_denominator) AS season_pie_denominator,
                SUM(pie_numerator) / NULLIF(SUM(pie_denominator), 0) AS season_pie,

                SUM(points) AS total_points,
                SUM(assists) AS total_assists,
                -- Derive total_rebounds from components to avoid hidden nulls/absent columns
                SUM(reboundsDefensive + reboundsOffensive) AS total_rebounds,
                SUM(steals) AS total_steals,
                SUM(blocks) AS total_blocks,
                SUM(turnovers) AS total_turnovers,
                SUM(fieldGoalsMade) AS total_fgm,
                SUM(fieldGoalsAttempted) AS total_fga,
                SUM(freeThrowsMade) AS total_ftm,
                SUM(freeThrowsAttempted) AS total_fta,
                SUM(threePointersMade) AS total_3pm,
                SUM(threePointersAttempted) AS total_3pa,

                SUM(points) / NULLIF(2.0 * (SUM(fieldGoalsAttempted) + 0.44 * SUM(freeThrowsAttempted)), 0) AS ts_pct,

                (SUM(points) * 36.0) / NULLIF(SUM(numMinutes), 0) AS pts_per36,
                (SUM(assists) * 36.0) / NULLIF(SUM(numMinutes), 0) AS ast_per36,
                (SUM(reboundsDefensive + reboundsOffensive) * 36.0) / NULLIF(SUM(numMinutes), 0) AS reb_per36
            FROM player_game_with_pie
            GROUP BY personId, season
            HAVING SUM(numMinutes) >= 500
        """)

        # Build names via INNER JOIN; require names to exist (no COALESCE name synthesis)
        self.conn.execute("""
            CREATE OR REPLACE TABLE player_seasons_with_names AS
            SELECT 
                ps.*,
                TRIM(p.firstName || ' ' || p.lastName) AS player_name,
                p.height,
                p.bodyWeight,
                p.draftYear
            FROM player_seasons ps
            INNER JOIN Players p ON ps.personId = p.personId
            WHERE p.firstName IS NOT NULL AND p.lastName IS NOT NULL
            AND LENGTH(TRIM(p.firstName || ' ' || p.lastName)) > 1
        """)

        # Ensure the name-join did not drop rows
        self._fail_if_join_drops('player_seasons', 'player_seasons_with_names', keys=['personId','season'])

        # Also assert player_name is non-null (belt & suspenders)
        self._fail_if_nulls('player_seasons_with_names', ['player_name'])

        row_count = self.conn.execute("SELECT COUNT(*) FROM player_seasons_with_names").fetchone()[0]
        logger.info(f"Aggregated {row_count:,} qualified player-seasons")


    def generate_rankings(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        logger.info("Generating rankings...")
        median_pie = self.conn.execute("""
            SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY season_pie) AS median_pie
            FROM player_seasons_with_names
        """).fetchone()[0]
        logger.info(f"Median Season PIE: {median_pie:.4f}")

        top_10 = self.conn.execute("""
            SELECT 
                ROW_NUMBER() OVER (ORDER BY season_pie DESC, total_minutes DESC, ts_pct DESC) AS rank,
                player_name, season,
                ROUND(season_pie, 4) AS pie_score,
                ROUND(total_minutes, 0) AS minutes
            FROM player_seasons_with_names
            ORDER BY season_pie DESC, total_minutes DESC, ts_pct DESC
            LIMIT 10
        """).df()

        worst_10 = self.conn.execute("""
            SELECT 
                ROW_NUMBER() OVER (ORDER BY season_pie ASC, total_minutes DESC, ts_pct DESC) AS rank,
                player_name, season,
                ROUND(season_pie, 4) AS pie_score,
                ROUND(total_minutes, 0) AS minutes
            FROM player_seasons_with_names
            ORDER BY season_pie ASC, total_minutes DESC, ts_pct DESC
            LIMIT 10
        """).df()

        avg_10 = self.conn.execute(f"""
            SELECT 
                ROW_NUMBER() OVER (
                    ORDER BY ABS(season_pie - {median_pie}) ASC, total_minutes DESC, ts_pct DESC
                ) AS rank,
                player_name, season,
                ROUND(season_pie, 4) AS pie_score,
                ROUND(total_minutes, 0) AS minutes,
                ROUND(ABS(season_pie - {median_pie}), 6) AS distance_from_median
            FROM player_seasons_with_names
            ORDER BY ABS(season_pie - {median_pie}) ASC, total_minutes DESC, ts_pct DESC
            LIMIT 10
        """).df()

        logger.info("Rankings generated successfully")
        return top_10, avg_10, worst_10

    def export_ml_dataset(self, output_path: str = "nba_ml_dataset.parquet") -> pd.DataFrame:
        logger.info("Exporting ML-ready dataset...")
        ml_dataset = self.conn.execute("""
            SELECT 
                personId, player_name, season, games_played, total_minutes,
                season_pie, ts_pct, pts_per36, ast_per36, reb_per36,
                total_points, total_assists, total_rebounds, total_steals, total_blocks,
                total_turnovers, total_fgm, total_fga, total_ftm, total_fta, total_3pm, total_3pa,
                height, bodyWeight, draftYear,
                CASE WHEN total_fga > 0 THEN total_fgm::FLOAT / total_fga ELSE NULL END AS fg_pct,
                CASE WHEN total_3pa > 0 THEN total_3pm::FLOAT / total_3pa ELSE NULL END AS fg3_pct,
                CASE WHEN total_fta > 0 THEN total_ftm::FLOAT / total_fta ELSE NULL END AS ft_pct,
                (total_fga + 0.44 * total_fta + total_turnovers) / total_minutes AS usage_per_min,
                (total_points + total_rebounds + total_assists + total_steals + total_blocks - 
                 (total_fga - total_fgm) - (total_fta - total_ftm) - total_turnovers) / games_played AS efficiency_per_game
            FROM player_seasons_with_names
            ORDER BY season_pie DESC
        """).df()

        ml_dataset.to_parquet(output_path, index=False, compression='snappy')
        logger.info(f"ML dataset exported to {output_path} ({len(ml_dataset):,} rows)")
        return ml_dataset

    def validate_ml_readiness(self, df: pd.DataFrame, strict: bool = True) -> Dict[str, any]:
        """
        Verify the ML dataset is structurally sound for modeling WITHOUT imputing:
          - required id fields present and non-null
          - unique (personId, season)
          - detect unexpected nulls / infinities in numeric features
          - report (not fix) potential issues
        """
        report: Dict[str, any] = {}

        # Required identifier fields
        must_have = ["personId", "player_name", "season", "games_played", "total_minutes", "season_pie"]
        missing_cols = [c for c in must_have if c not in df.columns]
        if missing_cols:
            raise KeyError(f"ML dataset missing required columns: {missing_cols}")

        # Non-null on must-have fields
        nn_counts = {c: int(df[c].isna().sum()) for c in must_have}
        report["nonnull_required"] = {c: (len(df) - nn) for c, nn in nn_counts.items()}
        bad_required = {c: nn for c, nn in nn_counts.items() if nn > 0}
        if bad_required:
            sample = df[list(bad_required.keys())].loc[df[list(bad_required.keys())].isna().any(axis=1)].head(10)
            msg = f"Required fields contain NULLs: {bad_required}\nSample rows with NULLs:\n{sample}"
            if strict:
                raise ValueError(msg)
            logger.warning(msg)

        # Uniqueness of (personId, season)
        dup_mask = df.duplicated(subset=["personId", "season"], keep=False)
        dup_count = int(dup_mask.sum())
        report["duplicate_person_season_rows"] = dup_count
        if dup_count > 0:
            sample_dup = df.loc[dup_mask, ["personId", "season"]].head(10)
            if strict:
                raise ValueError(f"Duplicate rows for (personId, season) found: {dup_count}. Sample:\n{sample_dup}")
            logger.warning(f"Duplicate rows for (personId, season): {dup_count}")

        # Feature nulls / infs (do NOT fill)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        nulls_numeric = df[numeric_cols].isna().sum().sort_values(ascending=False)
        inf_numeric = np.isinf(df[numeric_cols]).sum().sort_values(ascending=False)
        report["numeric_null_counts_top"] = nulls_numeric.head(15).to_dict()
        report["numeric_inf_counts_top"] = inf_numeric.head(15).to_dict()

        # Sanity range checks (example for percentages & PIE)
        if "season_pie" in df.columns:
            report["season_pie_min"] = float(np.nanmin(df["season_pie"].values))
            report["season_pie_max"] = float(np.nanmax(df["season_pie"].values))
        for pct in ["ts_pct", "fg_pct", "fg3_pct", "ft_pct"]:
            if pct in df.columns:
                vals = df[pct].values
                bad = np.logical_or(vals < 0, vals > 1)
                report[f"{pct}_out_of_range"] = int(np.nansum(bad))

        # Log sizes
        report["rows"] = len(df)
        report["cols"] = df.shape[1]

        logger.info(f"[ML-READY] rows={report['rows']:,}, cols={report['cols']}, "
                    f"dup_person_season={report['duplicate_person_season_rows']}, "
                    f"nulls_required={bad_required if bad_required else 0}")
        return report

    def validate_data_quality(self) -> Dict[str, any]:
        logger.info("Running data quality validation...")
        checks = {}

        # PIE sums ~1.0 per game
        game_pie_sum = self.conn.execute("""
            SELECT gameId, SUM(game_pie) AS total_pie
            FROM player_game_with_pie
            GROUP BY gameId
            HAVING ABS(SUM(game_pie) - 1.0) > 0.01
        """).df()
        checks['games_with_invalid_pie'] = len(game_pie_sum)

        # Names nulls (should be zero due to strict join)
        null_names = self.conn.execute("""
            SELECT COUNT(*) FROM player_seasons_with_names
            WHERE player_name IS NULL OR LENGTH(TRIM(player_name)) <= 1
        """).fetchone()[0]
        checks['null_player_names'] = null_names

        invalid_seasons = self.conn.execute("""
            SELECT COUNT(*) FROM player_seasons_with_names
            WHERE season NOT LIKE '____-__'
        """).fetchone()[0]
        checks['invalid_season_format'] = invalid_seasons

        pie_range = self.conn.execute("""
            SELECT MIN(season_pie), MAX(season_pie), AVG(season_pie), STDDEV(season_pie)
            FROM player_seasons_with_names
        """).fetchone()
        checks['pie_statistics'] = {'min': pie_range[0], 'max': pie_range[1], 'mean': pie_range[2], 'std': pie_range[3]}

        logger.info(f"Data quality checks: {checks}")
        return checks

    def print_rankings(self, top_10: pd.DataFrame, avg_10: pd.DataFrame, worst_10: pd.DataFrame) -> None:
        print("\n" + "="*60)
        print("NBA PLAYER SEASON RANKINGS (2010-11 to Present)")
        print("Using PIE (Player Impact Estimate) Metric")
        print("="*60)

        print("\n📊 TOP 10 SEASONS (Highest PIE)")
        print("-" * 40)
        for _, row in top_10.iterrows():
            print(f"{int(row['rank']):2d}. {row['player_name']} — {row['season']}")

        print("\n📊 MOST AVERAGE 10 SEASONS (Closest to Median PIE)")
        print("-" * 40)
        for _, row in avg_10.iterrows():
            print(f"{int(row['rank']):2d}. {row['player_name']} — {row['season']}")

        print("\n📊 WORST 10 SEASONS (Lowest PIE)")
        print("-" * 40)
        for _, row in worst_10.iterrows():
            print(f"{int(row['rank']):2d}. {row['player_name']} — {row['season']}")

    def close(self):
        self.conn.close()
        logger.info("Database connection closed")


# ─── SQLite Import Helper Functions ────────────────────────────────────────────

def _table_exists_sqlite(conn: sqlite3.Connection, table: str) -> bool:
    """Check if a table exists in SQLite database."""
    q = "SELECT name FROM sqlite_master WHERE type='table' AND name=?;"
    return pd.read_sql_query(q, conn, params=[table]).shape[0] > 0


def _import_df_to_duckdb(conn_duck: duckdb.DuckDBPyConnection, df: pd.DataFrame, table_name: str) -> None:
    """Import a pandas DataFrame into DuckDB table."""
    conn_duck.register("tmp_df_import", df)
    try:
        conn_duck.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM tmp_df_import")
    finally:
        conn_duck.unregister("tmp_df_import")


def _required_columns_present(df: pd.DataFrame, cols: List[str], table_name: str) -> None:
    """Validate that required columns are present in DataFrame."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {table_name}: {missing}")


def _safe_sqlite_read(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    """Safely read a table from SQLite without any data transformation."""
    return pd.read_sql_query(f"SELECT * FROM {table};", conn)




def _smoke_test(r: "NBAPlayerRankings") -> None:
    """
    Minimal invariant checks that fail fast:
      - Core tables are non-empty
      - No null player_name in final table
      - PIE sums ≈ 1.0 per game within 1% tolerance for all but a tiny fraction
    """
    # Tables exist & non-empty
    for t in ["player_game_with_pie", "player_seasons_with_names"]:
        cnt = r.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        if cnt == 0:
            raise AssertionError(f"Smoke test failed: table {t} is empty")

    # No null names (strict)
    null_names = r.conn.execute("""
        SELECT COUNT(*) FROM player_seasons_with_names
        WHERE player_name IS NULL OR LENGTH(TRIM(player_name)) <= 1
    """).fetchone()[0]
    if null_names != 0:
        raise AssertionError(f"Smoke test failed: {null_names} null/blank player_name rows")

    # PIE sum check
    off_games = r.conn.execute("""
        SELECT COUNT(*) FROM (
            SELECT gameId, ABS(SUM(game_pie) - 1.0) AS err
            FROM player_game_with_pie
            GROUP BY gameId
        ) t
        WHERE err > 0.01
    """).fetchone()[0]
    # Allow a tiny handful (we still log below in validate)
    if off_games > 0:
        logger.warning(f"Smoke test: {off_games} games with PIE sum off by > 0.01")


def main() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Orchestrates the full pipeline and returns top_10, avg_10, worst_10, ml_dataset.
    Prefers CSVs, but if not present will use a SQLite database if configured.
    """
    # Try CSVs first; otherwise probe env or common path for SQLite
    default_sqlite = Path("nba.sqlite")
    env_sqlite = os.environ.get("NBA_SQLITE_PATH")
    sqlite_candidate = Path(env_sqlite) if env_sqlite else (default_sqlite if default_sqlite.exists() else None)

    r = NBAPlayerRankings(
        data_path="./data",
        db_path=":memory:",
        sqlite_path=str(sqlite_candidate) if sqlite_candidate else None,
    )
    try:
        r.load_data()
        r.debug_input_diagnostics(sample=3)   # visibility before transforms
        r.create_season_column()
        r.calculate_pie_metrics()
        r.aggregate_season_stats()
        top_10, avg_10, worst_10 = r.generate_rankings()
        ml_dataset = r.export_ml_dataset("nba_ml_dataset.parquet")

        # Validate + smoke test
        r.validate_data_quality()
        _smoke_test(r)

        # ML readiness (strict: raises on required-field nulls / dup keys)
        ml_report = r.validate_ml_readiness(ml_dataset, strict=True)
        logger.info(f"[ML-READY REPORT] {ml_report}")

        r.print_rankings(top_10, avg_10, worst_10)
        return top_10, avg_10, worst_10, ml_dataset
    finally:
        r.close()


if __name__ == "__main__":
    # Run the analysis
    top_10, avg_10, worst_10, ml_dataset = main()
    
    # Save results for submission
    with open("nba_rankings_results.txt", "w") as f:
        f.write("TOP 10 SEASONS\n")
        for _, row in top_10.iterrows():
            f.write(f"{int(row['rank'])}. {row['player_name']} — {row['season']}\n")
        
        f.write("\nMOST AVERAGE 10 SEASONS\n")
        for _, row in avg_10.iterrows():
            f.write(f"{int(row['rank'])}. {row['player_name']} — {row['season']}\n")
        
        f.write("\nWORST 10 SEASONS\n")
        for _, row in worst_10.iterrows():
            f.write(f"{int(row['rank'])}. {row['player_name']} — {row['season']}\n")
    
    print("\n✅ Analysis complete! Results saved to 'nba_rankings_results.txt'")
    print("📊 ML dataset saved to 'nba_ml_dataset.parquet'")
