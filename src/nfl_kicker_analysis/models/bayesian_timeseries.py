"""
Time-Series Bayesian Models for NFL Kicker Analysis
==================================================
NEW IN v1.1.0  (2025-06-30)

* Hierarchical dynamic-linear model (level + trend) **or** SARIMA
  built on PyMC 5 (+ pymc-experimental).
* Mirrors the API of BayesianModelSuite so the rest of the pipeline
  stays unchanged.
"""
from __future__ import annotations
import warnings
from typing import Dict, Optional, List, Union

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

import jax  # keep for device detection
import numpyro

# Silence ArviZ HDI FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning, module="arviz")

# Optional fast Kalman-filter backend
try:
    from pymc_experimental.statespace import SARIMA   # type: ignore
except ImportError:  # graceful fallback for CI
    SARIMA = None

# -------------------------------------------------------------- #
# utils – choose chain layout safely                              #
# -------------------------------------------------------------- #
def _choose_chain_config(requested: int, use_jax: bool) -> dict:
    """
    Return a dict of kwargs for PyMC/JAX samplers guaranteeing ≥2 chains.
    On a single-device JAX setup we vectorise chains; on CPU we parallelise.
    """
    if use_jax:
        # run all requested chains on the same GPU
        return dict(chains=max(2, requested), chain_method="vectorized")
    else:
        # let PyMC fork processes; fall back to sequential if memory is tight
        return dict(chains=max(2, requested), cores=min(4, max(2, requested)))

# ------------------------------------------------------------------ #
# Helper – aggregate attempt-level rows → weekly counts per kicker   #
# ------------------------------------------------------------------ #
def _prep_series(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Collapse attempt-level data to (attempts, made) per kicker & period.
    
    Also preserves age-related columns by taking the mean within each period.

    * Ensures ``game_date`` is datetime64[ns] so `.dt` accessors work.
    * Returns a tidy frame sorted by kicker & date_key.
    """
    if "game_date" not in df.columns:
        raise KeyError("Column 'game_date' required for time-series modelling")

    if not np.issubdtype(df["game_date"].dtype, np.datetime64):
        df = df.copy()
        df["game_date"] = pd.to_datetime(df["game_date"])

    # Build aggregation dictionary  
    agg_dict = {
        "success": ["size", "sum"]  # attempts = size, made = sum
    }
    
    # Add age-related columns if they exist
    age_cols = ["age_at_attempt", "career_year", "seasons_of_experience"]
    for col in age_cols:
        if col in df.columns:
            agg_dict[col] = ["mean"]  # Take mean within period

    out = (
        df.assign(date_key=df["game_date"].dt.to_period(freq).dt.start_time)
          .groupby(["player_name", "player_id", "date_key"], sort=False)
          .agg(agg_dict)
          .reset_index()
    )
    
    # Flatten multi-level columns and rename
    if isinstance(out.columns, pd.MultiIndex):
        new_cols = []
        for col in out.columns:
            if col[0] == "success" and col[1] == "size":
                new_cols.append("attempts")
            elif col[0] == "success" and col[1] == "sum":
                new_cols.append("made")
            elif col[1] == "":
                new_cols.append(col[0])  # groupby columns
            else:
                new_cols.append(col[0])  # age columns
        out.columns = new_cols
    
    out["success_rate"] = out["made"] / out["attempts"]
    return out.sort_values(["player_name", "date_key"], ignore_index=True)

# ------------------------------------------------------------------ #
# Main class                                                         #
# ------------------------------------------------------------------ #
class TimeSeriesBayesianModelSuite:
    """
    Hierarchical DLM / SARIMA wrapper for weekly kicker make-rates.

    Parameters
    ----------
    freq          : str, default "W-MON"
    draws, tune,
    target_accept : passed to `pm.sample`
    random_seed   : int | None
    use_sarima    : bool – if True **and** pymc_experimental is installed,
                    approximates the latent process via SARIMA for speed.
    """

    def __init__(
        self,
        *,
        freq: str = "W-MON",
        draws: int = 1_000,
        tune: int = 1_000,
        target_accept: float = 0.9,
        random_seed: int | None = 42,
        use_sarima: bool = False,
        debug: bool = False,
        diag_vars: Optional[List[str]] = None,
        use_jax: Optional[bool] = None,
        chains: int = 4,                          # NEW: desired chains
        init_sigma: float = 5.0,                  # NEW: σ for initial state
    ):
        self.freq, self.draws, self.tune = freq, draws, tune
        self.target_accept, self.random_seed = target_accept, random_seed
        self.use_sarima = use_sarima and SARIMA is not None
        self.debug = debug
        self.diag_vars = diag_vars or ["sigma_level", "sigma_trend", "lvl", "trd"]
        
        self.init_sigma = init_sigma
        self.chains = chains
        # Auto-detect GPU: use JAX if any CUDA devices available
        self.use_jax = bool(jax.devices()) if use_jax is None else use_jax

        self._model: Optional[pm.Model] = None
        self._trace: Optional[az.InferenceData] = None
        self._meta: Optional[pd.DataFrame] = None

    # -------------------------------------------------------------- #
    # tiny helper so we never sprinkle print() everywhere
    def _log(self, *msg):
        if self.debug:
            print("[TS-Bayes]", *msg)

    # -------------------------------------------------------------- #
    # Fit                                                            #
    # -------------------------------------------------------------- #
    def fit(
        self,
        df: pd.DataFrame,
        *,
        preprocessor=None,
        min_attempts: int = 5,
    ) -> None:
        """
        Fit the hierarchical time-series model on attempt-level ``df``.

        Parameters
        ----------
        df            : long DataFrame (one row per FG attempt)
        preprocessor  : optional ``DataPreprocessor`` to mirror BayesianModelSuite
        min_attempts  : kickers with < ``min_attempts`` total are dropped
                        to keep the latent state-space well-conditioned.
        """
        # 0️⃣ optional slice-level preprocessing – keeps pipeline symmetry
        if preprocessor is not None:
            df = preprocessor.preprocess_slice(df)

        # 1️⃣ drop ultra-low-volume kickers
        keep = df.groupby("player_name")["success"].size().loc[lambda s: s >= min_attempts].index
        df = df[df["player_name"].isin(keep)]

        # 2️⃣ aggregate to weekly (or monthly) counts
        ts = _prep_series(df, self.freq)
        self._meta = (
            ts[["player_name", "player_id"]].drop_duplicates().set_index("player_name")
        )
        players = ts["player_name"].unique()
        P       = len(players)

        # 3️⃣ build rectangular player × time grid
        full_idx = (
            ts.groupby("player_name", sort=False)["date_key"]
            .apply(lambda s: pd.date_range(s.min(), s.max(), freq=self.freq))
            .explode()
            .reset_index()
            .rename(columns={0: "date_key"})
        )
        ts_full = full_idx.merge(ts, how="left", on=["player_name", "date_key"])
        ts_full[["attempts", "made"]] = ts_full[["attempts", "made"]].fillna(0.0)

        y = ts_full["made"].to_numpy(int)        # successes
        n = ts_full["attempts"].to_numpy(int)    # trials

        player_idx = pd.Categorical(ts_full["player_name"], categories=players).codes
        time_idx   = ts_full.groupby("player_name").cumcount()
        T_max      = int(time_idx.max()) + 1

        # 4️⃣ Static covariates ------------------------------------
        #   Enhanced age modeling: prefer career_year over age, then age, then zeros
        if "career_year" in ts_full.columns:
            # Use career_year for smooth aging curves (preferred)
            age_covariate = ts_full["career_year"].to_numpy(float)
            age_covariate_name = "career_year"
            self._log("Using career_year for age modeling")
        elif "age" in ts_full.columns:
            # Fallback to age_at_attempt
            age_covariate = ts_full["age"].to_numpy(float) 
            age_covariate_name = "age"
            self._log("Using age for age modeling")
        else:
            # No age information available
            age_covariate = np.zeros(len(ts_full), dtype=float)
            age_covariate_name = "none"
            self._log("No age information available - using zeros")

        # 5️⃣ build PyMC model
        with pm.Model() as self._model:
            if self.use_sarima:
                if SARIMA is None:
                    raise RuntimeError(
                        "pymc-experimental not installed – "
                        "set use_sarima=False or install the extra dependency."
                    )
                sarima = SARIMA(
                    name="sarima",
                    endog=y,
                    exog=None,
                    order=(0, 1, 1),
                    seasonal_order=(0, 0, 0, 0),
                    measurement_error=True,
                )
                mu_mat = sarima.states.reshape((1, -1))  # flattened for indexing
            else:
                init_0 = pm.Normal.dist(0.0, self.init_sigma)  # explicit init_dist
                σ_lvl = pm.Exponential("sigma_level", 1.0)
                σ_trd = pm.Exponential("sigma_trend", 1.0)
                lvl = pm.GaussianRandomWalk(
                    "lvl", sigma=σ_lvl, init_dist=init_0, shape=(P, T_max)
                )
                trd = pm.GaussianRandomWalk(
                    "trd", sigma=σ_trd, init_dist=init_0, shape=(P, T_max)
                )
                # identical trailing dimension – avoids broadcast crash
                step_frac = pm.math.constant(np.arange(T_max) / T_max)
                mu_mat = lvl + trd * step_frac          # ✅ broadcast-safe

            β_age = pm.Normal("beta_age", 0.0, 1.0)          # NEW: age/career_year effect
            
            # Apply age effect directly at observation level
            mu_obs = mu_mat[player_idx, time_idx] + β_age * age_covariate

            theta = pm.Deterministic(
                "theta", pm.math.invlogit(mu_obs)
            )
            pm.Binomial("obs", n=n, p=theta, observed=y)

            # -------- sampling ------------------------------------
            chain_cfg = _choose_chain_config(self.chains, self.use_jax)

            if self.use_jax:
                numpyro.set_host_device_count(chain_cfg["chains"])
                from pymc.sampling.jax import sample_numpyro_nuts
                self._trace = sample_numpyro_nuts(
                    draws=self.draws,
                    tune=self.tune,
                    target_accept=self.target_accept,
                    random_seed=self.random_seed,
                    progressbar=True,
                    **chain_cfg,
                )
            else:
                self._trace = pm.sample(
                    draws=self.draws,
                    tune=self.tune,
                    target_accept=self.target_accept,
                    random_seed=self.random_seed,
                    idata_kwargs={"log_likelihood": False},
                    progressbar=True,
                    **chain_cfg,
                )


    # -------------------------------------------------------------- #
    # Forecast                                                       #
    # -------------------------------------------------------------- #
    def forecast(self, *, steps: int = 6, hdi_prob: float = 0.9) -> pd.DataFrame:
        """
        Monte-Carlo simulate `steps` weeks ahead for **all** fitted kickers.

        Returns
        -------
        DataFrame with columns ['player_name', 'step', 'p_mean', 'hdi_lower', 'hdi_upper']
        """
        if any(obj is None for obj in (self._trace, self._meta, self._model)):
            raise RuntimeError("Call `.fit()` before forecasting")

        players = self._meta.index.tolist()
        lvl = self._trace.posterior["lvl"].values
        trd = self._trace.posterior["trd"].values
        chains, draws, P, T = lvl.shape
        lvl_last = lvl[:, :, :, -1].reshape(chains*draws, P)
        trd_last = trd[:, :, :, -1].reshape(chains*draws, P)

        rng = np.random.default_rng(self.random_seed)
        records: List[Dict[str, Union[str, int, float]]] = []
        for p_idx, name in enumerate(players):
            noise = rng.normal(0, 0.1, size=(chains*draws, steps)).cumsum(axis=1)
            lvl_path = lvl_last[:, p_idx][:, None] + noise
            trd_path = trd_last[:, p_idx][:, None]
            mu = lvl_path + trd_path * np.arange(1, steps+1)
            theta = 1 / (1 + np.exp(-mu))
            mean = theta.mean(axis=0)
            hdi  = az.hdi(theta, hdi_prob=hdi_prob)
            for s, (m, lo, hi) in enumerate(zip(mean, hdi[:,0], hdi[:,1]), 1):
                records.append({"player_name": name,
                                "step": s,
                                "p_mean": float(m),
                                "hdi_lower": float(lo),
                                "hdi_upper": float(hi)})
        return pd.DataFrame(records)

    # -------------------------------------------------------------- #
    # Diagnostics                                                    #
    # -------------------------------------------------------------- #
    def diagnostics(self, *, thin: int = 5) -> Dict[str, float]:
        if self._trace is None:
            raise RuntimeError("Model not fitted")

        if self._trace.posterior.dims["chain"] < 2:
            # r̂ undefined → return sentinel values and warn once
            warnings.warn("Only one chain present – r̂ unavailable; run ≥2 chains for convergence diagnostics.")
            return {"rhat_max": float("nan"), "ess_min": float("nan")}

        rhat_max, ess_min = -np.inf, np.inf
        for var in self.diag_vars:
            if var not in self._trace.posterior:
                continue
            data = self._trace.posterior[var]
            if thin > 1 and data.ndim > 2:
                slc = (slice(None), slice(None)) + (slice(None, None, thin),) * (data.ndim - 2)
                data = data[slc]
            rhat_max = max(rhat_max, az.rhat(data).to_array().max().item())
            ess_min  = min(ess_min,  az.ess(data, method="bulk").to_array().min().item())
        return {"rhat_max": float(rhat_max), "ess_min": float(ess_min)}

    # -------------------------------------------------------------- #
    # Quick plot                                                     #
    # -------------------------------------------------------------- #
    def plot_forecast(self, player_name: str, *, ax=None):
        """
        Plot historical weekly make-probability with 90 % HDI for one kicker.
        """
        import matplotlib.pyplot as plt
        if self._trace is None or self._meta is None:
            raise RuntimeError("Fit the model first")
        if player_name not in self._meta.index:
            raise ValueError(f"Unknown player '{player_name}'")

        p_idx = list(self._meta.index).index(player_name)
        lvl = self._trace.posterior["lvl"].sel(lvl_dim_0=p_idx).stack(draws=("chain","draw"))
        trd = self._trace.posterior["trd"].sel(trd_dim_0=p_idx).stack(draws=("chain","draw"))
        T = lvl.shape[-1]
        x = np.arange(T)
        mu = lvl + trd * (x / T)
        p  = 1 / (1 + np.exp(-mu))
        mean = p.mean("draws")
        hdi  = az.hdi(p, hdi_prob=0.9)

        if ax is None:
            _, ax = plt.subplots(figsize=(8,4))
        ax.plot(x, mean, label="p_mean")
        ax.fill_between(x, hdi.sel(hdi="lower"), hdi.sel(hdi="upper"), alpha=0.3,
                        label="90 % HDI")
        ax.set_xlabel("Week index")
        ax.set_ylabel("Make probability")
        ax.set_title(f"Historical make-probability – {player_name}")
        ax.legend()
        return ax

# ------------------------------------------------------------------ #
# Smoke-test CLI                                                     #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    from src.nfl_kicker_analysis.data.loader import DataLoader
    from src.nfl_kicker_analysis.data.feature_engineering import FeatureEngineer

    print("🏈  Time-Series Bayesian demo (weekly make-rates)")
    df_raw  = DataLoader().load_complete_dataset()
    df_feat = FeatureEngineer().create_all_features(df_raw)

    ts = TimeSeriesBayesianModelSuite(freq="W-MON",
                                      draws=250, 
                                      tune=250,
                                      use_sarima=True,
                                      target_accept=.85,
                                      )
    ts.fit(df_feat[df_feat["season"] <= 2018])             # train through 2019
    fcst = ts.forecast(steps=6)                            # next six weeks
    print(fcst.head())

    diag = ts.diagnostics()
    print(f"R-hat ≤ 1.01 ? {diag['rhat_max']:.3f} | ESS ≥ 100 ? {diag['ess_min']:.0f}")




