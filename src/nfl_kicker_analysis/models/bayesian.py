"""
Bayesian models for NFL kicker analysis.
Provides hierarchical Bayesian logistic regression and evaluation utilities using PyMC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt        # NEW – for PPC plot
from scipy import stats                # NEW – ECDF sampling for EPA
from typing import Dict, Any, TYPE_CHECKING

from src.nfl_kicker_analysis.utils.metrics import ModelEvaluator
from src.nfl_kicker_analysis.config import FEATURE_LISTS, config

if TYPE_CHECKING:
    from src.nfl_kicker_analysis.data.preprocessor import DataPreprocessor

__all__ = [
    "BayesianModelSuite",
]


class BayesianModelSuite:
    """Hierarchical Bayesian logistic‑regression models for kicker analysis."""

    def __init__(
        self,
        *,
        draws: int = 1_000,
        tune: int = 1_000,
        target_accept: float = 0.95,  # Increased from 0.9 to reduce divergences
        include_random_slope: bool = False,
        random_seed: int | None = 42,
    ) -> None:
        self.draws = draws
        self.tune = tune
        self.target_accept = target_accept
        self.include_random_slope = include_random_slope
        self.random_seed = random_seed

        # Model components - set during fit()
        self._model = None
        self._trace = None
        self._kicker_map = {}
        self._distance_mu = 0.0
        self._distance_sigma = 1.0
        self.baseline_probs = {}  # For consistent EPA baselines with EPACalculator
        self.evaluator = ModelEvaluator()

    def _bootstrap_distances(
        self,
        distances: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
        weights: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Bootstrap helper using `np.random.Generator.choice`.
        
        Simple, testable wrapper for bootstrap resampling of field-goal distances.
        Keeps all distance logic in one place and follows NumPy best practices.
        """
        if distances.size == 0:
            raise ValueError("No distances available for bootstrap.")
        return rng.choice(distances, size=n_samples, replace=True, p=weights)

    # ---------------------------------------------------------------------
    # 🛠️  Helper utilities
    # ---------------------------------------------------------------------
    def _standardize(self, x: np.ndarray, *, fit: bool = False) -> np.ndarray:
        if fit:
            self._distance_mu = float(x.mean())
            self._distance_sigma = float(x.std())
        return (x - self._distance_mu) / self._distance_sigma

    def _encode_kicker(self, raw_ids: np.ndarray, *, fit: bool = False,
                       unknown_action: str = "average") -> np.ndarray:
        """
        Map raw kicker IDs → compact indices (kicker_idx).
        """
        if fit:
            self._kicker_map = {pid: i for i, pid in enumerate(np.unique(raw_ids))}
        idx = np.array([self._kicker_map.get(pid, -1) for pid in raw_ids], int)

        if (idx == -1).any():
            n_unseen = (idx == -1).sum()
            msg = f"{n_unseen} unseen kicker IDs – mapped to league mean."
            if unknown_action == "raise":
                raise ValueError(msg)
            elif unknown_action == "warn":
                print("⚠️ ", msg)

        return idx




    # ---------------------------------------------------------------------
    # 🔨  Model construction
    # ---------------------------------------------------------------------
    def _build_model(
        self,
        distance_std: np.ndarray,
        age_c: np.ndarray,           # <-- NEW: centered age
        age_c2: np.ndarray,          # <-- NEW: quadratic age
        exp_std: np.ndarray,
        success: np.ndarray,
        kicker_idx: np.ndarray,
        n_kickers: int,
    ) -> pm.Model:
        with pm.Model() as model:
            # Population-level effects
            alpha = pm.Normal("alpha", 1.5, 1.0)
            beta_dist = pm.Normal("beta_dist", -1.5, 0.8)
            
            # Age effects (linear + quadratic)
            beta_age  = pm.Normal("beta_age",  0.0, 0.5)
            beta_age2 = pm.Normal("beta_age2", 0.0, 0.5)
            beta_exp  = pm.Normal("beta_exp",  0.0, 0.5)

            # Per-kicker random intercepts (non-centered)
            σ_u   = pm.HalfNormal("sigma_u", 0.8)
            u_raw = pm.Normal("u_raw", 0.0, 1.0, shape=n_kickers)
            u     = pm.Deterministic("u", σ_u * u_raw)

            # Per-kicker random aging slopes (optional enhancement)
            if self.include_random_slope:
                σ_age = pm.HalfNormal("sigma_age", 0.5)
                a_raw = pm.Normal("a_raw", 0.0, 1.0, shape=n_kickers)
                a_k   = pm.Deterministic("a_k", σ_age * a_raw)
                age_slope_effect = a_k[kicker_idx] * age_c
            else:
                age_slope_effect = 0.0

            # Linear predictor
            lin_pred = (
                alpha
                + (beta_dist * distance_std)
                + (beta_age * age_c) + age_slope_effect
                + (beta_age2 * age_c2)
                + (beta_exp * exp_std)
                + u[kicker_idx]
            )

            θ = pm.Deterministic("theta", pm.invlogit(lin_pred))
            pm.Bernoulli("obs", p=θ, observed=success)
        return model

    # ---------------------------------------------------------------------
    # 📈  Public API
    # ---------------------------------------------------------------------
    def fit(self, df, *, preprocessor=None):
        # ------------------------------------------------------------------
        # 0️⃣  Exactly one preprocessing pass
        if df.attrs.get("engineered", False):
            processed = df.copy()
        elif preprocessor is not None:
            processed = preprocessor.preprocess_slice(df)
        else:
            # 🎯 AUTO-CREATE BAYESIAN-MINIMAL PREPROCESSOR 
            from src.nfl_kicker_analysis.data.preprocessor import DataPreprocessor
            from src.nfl_kicker_analysis import config
            
            bayes_preprocessor = DataPreprocessor()
            bayes_preprocessor.bayes_minimal = True  # Enable minimal preprocessing
            bayes_preprocessor.update_config(
                min_distance=20, max_distance=60, 
                min_kicker_attempts=5,
                season_types=['Reg', 'Post'],
                include_performance_history=False,  # Not needed for Bayesian
                include_statistical_features=False,  # Avoid complex features
                include_player_status=True,  # Enable player status features
                performance_window=12
            )
            processed = bayes_preprocessor.preprocess_slice(df)
        # ------------------------------------------------------------------
        # 1️⃣  Predictors
        dist_std = self._standardize(processed["attempt_yards"].to_numpy(float), fit=True)
        
        # Age variables (centered & scaled)
        age_c  = processed["age_c"].to_numpy(float) if "age_c" in processed.columns else np.zeros(len(processed), dtype=float)
        age_c2 = processed["age_c2"].to_numpy(float) if "age_c2" in processed.columns else np.zeros(len(processed), dtype=float)

        # Handle experience standardization
        if "exp_100" in processed.columns:
            exp_std = (
                (processed["exp_100"] - processed["exp_100"].mean()) /
                processed["exp_100"].std()
            ).to_numpy(float)
        else:
            exp_std = np.zeros(len(processed), dtype=float)
            
        success    = processed["success"].to_numpy(int)
        kicker_idx = self._encode_kicker(processed["kicker_id"].to_numpy(int), fit=True)
        n_kickers  = len(self._kicker_map)

        # ---- model & sampling -------------------------------------------
        self._model = self._build_model(
            dist_std, age_c, age_c2, exp_std, success, kicker_idx, n_kickers
        )
        
        # ── FIX: ensure pm.sample knows which model to use ────────────
        with self._model:
            self._trace = pm.sample(
                draws=self.draws, tune=self.tune,
                chains=4, target_accept=self.target_accept,
                random_seed=self.random_seed,
                return_inferencedata=True
            )

    def predict(
        self,
        df: pd.DataFrame,
        *,
        return_ci: bool = False,
        preprocessor=None
    ):
        if self._trace is None or self._model is None:
            raise RuntimeError("Model not yet fitted.")

        # -- preprocessing --------------------------------------------------
        if preprocessor is not None:
            df = preprocessor.preprocess_slice(df)

        dist_std = (
            df["distance_zscore"].to_numpy(float)
            if "distance_zscore" in df.columns
            else self._standardize(df["attempt_yards"].to_numpy(float), fit=False)
        )
        kid_idx = self._encode_kicker(df["kicker_id"].to_numpy(int),
                                    fit=False, unknown_action="average")

        # -- posterior samples (non-centered model) ---------------------
        a = self._trace.posterior["alpha"].values.flatten()
        b = self._trace.posterior["beta_dist"].values.flatten()
        u = self._trace.posterior["u"].values.reshape(a.size, -1)  # use u not u_raw

        # Age and experience effects (if available)
        age_effect = 0.0
        if "age_c" in df.columns:
            age_c = df["age_c"].to_numpy(float)
            age_c2 = df["age_c2"].to_numpy(float) if "age_c2" in df.columns else np.zeros_like(age_c)
            
            beta_age = self._trace.posterior["beta_age"].values.flatten()
            beta_age2 = self._trace.posterior["beta_age2"].values.flatten()
            
            age_effect = (beta_age[:, None] * age_c + 
                         beta_age2[:, None] * age_c2)
            
            # Add random age slopes if available
            if "a_k" in self._trace.posterior:
                a_k = self._trace.posterior["a_k"].values.reshape(a.size, -1)
                a_k = np.pad(a_k, ((0, 0), (0, 1)), constant_values=0.0)
                idx_age = np.where(kid_idx == -1, a_k.shape[1] - 1, kid_idx)
                age_effect += a_k[:, idx_age] * age_c
        
        exp_effect = 0.0
        if "exp_100" in df.columns:
            exp_std = ((df["exp_100"] - df["exp_100"].mean()) / df["exp_100"].std()).to_numpy(float)
            beta_exp = self._trace.posterior["beta_exp"].values.flatten()
            exp_effect = beta_exp[:, None] * exp_std

        # Pad league-mean column for unseen kickers
        u = np.pad(u, ((0, 0), (0, 1)), constant_values=0.0)
        idx = np.where(kid_idx == -1, u.shape[1] - 1, kid_idx)

        lin = (a[:, None] + b[:, None] * dist_std + 
               age_effect + exp_effect + u[:, idx])
        theta = 1 / (1 + np.exp(-lin))
        mean = theta.mean(axis=0)

        if not return_ci:
            return mean

        lower, upper = np.percentile(theta, [2.5, 97.5], axis=0)
        return mean, lower, upper


    def evaluate(
        self, 
        df: pd.DataFrame, 
        *, 
        preprocessor=None
    ) -> Dict[str, float]:
        """Compute AUC, Brier score & log‑loss on provided data.
        
        Args:
            df: Data to evaluate on
            preprocessor: Optional DataPreprocessor instance. If provided, will
                         use it to preprocess the data before evaluation.
        """
        # Apply preprocessing if provided  
        if preprocessor is not None:
            df = preprocessor.preprocess_slice(df)
                
        y_true = df["success"].to_numpy(dtype=int)
        y_pred_result = self.predict(df)  # predict() will handle its own preprocessing if needed
        
        # Handle both single prediction and CI tuple returns
        if isinstance(y_pred_result, tuple):
            y_pred = y_pred_result[0]  # Just use mean predictions for evaluation
        else:
            y_pred = y_pred_result
            
        return self.evaluator.calculate_classification_metrics(y_true, y_pred)


    def diagnostics(self, *, return_scalars: bool = False) -> Dict[str, Any]:
        """
        Compute and return MCMC diagnostics.

        Parameters
        ----------
        return_scalars : bool, default False
            If True, also include convenience keys
            ``rhat_max`` and ``ess_min`` for quick threshold checks.

        Returns
        -------
        dict
            Keys: rhat, ess (xarray.Dataset), rhat_vals, ess_vals (np.ndarray),
            summary_ok (bool), and optionally rhat_max, ess_min (float).
        """
        if self._trace is None:
            raise RuntimeError("Model not yet fitted.")

        # ArviZ calls (collapse chain/draw)
        rhats = az.rhat(self._trace)
        ess   = az.ess(self._trace)

        # Flatten → numpy for easy thresholding
        rhat_vals = rhats.to_array().values.ravel()
        ess_vals  = ess.to_array().values.ravel()

        summary_ok = (rhat_vals <= 1.01).all() and (ess_vals >= 100).all()
        if not summary_ok:
            print("⚠️  Sampling diagnostics outside recommended thresholds.")

        out = {
            "rhat": rhats,
            "ess": ess,
            "rhat_vals": rhat_vals,
            "ess_vals": ess_vals,
            "summary_ok": summary_ok,
        }
        if return_scalars:
            out["rhat_max"] = rhat_vals.max()
            out["ess_min"] = ess_vals.min()
        return out




    # -----------------------------------------------------------------
    # 🌟  NEW 1: kicker-level credible interval
    # -----------------------------------------------------------------
    def kicker_interval(
        self,
        kicker_id: int,
        distance: float | None = None,
        ci: float = 0.95,
    ) -> Dict[str, float]:
        """
        Return mean, lower, upper success probability for a *single* kicker.

        Args
        ----
        kicker_id : raw ID as in dataframe
        distance  : yards; if None, uses the empirical mean distance of
                    training data, transformed with stored μ/σ.
        ci        : central credible-interval mass (default 0.95)
        """
        if self._trace is None:
            raise RuntimeError("Model must be fitted first")

        # 1 → index or league-mean column
        k_idx = self._kicker_map.get(kicker_id, -1)
        pad_col = len(self._kicker_map)   # after pad in predict()

        # 2 → choose distance
        if distance is None:
            distance_std = 0.0            # z-score of mean is 0
        else:
            distance_std = (distance - self._distance_mu) / self._distance_sigma

        a = self._trace.posterior["alpha"].values.flatten()
        
        # Robust lookup for the distance slope parameter (handles naming changes)
        slope_name = "beta_dist" if "beta_dist" in self._trace.posterior else "beta"
        b = self._trace.posterior[slope_name].values.flatten()
        
        u = self._trace.posterior["u"].values.reshape(a.size, -1)

        # pad league-mean
        u = np.pad(u, ((0, 0), (0, 1)), constant_values=0.0)
        idx = pad_col if k_idx == -1 else k_idx

        logit_p = a + b * distance_std + u[:, idx]
        p = 1 / (1 + np.exp(-logit_p))

        lower, upper = np.quantile(p, [(1-ci)/2, 1-(1-ci)/2])
        return {"mean": p.mean(), "lower": lower, "upper": upper,
                "n_draws": p.size, "distance_std": distance_std}

    # -----------------------------------------------------------------
    # 🌟  NEW 2: posterior-predictive plot across 5-yd bins
    # -----------------------------------------------------------------
    def plot_distance_ppc(
        self,
        df: pd.DataFrame,
        *,
        bin_width: int = 5,
        preprocessor = None,
        ax = None
    ):
        """
        Bin attempts by distance and overlay actual vs posterior mean make-rate.
        """
        if preprocessor is not None:
            df = preprocessor.preprocess_slice(df)

        # 1 Actual success by bin
        df = df.copy()
        df["bin"] = (df["attempt_yards"] // bin_width) * bin_width
        actual = df.groupby("bin")["success"].mean()

        # 2 Posterior mean per attempt → group
        preds = self.predict(df)
        df["pred"] = preds
        posterior = df.groupby("bin")["pred"].mean()

        # 3 Plot
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(actual.index, actual.values, marker="o",
                label="Actual", linewidth=2)
        ax.plot(posterior.index, posterior.values, marker="s",
                label="Posterior mean", linestyle="--")
        ax.set_xlabel("Distance bin (yards)")
        ax.set_ylabel("FG make probability")
        ax.set_title("Posterior-Predictive Check ({}-yd bins)".format(bin_width))
        ax.legend()
        plt.tight_layout()
        return ax

    # -----------------------------------------------------------------
    # 🌟  NEW 3: age-binned posterior-predictive check
    # -----------------------------------------------------------------
    def plot_age_ppc(
        self,
        df: pd.DataFrame,
        *,
        bin_width: float = 2.0,
        preprocessor = None,
        ax = None
    ):
        """
        Bin attempts by age and overlay actual vs posterior mean make-rate.
        """
        if preprocessor is not None:
            df = preprocessor.preprocess_slice(df)

        # Use raw age for binning (more interpretable)
        age_col = "age_at_attempt" if "age_at_attempt" in df.columns else "age_c"
        df = df.copy()
        
        if age_col == "age_c":
            # Convert back to raw age for binning
            df["age_bin"] = ((df["age_c"] * 10 + 30) // bin_width) * bin_width
        else:
            df["age_bin"] = (df[age_col] // bin_width) * bin_width
            
        # Actual success by age bin
        actual = df.groupby("age_bin")["success"].mean()

        # Posterior mean per attempt → group by age
        preds = self.predict(df)
        df["pred"] = preds
        posterior = df.groupby("age_bin")["pred"].mean()

        # Plot
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(actual.index, actual.values, marker="o",
                label="Actual", linewidth=2)
        ax.plot(posterior.index, posterior.values, marker="s",
                label="Posterior mean", linestyle="--")
        ax.set_xlabel("Age bin (years)")
        ax.set_ylabel("FG make probability")
        ax.set_title("Age-Based Posterior-Predictive Check ({:.1f}-yr bins)".format(bin_width))
        ax.legend()
        plt.tight_layout()
        return ax

    # ───────────────────────────────────────────────────────────
    # Helper: draw-level EPA simulation  (fully replaced)
    # ───────────────────────────────────────────────────────────
    def _epa_fg_plus_draws(
        self,
        league_df: pd.DataFrame,
        *,
        kicker_ids: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
        distance_strategy: str = "kicker",
        τ: float = 20.0,
        compute_league_avg_only: bool = False,
        full_league_avg: float | None = None,
    ) -> np.ndarray:
        """
        Monte-Carlo simulate expected points per attempt for every posterior
        draw *and* every kicker in *kicker_ids*.

        Parameters
        ----------
        distance_strategy : {"kicker","league"}
            • "kicker" (default) ─ bootstrap distances **from the kicker's own
              historical attempts** – fairer when kick distributions differ.
            • "league" ─ previous behaviour (single shared pool).
        τ : float, default 20.0
            Empirical-Bayes shrinkage parameter. Higher values = less shrinkage.
        compute_league_avg_only : bool, default False
            If True, only compute the league average without shrinkage.
        full_league_avg : float or None, default None
            If provided, use this as the league average for shrinkage calculation.

        Returns
        -------
        np.ndarray
            Shape = (n_draws, len(kicker_ids)); each entry is the expected
            points per attempt for one posterior draw.
        """
        if self._trace is None:
            raise RuntimeError("Model not yet fitted.")

        if distance_strategy not in {"kicker", "league"}:
            raise ValueError("distance_strategy must be 'kicker' or 'league'")

        n_kickers = len(kicker_ids)

        # ------------------------------------------------------------------
        # 1️⃣ choose a distance pool for each kicker
        # ------------------------------------------------------------------
        if distance_strategy == "league":
            pool = league_df["attempt_yards"].to_numpy(float)
            distance_sets = [pool] * n_kickers
        else:  # "kicker"
            distance_sets = [
                league_df.loc[league_df["kicker_id"] == kid, "attempt_yards"]
                          .to_numpy(float)
                for kid in kicker_ids
            ]

        # ------------------------------------------------------------------
        # 2️⃣ build the simulated attempt frame
        # ------------------------------------------------------------------
        sampled_distances = []
        sampled_kicker_ids = []
        for kid, dists in zip(kicker_ids, distance_sets):
            if dists.size == 0:
                raise ValueError(f"No distance data for kicker_id={kid}")
            sampled = rng.choice(dists, size=n_samples, replace=True)
            sampled_distances.append(sampled)
            sampled_kicker_ids.append(np.full(n_samples, kid, int))

        sim = pd.DataFrame({
            "attempt_yards": np.concatenate(sampled_distances),
            "kicker_id":     np.concatenate(sampled_kicker_ids),
        })

        # ------------------------------------------------------------------
        # 3️⃣ forward pass through the posterior
        # ------------------------------------------------------------------
        dist_std = self._standardize(sim["attempt_yards"].to_numpy(float),
                                     fit=False)
        kid_idx  = self._encode_kicker(sim["kicker_id"].to_numpy(int),
                                       fit=False, unknown_action="average")

        a = self._trace.posterior["alpha"].values.flatten()
        
        # Robust lookup for the distance slope parameter (handles naming changes)
        slope_name = "beta_dist" if "beta_dist" in self._trace.posterior else "beta"
        b = self._trace.posterior[slope_name].values.flatten()
        
        u = self._trace.posterior["u"].values.reshape(a.size, -1)
        u = np.pad(u, ((0, 0), (0, 1)), constant_values=0.0)  # league-mean slot
        idx = np.where(kid_idx == -1, u.shape[1] - 1, kid_idx)

        lin   = a[:, None] + b[:, None] * dist_std + u[:, idx]
        theta = 1 / (1 + np.exp(-lin))              # shape (draws , K·S)
        theta = theta.reshape(a.size, n_kickers, n_samples)
        pts   = theta.mean(axis=-1) * 3.0                    # draws × K

        # ----- Empirical-Bayes shrink toward league avg --------------------
        # Use pre-computed full league average or compute from current data
        if full_league_avg is not None:
            league_avg = full_league_avg                     # Use pre-computed full league average
        else:
            league_avg = pts.mean()                          # Fallback: compute from current data
        
        # If only computing league average, return early
        if compute_league_avg_only:
            return pts  # Return raw points for league average calculation
        
        # Get attempt counts for shrinkage calculation
        n_i = league_df.groupby("kicker_id")["success"].size().reindex(kicker_ids).to_numpy(float)
        
        # Standard Empirical Bayes shrinkage toward league average:
        # shrunk = B * raw + (1-B) * prior, where B = n/(n+τ) and prior = league_avg
        B = n_i / (n_i + τ)                                  # shrinkage weights (K,)
        
        # Apply shrinkage: each kicker's points shrunk toward league average
        shrunk_pts = (B[np.newaxis, :] * pts +               # B * raw_points  
                      (1 - B)[np.newaxis, :] * league_avg)   # (1-B) * league_avg
        
        # Return EPA as deviation from league average
        return shrunk_pts - league_avg                        # EPA relative to league avg

    # ───────────────────────────────────────────────────────────
    # Public: EPA-FG⁺ leaderboard  (patched 2025-06-30)
    # ───────────────────────────────────────────────────────────
    def epa_fg_plus(
        self,
        league_df: pd.DataFrame,
        *,
        n_samples: int = 1_000,
        min_attempts: int = config.MIN_KICKER_ATTEMPTS,
        seed: int | None = None,
        weights: np.ndarray | None = None,
        cred_mass: float = 0.95,
        return_ci: bool = True,
        distance_strategy: str = "kicker",
        τ: float = 20.0,
    ) -> pd.DataFrame:
        """
        Bayesian EPA-FG⁺ leaderboard with minimum attempts filter.

        Parameters
        ----------
        min_attempts : int, default config.MIN_KICKER_ATTEMPTS
            Minimum attempts required for kicker inclusion in leaderboard
        distance_strategy : {"kicker","league"}
            Sampling scheme for the synthetic attempt set:
            • "kicker" (default) ─ bootstrap distances **from the kicker's own
              historical attempts** – fairer when kick distributions differ.
            • "league" ─ previous behaviour (single shared pool).
        τ : float, default 20.0
            Empirical-Bayes shrinkage parameter. Higher values = less shrinkage.

        Returns
        -------
        pandas.DataFrame
            Kicker leaderboard with EPA-FG⁺ and credible intervals.
        """
        # ❶ Baseline must be computed on full-league (reuse EPACalculator for consistency)
        if not self.baseline_probs:
            from src.nfl_kicker_analysis.utils.metrics import EPACalculator
            epa_calc = EPACalculator()
            self.baseline_probs = epa_calc.calculate_baseline_probs(league_df)
        
        # ❷ Compute full league average FIRST (before any filtering)
        all_kicker_ids = league_df["kicker_id"].unique()
        rng = np.random.default_rng(seed)
        
        # Get league average from ALL kickers
        full_league_draws = self._epa_fg_plus_draws(
            league_df,  # FULL league data
            kicker_ids=all_kicker_ids,
            n_samples=n_samples,
            rng=rng,
            distance_strategy=distance_strategy,
            τ=τ,
            compute_league_avg_only=True  # New flag to only compute league avg
        )
        full_league_avg = full_league_draws.mean()
        
        # ❷ NOW apply minimum attempts filter
        qual_ids = (
            league_df.groupby("kicker_id")["success"]
                .size()
                .loc[lambda s: s >= min_attempts]
                .index
        )
        filtered_df = league_df[league_df["kicker_id"].isin(qual_ids)].copy()
        
        # ❸ Continue with existing logic on filtered data
        kicker_ids = filtered_df["kicker_id"].unique()
        
        # Kicker metadata
        meta = (
            filtered_df.groupby("kicker_id")["player_name"]
                .first()
                .to_frame()
        )

        # Monte Carlo simulation with pre-computed league average
        draws = self._epa_fg_plus_draws(
            filtered_df,  # Use filtered data for draws
            kicker_ids=kicker_ids,
            n_samples=n_samples,
            rng=rng,
            distance_strategy=distance_strategy,
            τ=τ,
            full_league_avg=full_league_avg  # Pass the pre-computed league average
        )
        assert draws.shape[1] == len(kicker_ids), "shape mismatch in posterior draws"

        # draws now contain EPA values directly (already relative to league average)
        epa_draws    = draws.T       # transpose to get (kickers, draws)
        mean_pts     = draws.mean(axis=0)  # keep for compatibility

        if not return_ci:
            out = pd.DataFrame({
                "epa_fg_plus_mean": epa_draws.mean(axis=1),
            }, index=kicker_ids)
            return out.join(meta).sort_values("epa_fg_plus_mean", ascending=False)

        hdi = az.hdi(epa_draws.T, hdi_prob=cred_mass)        #  (K,2)
        width = hdi[:, 1] - hdi[:, 0]
        q33, q66 = np.quantile(width, [0.33, 0.66])
        certainty = np.where(width < q33, "high",
                     np.where(width < q66, "medium", "low"))

        tbl = pd.DataFrame({
            "epa_fg_plus_mean": epa_draws.mean(axis=1),
            "hdi_lower": hdi[:, 0],
            "hdi_upper": hdi[:, 1],
            "certainty_level": certainty,
            "expected_pts_per_att": mean_pts,
        }, index=kicker_ids)

        return (
            tbl.join(meta)                      # add player_name
               .sort_values("epa_fg_plus_mean", ascending=False)
        )

    # ---------------------------------------------------------------------
    # 🔍  Helper methods for kicker ID/name conversion
    # ---------------------------------------------------------------------
    def get_kicker_id_by_name(self, df: pd.DataFrame, player_name: str) -> int | None:
        """
        Get kicker_id for a given player_name from the dataset.
        
        Args:
            df: DataFrame containing kicker_id and player_name columns
            player_name: Name of the kicker to look up
            
        Returns:
            kicker_id if found, None otherwise
        """
        matches = df[df["player_name"] == player_name]["kicker_id"].unique()
        return matches[0] if len(matches) > 0 else None
    
    def get_kicker_name_by_id(self, df: pd.DataFrame, kicker_id: int) -> str | None:
        """
        Get player_name for a given kicker_id from the dataset.
        
        Args:
            df: DataFrame containing kicker_id and player_name columns
            kicker_id: ID of the kicker to look up
            
        Returns:
            player_name if found, None otherwise
        """
        matches = df[df["kicker_id"] == kicker_id]["player_name"].unique()
        return matches[0] if len(matches) > 0 else None
    
    def kicker_interval_by_name(
        self,
        df: pd.DataFrame,
        player_name: str,
        distance: float | None = None,
        ci: float = 0.95,
    ) -> Dict[str, float]:
        """
        Return mean, lower, upper success probability for a kicker by name.
        
        Args:
            df: DataFrame containing kicker mappings
            player_name: Name of the kicker
            distance: yards; if None, uses empirical mean
            ci: central credible-interval mass (default 0.95)
        """
        kicker_id = self.get_kicker_id_by_name(df, player_name)
        if kicker_id is None:
            raise ValueError(f"Kicker '{player_name}' not found in dataset")
        return self.kicker_interval(kicker_id, distance, ci)






if __name__ == "__main__":
    from src.nfl_kicker_analysis.data.loader import DataLoader
    from src.nfl_kicker_analysis.data.feature_engineering import FeatureEngineer
    from src.nfl_kicker_analysis.data.preprocessor import DataPreprocessor
    from src.nfl_kicker_analysis.models.bayesian import BayesianModelSuite
    import numpy as np
    import arviz as az
    import matplotlib.pyplot as plt

    import jax
    print(jax.devices())        # should list "GpuDevice" (CUDA) or "METAL" (Apple)  
    print(jax.default_backend()) # should print 'gpu', not 'cpu'

    print("🏈 Bayesian Model Suite with DataPreprocessor Integration Demo")
    print("=" * 60)
    
    # 1️⃣ Loading and engineering features
    print("\n1️⃣ Loading and engineering features...")
    loader = DataLoader()
    df_raw = loader.load_complete_dataset()
    engineer = FeatureEngineer()
    df_feat = engineer.create_all_features(df_raw)
    print(f"   Raw data shape: {df_raw.shape}")
    print(f"   Engineered data shape: {df_feat.shape}")
    
    # 2️⃣ Configuring preprocessing
    print("\n2️⃣ Configuring preprocessing...")
    CONFIG = {
        'min_distance': 20,
        'max_distance': 60,
        'min_kicker_attempts': 8,
        'season_types': ['Reg', 'Post'],
        'include_performance_history': True,
        'include_statistical_features': False,
        'include_player_status': True,  # ✅ FIX: Added missing parameter
        'performance_window': 12,
    }
    
    preprocessor = DataPreprocessor()
    preprocessor.update_config(**CONFIG)
    preprocessor.update_feature_lists(**FEATURE_LISTS)
    
    # 3️⃣ Method A: Manual preprocessing then passing to BayesianModelSuite
    print("\n3️⃣ Method A: Manual preprocessing then passing to BayesianModelSuite")
    print("-" * 50)
    df_processed = preprocessor.preprocess_complete(df_feat)
    print(f"   Processed data shape: {df_processed.shape}")
    train_data = df_processed[df_processed["season"] <= 2017]
    test_data = df_processed[df_processed["season"] == 2018]
    print(f"   Train data shape: {train_data.shape}")
    print(f"   Test data shape: {test_data.shape}")
    suite_a = BayesianModelSuite(draws=1000,
                                 tune=1000, 
                                 include_random_slope=False, 
                                 random_seed=42)
    suite_a.fit(train_data)
    metrics_a = suite_a.evaluate(test_data)
    print("   Method A Results:")
    for metric, value in metrics_a.items():
        print(f"     {metric}: {value:.4f}")
    
    # 4️⃣ Method B: Automatic preprocessing within BayesianModelSuite
    print("\n4️⃣ Method B: Automatic preprocessing within BayesianModelSuite")
    print("-" * 50)
    train_raw = df_feat[df_feat["season"] <= 2017]
    test_raw = df_feat[df_feat["season"] == 2018]
    suite_b = BayesianModelSuite(draws=1000,
                                 tune=1000, 
                                 include_random_slope=False, 
                                 random_seed=42)
    suite_b.fit(train_raw, preprocessor=preprocessor)
    metrics_b = suite_b.evaluate(test_raw, preprocessor=preprocessor)
    print("   Method B Results:")
    for metric, value in metrics_b.items():
        print(f"     {metric}: {value:.4f}")
    
    # 5️⃣ Comparing both methods
    print("\n5️⃣ Comparing both methods")
    print("-" * 50)
    print("Both methods should produce identical results since they use the same preprocessing pipeline and the same random seed.")
    print("\nMethod A vs Method B comparison:")
    for metric in metrics_a.keys():
        diff = abs(metrics_a[metric] - metrics_b[metric])
        print(f"   {metric}: {diff:.6f} (difference)")
    
    print("\n✅ Integration complete! The BayesianModelSuite now supports both:")
    print("   • Direct use with preprocessed data")
    print("   • Automatic preprocessing with DataPreprocessor integration")
    print("   • Consistent results across all model families")
    print("   • No performance penalty from preprocessing in MCMC loop")
    
    # ------------------
    # ▶️ Validation Checks
    # ------------------
    print("\n✅ Running validation checks...")

    # 1️⃣ Credible interval sanity
    cid = suite_b.kicker_interval_by_name(df_feat, "JUSTIN TUCKER", distance=40)
    assert cid["lower"] <= cid["mean"] <= cid["upper"], \
        f"Credible interval ordering failed: {cid}"
    print("• Credible interval check passed.")

    # 2️⃣ Posterior-Predictive Check correlation
    ax = suite_b.plot_distance_ppc(test_data, bin_width=5, preprocessor=preprocessor)
    df_ppc = preprocessor.preprocess_slice(test_data).copy()
    df_ppc["bin"] = (df_ppc["attempt_yards"] // 5) * 5
    actual = df_ppc.groupby("bin")["success"].mean().values
    posterior = df_ppc.assign(pred=suite_b.predict(df_ppc)) \
                   .groupby("bin")["pred"].mean().values
    corr = np.corrcoef(actual, posterior)[0, 1]
    assert corr > 0.9, f"PPC correlation too low: {corr:.3f}"
    print(f"• PPC correlation check passed (r={corr:.3f}).")

    # 3️⃣ EPA-FG+ leaderboard consistency
    epa_tbl = suite_b.epa_fg_plus(df_feat, n_samples=500, return_ci=True)
    assert {"hdi_lower", "hdi_upper", "certainty_level"}.issubset(epa_tbl.columns)
    print(epa_tbl.head())
    
    # Check that Justin Tucker is among the top kickers using name lookup
    justin_tucker_id = suite_b.get_kicker_id_by_name(df_feat, "JUSTIN TUCKER")
    top_kickers = epa_tbl.index.tolist()[:3]
    if justin_tucker_id is not None:
        assert justin_tucker_id in top_kickers, \
            f"Expected JUSTIN TUCKER (ID: {justin_tucker_id}) in top 3: {top_kickers}"
        print("• EPA-FG+ leaderboard check passed.")
    else:
        print("• WARNING: JUSTIN TUCKER not found in dataset, skipping leaderboard check.")

    # 4️⃣ MCMC diagnostics
    diag = suite_b.diagnostics(return_scalars=True)
    assert diag["summary_ok"], (
        f"R-hat max={diag['rhat_max']:.3f}, "
        f"ESS min={diag['ess_min']:.0f}"
    )
    print("• Diagnostics check passed (R-hat ≤1.01, ESS ≥100).")

    print("\n🎉 All validation checks passed!")
