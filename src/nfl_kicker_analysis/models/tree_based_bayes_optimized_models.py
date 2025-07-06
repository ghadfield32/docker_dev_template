"""
Traditional ML models for NFL kicker analysis.
Includes simple logistic regression, ridge logistic regression, and random forest.
Each model can be optionally tuned using Bayesian optimization with Optuna.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, Tuple, Optional, Union, Any
import optuna
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from numpy.typing import NDArray

from src.nfl_kicker_analysis.utils.metrics import ModelEvaluator

class TreeBasedModelSuite:
    """Suite of traditional ML models with optional Bayesian optimization."""
    
    def __init__(self):
        """Initialize the model suite."""
        self.fitted_models: Dict[str, Union[LogisticRegression, RandomForestClassifier, XGBClassifier, CatBoostClassifier]] = {}
        self.evaluator = ModelEvaluator()
        self._tss = TimeSeriesSplit(n_splits=3)  # 3-fold CV that respects time order
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.int_], OneHotEncoder]:
        """
        Prepare feature matrices for modeling.
        
        Args:
            data: DataFrame with attempt_yards and kicker_id
            
        Returns:
            Tuple of:
            - Distance-only features
            - Combined features (distance + one-hot kicker)
            - Kicker IDs
            - OneHotEncoder for kickers
        """
        # Distance features
        X_distance = data['attempt_yards'].values.astype(np.float_).reshape(-1, 1)
        
        # Kicker IDs for tree models
        kicker_ids = data['kicker_id'].values.astype(np.int_).reshape(-1, 1)
        
        # One-hot encode kickers for linear models
        encoder = OneHotEncoder(sparse_output=True)
        kicker_onehot = encoder.fit_transform(kicker_ids)
        X_combined = np.hstack([X_distance, kicker_onehot.toarray()])
        
        return X_distance, X_combined, kicker_ids, encoder
    
    def create_time_split(self, data: pd.DataFrame) -> Tuple[NDArray[np.int_], NDArray[np.int_]]:
        """
        Create train/test split by time.
        
        Args:
            data: DataFrame with game_date
            
        Returns:
            Train and test indices
        """
        train_mask = data['season'] <= 2017
        test_mask = data['season'] == 2018
        
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        
        print(f"Train: {len(train_idx):,} attempts ({train_mask.mean():.1%})")
        print(f"Test: {len(test_idx):,} attempts ({test_mask.mean():.1%})")
        
        return train_idx, test_idx

    def _tune_simple_logistic_optuna(
        self,
        X: NDArray[np.float_],
        y: NDArray[np.int_],
        n_trials: int = 50,
    ) -> LogisticRegression:
        """
        Bayesian-optimize and fit a simple LogisticRegression.
        Returns the fitted best model.
        """
        def objective(trial: optuna.Trial) -> float:
            params = {
                "C": trial.suggest_float("C", 1e-5, 100, log=True),
                "max_iter": 1000,
                "random_state": 42,
            }
            aucs = []
            for tr_idx, val_idx in self._tss.split(X):
                model = LogisticRegression(**params)
                model.fit(X[tr_idx], y[tr_idx])
                preds = model.predict_proba(X[val_idx])[:, 1].astype(np.float_)
                aucs.append(self.evaluator.calculate_auc(y[val_idx], preds))
            return float(np.mean(aucs))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_params.update(dict(max_iter=1000, random_state=42))
        best_model = LogisticRegression(**best_params)
        best_model.fit(X, y)
        return best_model

    def _tune_ridge_logistic_optuna(
        self,
        X: NDArray[np.float_],
        y: NDArray[np.int_],
        n_trials: int = 50,
    ) -> LogisticRegression:
        """
        Bayesian-optimize and fit a ridge LogisticRegression.
        Returns the fitted best model.
        """
        def objective(trial: optuna.Trial) -> float:
            params = {
                "C": trial.suggest_float("C", 1e-5, 100, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0, 1),
                "penalty": "elasticnet",
                "solver": "saga",
                "max_iter": 1000,
                "random_state": 42,
            }
            aucs = []
            for tr_idx, val_idx in self._tss.split(X):
                model = LogisticRegression(**params)
                model.fit(X[tr_idx], y[tr_idx])
                preds = model.predict_proba(X[val_idx])[:, 1].astype(np.float_)
                aucs.append(self.evaluator.calculate_auc(y[val_idx], preds))
            return float(np.mean(aucs))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_params.update(dict(penalty="elasticnet", solver="saga", max_iter=1000, random_state=42))
        best_model = LogisticRegression(**best_params)
        best_model.fit(X, y)
        return best_model

    def _tune_random_forest_optuna(
        self,
        X: NDArray[np.float_],
        y: NDArray[np.int_],
        n_trials: int = 50,
    ) -> RandomForestClassifier:
        """
        Bayesian-optimize and fit a RandomForestClassifier.
        Returns the fitted best model.
        """
        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_float("max_features", 0.1, 1.0),
                "n_jobs": -1,
                "random_state": 42,
            }
            aucs = []
            for tr_idx, val_idx in self._tss.split(X):
                model = RandomForestClassifier(**params)
                model.fit(X[tr_idx], y[tr_idx])
                preds = model.predict_proba(X[val_idx])[:, 1].astype(np.float_)
                aucs.append(self.evaluator.calculate_auc(y[val_idx], preds))
            return float(np.mean(aucs))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_params.update(dict(n_jobs=-1, random_state=42))
        best_model = RandomForestClassifier(**best_params)
        best_model.fit(X, y)
        return best_model

    def _tune_xgboost_optuna(
        self,
        X: NDArray[np.float_],
        y: NDArray[np.int_],
        n_trials: int = 50,
    ) -> XGBClassifier:
        """
        Bayesian-optimize and fit an XGBClassifier.
        Returns the fitted best model.
        """
        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 600),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "n_jobs": -1,
                "random_state": 42,
            }
            aucs = []
            for tr_idx, val_idx in self._tss.split(X):
                model = XGBClassifier(**params)
                model.fit(X[tr_idx], y[tr_idx])
                preds = model.predict_proba(X[val_idx])[:, 1].astype(np.float_)
                aucs.append(self.evaluator.calculate_auc(y[val_idx], preds))
            return float(np.mean(aucs))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_params.update(dict(objective="binary:logistic", eval_metric="logloss", n_jobs=-1, random_state=42))
        best_model = XGBClassifier(**best_params)
        best_model.fit(X, y)
        return best_model

    def _tune_catboost_optuna(
        self,
        df_train: pd.DataFrame,          # ←  NOW a DataFrame, not ndarray
        y_train: NDArray[np.int_],
        n_trials: int = 50,
    ) -> CatBoostClassifier:
        """
        Bayesian-optimise and fit a CatBoostClassifier on a mixed-dtype DataFrame.
        The column ``'kicker_id'`` is treated as categorical automatically.
        """
        cat_cols = ["kicker_id"]         # name-based is safer than index-based

        def objective(trial: optuna.Trial) -> float:
            params = {
                "iterations":        trial.suggest_int ("iterations",        300, 800),
                "depth":             trial.suggest_int ("depth",             4,   10),
                "learning_rate":     trial.suggest_float("learning_rate",    0.01, 0.3, log=True),
                "l2_leaf_reg":       trial.suggest_float("l2_leaf_reg",      1e-3, 10.0, log=True),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
                "random_strength":   trial.suggest_float("random_strength",  0.1, 10.0),
                "border_count":      trial.suggest_int ("border_count",      32,  255),
                "loss_function":     "Logloss",
                "eval_metric":       "AUC",
                "verbose":           False,
                "random_seed":       42,
            }
            aucs = []
            for tr_idx, val_idx in self._tss.split(df_train):
                model = CatBoostClassifier(**params)
                model.fit(
                    df_train.iloc[tr_idx],
                    y_train[tr_idx],
                    cat_features=cat_cols,
                    verbose=False,
                )
                preds = model.predict_proba(df_train.iloc[val_idx])[:, 1].astype(np.float_)
                aucs.append(self.evaluator.calculate_auc(y_train[val_idx], preds))
            return float(np.mean(aucs))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_params.update(dict(loss_function="Logloss",
                                eval_metric="AUC",
                                verbose=False,
                                random_seed=42))
        best_model = CatBoostClassifier(**best_params)
        best_model.fit(df_train, y_train, cat_features=cat_cols, verbose=False)
        return best_model

    def fit_all_models(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Fit all traditional + boosted models and return their metrics.
        """
        print("Fitting traditional & boosted ML models with Bayesian optimization...")

        # Feature prep
        X_distance, X_combined, kicker_ids, kicker_encoder = self.prepare_features(data)
        y = data["success"].values.astype(np.int_)

        # Time-based split
        train_idx, test_idx = self.create_time_split(data)
        X_dist_train, X_dist_test = X_distance[train_idx], X_distance[test_idx]
        X_comb_train, X_comb_test = X_combined[train_idx], X_combined[test_idx]
        kicker_train, kicker_test = kicker_ids[train_idx], kicker_ids[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        results = {}

        # 1. Simple Logistic with Optuna
        print("  ▸ simple logistic + Optuna")
        simple_lr = self._tune_simple_logistic_optuna(X_dist_train, y_train)
        self.fitted_models["simple_logistic"] = simple_lr
        y_pred = simple_lr.predict_proba(X_dist_test)[:, 1].astype(np.float_)
        results["Simple Logistic"] = self.evaluator.calculate_classification_metrics(y_test, y_pred)

        # 2. Ridge Logistic with Optuna
        print("  ▸ ridge logistic + Optuna")
        ridge_lr = self._tune_ridge_logistic_optuna(X_comb_train, y_train)
        self.fitted_models["ridge_logistic"] = ridge_lr
        y_pred = ridge_lr.predict_proba(X_comb_test)[:, 1].astype(np.float_)
        results["Ridge Logistic"] = self.evaluator.calculate_classification_metrics(y_test, y_pred)

        # 3. Random Forest with Optuna
        print("  ▸ random forest + Optuna")
        X_rf_train = np.column_stack([X_dist_train, kicker_train]).astype(np.float_)
        X_rf_test = np.column_stack([X_dist_test, kicker_test]).astype(np.float_)
        rf_model = self._tune_random_forest_optuna(X_rf_train, y_train)
        self.fitted_models["random_forest"] = rf_model
        y_pred = rf_model.predict_proba(X_rf_test)[:, 1].astype(np.float_)
        results["Random Forest"] = self.evaluator.calculate_classification_metrics(y_test, y_pred)

        # 4. XGBoost with Optuna
        print("  ▸ xgboost + Optuna")
        xgb_model = self._tune_xgboost_optuna(X_rf_train, y_train)
        self.fitted_models["xgboost"] = xgb_model
        y_pred = xgb_model.predict_proba(X_rf_test)[:, 1].astype(np.float_)
        results["XGBoost"] = self.evaluator.calculate_classification_metrics(y_test, y_pred)

        # 5. CatBoost with Optuna  ←-- NEW implementation
        print("  ▸ catboost + Optuna")
        df_cat_train = pd.DataFrame({
            "attempt_yards": X_dist_train.ravel().astype(np.float32),
            "kicker_id":     kicker_train.ravel().astype("int32"),
        })
        df_cat_test = pd.DataFrame({
            "attempt_yards": X_dist_test.ravel().astype(np.float32),
            "kicker_id":     kicker_test.ravel().astype("int32"),
        })

        cat_model = self._tune_catboost_optuna(df_cat_train, y_train)
        self.fitted_models["catboost"] = cat_model
        y_pred = cat_model.predict_proba(df_cat_test)[:, 1].astype(np.float_)
        results["CatBoost"] = self.evaluator.calculate_classification_metrics(y_test, y_pred)

        print("***** all models fitted & scored *****")
        return results

    def predict(self, model_name: str, data: pd.DataFrame) -> NDArray[np.float_]:
        """
        Predict probabilities with any fitted model in the suite.
        """
        if model_name not in self.fitted_models:
            raise ValueError(f"Model {model_name} not fitted")

        model = self.fitted_models[model_name]

        if model_name == "simple_logistic":
            X = data["attempt_yards"].values.astype(np.float_).reshape(-1, 1)
        elif model_name in {"ridge_logistic"}:
            # Distance + one-hot kicker – need encoder state (TODO: save encoder)
            raise NotImplementedError("Pass encoded matrix for ridge predictions")
        elif model_name in {"random_forest", "xgboost"}:
            X = np.column_stack([
                data["attempt_yards"].values.astype(np.float_),
                data["kicker_id"].values.astype(np.float_)  # Convert to float for tree models
            ])
        elif model_name == "catboost":
            X = pd.DataFrame({
                "attempt_yards": data["attempt_yards"].values.astype(np.float32),
                "kicker_id": data["kicker_id"].values.astype("int32"),
            })
        else:
            raise ValueError(f"Unknown model: {model_name}")

        return model.predict_proba(X)[:, 1].astype(np.float_)

    def get_feature_importance(self, model_name: str) -> Optional[NDArray[np.float_]]:
        """
        Get feature importance for tree-based models.
        """
        if model_name not in self.fitted_models:
            return None

        model = self.fitted_models[model_name]

        if model_name in {"catboost"}:
            return model.get_feature_importance().astype(np.float_)
        elif hasattr(model, "feature_importances_"):
            return model.feature_importances_.astype(np.float_)
        else:
            return None

if __name__ == "__main__":
    from src.nfl_kicker_analysis.data.loader import DataLoader
    from src.nfl_kicker_analysis.data.feature_selection import (
        DynamicSchema,
        filter_to_final_features,
        update_schema_numerical,
    )
    from src.nfl_kicker_analysis.data.preprocessor import DataPreprocessor
    from src.nfl_kicker_analysis.data.feature_engineering import FeatureEngineer

    # ─── 1 Load raw data ──────────────────────────────────────────
    loader = DataLoader()
    df_raw = loader.load_complete_dataset()
    
    # ─── 2 Feature engineering pass ───────────────────────────────
    engineer = FeatureEngineer()
    df_feat = engineer.create_all_features(df_raw)

    for category, details in engineer.get_available_features(df_feat).items():
        print(f"-- {category} --")
        for feat, uniques in details.items():
            print(f"   {feat}: {len(uniques)} unique | sample {uniques[:5] if uniques else '...'}")

    # ─── 3 Define all tunables in one place ─────────────────────
    CONFIG = {
        'min_distance': 20,
        'max_distance': 60,
        'min_kicker_attempts': 8,
        'season_types': ['Reg', 'Post'],  # now include playoffs
        'include_performance_history': True,
        'include_statistical_features': False,
        'include_player_status': True,  # ✅ FIX: Added missing parameter
        'performance_window': 12,
    }


    # ------------------------------------------------------------------
    # 🔧 Single source of truth for column roles – edit freely
    # ------------------------------------------------------------------
    FEATURE_LISTS = {
        "numerical": [
            "attempt_yards", "age_at_attempt", "distance_squared",
            "career_length_years", "season_progress", "rolling_success_rate",
            "current_streak", "distance_zscore", "distance_percentile",
        ],
        "ordinal":  ["season", "week", "month", "day_of_year"],
        "nominal":  [
            "kicker_id", "is_long_attempt", "is_very_long_attempt",
            "is_rookie_attempt", "distance_category", "experience_category",
        ],
        "y_variable": ["success"],
    }

    # ➊  Build schema from the dict
    schema = DynamicSchema(FEATURE_LISTS)
    
    # read final_features.txt
    with open("data/models/features/final_features.txt", "r") as f:
        final_features = [line.strip() for line in f]
    print(f"---------------final_features---------------")
    print(final_features)
    numeric_final = [f for f in final_features if f in schema.numerical]

    print(f"\n✨ Final feature count: {len(numeric_final)}")
    print("Selected features:")
    for feat in numeric_final:
        print(f"  • {feat}")

    # 🔄 Push into schema so every later stage sees the new list
    update_schema_numerical(schema, numeric_final)

    # output final_features from schema
    FEATURE_LISTS = schema.lists
    print(f"---------------FEATURE_LISTS---------------")
    print(FEATURE_LISTS)

    pre = DataPreprocessor()
    pre.update_config(**CONFIG)
    pre.update_feature_lists(**FEATURE_LISTS)
    _ = pre.preprocess_complete(df_feat)
    X, y = pre.fit_transform_features()

    print("First 5 rows after inverse‑transform round‑trip →")
    print(pre.invert_preprocessing(X[:5]).head())

    # ─── 4  Fit traditional models on real data ───────────────────────────────
    print("Testing Traditional Models…")
    model_suite = TreeBasedModelSuite()

    try:
        results = model_suite.fit_all_models(pre.processed_data)
        print("\n***** Metrics *****")
        for model, metric_dict in results.items():
            print(f"{model}:")
            for k, v in metric_dict.items():
                print(f"   {k:>12}: {v:.4f}")
        print("******* Traditional models tests passed!")

    except Exception as e:
        print(f"------------- Error testing traditional models: {e}")


