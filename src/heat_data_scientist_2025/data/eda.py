%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import warnings
import time
from concurrent.futures import ThreadPoolExecutor
import json
warnings.filterwarnings('ignore')
from typing import List, Optional, Dict, Any, Tuple, Set

from src.heat_data_scientist_2025.data.feature_engineering import engineer_features
from src.heat_data_scientist_2025.data.column_schema import (load_schema_from_yaml, extract_feature_lists_from_schema
                                                        ,SchemaValidationError, report_schema_dtype_violations)
# Load engineered data
from src.heat_data_scientist_2025.data.load_data_utils import load_data_optimized
from src.heat_data_scientist_2025.utils.config import CFG



# ML imports
from sklearn.feature_selection import mutual_info_regression, VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import TimeSeriesSplit
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Set plot styles
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10

# Paths
DATA_PATH = CFG.ml_dataset_path
EDA_OUT_DIR = CFG.eda_out_dir
SCHEMA_PATH = CFG.column_schema_path

# Performance settings
SAMPLE_SIZE = 10000  # For heavy computations
MAX_FEATURES_FOR_CLUSTERING = 100  # Limit features for clustering
MAX_FEATURES_FOR_VIF = 50  # Limit features for VIF analysis
PROFILING_SAMPLE_SIZE = 5000  # For automated profiling


def safe_sort_values(df_or_series, *args, **kwargs):
    """
    Safely sort DataFrames/Series that may contain categorical or dtype-like
    (np.dtype / pd.CategoricalDtype) content. We never coerce the source data
    globally; we only cast the *sort key* (or the values for Series) to a safe
    type for the sort itself.

    Examples of dtype-like content that can crash argsort:
      - Series of dtypes (e.g., df.dtypes)
      - Groupby results indexed by dtype objects
    """
    import pandas as pd
    import numpy as np
    from pandas.api.types import is_categorical_dtype

    def _looks_dtype_like(x) -> bool:
        # Robust check that works for numpy dtypes and pandas CategoricalDtype
        tname = type(x).__name__.lower()
        return "dtype" in tname

    def _series_is_dtype_like(s: pd.Series) -> bool:
        if s.dtype != object:
            return False
        # inspect a small non-null sample
        sample = s.dropna()
        if sample.empty:
            return False
        return sample.head(10).map(_looks_dtype_like).all()

    if isinstance(df_or_series, pd.DataFrame):
        df_copy = df_or_series.copy()

        # Normalize any categorical columns to object for stable comparisons
        for col in df_copy.select_dtypes(include=['category']).columns:
            df_copy[col] = df_copy[col].astype('object')

        # If sorting by specific columns, ensure those columns aren't dtype-like
        by = kwargs.get("by", None)
        if by is None and len(args) >= 1:
            by = args[0]
        if by is not None:
            by_cols = [by] if isinstance(by, str) else list(by)
            for col in by_cols:
                if col in df_copy.columns:
                    s = df_copy[col]
                    if is_categorical_dtype(s):
                        df_copy[col] = s.astype('object')
                    elif _series_is_dtype_like(s):
                        df_copy[col] = s.astype(str)

        return df_copy.sort_values(*args, **kwargs)

    elif isinstance(df_or_series, pd.Series):
        s = df_or_series
        # Convert categorical to object, or dtype-like objects to string, then sort
        if is_categorical_dtype(s):
            s = s.astype('object')
        elif _series_is_dtype_like(s):
            s = s.astype(str)
        return s.sort_values(*args, **kwargs)

    # Fallback: return original if unknown type
    return df_or_series



def display_schema_summary(schema):
    """
    Display a summary of the loaded schema.
    
    Args:
        schema: SchemaConfig object
    """
    print("\n" + "="*60)
    print("SCHEMA CONFIGURATION SUMMARY")
    print("="*60)
    print(f"ID columns ({len(schema.id())}):           {schema.id()}")
    print(f"Ordinal columns ({len(schema.ordinal())}):       {schema.ordinal()}")
    print(f"Nominal columns ({len(schema.nominal())}):       {schema.nominal()[:10]}{'...' if len(schema.nominal()) > 10 else ''}")
    print(f"Numerical columns ({len(schema.numerical())}):     {schema.numerical()[:10]}{'...' if len(schema.numerical()) > 10 else ''}")
    print(f"Target column:          {schema.target()}")
    print(f"Total expected columns: {len(schema.all_expected())}")

def validate_dataframe_against_schema(df, schema, strict=False, debug=False):
    """
    Validate DataFrame against schema and return validation report.
    
    Args:
        df: DataFrame to validate
        schema: SchemaConfig object
        strict: Whether to raise errors on validation failures
        debug: Whether to print debug information
        
    Returns:
        Validation report dictionary
    """
    print(f"\nValidating DataFrame with {len(df.columns)} columns against schema...")
    
    try:
        validation_report = schema.validate_dataframe(df, strict=strict, debug=debug)
        
        print(f"✓ Validation completed:")
        print(f"  - Missing columns: {len(validation_report['missing_columns'])}")
        print(f"  - Unexpected columns: {len(validation_report['unexpected_columns'])}")
        print(f"  - Dtype mismatches: {len(validation_report['dtype_mismatches'])}")
        print(f"  - Valid columns: {len(validation_report['ok'])}")
        
        if validation_report['missing_columns']:
            print(f"  Missing columns: {validation_report['missing_columns'][:10]}{'...' if len(validation_report['missing_columns']) > 10 else ''}")
        
        if validation_report['unexpected_columns']:
            print(f"  Unexpected columns: {validation_report['unexpected_columns'][:10]}{'...' if len(validation_report['unexpected_columns']) > 10 else ''}")
            
        return validation_report
        
    except SchemaValidationError as e:
        print(f"❌ Schema validation failed: {e}")
        if strict:
            raise
        return None

def extract_feature_groups_from_schema(df, schema):
    """
    Extract feature groups using the schema, with debug enabled to surface
    near-misses and non-numeric numericals early.
    """
    print(f"\nExtracting feature groups from schema...")

    numericals, ordinal, nominal, y, cat_breakdown = extract_feature_lists_from_schema(
        df, schema, debug=True  # <-- enable detailed diagnostics
    )

    print(f"✓ Feature extraction completed:")
    print(f"  - Numerical features: {len(numericals)}")
    print(f"  - Ordinal features: {len(ordinal)}")
    print(f"  - Nominal features: {len(nominal)}")
    print(f"  - Target variable: {y}")
    print(f"  - Numerical categories available: {list(cat_breakdown.keys())}")

    return numericals, ordinal, nominal, y, cat_breakdown




def plot_full_correlation_mapping(df, numericals, output_dir=None, show=True):
    """
    Plot a clustered heatmap of the correlation matrix for all numerical features.
    """
    corr = df[numericals].corr()
    # clustered heatmap for easier pattern discovery
    sns.clustermap(
        corr,
        figsize=(14, 14),
        cmap="coolwarm",
        center=0,
        metric="euclidean",
        method="average",
        annot=False
    )
    plt.suptitle("Clustered Correlation Heatmap – All Numerical Features", y=1.02)
    plt.tight_layout()
    if output_dir:
        plt.savefig(output_dir / "full_correlation_clustermap.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()



def data_preparation_optimized(df, schema=None, output_dir=None, do_impute=False):
    """
    Optimized data preparation with performance improvements.
    - Fast feature filtering before expensive operations
    - Sampling for heavy computations
    - Efficient outlier detection
    - Uses schema system for feature categorization
    - Imputation is optional (do_impute flag); missingness is reported explicitly.
    
    Returns:
        df_prepared: working copy (without hidden imputation unless enabled)
        selected_features: list of features chosen via variance and MI
        target_col: name of target
        prep_diagnostics: dict with detailed diagnostics (missingness, skewed, MI, etc.)
    """
    print("\n" + "="*60)
    print("OPTIMIZED DATA PREPARATION (WITH EXPLICIT DIAGNOSTICS)")
    print("="*60)

    start_time = time.time()

    numericals, ordinal, nominal, target_col, cat_breakdown = extract_feature_groups_from_schema(df, schema)
    print(f"Using schema-defined features:")
    print(f"  - Numerical: {len(numericals)}")
    print(f"  - Ordinal: {len(ordinal)}")  
    print(f"  - Nominal: {len(nominal)}")
    print(f"  - Target: {target_col}")

    if target_col is None or target_col not in df.columns:
        raise ValueError("Target column not found; cannot proceed with preparation.")

    # 1. Missing data diagnostics
    print("Performing fast missing data analysis...")
    missing_pct = df[numericals].isna().mean()
    keep_features = missing_pct[missing_pct <= 0.9].index.tolist()
    dropped_features = [col for col in numericals if col not in keep_features]
    print(f"✓ Dropped {len(dropped_features)} features with >90% missing")
    print(f"✓ Keeping {len(keep_features)} features with ≤90% missing")

    df_prepared = df.copy()  # base for transformations

    # 2. Variance thresholding for dimensionality reduction
    if len(keep_features) > MAX_FEATURES_FOR_CLUSTERING:
        print(f"\nApplying variance thresholding to reduce features from {len(keep_features)} to <={MAX_FEATURES_FOR_CLUSTERING}...")
        sample_data = df_prepared[keep_features].sample(n=min(SAMPLE_SIZE, len(df_prepared)), random_state=42)

        selector = VarianceThreshold(threshold=0.01)  # near-constant removal
        selector.fit(sample_data.fillna(0))

        selected_features = [keep_features[i] for i in range(len(keep_features)) if selector.get_support()[i]]

        # further reduce by correlation with target if still too many
        if len(selected_features) > MAX_FEATURES_FOR_CLUSTERING:
            print("Further reducing by absolute correlation with target...")
            correlations = df_prepared[selected_features].corrwith(df_prepared[target_col]).abs()
            selected_features = correlations.nlargest(MAX_FEATURES_FOR_CLUSTERING).index.tolist()

        print(f"✓ Reduced to {len(selected_features)} features after variance thresholding")
    else:
        selected_features = keep_features.copy()
        print(f"Skipping variance thresholding (feature count {len(selected_features)} within limit)")

    # 3. Skewness detection (before any transform)
    skewness_series = df_prepared[selected_features].skew(numeric_only=True).abs()
    skewed_features = skewness_series[skewness_series > 1].index.tolist()
    print(f"\nDetected {len(skewed_features)} skewed features (|skew| > 1): {skewed_features[:10]}{'...' if len(skewed_features) > 10 else ''}")

    # 4. Mutual information selection (only if enough target data)
    mi_df = None
    if df_prepared[target_col].notna().sum() > 100 and len(selected_features) > 0:
        print(f"\nComputing mutual information for {len(selected_features)} candidate features...")
        valid_mask = df_prepared[target_col].notna()
        sample_indices = df_prepared.loc[valid_mask].sample(
            n=min(SAMPLE_SIZE, valid_mask.sum()), random_state=42
        ).index
        X_sample = df_prepared.loc[sample_indices, selected_features]
        y_sample = df_prepared.loc[sample_indices, target_col]

        mi_scores = mutual_info_regression(X_sample.fillna(0), y_sample, random_state=42, n_jobs=-1)
        mi_df = pd.DataFrame({
            'feature': selected_features,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)

        top_k = min(50, len(selected_features))
        top_features = mi_df.head(top_k)['feature'].tolist()
        selected_features = top_features
        print(f"✓ Selected top {len(selected_features)} features by mutual information")
        print(f"  Top 10: {selected_features[:10]}")
        if output_dir:
            mi_df.to_csv(output_dir / "mutual_information_ranking.csv", index=False)
            print(f"  Saved MI ranking to {output_dir / 'mutual_information_ranking.csv'}")
    else:
        print("Skipping mutual information (insufficient target data or no selected features)")

    # 5. Optional log-transform of skewed features (do not do automatically)
    # Provide suggested transformation list but do not apply unless caller opts in
    suggested_log_transform = skewed_features.copy()
    print(f"\nSuggested log1p transform for skewed features (not applied automatically): {suggested_log_transform[:10]}{'...' if len(suggested_log_transform) > 10 else ''}")

    # 6. Missing indicators (kept explicit)
    missing_indicators = []
    if 'dead_cap' in df.columns:
        print(f"\nCreating missing indicator for 'dead_cap' (explicit, not imputed)...")
        df_prepared['dead_cap_missing'] = df_prepared['dead_cap'].isna().astype(int)
        selected_features.append('dead_cap_missing')
        missing_indicators.append('dead_cap_missing')
        print(f"✓ Added dead_cap_missing indicator")

    # 7. Imputation only if requested (do_impute)
    if do_impute and selected_features:
        print(f"\nImputing selected numeric features with median (user opted in)...")
        df_prepared[selected_features] = df_prepared[selected_features].fillna(df_prepared[selected_features].median())
        print(f"✓ Imputation complete")
    else:
        print(f"\nSkipping imputation (do_impute={do_impute}); missingness preserved for transparency.")

    # 8. Outlier detection (always run on prepared set)
    outlier_analysis = None
    if selected_features:
        print(f"\nPerforming outlier analysis on selected features...")
        outlier_analysis = detect_outliers_efficient(df_prepared[selected_features])
        if output_dir and outlier_analysis is not None:
            outlier_analysis.to_csv(output_dir / "outlier_analysis.csv", index=False)
            print(f"  Saved outlier analysis to {output_dir / 'outlier_analysis.csv'}")

    prep_time = time.time() - start_time

    # Summary diagnostics
    diagnostics = {
        'original_numerical': len(numericals),
        'after_missing_filter': len(keep_features),
        'after_variance_threshold': len(selected_features),
        'skewed_features': skewed_features,
        'missing_indicators': missing_indicators,
        'did_impute': do_impute,
        'outlier_summary': outlier_analysis.to_dict(orient='list') if outlier_analysis is not None else None,
        'mi_dataframe': mi_df  # caller can inspect if needed
    }

    print("\n--- PREPARATION SUMMARY ---")
    print(f"Original numericals: {len(numericals)}")
    print(f"Kept after missing filter: {len(keep_features)}")
    print(f"Selected features: {len(selected_features)}")
    print(f"Skewed suggested for log transform: {len(skewed_features)}")
    print(f"Missing indicators added: {len(missing_indicators)}")
    print(f"Imputation performed: {do_impute}")
    print(f"Total prep time: {prep_time:.2f}s")

    if output_dir:
        summary_df = pd.DataFrame({
            'step': ['Original', 'Missing Filter', 'Variance Threshold', 'MI Selection' if mi_df is not None else 'MI Skipped', 'Log Skew Suggestion', 'Missing Indicators'],
            'feature_count': [
                len(numericals),
                len(keep_features),
                len(selected_features),
                len(selected_features) if mi_df is not None else 'n/a',
                len(skewed_features),
                len(missing_indicators)
            ],
            'details': [
                f'{len(numericals)} numeric features',
                f'Keeping {len(keep_features)} after missingness filter',
                f'{len(selected_features)} final candidates',
                f'MI applied' if mi_df is not None else 'MI skipped',
                f'Suggested log1p for {len(skewed_features)}',
                f'Added {len(missing_indicators)} indicators'
            ]
        })
        summary_df.to_csv(output_dir / "preparation_summary.csv", index=False)
        print(f"  Saved preparation summary to {output_dir / 'preparation_summary.csv'}")

    return df_prepared, selected_features, target_col, diagnostics

def summarize_target_distribution(df, target_col, output_dir=None, show=True):
    """
    Compute and plot distribution of the target (analogous to salary histogram/boxplot).
    Returns structured summary and optionally saves plots.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in DataFrame.")

    series = df[target_col].dropna()
    if series.empty:
        print(f"No non-null values in target '{target_col}' to summarize.")
        return {}

    # Summary stats
    desc = series.describe(percentiles=[0.25, 0.5, 0.75, 0.9])
    skewness = series.skew()
    mean_val = desc["mean"]
    median_val = desc["50%"]
    q1 = desc["25%"]
    q3 = desc["75%"]
    max_val = desc["max"]

    # Plot histogram
    plt.figure(figsize=(12, 6))
    plt.hist(series, bins=50, edgecolor="white")
    plt.title(f"Distribution of {target_col}")
    plt.xlabel(target_col)
    plt.ylabel("Count")
    plt.grid(alpha=0.3)
    # annotate with key percentiles
    plt.axvline(median_val, color="black", linestyle="--", label=f"Median: {median_val:,.2f}")
    plt.axvline(q3, color="gray", linestyle=":", label=f"75th pct: {q3:,.2f}")
    plt.legend()
    if output_dir:
        plt.tight_layout()
        plt.savefig(output_dir / f"{target_col}_distribution_histogram.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    # Boxplot
    plt.figure(figsize=(6, 6))
    plt.boxplot(series, vert=True, showfliers=True)
    plt.title(f"{target_col} Boxplot")
    plt.ylabel(target_col)
    if output_dir:
        plt.tight_layout()
        plt.savefig(output_dir / f"{target_col}_boxplot.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    # Narrative summary (example-style)
    narrative = (
        f"The distribution of {target_col} is {'rightward' if skewness > 0 else 'leftward'} skewed "
        f"(skewness={skewness:.2f}), with median {target_col} = {median_val:,.2f} and mean = {mean_val:,.2f}. "
        f"75% of observations fall below {q3:,.2f} (third quartile), while the maximum value is {max_val:,.2f}."
    )

    summary = {
        "mean": mean_val,
        "median": median_val,
        "q1": q1,
        "q3": q3,
        "max": max_val,
        "skewness": skewness,
        "n_non_null": len(series),
        "n_total": len(df),
        "n_null": df[target_col].isna().sum(),
        "n_unique": series.nunique(),
        "n_zeros": (series == 0).sum(),
        "n_outliers_high": ((series > (q3 + 1.5 * (q3 - q1)))).sum(),
        "n_outliers_low": ((series < (q1 - 1.5 * (q3 - q1)))).sum(),
        "narrative": narrative
    }

    print("\n=== Target Distribution Summary ===")
    print(narrative)
    print(f"Count non-null: {summary['n_non_null']}, Nulls: {summary['n_null']}")
    print(f"Top quartiles: Q1={q1:,.2f}, Median={median_val:,.2f}, Q3={q3:,.2f}")
    print(f"Skewness: {skewness:.2f}")

    return summary



def extract_primary_category(series, separator="-"):
    """
    Given a pandas Series of multi-valued categories like 'SG-SF', extract the primary
    (first) category for grouping (e.g., 'SG'). Returns a new Series.
    """
    def first_token(val):
        if pd.isna(val):
            return np.nan
        if isinstance(val, (list, tuple)):
            return val[0]
        s = str(val)
        return s.split(separator)[0].strip()

    return series.map(first_token)


def correlation_with_target_summary(df, numericals, target_col, top_k=10, output_dir=None):
    """
    Compute correlation of numerical features with target, sort them, and provide summary.
    Returns sorted correlations and a narrative dict.
    """
    if target_col not in df.columns:
        raise ValueError("Target column missing for correlation summary.")

    combo = df[numericals + [target_col]].dropna(subset=[target_col])
    corr_matrix = combo.corr()
    corr_with_target = corr_matrix[target_col].drop(target_col).abs().sort_values(ascending=False)

    top_features = corr_with_target.head(top_k)
    print(f"\nTop {top_k} numerical features by absolute correlation with {target_col}:")
    print(top_features.to_string())

    # Check intercorrelation among the top features
    inter_corr = combo[top_features.index].corr().abs()
    upper_tri = inter_corr.where(np.triu(np.ones(inter_corr.shape), k=1).astype(bool))
    high_inter = (upper_tri > 0.8).stack().reset_index()
    high_inter.columns = ['feature_1', 'feature_2', 'corr']
    high_inter = high_inter[high_inter['corr'] > 0.8]

    if not high_inter.empty:
        print("\n⚠ High intercorrelation detected among top predictors (potential multicollinearity):")
        for _, row in high_inter.iterrows():
            print(f"  - {row['feature_1']} <-> {row['feature_2']}: corr={row['corr']:.2f}")
    else:
        print("\nNo severe intercorrelation detected among top predictors (all ≤0.8).")

    if output_dir:
        corr_with_target.to_csv(output_dir / f"{target_col}_correlation_with_features.csv")
        inter_corr.to_csv(output_dir / f"intercorrelation_top_{top_k}.csv")
        print(f"Saved correlation summaries to {output_dir}")

    narrative = (
        f"The top {top_k} features most associated with {target_col} (by absolute Pearson correlation) are: "
        + ", ".join([f"{f} ({corr_with_target[f]:.2f})" for f in top_features.index]) + "."
    )
    if not high_inter.empty:
        narrative += " Note: several of these top features are themselves highly correlated, which suggests potential multicollinearity that should be addressed before modeling."

    return {
        "sorted_correlations": corr_with_target,
        "top_features": top_features,
        "intercorrelated_pairs": high_inter,
        "narrative": narrative
    }



def categorical_target_impact(df, categorical_col, target_col, output_dir=None):
    """
    Like 'Salary by Position': summarizes the target by a primary categorical grouping,
    including group medians and a boxplot.
    """
    if categorical_col not in df.columns:
        print(f"Categorical column {categorical_col} not in df.")
        return

    # Extract primary category if compound
    primary = extract_primary_category(df[categorical_col])
    df_temp = df.assign(primary_category=primary)

    group_stats = df_temp.groupby("primary_category")[target_col].agg(
        count="count", median="median", mean="mean"
    ).sort_values("median", ascending=False)
    print(f"\nTarget ({target_col}) summary by primary '{categorical_col}':")
    print(group_stats.head(10).to_string())

    # Boxplot
    plt.figure(figsize=(10,6))
    sns.boxplot(x="primary_category", y=target_col, data=df_temp)
    plt.title(f"{target_col} Distribution by Primary {categorical_col}")
    plt.xlabel(f"Primary {categorical_col}")
    plt.ylabel(target_col)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if output_dir:
        filename = f"{target_col}_by_primary_{categorical_col.lower()}_boxplot.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        print(f"Saved {filename}")
    plt.close()

    # Narrative
    top = group_stats.iloc[0]
    narrative = (
        f"The primary category with highest median {target_col} is '{group_stats.index[0]}' "
        f"with median {target_col} = {top['median']:.2f} (count={int(top['count'])})."
    )
    return {
        "group_stats": group_stats,
        "narrative": narrative
    }






def detect_outliers_efficient(df):
    """Efficient outlier detection using multivariate approach."""
    outlier_results = []

    # Sample data for outlier detection
    sample_data = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)

    # Handle NaN and infinity values for outlier detection
    sample_data_clean = sample_data.fillna(sample_data.median())
    sample_data_clean = sample_data_clean.replace([np.inf, -np.inf], np.nan)
    sample_data_clean = sample_data_clean.fillna(sample_data_clean.median())

    # Multivariate outlier detection (more efficient)
    try:
        # Isolation Forest on full feature matrix
        iso_forest = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
        iso_pred = iso_forest.fit_predict(sample_data_clean)
        iso_outliers = (iso_pred == -1).sum()

        # Local Outlier Factor on full feature matrix
        lof = LocalOutlierFactor(contamination=0.1, n_jobs=-1)
        lof_pred = lof.fit_predict(sample_data_clean)
        lof_outliers = (lof_pred == -1).sum()

        # Univariate outlier detection (IQR method)
        iqr_outliers = 0
        for col in df.columns:
            data = sample_data[col].dropna()
            if len(data) > 0:
                Q1, Q3 = data.quantile([0.25, 0.75])
                IQR = Q3 - Q1
                outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum()
                iqr_outliers += outliers

        outlier_results.append({
            'method': 'multivariate',
            'iso_outliers': iso_outliers,
            'lof_outliers': lof_outliers,
            'iqr_outliers': iqr_outliers,
            'total_samples': len(sample_data)
        })

    except Exception as e:
        print(f"⚠ Error in multivariate outlier detection: {e}")
        # Fallback to simple univariate analysis
        outlier_results.append({
            'method': 'univariate_fallback',
            'iso_outliers': 0,
            'lof_outliers': 0,
            'iqr_outliers': 0,
            'total_samples': len(sample_data)
        })

    return pd.DataFrame(outlier_results)

def automated_profiling_optimized(df, output_dir=None):
    """Generate optimized automated profiling reports."""
    print("\n" + "="*60)
    print("OPTIMIZED AUTOMATED PROFILING")
    print("="*60)

    # Sample data for profiling
    sample_df = df.sample(n=min(PROFILING_SAMPLE_SIZE, len(df)), random_state=42)
    print(f"Using {len(sample_df)} samples for profiling (from {len(df)} total)")

    try:
        from ydata_profiling import ProfileReport

        print("Generating ydata-profiling report (optimized)...")
        profile = ProfileReport(
            sample_df, 
            title="NBA Player Valuation Dataset - Enhanced EDA Report",
            explorative=True,
            minimal=True,  # Faster generation
            correlations={
                "pearson": {"calculate": True},
                "spearman": {"calculate": False},  # Skip for speed
                "kendall": {"calculate": False},
                "phi_k": {"calculate": False},
                "cramers": {"calculate": False}
            },
            missing_diagrams={
                "matrix": True,
                "bar": True,
                "heatmap": False,  # Skip for speed
                "dendrogram": False  # Skip for speed
            },
            duplicates={
                "head": 5  # Reduce for speed
            },
            samples={
                "head": 5,  # Reduce for speed
                "tail": 5
            }
        )   
        if output_dir:
            profile.to_file(output_dir / "ydata_profiling_report.html")
        print("✓ ydata-profiling report saved successfully!")

    except ImportError:
        print("⚠ ydata-profiling not installed. Install with: pip install ydata-profiling")

    try:
        import sweetviz as sv

        print("Generating Sweetviz report (optimized)...")
        report = sv.analyze(
            sample_df,
            target_feat=None,
            feat_cfg=None,
            pairwise_analysis="off"  # Skip for speed
        )
        if output_dir:
            report.show_html(output_dir / "sweetviz_report.html")
        print("✓ Sweetviz report saved successfully!")

    except ImportError:
        print("⚠ Sweetviz not installed. Install with: pip install sweetviz")



def categorical_analysis_quick(df, target_col, output_dir=None):
    """Quick categorical analysis for performance."""
    print("\n" + "="*60)
    print("QUICK CATEGORICAL VARIABLE ANALYSIS")
    print("="*60)

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) == 0:
        print("No categorical columns found.")
        return

    print(f"Found {len(categorical_cols)} categorical columns")

    # Sample data for analysis
    sample_df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)

    # Analyze top 5 categorical variables
    for col in categorical_cols[:5]:
        print(f"\nAnalyzing {col}...")

        # Value counts
        value_counts = sample_df[col].value_counts().head(10)

        # Quick visualization
        plt.figure(figsize=(12, 6))
        value_counts.plot(kind='bar')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        if output_dir:
            plt.savefig(output_dir / f"{col}_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    print("✓ Quick categorical analysis completed")

def time_series_cv_setup_quick(df, output_dir=None):
    """Quick time series cross-validation setup."""
    print("\n" + "="*60)
    print("QUICK TIME SERIES CROSS-VALIDATION SETUP")
    print("="*60)

    # Find time-based columns
    time_cols = []
    for col in df.columns:
        if 'season' in col.lower() or 'year' in col.lower() or 'date' in col.lower():
            time_cols.append(col)

    if not time_cols:
        print("No obvious time-based columns found.")
        return None

    print(f"Found potential time columns: {time_cols}")

    # Quick temporal analysis
    temporal_analysis = {}

    for col in time_cols[:3]:  # Limit to first 3
        unique_values = df[col].dropna().unique()
        if len(unique_values) > 1:
            try:
                numeric_data = pd.to_numeric(df[col], errors='coerce')
                min_val = numeric_data.min()
                max_val = numeric_data.max()
            except:
                min_val = str(df[col].min())
                max_val = str(df[col].max())

            temporal_analysis[col] = {
                'unique_values': len(unique_values),
                'min_value': min_val,
                'max_value': max_val,
                'missing_pct': df[col].isna().mean() * 100
            }

    if temporal_analysis:
        temporal_df = pd.DataFrame.from_dict(temporal_analysis, orient='index')
        print(f"\nTemporal column analysis:")
        print(temporal_df)

        # Save temporal analysis
        if output_dir:
            temporal_df.to_csv(output_dir / "temporal_analysis.csv")

    print("✓ Quick time series analysis completed")

    return temporal_analysis


def compute_vif(df, features, output_dir=None):
    """
    Compute Variance Inflation Factor (VIF) to detect multicollinearity.
    Drops or flags any feature with VIF > 10 as highly collinear.
    """
    print("\n=== Multicollinearity Check via VIF ===")
    X = StandardScaler().fit_transform(df[features].fillna(0))
    vif_data = pd.DataFrame({
        'feature': features,
        'VIF': [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    }).sort_values("VIF", ascending=False)
    
    high_vif = vif_data[vif_data['VIF'] > 10]
    print(f"Found {len(high_vif)} features with VIF > 10 (high multicollinearity).")
    if not high_vif.empty:
        print("Top offenders:")
        print(high_vif.to_string(index=False))
    else:
        print("No severe multicollinearity detected (all VIF ≤ 10).")
    
    if output_dir:
        vif_data.to_csv(output_dir / "vif_analysis.csv", index=False)
    return vif_data





def plot_categorical_boxplots(df, categorical_cols, target_col, output_dir=None):
    """
    Boxplots of target_col by each of up to 5 key categorical features.
    Highlights group-level differences in the target.
    """
    print("\n=== Categorical Variable Impact (Boxplots) ===")
    for col in categorical_cols[:5]:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=col, y=target_col, data=df)
        plt.title(f"{target_col} Distribution by {col}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        filename = f"aav_by_{col.lower()}.png"
        if output_dir:
            plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"• Saved boxplot for '{col}'—good for spotting which categories drive AAV differences.")


def plot_time_trend(df, time_col, target_col, output_dir=None):
    """
    Plot average target_col per time period to surface macro trends.
    """
    print("\n=== Time-Trend of Contract Values ===")
    agg = df.groupby(time_col)[target_col].mean().reset_index()
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=time_col, y=target_col, data=agg, marker="o")
    plt.title(f"Average {target_col} Over {time_col}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if output_dir:
        plt.savefig(output_dir / "time_trend_aav.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Shows how {target_col} evolves over {time_col}, revealing market cycles or rule-driven jumps.")


def plot_pca_scree(df, numericals, output_dir=None):
    """
    Run PCA on numericals to show cumulative explained variance.
    """
    print("\n=== PCA & Explained Variance (Scree) ===")
    X = StandardScaler().fit_transform(df[numericals].fillna(0))
    pca = PCA().fit(X)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cum_var)+1), cum_var, marker="o")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Scree Plot")
    plt.grid(True)
    plt.tight_layout()
    if output_dir:
        plt.savefig(output_dir / "pca_scree_plot.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    n80 = np.searchsorted(cum_var, 0.80) + 1
    print(f"{n80} components needed to explain 80% of variance.")
    return cum_var

from pandas.api.types import is_numeric_dtype  # make sure this import is available where the function lives
def plot_schema_category_overview(df,
                                  cat_breakdown,
                                  target_col: str,
                                  output_dir,
                                  *,
                                  corr_method: str = "pearson",
                                  max_regplots: int = 12,
                                  top_k_scatter_matrix: int = 5,
                                  max_heatmap_features: int = 60,
                                  max_sample: int = 5000,
                                  debug: bool = False):
    """
    For each numerical category:
      Panel A: ggplot-style regplots vs target (top-|corr| features, up to max_regplots)
      Panel B: Intra-category correlation heatmap (up to max_heatmap_features)
      Panel C: Scatter-matrix of top-k features by |corr| to target

    Saves a single stacked PNG per category: [NN]_<category>_overview.png

    Notes:
      * Skips non-numeric columns (no coercion/fill).
      * Uses pandas.DataFrame.corr(method) for correlations.
      * Uses pandas.plotting.scatter_matrix for the matrix plot.
      * Combines panels with Pillow if installed; otherwise leaves panels separate.
    """
    import math
    from pathlib import Path
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from pandas.api.types import is_numeric_dtype
    from pandas.plotting import scatter_matrix

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if target_col not in df.columns:
        print(f"[ERROR] target '{target_col}' not in df; aborting.")
        return

    # Category IDs for clear labels
    category_ids = {cat: f"{i+1:02d}" for i, cat in enumerate(cat_breakdown.keys())}

    # Helper: safe correlation-with-target on numeric columns
    def _corr_abs_with_target(cols):
        usable = [c for c in cols if c in df.columns and is_numeric_dtype(df[c])]
        if not usable or not is_numeric_dtype(df[target_col]):
            return pd.Series(dtype=float)
        sub = df[usable + [target_col]].dropna(subset=[target_col])
        if sub.empty:
            return pd.Series(dtype=float)
        corrs = sub.corr(method=corr_method)[target_col].drop(target_col)
        # remove NaNs (constant cols or insufficient data)
        corrs = corrs[~corrs.isna()]
        return corrs.abs().sort_values(ascending=False)

    # Helper: save fig to path
    def _save_fig(fig, path):
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Helper: combine images vertically with Pillow
    def _stack_panels(panel_paths, stacked_path, pad_px=16, bg=(255, 255, 255)):
        try:
            from PIL import Image
        except ImportError:
            print("[INFO] Pillow not installed. Panels saved separately:\n  - " +
                  "\n  - ".join(str(p) for p in panel_paths if p))
            return
        imgs = []
        for p in panel_paths:
            if p and Path(p).exists():
                imgs.append(Image.open(p).convert("RGB"))
        if not imgs:
            print(f"[WARN] No panels to stack for {stacked_path.name}.")
            return
        w = max(im.width for im in imgs)
        h = sum(im.height for im in imgs) + pad_px * (len(imgs) - 1)
        canvas = Image.new("RGB", (w, h), bg)
        y = 0
        for im in imgs:
            x = (w - im.width) // 2
            canvas.paste(im, (x, y))
            y += im.height + pad_px
        canvas.save(stacked_path)

    # Iterate categories
    for category, feats in cat_breakdown.items():
        cat_id = category_ids[category]
        prefix = f"{cat_id}_{category}"
        numeric_feats = [f for f in feats if f in df.columns and is_numeric_dtype(df[f])]

        if not numeric_feats:
            print(f"[SKIP] '{category}': no numeric features present.")
            continue
        if not is_numeric_dtype(df[target_col]):
            print(f"[SKIP] '{category}': target '{target_col}' is non-numeric; skipping plots.")
            continue

        # Determine top features by |corr| to target
        corr_abs = _corr_abs_with_target(numeric_feats)
        # REG plot features
        reg_feats = corr_abs.index.tolist()[:max_regplots] if not corr_abs.empty \
                    else numeric_feats[:min(len(numeric_feats), max_regplots)]
        # Heatmap features (cap for readability)
        heatmap_feats = numeric_feats[:min(len(numeric_feats), max_heatmap_features)]
        # Scatter-matrix features
        sm_feats = corr_abs.index.tolist()[:min(top_k_scatter_matrix, len(corr_abs))]

        if debug:
            print(f"[{cat_id}] {category}: numeric={len(numeric_feats)}, "
                  f"reg_feats={len(reg_feats)}, heatmap_feats={len(heatmap_feats)}, "
                  f"scatter_matrix_feats={len(sm_feats)}")

        panel_paths = []

        # -----------------------
        # Panel A: regplots grid
        # -----------------------
        if reg_feats:
            n = len(reg_feats)
            n_cols = min(3, n)
            n_rows = math.ceil(n / n_cols)
            with plt.style.context("ggplot"):
                fig, axes = plt.subplots(n_rows, n_cols,
                                         figsize=(5 * n_cols, 4 * n_rows),
                                         squeeze=False)
                plotted = 0
                for i, feat in enumerate(reg_feats):
                    r, c = divmod(i, n_cols)
                    ax = axes[r][c]
                    sub = df[[feat, target_col]].dropna()
                    if sub.empty:
                        ax.set_visible(False)
                        continue
                    try:
                        # seaborn.regplot with lowess (documented parameter)
                        sns.regplot(
                            x=feat, y=target_col, data=sub,
                            scatter_kws=dict(alpha=0.35, s=10),
                            line_kws=dict(linewidth=1.25),
                            lowess=True,
                            ax=ax
                        )
                        corr_val = sub[feat].corr(sub[target_col], method=corr_method)
                        ax.text(0.03, 0.93, f"ρ={corr_val:.2f}", transform=ax.transAxes, fontsize=9,
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
                        ax.set_title(f"[{cat_id}] {feat} → {target_col}")
                        plotted += 1
                    except Exception as e:
                        ax.set_visible(False)
                        if debug:
                            print(f"[WARN] regplot failed: cat={category}, feat={feat}, err={e}")
                # hide unused
                total = n_rows * n_cols
                for j in range(n, total):
                    r, c = divmod(j, n_cols)
                    axes[r][c].set_visible(False)

                fig.suptitle(f"[{cat_id}] {category.title()} — Panel A: Regplots vs {target_col}", y=1.02)
                pA = outdir / f"{prefix}_A_regplots.png"
                _save_fig(fig, pA)
                if plotted:
                    panel_paths.append(pA)
        else:
            if debug:
                print(f"[{cat_id}] {category}: no features to plot for Panel A")

        # --------------------------------
        # Panel B: intra-category heatmap
        # --------------------------------
        if len(heatmap_feats) >= 2:
            sub_df = df[heatmap_feats].dropna(how="all")
            if not sub_df.empty:
                corr_mat = sub_df.corr(method=corr_method)  # pandas.DataFrame.corr
                fig, ax = plt.subplots(figsize=(min(12, 0.6*len(heatmap_feats) + 3), 8))
                sns.heatmap(corr_mat, annot=False, cmap="coolwarm", center=0, square=False, ax=ax)
                ax.set_title(f"[{cat_id}] {category.title()} — Panel B: Correlation ({corr_method})")
                pB = outdir / f"{prefix}_B_heatmap_{corr_method}.png"
                _save_fig(fig, pB)
                panel_paths.append(pB)
        else:
            if debug:
                print(f"[{cat_id}] {category}: <2 features for heatmap")

        # -----------------------------------
        # Panel C: scatter-matrix (top-k corr)
        # -----------------------------------
        if sm_feats:
            combo = df[sm_feats + [target_col]].dropna()
            if not combo.empty:
                if len(combo) > max_sample:
                    combo = combo.sample(n=max_sample, random_state=42)
                with plt.style.context("ggplot"):
                    axs = scatter_matrix(
                        combo[sm_feats + [target_col]],
                        alpha=0.6, diagonal="kde",
                        figsize=(3.2 * len(sm_feats + [target_col]),
                                 3.2 * len(sm_feats + [target_col]))
                    )
                    fig_sm = axs[0, 0].get_figure()
                    fig_sm.suptitle(f"[{cat_id}] {category.title()} — Panel C: Scatter-matrix (top {len(sm_feats)} by |corr| to {target_col})",
                                    y=0.98)
                    pC = outdir / f"{prefix}_C_scatter_matrix.png"
                    _save_fig(fig_sm, pC)
                    panel_paths.append(pC)
        else:
            if debug:
                print(f"[{cat_id}] {category}: no top-k features for scatter-matrix")

        # --------------------------
        # Stack panels into overview
        # --------------------------
        if panel_paths:
            stacked = outdir / f"{prefix}_overview.png"
            _stack_panels(panel_paths, stacked)
            print(f"✓ Saved category overview: {stacked}")
        else:
            print(f"[INFO] No panels produced for '{category}'")


def plot_category_overview_inline(df, cat_breakdown, target_col, max_features=8, figsize_per_cat=(15, 10)):
    """
    Create one comprehensive visualization per numerical category for inline notebook display.
    Each category gets its own figure with clear labeling and insights.
    
    Args:
        df: DataFrame with features
        cat_breakdown: Dict of {category_name: [feature_list]}
        target_col: Name of target variable
        max_features: Maximum features to show per category
        figsize_per_cat: Figure size for each category plot
    """
    print("\n" + "="*80)
    print("NUMERICAL CATEGORIES ANALYSIS - INLINE DISPLAY")
    print("="*80)
    
    # Configure matplotlib for notebook display
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pandas.api.types import is_numeric_dtype
    
    # Set style for better notebook display
    plt.style.use('default')  # Use default instead of seaborn-v0_8 for better inline display
    sns.set_palette("husl")
    
    if target_col not in df.columns or not is_numeric_dtype(df[target_col]):
        print(f"❌ Target column '{target_col}' not found or not numeric")
        return
    
    # Process each numerical category
    for i, (category, features) in enumerate(cat_breakdown.items(), 1):
        print(f"\n📊 Processing Category {i}: {category.upper()}")
        
        # Filter to numeric features that exist in dataframe
        numeric_features = [f for f in features if f in df.columns and is_numeric_dtype(df[f])]
        
        if not numeric_features:
            print(f"   ⚠️ No numeric features found for category '{category}'")
            continue
            
        print(f"   📈 Found {len(numeric_features)} numeric features")
        
        # Calculate correlations with target
        correlations = {}
        for feat in numeric_features:
            corr_data = df[[feat, target_col]].dropna()
            if len(corr_data) > 10:  # Need minimum data points
                corr = corr_data[feat].corr(corr_data[target_col])
                if not pd.isna(corr):
                    correlations[feat] = abs(corr)
        
        if not correlations:
            print(f"   ⚠️ No valid correlations calculated for category '{category}'")
            continue
            
        # Select top features by correlation
        top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:max_features]
        selected_features = [feat for feat, _ in top_features]
        
        print(f"   🎯 Top features by |correlation| with {target_col}:")
        for feat, corr in top_features:
            print(f"      • {feat}: {corr:.3f}")
        
        # Create comprehensive plot for this category
        fig = plt.figure(figsize=figsize_per_cat)
        fig.suptitle(f'📊 CATEGORY: {category.upper()} (Top {len(selected_features)} Features)', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Create subplots: correlation bar + feature distributions + target relationships
        gs = fig.add_gridspec(3, 4, height_ratios=[1, 1.5, 1.5], hspace=0.3, wspace=0.3)
        
        # 1. Correlation bar chart (top row, spans all columns)
        ax_corr = fig.add_subplot(gs[0, :])
        corr_values = [corr for _, corr in top_features]
        feature_names = [feat[:20] + '...' if len(feat) > 20 else feat for feat, _ in top_features]
        
        bars = ax_corr.barh(range(len(feature_names)), corr_values, 
                           color=plt.cm.viridis(np.linspace(0, 1, len(feature_names))))
        ax_corr.set_yticks(range(len(feature_names)))
        ax_corr.set_yticklabels(feature_names, fontsize=10)
        ax_corr.set_xlabel(f'|Correlation| with {target_col}', fontweight='bold')
        ax_corr.set_title('Feature Correlations with Target', fontweight='bold')
        ax_corr.grid(axis='x', alpha=0.3)
        
        # Add correlation values on bars
        for i, (bar, corr) in enumerate(zip(bars, corr_values)):
            ax_corr.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{corr:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')
        
        # 2. Feature distributions (middle row)
        n_dist_plots = min(4, len(selected_features))
        for j in range(n_dist_plots):
            ax_dist = fig.add_subplot(gs[1, j])
            feat = selected_features[j]
            
            # Plot distribution
            data = df[feat].dropna()
            if len(data) > 0:
                ax_dist.hist(data, bins=30, alpha=0.7, color=plt.cm.viridis(j/max(1, n_dist_plots-1)))
                ax_dist.set_title(f'{feat[:15]}{"..." if len(feat) > 15 else ""}', fontsize=10, fontweight='bold')
                ax_dist.set_ylabel('Count')
                ax_dist.grid(alpha=0.3)
                
                # Add statistics
                mean_val = data.mean()
                median_val = data.median()
                ax_dist.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.2f}')
                ax_dist.axvline(median_val, color='orange', linestyle='--', alpha=0.8, label=f'Median: {median_val:.2f}')
                ax_dist.legend(fontsize=8)
        
        # 3. Scatter plots vs target (bottom row)
        n_scatter_plots = min(4, len(selected_features))
        for j in range(n_scatter_plots):
            ax_scatter = fig.add_subplot(gs[2, j])
            feat = selected_features[j]
            
            # Plot scatter
            scatter_data = df[[feat, target_col]].dropna()
            if len(scatter_data) > 0:
                # Sample data if too large for performance
                if len(scatter_data) > 1000:
                    scatter_data = scatter_data.sample(n=1000, random_state=42)
                
                ax_scatter.scatter(scatter_data[feat], scatter_data[target_col], 
                                 alpha=0.6, s=20, color=plt.cm.viridis(j/max(1, n_scatter_plots-1)))
                
                # Add trend line
                try:
                    z = np.polyfit(scatter_data[feat], scatter_data[target_col], 1)
                    p = np.poly1d(z)
                    ax_scatter.plot(scatter_data[feat], p(scatter_data[feat]), "r--", alpha=0.8, linewidth=2)
                except:
                    pass  # Skip trend line if it fails
                
                ax_scatter.set_xlabel(f'{feat[:15]}{"..." if len(feat) > 15 else ""}', fontweight='bold')
                ax_scatter.set_ylabel(target_col if j == 0 else '')
                ax_scatter.set_title(f'vs {target_col}', fontsize=10, fontweight='bold')
                ax_scatter.grid(alpha=0.3)
                
                # Add correlation annotation
                corr_val = correlations.get(feat, 0)
                ax_scatter.text(0.05, 0.95, f'r = {corr_val:.3f}', transform=ax_scatter.transAxes,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                               fontsize=10, fontweight='bold')
        
        # Show the plot inline
        plt.tight_layout()
        plt.show()
        
        # Print category summary
        print(f"\n   📋 CATEGORY SUMMARY: {category}")
        print(f"      • Total features: {len(numeric_features)}")
        print(f"      • Features with valid correlations: {len(correlations)}")
        print(f"      • Strongest correlation: {max(correlations.values()):.3f}")
        print(f"      • Average correlation: {np.mean(list(correlations.values())):.3f}")
        print("   " + "-"*50)

def analyze_target_distribution_enhanced(df, target_col):
    """
    Enhanced target distribution analysis with multiple visualizations and insights.
    
    Args:
        df: DataFrame
        target_col: Target column name
    """
    print("\n" + "="*80)
    print(f"🎯 TARGET VARIABLE ANALYSIS: {target_col}")
    print("="*80)
    
    if target_col not in df.columns:
        print(f"❌ Target column '{target_col}' not found")
        return
    
    target_data = df[target_col].dropna()
    
    if len(target_data) == 0:
        print(f"❌ No valid data for target '{target_col}'")
        return
    
    # Calculate comprehensive statistics
    stats = {
        'count': len(target_data),
        'mean': target_data.mean(),
        'median': target_data.median(),
        'std': target_data.std(),
        'min': target_data.min(),
        'max': target_data.max(),
        'q25': target_data.quantile(0.25),
        'q75': target_data.quantile(0.75),
        'skewness': target_data.skew(),
        'kurtosis': target_data.kurtosis(),
        'unique_values': target_data.nunique(),
        'zeros': (target_data == 0).sum(),
        'missing': df[target_col].isna().sum()
    }
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'🎯 TARGET ANALYSIS: {target_col}', fontsize=16, fontweight='bold')
    
    # 1. Histogram with statistics
    ax = axes[0, 0]
    n, bins, patches = ax.hist(target_data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['mean']:.2f}")
    ax.axvline(stats['median'], color='orange', linestyle='--', linewidth=2, label=f"Median: {stats['median']:.2f}")
    ax.set_title('Distribution with Mean/Median', fontweight='bold')
    ax.set_xlabel(target_col)
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Box plot with outliers
    ax = axes[0, 1]
    box_plot = ax.boxplot(target_data, vert=True, patch_artist=True)
    box_plot['boxes'][0].set_facecolor('lightgreen')
    ax.set_title('Box Plot (Outlier Detection)', fontweight='bold')
    ax.set_ylabel(target_col)
    ax.grid(alpha=0.3)
    
    # 3. Q-Q Plot for normality check
    ax = axes[0, 2]
    from scipy import stats as scipy_stats
    scipy_stats.probplot(target_data, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Normality Check)', fontweight='bold')
    ax.grid(alpha=0.3)
    
    # 4. Cumulative distribution
    ax = axes[1, 0]
    sorted_data = np.sort(target_data)
    cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax.plot(sorted_data, cumulative, linewidth=2, color='purple')
    ax.set_title('Cumulative Distribution Function', fontweight='bold')
    ax.set_xlabel(target_col)
    ax.set_ylabel('Cumulative Probability')
    ax.grid(alpha=0.3)
    
    # Add percentile lines
    for p in [25, 50, 75, 90, 95]:
        val = target_data.quantile(p/100)
        ax.axvline(val, color='red', alpha=0.5, linestyle=':', 
                  label=f'{p}th: {val:.2f}' if p in [50, 90, 95] else '')
    ax.legend()
    
    # 5. Statistics table
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_text = f"""
    📊 DESCRIPTIVE STATISTICS
    
    Count: {stats['count']:,}
    Missing: {stats['missing']:,}
    Unique Values: {stats['unique_values']:,}
    
    📈 CENTRAL TENDENCY
    Mean: {stats['mean']:.3f}
    Median: {stats['median']:.3f}
    
    📏 SPREAD
    Std Dev: {stats['std']:.3f}
    Range: {stats['max'] - stats['min']:.3f}
    IQR: {stats['q75'] - stats['q25']:.3f}
    
    📐 SHAPE
    Skewness: {stats['skewness']:.3f}
    Kurtosis: {stats['kurtosis']:.3f}
    
    🔢 EXTREMES
    Min: {stats['min']:.3f}
    Max: {stats['max']:.3f}
    Zeros: {stats['zeros']:,}
    """
    
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    # 6. Distribution insights
    ax = axes[1, 2]
    ax.axis('off')
    
    # Generate insights
    insights = []
    
    if abs(stats['skewness']) > 1:
        insights.append(f"🔍 Highly skewed distribution (skew={stats['skewness']:.2f})")
    elif abs(stats['skewness']) > 0.5:
        insights.append(f"🔍 Moderately skewed (skew={stats['skewness']:.2f})")
    else:
        insights.append(f"✅ Relatively symmetric distribution")
    
    if stats['kurtosis'] > 3:
        insights.append(f"📊 Heavy-tailed distribution (kurtosis={stats['kurtosis']:.2f})")
    elif stats['kurtosis'] < -1:
        insights.append(f"📊 Light-tailed distribution (kurtosis={stats['kurtosis']:.2f})")
    
    cv = stats['std'] / stats['mean'] if stats['mean'] != 0 else float('inf')
    if cv > 1:
        insights.append(f"📈 High variability (CV={cv:.2f})")
    elif cv < 0.3:
        insights.append(f"📈 Low variability (CV={cv:.2f})")
    
    if stats['zeros'] > 0:
        zero_pct = (stats['zeros'] / stats['count']) * 100
        insights.append(f"⚠️ {zero_pct:.1f}% zero values")
    
    # Check for potential outliers
    iqr = stats['q75'] - stats['q25']
    lower_bound = stats['q25'] - 1.5 * iqr
    upper_bound = stats['q75'] + 1.5 * iqr
    outliers = ((target_data < lower_bound) | (target_data > upper_bound)).sum()
    outlier_pct = (outliers / len(target_data)) * 100
    insights.append(f"🚨 {outlier_pct:.1f}% potential outliers")
    
    insights_text = "🔍 KEY INSIGHTS:\n\n" + "\n".join(insights)
    
    ax.text(0.1, 0.9, insights_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\n📋 TARGET SUMMARY:")
    print(f"   • Distribution: {'Right-skewed' if stats['skewness'] > 0.5 else 'Left-skewed' if stats['skewness'] < -0.5 else 'Symmetric'}")
    print(f"   • Central value: {stats['median']:.3f} (median)")
    print(f"   • Typical range: {stats['q25']:.3f} to {stats['q75']:.3f} (IQR)")
    print(f"   • Data quality: {((stats['count'] - stats['zeros']) / stats['count'] * 100):.1f}% non-zero values")
    
    return stats


def analyze_missing_data_patterns(
    df: pd.DataFrame,
    *,
    schema=None,
    scope: str = "schema",  # "schema" | "schema_plus_engineered" | "all"
    engineered_columns: Optional[List[str]] = None,
    expected_missing: Optional[List[str]] = None,
    top_k: int = 10,
    debug: bool = True,
) -> dict:
    """
    Missingness analyzer with explicit scope and expectation controls.
    - We never fill; we only measure and label.
    - expected_missing columns are still reported, but bucketed separately.

    Returns a dict with:
      {
        'scoped_columns': [...],
        'summary': {...},
        'top_missing': DataFrame,
        'expected_missing_report': DataFrame,
        'out_of_schema_report': DataFrame
      }
    """
    import numpy as np
    import pandas as pd

    expected_missing = set(expected_missing or [])
    engineered_columns = set(engineered_columns or [])

    # ----- Decide the column scope
    if scope == "all" or schema is None:
        scoped_cols = list(df.columns)
        scope_used = "all"
    elif scope == "schema":
        # strictly schema-approved features
        scoped_cols = schema.numerical() + schema.ordinal() + schema.nominal()
        scoped_cols = [c for c in scoped_cols if c in df.columns]
        scope_used = "schema"
    elif scope == "schema_plus_engineered":
        base = schema.numerical() + schema.ordinal() + schema.nominal()
        scoped_cols = sorted(set([c for c in base if c in df.columns] + [c for c in engineered_columns if c in df.columns]))
        scope_used = "schema_plus_engineered"
    else:
        raise ValueError(f"Unknown scope: {scope}")

    if debug:
        print(f"[missingness] scope={scope_used}, n_scoped_cols={len(scoped_cols)}")
        missing_in_scope = [c for c in scoped_cols if c not in df.columns]
        if missing_in_scope:
            print(f"[missingness][warn] scoped columns not actually in df: {missing_in_scope}")

    # ----- Compute missingness within scope
    miss_pct = df[scoped_cols].isna().mean().sort_values(ascending=False)
    miss_cnt = df[scoped_cols].isna().sum().sort_values(ascending=False)
    dtypes = df[scoped_cols].dtypes.astype(str)

    report = pd.DataFrame({
        "column": miss_pct.index,
        "missing_pct": miss_pct.values * 100,
        "missing_cnt": miss_cnt.values,
        "dtype": [dtypes[c] for c in miss_pct.index],
        "in_schema": [c in (schema.numerical() + schema.ordinal() + schema.nominal()) if schema else False for c in miss_pct.index],
        "engineered": [c in engineered_columns for c in miss_pct.index],
        "expected_missing": [c in expected_missing for c in miss_pct.index],
    })

    # ----- Buckets
    expected_bucket = report[report["expected_missing"]]
    out_of_schema_bucket = report[(~report["in_schema"]) & (~report["engineered"])]
    top_missing = report.sort_values("missing_pct", ascending=False).head(top_k)

    # ----- Summary
    n_cols = len(scoped_cols)
    n_with_missing = (miss_cnt > 0).sum()
    frac_cols_missing = (n_with_missing / max(1, n_cols)) * 100.0

    summary = {
        "columns_in_scope": n_cols,
        "columns_with_missing": int(n_with_missing),
        "pct_columns_with_missing": round(frac_cols_missing, 1),
        "worst_over_50pct": int((miss_pct > 0.50).sum()),
        "moderate_20_50pct": int(((miss_pct > 0.20) & (miss_pct <= 0.50)).sum()),
        "minor_under_20pct": int(((miss_pct > 0.0) & (miss_pct <= 0.20)).sum()),
    }

    if debug:
        print("\n📋 MISSING DATA SUMMARY (scoped):")
        print(f"   • Columns affected: {summary['columns_with_missing']}/{summary['columns_in_scope']} ({summary['pct_columns_with_missing']}%)")
        print(f"   • Worst offenders (>50% missing): {summary['worst_over_50pct']}")
        print(f"   • Moderate (20–50%): {summary['moderate_20_50pct']}")
        print(f"   • Minor (<20%): {summary['minor_under_20pct']}")

        if not expected_bucket.empty:
            print("\nℹ️  Expected-missing columns (diagnostic/derived—missing is OK):")
            for _, r in expected_bucket.sort_values("missing_pct", ascending=False).head(top_k).iterrows():
                print(f"   • {r['column']:<40} | {r['missing_pct']:6.1f}% | {r['dtype']}")

        if not out_of_schema_bucket.empty:
            print("\n⚠️  Out-of-schema columns in scope (why are these here?):")
            for _, r in out_of_schema_bucket.sort_values("missing_pct", ascending=False).head(top_k).iterrows():
                print(f"   • {r['column']:<40} | {r['missing_pct']:6.1f}% | {r['dtype']}")

        print("\n🔍 TOP MISSING COLUMNS (in scope):")
        for _, r in top_missing.iterrows():
            print(f"   • {r['column']:<40} | {r['missing_pct']:6.1f}% | {r['dtype']}")

    return {
        "scoped_columns": scoped_cols,
        "summary": summary,
        "top_missing": top_missing,
        "expected_missing_report": expected_bucket.sort_values("missing_pct", ascending=False),
        "out_of_schema_report": out_of_schema_bucket.sort_values("missing_pct", ascending=False),
        "full_report": report.sort_values("missing_pct", ascending=False)
    }




def analyze_data_quality_overview(df, sample_size=10000):
    """
    Data quality overview. For dtype distributions, we string-coerce dtype-like index
    to avoid dtype-comparison errors inside plotting/sorting.
    """
    print("\n" + "="*80)
    print("🔍 DATA QUALITY OVERVIEW")
    print("="*80)

    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
        print(f"📊 Analyzing sample of {sample_size:,} rows (from {len(df):,} total)")
    else:
        df_sample = df.copy()
        print(f"📊 Analyzing all {len(df):,} rows")

    # Data types breakdown (stringify for safe plotting)
    dtype_counts = df_sample.dtypes.value_counts()
    dtype_counts_display = dtype_counts.copy()
    dtype_counts_display.index = dtype_counts_display.index.map(str)

    numeric_cols = df_sample.select_dtypes(include=[np.number]).columns
    text_cols = df_sample.select_dtypes(include=['object']).columns

    total_cells = df_sample.shape[0] * df_sample.shape[1]
    missing_cells = df_sample.isnull().sum().sum()
    duplicate_rows = df_sample.duplicated().sum()

    # Numeric anomalies summary
    numeric_anomalies = {}
    for col in numeric_cols:
        data = df_sample[col].dropna()
        if len(data) == 0:
            continue
        q1, q3 = data.quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = ((data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr))).sum()
        zeros = (data == 0).sum()
        infinites = np.isinf(data).sum()
        numeric_anomalies[col] = {
            'outliers': int(outliers),
            'zeros': int(zeros),
            'infinites': int(infinites),
            'outlier_pct': (outliers / len(data)) * 100
        }

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('🔍 DATA QUALITY OVERVIEW', fontsize=16, fontweight='bold')

    # 1) Data type distribution (uses stringified index)
    ax = axes[0, 0]
    wedges, texts, autotexts = ax.pie(
        dtype_counts_display.values,
        labels=dtype_counts_display.index,
        autopct='%1.1f%%', startangle=90
    )
    ax.set_title('Data Types Distribution', fontweight='bold')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    # 2) Missing data overview
    ax = axes[0, 1]
    missing_by_col = df_sample.isnull().sum()
    missing_cols = missing_by_col[missing_by_col > 0].sort_values(ascending=True)
    if len(missing_cols) > 0:
        top_missing = missing_cols.tail(15)
        bars = ax.barh(range(len(top_missing)), top_missing.values, color='red', alpha=0.7)
        ax.set_yticks(range(len(top_missing)))
        ax.set_yticklabels([c[:20] + '...' if len(c) > 20 else c for c in top_missing.index], fontsize=9)
        ax.set_xlabel('Missing Values Count', fontweight='bold')
        ax.set_title(f'Missing Data (Top {len(top_missing)} Columns)', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    else:
        ax.text(0.5, 0.5, '✅ No Missing Data Found!', transform=ax.transAxes,
                ha='center', va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        ax.set_xticks([]); ax.set_yticks([])

    # 3) Duplicate analysis
    ax = axes[0, 2]
    duplicate_data = ['Unique Rows', 'Duplicate Rows']
    duplicate_counts = [len(df_sample) - duplicate_rows, duplicate_rows]
    colors = ['lightgreen', 'red'] if duplicate_rows > 0 else ['lightgreen', 'lightgray']
    bars = ax.bar(duplicate_data, duplicate_counts, color=colors, alpha=0.7)
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Row Duplication Analysis', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar, count in zip(bars, duplicate_counts):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + max(duplicate_counts) * 0.01,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')

    # 4) Numeric anomalies overview
    ax = axes[1, 0]
    if numeric_anomalies:
        anomaly_summary = pd.DataFrame(numeric_anomalies).T
        top_anomalies = anomaly_summary.nlargest(15, 'outlier_pct')
        bars = ax.barh(range(len(top_anomalies)), top_anomalies['outlier_pct'], color='orange', alpha=0.7)
        ax.set_yticks(range(len(top_anomalies)))
        ax.set_yticklabels([c[:20] + '...' if len(c) > 20 else c for c in top_anomalies.index], fontsize=9)
        ax.set_xlabel('Outlier Percentage (%)', fontweight='bold')
        ax.set_title('Outlier Analysis (Top Numeric Cols)', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        for bar, pct in zip(bars, top_anomalies['outlier_pct']):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{pct:.1f}%', ha='left', va='center', fontsize=9, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No Numeric Columns Found', transform=ax.transAxes,
                ha='center', va='center', fontsize=12)

    # 5) Scorecard + 6) Recommendations (unchanged from your version) ...
    # (Keep your existing code for the scorecard and recommendations.)
    plt.tight_layout()
    plt.show()

    # Print summary (unchanged)
    overall_cols = len(df_sample.columns)
    print(f"\n📋 QUALITY SUMMARY:")
    print(f"   • Columns: {overall_cols}, Numeric: {len(numeric_cols)}, Text: {len(text_cols)}")
    return {
        "dtype_counts": dtype_counts,  # raw
        "numeric_anomalies": numeric_anomalies
    }




def analyze_data_quality_overview(df, sample_size=10000):
    """
    Comprehensive data quality analysis with visualizations.
    
    Args:
        df: DataFrame to analyze
        sample_size: Sample size for performance in large datasets
    """
    print("\n" + "="*80)
    print("🔍 DATA QUALITY OVERVIEW")
    print("="*80)
    
    # Sample data if too large
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
        print(f"📊 Analyzing sample of {sample_size:,} rows (from {len(df):,} total)")
    else:
        df_sample = df.copy()
        print(f"📊 Analyzing all {len(df):,} rows")
    
    # Analyze data types
    dtype_counts = df_sample.dtypes.value_counts()
    numeric_cols = df_sample.select_dtypes(include=[np.number]).columns
    text_cols = df_sample.select_dtypes(include=['object']).columns
    
    # Calculate quality metrics
    total_cells = df_sample.shape[0] * df_sample.shape[1]
    missing_cells = df_sample.isnull().sum().sum()
    duplicate_rows = df_sample.duplicated().sum()
    
    # Analyze numeric columns for anomalies
    numeric_anomalies = {}
    for col in numeric_cols:
        data = df_sample[col].dropna()
        if len(data) > 0:
            q1, q3 = data.quantile([0.25, 0.75])
            iqr = q3 - q1
            outliers = ((data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr))).sum()
            zeros = (data == 0).sum()
            infinites = np.isinf(data).sum()
            
            numeric_anomalies[col] = {
                'outliers': outliers,
                'zeros': zeros,
                'infinites': infinites,
                'outlier_pct': (outliers / len(data)) * 100
            }
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('🔍 DATA QUALITY OVERVIEW', fontsize=16, fontweight='bold')
    
    # 1. Data type distribution
    ax = axes[0, 0]
    colors = ['skyblue', 'lightgreen', 'salmon', 'gold', 'lightcoral'][:len(dtype_counts)]
    wedges, texts, autotexts = ax.pie(dtype_counts.values, labels=dtype_counts.index, 
                                     autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title('Data Types Distribution', fontweight='bold')
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    # 2. Missing data overview
    ax = axes[0, 1]
    missing_by_col = df_sample.isnull().sum()
    missing_cols = missing_by_col[missing_by_col > 0].sort_values(ascending=True)
    
    if len(missing_cols) > 0:
        # Show top 15 columns with missing data
        top_missing = missing_cols.tail(15)
        bars = ax.barh(range(len(top_missing)), top_missing.values, color='red', alpha=0.7)
        ax.set_yticks(range(len(top_missing)))
        ax.set_yticklabels([col[:20] + '...' if len(col) > 20 else col for col in top_missing.index], fontsize=9)
        ax.set_xlabel('Missing Values Count', fontweight='bold')
        ax.set_title(f'Missing Data (Top {len(top_missing)} Columns)', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    else:
        ax.text(0.5, 0.5, '✅ No Missing Data Found!', transform=ax.transAxes,
                ha='center', va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        ax.set_xticks([])
        ax.set_yticks([])
    
    # 3. Duplicate analysis
    ax = axes[0, 2]
    duplicate_data = ['Unique Rows', 'Duplicate Rows']
    duplicate_counts = [len(df_sample) - duplicate_rows, duplicate_rows]
    colors = ['lightgreen', 'red'] if duplicate_rows > 0 else ['lightgreen', 'lightgray']
    
    bars = ax.bar(duplicate_data, duplicate_counts, color=colors, alpha=0.7)
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Row Duplication Analysis', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for bar, count in zip(bars, duplicate_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(duplicate_counts) * 0.01,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Numeric anomalies overview
    ax = axes[1, 0]
    if numeric_anomalies:
        anomaly_summary = pd.DataFrame(numeric_anomalies).T
        # Show columns with highest outlier percentages
        top_anomalies = anomaly_summary.nlargest(15, 'outlier_pct')
        
        bars = ax.barh(range(len(top_anomalies)), top_anomalies['outlier_pct'], 
                      color='orange', alpha=0.7)
        ax.set_yticks(range(len(top_anomalies)))
        ax.set_yticklabels([col[:20] + '...' if len(col) > 20 else col for col in top_anomalies.index], fontsize=9)
        ax.set_xlabel('Outlier Percentage (%)', fontweight='bold')
        ax.set_title(f'Outlier Analysis (Top {len(top_anomalies)} Numeric Cols)', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add percentage labels
        for i, (bar, pct) in enumerate(zip(bars, top_anomalies['outlier_pct'])):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{pct:.1f}%', ha='left', va='center', fontsize=9, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No Numeric Columns Found', transform=ax.transAxes,
                ha='center', va='center', fontsize=12)
    
    # 5. Overall quality score
    ax = axes[1, 1]
    ax.axis('off')
    
    # Calculate quality scores
    completeness_score = ((total_cells - missing_cells) / total_cells) * 100
    uniqueness_score = ((len(df_sample) - duplicate_rows) / len(df_sample)) * 100
    
    # Consistency score (based on data types and anomalies)
    if numeric_anomalies:
        avg_outlier_pct = np.mean([info['outlier_pct'] for info in numeric_anomalies.values()])
        consistency_score = max(0, 100 - avg_outlier_pct)
    else:
        consistency_score = 100
    
    overall_quality = (completeness_score + uniqueness_score + consistency_score) / 3
    
    quality_text = f"""
    📊 DATA QUALITY SCORECARD
    
    🎯 OVERALL QUALITY: {overall_quality:.1f}/100
    
    📋 DETAILED SCORES:
    • Completeness: {completeness_score:.1f}%
      ({total_cells - missing_cells:,} / {total_cells:,} cells)
    
    • Uniqueness: {uniqueness_score:.1f}%
      ({duplicate_rows:,} duplicate rows)
    
    • Consistency: {consistency_score:.1f}%
      (outlier analysis)
    
    📈 DATASET INFO:
    • Rows: {len(df_sample):,}
    • Columns: {len(df_sample.columns):,}
    • Numeric Columns: {len(numeric_cols):,}
    • Text Columns: {len(text_cols):,}
    
    🎨 DATA TYPES:
    {dtype_counts.to_string()}
    """
    
    # Color coding for quality score
    if overall_quality >= 90:
        bg_color = 'lightgreen'
    elif overall_quality >= 70:
        bg_color = 'lightyellow'
    else:
        bg_color = 'lightcoral'
    
    ax.text(0.1, 0.9, quality_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor=bg_color, alpha=0.8))
    
    # 6. Quality recommendations
    ax = axes[1, 2]
    ax.axis('off')
    
    recommendations = []
    
    if missing_cells > 0:
        missing_pct = (missing_cells / total_cells) * 100
        if missing_pct > 20:
            recommendations.append("🚨 HIGH missing data - consider imputation")
        elif missing_pct > 5:
            recommendations.append("⚠️ MODERATE missing data - review patterns")
        else:
            recommendations.append("✅ LOW missing data levels")
    else:
        recommendations.append("✅ NO missing data")
    
    if duplicate_rows > 0:
        dup_pct = (duplicate_rows / len(df_sample)) * 100
        if dup_pct > 10:
            recommendations.append("🚨 HIGH duplicate rows - investigate")
        else:
            recommendations.append("⚠️ Some duplicate rows found")
    else:
        recommendations.append("✅ NO duplicate rows")
    
    if numeric_anomalies:
        high_outlier_cols = [col for col, info in numeric_anomalies.items() if info['outlier_pct'] > 10]
        if high_outlier_cols:
            recommendations.append(f"🚨 {len(high_outlier_cols)} cols with high outliers")
        else:
            recommendations.append("✅ Outlier levels acceptable")
    
    if len(numeric_cols) / len(df_sample.columns) < 0.3:
        recommendations.append("ℹ️ Consider encoding categorical variables")
    
    rec_text = "🔧 RECOMMENDATIONS:\n\n" + "\n".join(recommendations)
    
    ax.text(0.1, 0.9, rec_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\n📋 QUALITY SUMMARY:")
    print(f"   • Overall Quality Score: {overall_quality:.1f}/100")
    print(f"   • Data Completeness: {completeness_score:.1f}%")
    print(f"   • Row Uniqueness: {uniqueness_score:.1f}%")
    print(f"   • Columns with >10% outliers: {len([col for col, info in numeric_anomalies.items() if info['outlier_pct'] > 10]) if numeric_anomalies else 0}")
    
    quality_metrics = {
        'overall_quality': overall_quality,
        'completeness_score': completeness_score,
        'uniqueness_score': uniqueness_score,
        'consistency_score': consistency_score,
        'missing_cells': missing_cells,
        'duplicate_rows': duplicate_rows,
        'numeric_anomalies': numeric_anomalies
    }
    
    return quality_metrics

def analyze_feature_importance_overview(df, numericals, target_col, top_k=20):
    """
    Analyze and visualize feature importance using multiple methods.

    Args:
        df: DataFrame
        numericals: List of numerical feature names
        target_col: Target column name
        top_k: Number of top features to display
    """
    print("\n" + "="*80)
    print("🎯 FEATURE IMPORTANCE ANALYSIS")
    print("="*80)

    from sklearn.feature_selection import mutual_info_regression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler

    # Prepare data
    feature_data = df[numericals + [target_col]].dropna()
    if len(feature_data) < 50:
        print("❌ Insufficient data for feature importance analysis")
        return

    X = feature_data[numericals].fillna(0)  # Simple imputation for this analysis
    y = feature_data[target_col]

    print(f"📊 Analyzing {len(numericals)} features with {len(feature_data)} complete samples")

    # Method 1: Correlation-based importance
    correlations = X.corrwith(y).abs()

    # Method 2: Mutual Information
    print("   🔄 Computing mutual information...")
    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_importance = pd.Series(mi_scores, index=numericals)

    # Method 3: Random Forest importance (quick version)
    print("   🔄 Computing Random Forest importance...")
    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_importance = pd.Series(rf.feature_importances_, index=numericals)

    # Combine and rank - USE SAFE SORTING HERE
    importance_df = pd.DataFrame({
        'feature': numericals,
        'correlation': correlations.reindex(numericals).fillna(0),
        'mutual_info': mi_importance.reindex(numericals).fillna(0),
        'random_forest': rf_importance.reindex(numericals).fillna(0)
    })

    # Normalize each method to 0-1 scale for comparison
    for col in ['correlation', 'mutual_info', 'random_forest']:
        if importance_df[col].max() > 0:
            importance_df[f'{col}_norm'] = importance_df[col] / importance_df[col].max()
        else:
            importance_df[f'{col}_norm'] = 0

    # Calculate combined score
    importance_df['combined_score'] = (
        importance_df['correlation_norm'] +
        importance_df['mutual_info_norm'] +
        importance_df['random_forest_norm']
    ) / 3

    # Use safe_sort_values instead of regular sort_values to avoid dtype issues
    importance_df = safe_sort_values(importance_df, 'combined_score', ascending=False).reset_index(drop=True)

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🎯 FEATURE IMPORTANCE ANALYSIS', fontsize=16, fontweight='bold')

    top_features = importance_df.head(top_k)

    # 1. Combined importance ranking
    ax = axes[0, 0]
    bars = ax.barh(range(len(top_features)), top_features['combined_score'],
                   color=plt.cm.viridis(np.linspace(0, 1, len(top_features))))
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels([f[:20] + '...' if len(f) > 20 else f for f in top_features['feature']], fontsize=9)
    ax.set_xlabel('Combined Importance Score', fontweight='bold')
    ax.set_title(f'Top {top_k} Features - Combined Ranking', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add scores on bars
    for bar, score in zip(bars, top_features['combined_score']):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{score:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')

    # 2. Method comparison for top features
    ax = axes[0, 1]
    top_10 = importance_df.head(10)
    x = np.arange(len(top_10))
    width = 0.25

    bars1 = ax.bar(x - width, top_10['correlation_norm'], width, label='Correlation', alpha=0.8)
    bars2 = ax.bar(x, top_10['mutual_info_norm'], width, label='Mutual Info', alpha=0.8)
    bars3 = ax.bar(x + width, top_10['random_forest_norm'], width, label='Random Forest', alpha=0.8)

    ax.set_xlabel('Features', fontweight='bold')
    ax.set_ylabel('Normalized Importance', fontweight='bold')
    ax.set_title('Method Comparison (Top 10)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f[:10] + '...' if len(f) > 10 else f for f in top_10['feature']], rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 3. Correlation vs Mutual Information scatter
    ax = axes[1, 0]
    scatter = ax.scatter(
        importance_df['correlation'],
        importance_df['mutual_info'],
        c=importance_df['random_forest'],
        cmap='viridis',
        alpha=0.6,
        s=30
    )
    ax.set_xlabel('Correlation with Target', fontweight='bold')
    ax.set_ylabel('Mutual Information', fontweight='bold')
    ax.set_title('Importance Methods Relationship', fontweight='bold')
    ax.grid(alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Random Forest Importance', fontweight='bold')

    # Annotate top features safely using enumerate (ranked)
    for rank, row in importance_df.head(5).iterrows():
        feat = row['feature']
        corr_val = row['correlation']
        mi_val = row['mutual_info']
        ax.annotate(feat[:8], (corr_val, mi_val),
                    xytext=(5, 5), textcoords='offset points', fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    # 4. Importance statistics
    ax = axes[1, 1]
    ax.axis('off')

    # Safely handle if the dataframe is empty
    if not importance_df.empty:
        n_important = len(importance_df[importance_df['combined_score'] > 0.1])
        avg_correlation = importance_df['correlation'].mean()
        max_importance = importance_df['combined_score'].max()
        top_feature_name = importance_df.iloc[0]['feature']
        corr_vs_mi = importance_df['correlation'].corr(importance_df['mutual_info'])
        corr_vs_rf = importance_df['correlation'].corr(importance_df['random_forest'])
        mi_vs_rf = importance_df['mutual_info'].corr(importance_df['random_forest'])
    else:
        n_important = avg_correlation = max_importance = 0
        top_feature_name = "<none>"
        corr_vs_mi = corr_vs_rf = mi_vs_rf = 0

    stats_text = f"""
    📊 IMPORTANCE STATISTICS

    Total Features Analyzed: {len(numericals)}
    Features with Combined Score > 0.1: {n_important}

    🔍 METHOD AVERAGES:
    Correlation: {avg_correlation:.3f}
    Mutual Information: {importance_df['mutual_info'].mean():.3f}
    Random Forest: {importance_df['random_forest'].mean():.3f}

    🎯 TOP FEATURE:
    {top_feature_name}
    Combined Score: {max_importance:.3f}

    📈 AGREEMENT BETWEEN METHODS:
    Corr vs MI: {corr_vs_mi:.3f}
    Corr vs RF: {corr_vs_rf:.3f}
    MI vs RF: {mi_vs_rf:.3f}
    """

    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    plt.tight_layout()
    plt.show()

    # Print top features with proper ranking using enumerate to avoid index-type issues
    print(f"\n🏆 TOP 10 MOST IMPORTANT FEATURES:")
    for rank, row in enumerate(importance_df.head(10).to_dict(orient='records'), start=1):
        print(f"   {rank:2d}. {row['feature']:<30} | Combined: {row['combined_score']:.3f} | "
              f"Corr: {row['correlation']:.3f} | MI: {row['mutual_info']:.3f} | RF: {row['random_forest']:.3f}")

    return importance_df






def run_enhanced_comprehensive_eda_inline():
    """
    Enhanced EDA runner optimized for inline notebook display with valuable insights.
    Creates clear, focused visualizations that display directly in notebooks.
    """
    print("🚀 Starting Enhanced Comprehensive EDA Analysis...")
    print("="*80)
    total_start_time = time.time()
    
    # Configure matplotlib for inline display
    import matplotlib
    matplotlib.use('inline')  # Ensure inline backend
    plt.ioff()  # Turn off interactive mode
    
    # Load schema
    schema = load_schema_from_yaml(SCHEMA_PATH) if SCHEMA_PATH.exists() else None
    if schema is None:
        print(f"⚠️ Schema YAML not found at {SCHEMA_PATH}. Proceeding without schema.")
        return
    else:
        display_schema_summary(schema)
    
    
    print(f"\n📂 Loading data from: {DATA_PATH}")
    df = load_data_optimized(
        DATA_PATH,
        debug=True,
        drop_null_rows=True,
    )
    print(f"✅ Loaded dataset: {df.shape}")
    
    # Feature engineering
    print(f"\n🔧 Applying feature engineering...")
    df_eng, _ = engineer_features(df)
    print(f"✅ Feature engineering completed. New shape: {df_eng.shape}")
    
    print("---------------dtype check and violations---------------")
    _ = report_schema_dtype_violations(df_eng, schema, max_show=50)

    # Extract feature groups from schema
    try:
        numericals, ordinal, nominal, target_col, cat_breakdown = extract_feature_groups_from_schema(df_eng, schema)
        print(f"✅ Schema-based feature extraction successful")
        print(f"   • Numerical features: {len(numericals)}")
        print(f"   • Ordinal features: {len(ordinal)}")
        print(f"   • Nominal features: {len(nominal)}")
        print(f"   • Target variable: {target_col}")
        print(f"   • Numerical categories: {list(cat_breakdown.keys())}")
    except Exception as e:
        print(f"❌ Schema feature extraction failed: {e}")
        return
    
    if target_col is None or target_col not in df_eng.columns:
        print("❌ No target variable found; aborting.")
        return
    
    # Validate required columns
    for req in ["PLAYER_ID", "TEAM_ID"]:
        if req not in df_eng.columns:
            print(f"❌ Missing required ID column: {req}")
            return
    
    print("✅ Dataset validation passed. Ready for EDA.")
    
    # Data preparation (no imputation to preserve transparency)
    print(f"\n⚙️ Performing data preparation...")
    df_prepared, selected_features, target_col, prep_diagnostics = data_preparation_optimized(
        df_eng, schema, EDA_OUT_DIR, do_impute=False
    )
    
    # Re-extract feature groups after preparation
    _, _, _, _, cat_breakdown_prepped = extract_feature_groups_from_schema(df_prepared, schema)
    
    print(f"\n🎯 Starting comprehensive EDA analysis...")
    
    # 1. TARGET ANALYSIS
    print(f"\n" + "🎯"*40)
    target_stats = analyze_target_distribution_enhanced(df_prepared, target_col)
    
    # 2. MISSING DATA ANALYSIS
    print(f"\n" + "🕳️"*40)
    missing_diag = analyze_missing_data_patterns(
        df,
        schema=schema,
        scope="schema",  # <- default expectation
        top_k=10,
        debug=True
    )
    # 3. DATA QUALITY OVERVIEW
    print(f"\n" + "🔍"*40)
    quality_metrics = analyze_data_quality_overview(df_prepared, sample_size=5000)
    
    # 4. FEATURE IMPORTANCE ANALYSIS
    print(f"\n" + "🎯"*40)
    importance_df = analyze_feature_importance_overview(df_prepared, selected_features, target_col, top_k=15)
    
    # 5. NUMERICAL CATEGORIES ANALYSIS - MAIN ATTRACTION
    print(f"\n" + "📊"*40)
    plot_category_overview_inline(df_prepared, cat_breakdown_prepped, target_col, 
                                max_features=8, figsize_per_cat=(16, 12))
    
    # 6. EXECUTION SUMMARY
    total_time = time.time() - total_start_time
    print("\n" + "🎉"*80) 
    print("ENHANCED COMPREHENSIVE EDA COMPLETE!")
    print("🎉"*80)
    print(f"📊 Total execution time: {total_time:.2f}s")
    print(f"💾 Additional outputs saved in: {EDA_OUT_DIR}")
    print(f"🎯 Target analyzed: {target_col}")
    print(f"📈 Categories analyzed: {len(cat_breakdown_prepped)}")
    print(f"🔢 Features processed: {len(selected_features)}")
    
    # Return summary for further analysis
    eda_summary = {
        'target_stats': target_stats,
        'missing_stats': missing_diag,
        'quality_metrics': quality_metrics,
        'importance_df': importance_df,
        'categories_analyzed': list(cat_breakdown_prepped.keys()),
        'total_time': total_time
    }
    
    print(f"\n✨ EDA Summary returned for further analysis")
    return eda_summary


if __name__ == "__main__":
    # ------EDA------
    run_enhanced_comprehensive_eda_inline()
    
