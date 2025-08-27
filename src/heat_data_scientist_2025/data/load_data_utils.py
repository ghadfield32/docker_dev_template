"""
load_data_utils
"""
import pandas as pd
import time

def load_data_optimized(
    DATA_PATH: str,
    debug: bool = False,
    drop_null_rows: bool = False,
    drop_null_how: str = 'any',  # 'any' or 'all'
    drop_null_subset: list | None = None,  # list of column names or None for all columns
    use_sample: bool = False,
    sample_size: int = 1000,
):
    """Load data with performance optimizations and enhanced debug diagnostics.

    Parameters:
    - DATA_PATH: Path to the parquet file.
    - debug: If True, prints detailed dataset diagnostics.
    - drop_null_rows: If True, drops rows based on null criteria.
    - drop_null_how: 'any' to drop rows with any nulls, 'all' to drop rows with all nulls.
    - drop_null_subset: List of columns to consider when dropping nulls; defaults to all.

    Returns:
    - df: Loaded (and optionally filtered) DataFrame.
    """
    print("Loading data for enhanced comprehensive EDA...")
    start_time = time.time()

    # 1. Load data
    if use_sample:
        print(f"⚡ Using sample data (n={sample_size}) instead of real parquet.")
        len_df = sample_size
        df = pd.read_parquet(DATA_PATH)
        #take only the len of the data
        df = df.head(len_df)
    else:
        if DATA_PATH is None:
            raise ValueError("DATA_PATH must be provided when not using sample data.")
        df = pd.read_parquet(DATA_PATH)


    # 2. Drop null rows if requested
    if drop_null_rows:
        before = len(df)
        # Determine which subset to use for dropna
        subset_desc = "all columns" if drop_null_subset is None else f"subset={drop_null_subset}"
        print(f"→ Applying null dropping: how='{drop_null_how}', {subset_desc}")
        if drop_null_subset is None:
            df = df.dropna(how=drop_null_how)
        else:
            # Defensive: ensure provided columns exist (warn if some missing)
            missing_cols = [c for c in drop_null_subset if c.upper() not in df.columns]
            if missing_cols:
                print(f"⚠️ Warning: drop_null_subset columns not found in dataframe and will be ignored: {missing_cols}")
            valid_subset = [c.upper() for c in drop_null_subset if c.upper() in df.columns]
            df = df.dropna(how=drop_null_how, subset=valid_subset if valid_subset else None)
        dropped = before - len(df)
        print(f"✓ Dropped {dropped:,} rows by null criteria (how='{drop_null_how}', subset={drop_null_subset}); remaining {len(df):,} rows")

    # 3. Debug diagnostics
    if debug:
        print("========== Dataset Debug Details ============")
        print(f"Total rows       : {df.shape[0]:,}")
        print(f"Total columns    : {df.shape[1]:,}")
        print(f"Columns          : {df.columns.tolist()}")

        total = len(df)
        null_counts = df.isnull().sum()
        non_null_counts = total - null_counts
        null_percent = (null_counts / total) * 100
        dtype_info = df.dtypes

        null_summary = pd.DataFrame({
            'dtype'          : dtype_info,
            'null_count'     : null_counts,
            'non_null_count' : non_null_counts,
            'null_percent'   : null_percent
        }).sort_values(by='null_percent', ascending=False)

        pd.set_option('display.max_rows', None)
        print("---- Nulls Summary (per column) ----")
        print(null_summary)

    load_time = time.time() - start_time
    print(f"✓ Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]:,} columns in {load_time:.2f}s")
    return df



if __name__ == "__main__":
    from src.heat_data_scientist_2025.utils.config import CFG
    df = load_data_optimized(
        CFG.ml_dataset_path,
        debug=True,
        # use_sample=True,
        # drop_null_rows=True,
        # drop_null_subset=['AAV']
    )
    print(df.columns.tolist())
    print(df.head())
    print(df.shape)
    # unique values for season
    print(df['season'].unique())
    

    # df = load_data_optimized(
    #     FINAL_DATA_PATH,
    #     debug=True,
    #     use_sample=True,
    # )
    # print(df.columns.tolist())
