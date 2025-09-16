import os
import runpy
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Utilities
SCRIPT_PATH = Path('/mnt/data/gold_analysis.py')

def write_csv(path: Path, df: pd.DataFrame):
    path.write_text(df.to_csv(index=False), encoding='utf-8')

def run_script_with_csv(tmp_path: Path, df: pd.DataFrame):
    # Prepare environment: CSV file at expected relative path
    csv_path = tmp_path / "gold_data_2015_25.csv"
    write_csv(csv_path, df)

    # Use a non-interactive backend and suppress plt.show()
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        plt.show = lambda *args, **kwargs: None
    except Exception:
        pass

    # Execute the script with tmp_path as CWD so relative CSV resolves
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        ns = runpy.run_path(str(SCRIPT_PATH), run_name="__main__")
    finally:
        os.chdir(cwd)
    return ns

def make_linear_df(n=200, noise=0.2, seed=0):
    rng = np.random.default_rng(seed)
    # We want regression: A ~ GLD with high R^2
    dates = pd.date_range("2016-01-01", periods=n, freq="D")
    A_true = rng.normal(loc=10.0, scale=2.0, size=n)
    GLD = 2.0 * A_true + 3.0 + rng.normal(0.0, noise, size=n)  # feature
    B = rng.normal(loc=0.0, scale=1.0, size=n)

    GLD = pd.Series(GLD)  # make it a Series so we can use .iloc safely
    GLD.iloc[::37] = np.nan  # some NaNs to test dropna grouping

    df = pd.DataFrame({"Date": dates.astype(str), "GLD": GLD, "A": A_true, "B": B})

    # Add an extreme outlier in GLD so quantile filter trims it
    outlier_row = pd.DataFrame({"Date": ["2017-06-01"], "GLD": [1000.0], "A": [0.0], "B": [0.0]})
    df = pd.concat([df, outlier_row], ignore_index=True)
    return df

def make_constant_target_df(n=50, seed=1):
    # gold_analysis does A ~ GLD, so make A constant -> ss_tot == 0
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n, freq="D")
    A = np.full(n, 42.0)         # constant target
    GLD = rng.normal(loc=125.0, scale=2.0, size=n)
    B = rng.normal(loc=0.0, scale=1.0, size=n)
    df = pd.DataFrame({"Date": dates.astype(str), "GLD": GLD, "A": A, "B": B})
    return df

# ---------------- Tests ---------------- #

def test_system_run_linear(tmp_path):
    '''System test: end-to-end execution on linear data; verifies core artifacts.'''
    df = make_linear_df()
    ns = run_script_with_csv(tmp_path, df)

    # Artifacts
    import pandas as pd
    assert "df" in ns and isinstance(ns["df"], pd.DataFrame)
    assert "filtered_df" in ns and isinstance(ns["filtered_df"], pd.DataFrame)
    assert "annual" in ns and isinstance(ns["annual"], pd.DataFrame)
    assert "numeric_cols" in ns and isinstance(ns["numeric_cols"], list)

    # Loading shape
    assert ns["df"].shape[0] == df.shape[0]

    # Filtering removed the outlier in first numeric col (should be GLD)
    first_numeric = ns["numeric_cols"][0]
    assert first_numeric == "GLD"
    assert ns["filtered_df"].shape[0] < ns["df"].shape[0]

    # Grouping by year on GLD
    annual = ns["annual"]
    assert set(annual["Year"].tolist()).issubset({2016, 2017})

    df_copy = df.copy()
    df_copy["Date"] = pd.to_datetime(df_copy["Date"], errors="coerce")
    expected = (
        df_copy.dropna(subset=["GLD"])
               .groupby(df_copy["Date"].dt.year)["GLD"]
               .size()
               .reset_index(name="GLD_count")
    )
    merged = pd.merge(annual[["Year", "GLD_count"]], expected, left_on="Year", right_on="Date", how="left")
    assert (merged["GLD_count_x"].fillna(-1).values == merged["GLD_count_y"].fillna(-1).values).all()

    # Regression sanity: A ~ GLD should have slope ~0.5 and high R^2
    assert "slope" in ns and "intercept" in ns and "r2" in ns
    import numpy as np
    assert np.isfinite(ns["slope"]) and np.isfinite(ns["intercept"])
    assert 0.4 < ns["slope"] < 0.6
    assert ns["r2"] > 0.95

def test_edgecase_constant_target_r2_nan(tmp_path):
    '''If target A is constant, implementation should return NaN R^2 (ss_tot == 0).'''
    df = make_constant_target_df()
    ns = run_script_with_csv(tmp_path, df)
    import numpy as np
    assert "r2" in ns
    assert np.isnan(ns["r2"]), f"Expected r2 to be NaN for constant target, got {ns['r2']}"

def test_preprocessing_date_coercion(tmp_path):
    '''Invalid dates coerced to NaT should not break grouping.'''
    df = make_linear_df(n=30)
    df.loc[0, "Date"] = "not-a-date"
    df.loc[5, "Date"] = "13/40/2020"
    ns = run_script_with_csv(tmp_path, df)
    annual = ns["annual"]
    import numpy as np
    yrs = annual["Year"].dropna().to_numpy()
    assert np.all(np.isfinite(yrs))
    assert np.allclose(yrs, np.round(yrs))

def test_filtering_respects_non_null(tmp_path):
    '''Filtering keeps only rows where the reference numeric col (GLD) is not NA and within quantiles.'''
    df = make_linear_df(n=100)
    df.loc[[1, 3, 5], "GLD"] = np.nan
    ns = run_script_with_csv(tmp_path, df)
    first_numeric = ns["numeric_cols"][0]
    assert ns["filtered_df"][first_numeric].isna().sum() == 0
