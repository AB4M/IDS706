from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


DATA_PATH = Path("/mnt/data/gold_data_2015_25.csv")
OUT_DIR = Path("/mnt/data")
PLOT_PATH = OUT_DIR / "scatter_gld_spx.png"
README_PATH = OUT_DIR / "README.md"


def load_data(path: Path) -> pd.DataFrame:
    """Load CSV, parse dates, sort by date."""
    df = pd.read_csv(path)
    # Parse the Date column if present
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df


def basic_inspect(df: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    """Return info string and head for quick overview."""
    # Capture .info() into a string
    from io import StringIO
    buf = StringIO()
    df.info(buf=buf)
    info_str = buf.getvalue()
    head_df = df.head()
    return info_str, head_df


def filter_and_group(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Example filtering:
      - Keep rows where GLD is above its median
      - Keep rows from 2018 and later (if Date exists)
    Example grouping:
      - Group by year and compute mean for numeric columns + count
    """
    work = df.copy()

    # Filtering by GLD median if GLD exists
    if "GLD" in work.columns:
        gld_median = work["GLD"].median()
        work = work[work["GLD"] > gld_median].copy()

    # Filtering by date range (2018+) if Date exists
    if "Date" in df.columns:
        work = work[work["Date"] >= pd.Timestamp("2018-01-01")]

    # Group by calendar year if Date exists
    grouped = pd.DataFrame()
    if "Date" in df.columns:
        tmp = df.copy()
        tmp["Year"] = tmp["Date"].dt.year
        # numeric-only means
        means = tmp.groupby("Year").mean(numeric_only=True).reset_index()
        counts = tmp.groupby("Year").size().reset_index(name="count")
        grouped = pd.merge(means, counts, on="Year", how="left")

    return work, grouped


def fit_regression(df: pd.DataFrame) -> Tuple[LinearRegression, dict]:
    """
    Fit a simple linear regression:
        target: GLD
        features: SPX, USO, SLV, EUR/USD (when available)
    Returns the model and a metrics dictionary.
    """
    # Ensure required columns exist
    target_col = "GLD"
    candidate_features = ["SPX", "USO", "SLV", "EUR/USD"]
    features = [c for c in candidate_features if c in df.columns]

    if target_col not in df.columns or not features:
        raise ValueError("Required columns missing for regression. Need 'GLD' and at least one of: " + ", ".join(candidate_features))

    # Drop rows with NA in the selected columns
    reg_df = df.dropna(subset=[target_col] + features).copy()

    X = reg_df[features].values
    y = reg_df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "r2_test": float(r2_score(y_test, y_pred)),
        "mae_test": float(mean_absolute_error(y_test, y_pred)),
        "rmse_test": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "coefficients": {feat: float(coef) for feat, coef in zip(features, model.coef_)},
        "intercept": float(model.intercept_),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "features": features,
        "target": target_col,
    }
    return model, metrics


def make_plot(df: pd.DataFrame, model: LinearRegression) -> None:
    """
    Create a single scatter plot: GLD vs SPX
    Add a simple fitted line using univariate fit on SPX.
    """
    if "GLD" not in df.columns or "SPX" not in df.columns:
        return  # silently skip if columns are missing

    plot_df = df.dropna(subset=["GLD", "SPX"])
    if plot_df.empty:
        return

    # For the regression line on SPX only, fit a simple 1D model
    x = plot_df[["SPX"]].values
    y = plot_df["GLD"].values
    uni = LinearRegression().fit(x, y)
    x_line = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
    y_line = uni.predict(x_line)

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, alpha=0.6, label="Data")
    plt.plot(x_line, y_line, linewidth=2, label="Fit (GLD ~ SPX)")
    plt.title("GLD vs SPX")
    plt.xlabel("SPX")
    plt.ylabel("GLD")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()


def write_readme(info_str: str, head_df: pd.DataFrame, grouped: pd.DataFrame, metrics: dict) -> None:
    """Write a concise README.md summarizing steps and findings."""
    # Format head and a small piece of grouped for readability
    head_md = head_df.to_markdown(index=False)
    grouped_preview = grouped.head(10).to_markdown(index=False) if not grouped.empty else "_No grouping available._"

    coeff_lines = "\n".join(f"- {k}: {v:.6f}" for k, v in metrics.get("coefficients", {}).items())
    content = f"""# Gold Dataset Mini Analysis

This README documents a quick, end‑to‑end analysis performed by `gold_analysis.py` on `gold_data_2015_25.csv`.

## 1) Import & Inspect

- Displayed `.head()` for a quick peek
- Captured `.info()` and `.describe()` to understand types and summary stats

**Data Info (excerpt):**
```
{info_str.strip()}
```

**Head (first 5 rows):**
{head_md}

## 2) Basic Filtering & Grouping

- Filter: kept rows with **GLD** above its median and **Date >= 2018-01-01**
- Grouped by **Year** and computed numeric means + row counts

**Grouped Preview:**
{grouped_preview}

## 3) Simple Regression

- Target: **GLD**
- Features: `{", ".join(metrics.get("features", []))}`
- Model: Linear Regression (train/test split 75/25)

**Test Metrics:**
- R²: **{metrics.get("r2_test", float('nan')):.4f}**
- MAE: **{metrics.get("mae_test", float('nan')):.4f}**
- RMSE: **{metrics.get("rmse_test", float('nan')):.4f}**
- Intercept: **{metrics.get("intercept", float('nan')):.6f}**

**Coefficients:**
{coeff_lines or "_(none)_"}

## 4) Visualization

A single matplotlib scatter plot was created:

- `scatter_gld_spx.png`: GLD vs SPX with a simple fitted line (GLD ~ SPX).

![Scatter](scatter_gld_spx.png)

## How to Re‑Run

```bash
python gold_analysis.py
```

This will regenerate the plot and this README.
"""
    README_PATH.write_text(content, encoding="utf-8")


def main() -> None:
    df = load_data(DATA_PATH)

    # Inspect
    info_str, head_df = basic_inspect(df)
    _ = df.describe(include="all")  # computed to console if needed

    # Filter & Group
    filtered, grouped = filter_and_group(df)

    # Regression
    model, metrics = fit_regression(df)

    # Visualization
    make_plot(df, model)

    # Documentation
    write_readme(info_str, head_df, grouped, metrics)

    # Console outputs to guide the user when running locally
    print("=== HEAD ===")
    print(head_df)
    print("\n=== INFO ===")
    print(info_str)
    print("\n=== DESCRIBE (numeric) ===")
    num_desc = df.select_dtypes(include=["number"]).describe().T
    print(num_desc)
    print("\n=== GROUPED (preview) ===")
    print(grouped.head(10))
    print("\n=== REGRESSION METRICS ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print(f"\nPlot saved to: {PLOT_PATH}")
    print(f"README saved to: {README_PATH}")


if __name__ == "__main__":
    main()
