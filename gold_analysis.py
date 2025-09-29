from __future__ import annotations

import os
import math
from typing import List, Tuple

import numpy as np
import pandas as pd

# 安全的无界面后端，避免 CI 中显示问题
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CSV_PATH = "./gold_data_2015_25.csv"


# ---------- Extracted helpers ----------

def load_raw_df(path: str = CSV_PATH) -> pd.DataFrame:
    """Load CSV to DataFrame."""
    return pd.read_csv(path)


def detect_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Return list of numeric columns (float/int), preserving order."""
    return [c for c, dt in df.dtypes.items() if np.issubdtype(dt, np.number)]


def clean_dataframe(
    df: pd.DataFrame,
    target_col: str,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> pd.DataFrame:
    """
    Basic cleaning:
      - drop NA in target_col
      - clip to [q1, q99] to remove extreme outliers
    """
    if target_col not in df.columns:
        raise KeyError(f"target_col '{target_col}' not in DataFrame")

    cleaned = df.copy()

    # 去除目标列中的 NaN
    cleaned = cleaned[cleaned[target_col].notna()].copy()

    # 分位数截断（仅对目标列）
    lo = cleaned[target_col].quantile(lower_q)
    hi = cleaned[target_col].quantile(upper_q)
    cleaned = cleaned[(cleaned[target_col] >= lo) & (cleaned[target_col] <= hi)].copy()

    return cleaned


def ensure_year_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a 'Year' column exists.
    - If a 'Date' column exists, try to parse year.
    - Otherwise, if already has 'Year', use it directly.
    """
    out = df.copy()
    if "Year" in out.columns:
        return out

    if "Date" in out.columns:
        out["Year"] = pd.to_datetime(out["Date"], errors="coerce").dt.year
    else:
        # 兜底：如果没有 Date/Year，尝试从 Index 构造（测试数据可能已提供 Year）
        if "Year" not in out.columns:
            out["Year"] = pd.Series([np.nan] * len(out), index=out.index)

    return out


def build_annual_agg(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Aggregate by Year for the given value column (mean)."""
    if "Year" not in df.columns:
        raise KeyError("DataFrame must contain 'Year' before aggregation.")
    annual = (
        df.dropna(subset=["Year"])
        .groupby("Year", as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: f"{value_col}_mean"})
    )
    return annual


def simple_linear_regression(
    df: pd.DataFrame, x_col: str, y_col: str
) -> Tuple[float, float, float]:
    """
    Fit y = intercept + slope * x using numpy.polyfit (deg=1).
    Return (slope, intercept, r2). If variance in y equals 0, r2 = NaN.
    """
    reg_df = df[[x_col, y_col]].dropna()
    if len(reg_df) < 2:
        return (np.nan, np.nan, np.nan)

    x = reg_df[x_col].to_numpy(np.float64)
    y = reg_df[y_col].to_numpy(np.float64)

    slope, intercept = np.polyfit(x, y, deg=1)

    y_pred = intercept + slope * x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r2 = np.nan if ss_tot == 0 else (1 - ss_res / ss_tot)
    return (float(slope), float(intercept), float(r2))


def plot_scatter_with_fit(df: pd.DataFrame, x_col: str, y_col: str) -> None:
    """Simple scatter + fitted line (optional for notebooks; CI will no-op show)."""
    reg_df = df[[x_col, y_col]].dropna()
    if reg_df.empty:
        return

    slope, intercept, _ = simple_linear_regression(reg_df, x_col, y_col)

    plt.figure()
    plt.scatter(reg_df[x_col], reg_df[y_col], alpha=0.7)
    if not (math.isnan(slope) or math.isnan(intercept)):
        xline = np.linspace(reg_df[x_col].min(), reg_df[x_col].max(), 100)
        yline = intercept + slope * xline
        plt.plot(xline, yline)
    plt.title(f"{y_col} vs {x_col}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.show()


# ---------- Main (script mode) ----------

# 1) Load
dataframe = load_raw_df(CSV_PATH)

# 2) Detect numeric columns
numeric_cols = detect_numeric_columns(dataframe)

# 3) Choose a reference numeric column for cleaning (first numeric by default)
ref_col = numeric_cols[0] if numeric_cols else None
if ref_col is None:
    # 没有数值列时，导出占位并提前结束
    df = dataframe.copy()
    filtered_df = df.copy()
    annual = pd.DataFrame(columns=["Year"])
    slope = intercept = r2 = np.nan
else:
    # 4) Clean dataframe on the reference column
    df = dataframe.copy()
    filtered_df = clean_dataframe(df, ref_col)

    # 5) Ensure Year and annual aggregation (mean of reference col)
    filtered_df = ensure_year_column(filtered_df)
    annual = build_annual_agg(filtered_df, ref_col)

    # 6) Regression on annual: y = ref_col_mean, x = Year
    slope, intercept, r2 = simple_linear_regression(
        annual.rename(columns={f"{ref_col}_mean": "Y"}),
        x_col="Year",
        y_col="Y",
    )

# ---------- Expose variables expected by tests ----------
# 保持与测试脚本一致的命名（不要改名）
# df: 原始数据副本
# filtered_df: 清洗后的数据（目标列去NA且分位数过滤）
# annual: 年度聚合（mean）
# numeric_cols: 识别出的数值列
# slope/intercept/r2: 年度回归结果
