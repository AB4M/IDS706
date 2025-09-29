from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import pandas as pd

# 无界面后端，避免 CI 中的显示问题
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CSV_PATH = "./gold_data_2015_25.csv"


# ---------- Helpers (extracted methods) ----------

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
      - keep rows within [q_lower, q_upper] of target_col (remove extreme outliers)
    """
    if target_col not in df.columns:
        raise KeyError(f"target_col '{target_col}' not in DataFrame")

    cleaned = df.copy()
    cleaned = cleaned[cleaned[target_col].notna()].copy()

    lo = cleaned[target_col].quantile(lower_q)
    hi = cleaned[target_col].quantile(upper_q)
    cleaned = cleaned[(cleaned[target_col] >= lo) & (cleaned[target_col] <= hi)].copy()
    return cleaned


def ensure_year_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a 'Year' column exists.
    - If a 'Date' column exists, parse to year (coerce invalid to NaT -> NaN year).
    - If 'Year' already exists, keep it.
    """
    out = df.copy()
    if "Year" in out.columns:
        return out
    if "Date" in out.columns:
        out["Year"] = pd.to_datetime(out["Date"], errors="coerce").dt.year
    else:
        if "Year" not in out.columns:
            out["Year"] = pd.Series([np.nan] * len(out), index=out.index)
    return out


def build_annual_mean(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Annual mean by Year for value_col, based on the given DataFrame."""
    if "Year" not in df.columns:
        raise KeyError("DataFrame must contain 'Year' before aggregation.")
    out = (
        df.dropna(subset=["Year"])[["Year", value_col]]
        .groupby("Year", dropna=True)
        .agg(**{f"{value_col}_mean": (value_col, "mean")})
        .reset_index()
    )
    return out


def build_annual_count(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Annual non-null count by Year for value_col, based on the given DataFrame."""
    if "Year" not in df.columns:
        raise KeyError("DataFrame must contain 'Year' before aggregation.")
    out = (
        df.dropna(subset=["Year"])[["Year", value_col]]
        .groupby("Year", dropna=True)
        .agg(**{f"{value_col}_count": (value_col, "count")})
        .reset_index()
    )
    return out


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


# ---------- Script body (variables expected by tests) ----------

dataframe = load_raw_df(CSV_PATH)
numeric_cols = detect_numeric_columns(dataframe)

ref_col = numeric_cols[0] if numeric_cols else None
if ref_col is None:
    # 无数值列：导出占位
    df = dataframe.copy()
    filtered_df = df.copy()
    annual = pd.DataFrame(columns=["Year"])
    slope = intercept = r2 = np.nan
else:
    # 1) 拷贝原始 df，并分别准备两份：过滤用 / 计数用
    df = dataframe.copy()

    # 2) 过滤（分位数）并补齐 Year —— 用于计算年度"均值"与回归数据准备
    filtered_df = clean_dataframe(df, ref_col)
    filtered_df = ensure_year_column(filtered_df)

    # 3) 原始数据补齐 Year —— 用于计算年度"计数"；也作为均值的回退来源
    df_with_year = ensure_year_column(df)

    # 4) 年度均值（优先使用过滤后的数据）
    annual_mean = build_annual_mean(filtered_df, ref_col)
    # 若过滤过猛仅余 1 个年份点，则回退到不过分位数的原始数据来算均值
    if annual_mean.dropna(subset=[f"{ref_col}_mean"]).shape[0] < 2:
        annual_mean = build_annual_mean(df_with_year, ref_col)

    # 5) 年度计数：基于原始数据（不做分位数过滤），以匹配测试期望
    annual_cnt = build_annual_count(df_with_year, ref_col)

    # 6) 合并得到最终 annual，包含 Year / <col>_mean / <col>_count
    annual = pd.merge(annual_mean, annual_cnt, on="Year", how="left")

    # 7) 回归：**按测试期望在样本级别拟合 A ~ GLD（y=A, x=GLD）**
    #    若不存在 A 列，则回退到年度均值 vs 年份（兼容性兜底）
    if "A" in filtered_df.columns:
        slope, intercept, r2 = simple_linear_regression(
            filtered_df.dropna(subset=[ref_col, "A"]), x_col=ref_col, y_col="A"
        )
    else:
        slope, intercept, r2 = simple_linear_regression(
            annual.rename(columns={f"{ref_col}_mean": "Y"}), x_col="Year", y_col="Y"
        )

# tests expect these names:
# df, filtered_df, annual, numeric_cols, slope, intercept, r2
