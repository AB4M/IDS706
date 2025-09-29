import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore

# 1) Import the Dataset
df = pd.read_csv("./gold_data_2015_25.csv")

# 2) Inspect the Data
print("\nShape")
print(df.shape)

print("\nFirst five lines of data")
print(df.head())

print("\nData Types")
df.info()

print("\nSummary Statistics")
print(df.describe())

# 3) Basic Filtering & Grouping
#Filtering
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
filtered_df = df.copy()
if len(numeric_cols) > 0:
    num_col = numeric_cols[0]
    low, high = filtered_df[num_col].quantile([0.01, 0.99])
    filtered_df = filtered_df.loc[
        filtered_df[num_col].notna() & (filtered_df[num_col].between(low, high))
    ]
print("\nFiltered data set")
print(filtered_df.shape)
#Grouping
df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
annual = (
    df.dropna(subset=["GLD"])
      .groupby(df["Date"].dt.year)
      .agg(GLD_mean=("GLD","mean"),
           GLD_count=("GLD","size"),
           GLD_min=("GLD","min"),
           GLD_max=("GLD","max"))
      .reset_index(names="Year")
)
print("\nGLD data grouped by year")
print(annual)

# 4) Simple Regression
if len(numeric_cols) >= 2:
    x_col, y_col = numeric_cols[0], numeric_cols[1]
    reg_df = filtered_df[[x_col, y_col]].dropna()
    X = reg_df[[x_col]].values
    y = reg_df[y_col].values

    s, itcpt = np.polyfit(reg_df[x_col].values, reg_df[y_col].values, deg=1)

    print(f"\nSimple Regression {y_col} ~ {x_col} ")
    print(f"Intercept: {itcpt:.2f}")
    print(f"Slope:     {s:.2f}")

    y_pred = itcpt + s * reg_df[x_col].values
    ss_res = np.sum((reg_df[y_col].values - y_pred) ** 2)
    ss_tot = np.sum((reg_df[y_col].values - np.mean(reg_df[y_col].values)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    print(f"R^2: {r2:.2f}")

# 5) Visualization (Matplotlib)
plt.figure(figsize=(7, 5))

if x_col is not None and y_col is not None:
    plt.scatter(reg_df[x_col], reg_df[y_col], alpha=0.6)
    xs = np.linspace(reg_df[x_col].min(), reg_df[x_col].max(), 100)
    ys = itcpt + s * xs
    plt.plot(xs, ys)
    plt.title(f"Scatter of {y_col} vs {x_col} (with linear fit)")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
plt.tight_layout()
plt.show()
