# Gold Dataset Mini Analysis

This README documents a quick, end‑to‑end analysis performed by `gold_analysis.py` on `gold_data_2015_25.csv`.

## 1) Import & Inspect

- Displayed `.head()` for a quick peek
- Captured `.info()` and `.describe()` to understand types and summary stats

**Data Info (excerpt):**
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2666 entries, 0 to 2665
Data columns (total 6 columns):
 #   Column   Non-Null Count  Dtype         
---  ------   --------------  -----         
 0   Date     2666 non-null   datetime64[ns]
 1   SPX      2666 non-null   float64       
 2   GLD      2666 non-null   float64       
 3   USO      2666 non-null   float64       
 4   SLV      2666 non-null   float64       
 5   EUR/USD  2666 non-null   float64       
dtypes: datetime64[ns](1), float64(5)
memory usage: 125.1 KB
```

**Head (first 5 rows):**
| Date                |     SPX |    GLD |    USO |   SLV |   EUR/USD |
|:--------------------|--------:|-------:|-------:|------:|----------:|
| 2015-01-02 00:00:00 | 2058.2  | 114.08 | 159.12 | 15.11 |   1.20894 |
| 2015-01-05 00:00:00 | 2020.58 | 115.8  | 150.32 | 15.5  |   1.19464 |
| 2015-01-06 00:00:00 | 2002.61 | 117.12 | 144.4  | 15.83 |   1.1939  |
| 2015-01-07 00:00:00 | 2025.9  | 116.43 | 146.96 | 15.85 |   1.18754 |
| 2015-01-08 00:00:00 | 2062.14 | 115.94 | 148.4  | 15.64 |   1.1836  |

## 2) Basic Filtering & Grouping

- Filter: kept rows with **GLD** above its median and **Date >= 2018-01-01**
- Grouped by **Year** and computed numeric means + row counts

**Grouped Preview:**
|   Year |     SPX |     GLD |      USO |     SLV |   EUR/USD |   count |
|-------:|--------:|--------:|---------:|--------:|----------:|--------:|
|   2015 | 2061.07 | 111.146 | 132.792  | 14.9939 |   1.11006 |     252 |
|   2016 | 2094.65 | 119.363 |  84.2219 | 16.2908 |   1.10753 |     252 |
|   2017 | 2448.62 | 119.715 |  84.0315 | 16.1423 |   1.12979 |     249 |
|   2018 | 2746.21 | 120.177 | 106.782  | 14.7769 |   1.18138 |     251 |
|   2019 | 2913.58 | 131.562 |  95.1822 | 15.1839 |   1.11976 |     251 |
|   2020 | 3217.86 | 166.654 |  40.5932 | 19.1611 |   1.14209 |     253 |
|   2021 | 4273.39 | 168.311 |  47.012  | 23.2808 |   1.18317 |     252 |
|   2022 | 4098.51 | 167.905 |  72.8495 | 20.0774 |   1.05311 |     251 |
|   2023 | 4283.73 | 180.45  |  69.5499 | 21.4601 |   1.08162 |     250 |
|   2024 | 5428.24 | 221.099 |  74.5037 | 25.826  |   1.08244 |     252 |

## 3) Simple Regression

- Target: **GLD**
- Features: `SPX, USO, SLV, EUR/USD`
- Model: Linear Regression (train/test split 75/25)

**Test Metrics:**
- R²: **0.9178**
- MAE: **8.7240**
- RMSE: **13.1102**
- Intercept: **61.285706**

**Coefficients:**
- SPX: 0.016341
- USO: -0.008564
- SLV: 5.297135
- EUR/USD: -55.430253

## 4) Visualization

A single matplotlib scatter plot was created:

- `scatter_gld_spx.png`: GLD vs SPX with a simple fitted line (GLD ~ SPX).

![Scatter](scatter_gld_spx.png)

## How to Re‑Run

```bash
python gold_analysis.py
```

This will regenerate the plot and this README.
