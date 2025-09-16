# Gold Data Analysis Project

## Project layout
.
├─ gold_analysis.py
├─ requirements.txt
├─ tests/
│  └─ test_gold_analysis.py
├─ Dockerfile
└─ .gitignore

## Docker — reproducible environment
Build the image: docker run --rm gold-analysis:dev
Run tests: docker run --rm gold-analysis:dev

## What the tests cover
- tests/test_gold_analysis.py runs an end-to-end execution of the script and checks:
- Loading & preprocessing (date parsing, shape, numeric detection)
- Filtering (quantile trimming + non-null handling)
- Grouping (yearly GLD aggregates)
- Simple regression behavior (slope & R² sanity; constant-target edge case)

## Project Goal
The goal of this project is to analyze gold-related financial data 2015 to 2025.  

## Data Source
- **File:** `gold_data_2015_25.csv`  
- **Content:**  
  - SPX – S&P 500 Index daily closing prices.
  - GLD – SPDR Gold Shares ETF daily adjusted closing prices.
  - USO – United States Oil Fund ETF daily adjusted closing prices.
  - SLV – iShares Silver Trust ETF daily adjusted closing prices.
  - EUR/USD – Daily Euro to US Dollar exchange rate.
  - Data covers **2666 trading days** between 2015 and 2025.  

## Setup Instructions
import numpy, pandas and matplotlib to process the data.

## Data Analysis Steps

1. Data Import & Inspection
- Load CSV file with pandas.
- Check dataset shape, column types, summary statistics, and missing values.
2. Filtering & Cleaning
- Identify numeric columns.
- Remove extreme outliers (1st and 99th percentile).
3. Grouping
- Convert Date column to datetime.
- Group GLD prices by year and compute statistics (mean, min, max, count).
4. Simple Regression
- Perform a linear regression between SPX (independent variable) and GLD (dependent variable).
- Calculate intercept, slope, and R² score.
5. Visualization
- Scatter plot of SPX vs GLD with regression line.

## Outcomes
- Dataset Overview: 2666 rows × 6 columns.
- Numeric Columns: SPX, GLD, USO, SLV, EUR/USD.
- Regression Result (GLD ~ SPX):
- Intercept ≈ 34.54
- Slope ≈ 0.0352
- R² ≈ 0.854 → strong linear relationship.
- Visualization: A clear upward trend between SPX and GLD, with fitted regression line.
<img width="690" height="490" alt="image" src="https://github.com/user-attachments/assets/ca0749cd-88b3-4db5-9c76-052cc356a84e" />
