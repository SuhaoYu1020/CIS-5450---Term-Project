# CIS-5450 Term Project
## IPO Success Prediction

Predicting company delisting using financial statement data with logistic regression.

## Installation

```bash
pip install wrds pandas openpyxl scikit-learn numpy
```

## Project Structure

```
├── src/
│   ├── dataset.py          # Fetch IPO/Delist data from WRDS
│   ├── process.py           # Process IPO and Delist data
│   ├── data_cleaning.py    # Clean and merge financial data
│   └── main.py             # Main training and prediction pipeline
├── models/
│   └── logistic_regression.py  # Logistic regression model
└── data/                   # Data files (Excel format)
```

## Features

### Data Fetching (`src/dataset.py`)
- Fetch CRSP IPO/Delist data from WRDS
- Support Compustat ipodate or CRSP earliest namedt
- Fetch financial statement data (ROA, ROE, etc.)
- Automatic fallback across multiple library tables

### Data Processing (`src/process.py`)
- Filter companies that delisted within N years after IPO
- Merge IPO and Delist data by permno
- Standardize column names

### Data Cleaning (`src/data_cleaning.py`)
- Clean financial data and merge with IPO/Delist mapping
- Handle missing values with forward/backward fill
- Merge multiple cleaned tables into final dataset
- Remove redundant columns and zero-variance features

### Model Training (`src/main.py`)
- Aggregate monthly financial data to yearly features
- Predict next year delisting using current year features
- Company-level time-series split (train/val/test)
- Feature engineering: last, mean, std, slope aggregations
- Logistic regression with preprocessing pipeline

### Model (`models/logistic_regression.py`)
- Logistic regression with sklearn pipeline
- Includes imputation, scaling, and classification
- Supports model saving and loading

## Usage

### 1. Fetch Data from WRDS
```bash
# Fetch IPO data
python src/dataset.py --start 1974-01-01 --end 2024-12-31 --exchange NYSE

# Fetch Delist data
python src/dataset.py --start 1974-01-01 --end 2024-12-31 --exchange NYSE --delist

# Fetch financial data
python src/dataset.py --start 1974-01-01 --end 2024-12-31 --exchange NYSE --fetch-financials
```

### 2. Process IPO/Delist Data
```bash
python src/process.py --ipo-path data/ipo_1974_2024_1.xlsx --delist-path data/delist_1974_2024_1.xlsx --years 5
```

### 3. Clean and Merge Data
```bash
# Clean individual tables
python src/data_cleaning.py --index 1
python src/data_cleaning.py --index 2
python src/data_cleaning.py --index 3

# Merge all tables
python src/data_cleaning.py --merge
```

### 4. Train Model
```bash
python src/main.py --data_path data/final_table_merged.xlsx
```

## Model Features

The model uses 16 base financial features:
- Revenue (revtq)
- P/E ratios (pe_op_basic, pe_exi, pe_inc)
- Market/Book ratios (ptb, bm)
- Profitability (roa, roe, npm)
- Market valuation (tobinq)
- Growth (revenue_growth)
- Liquidity (quick_ratio, curr_ratio)
- Leverage (de_ratio)
- Turnover (at_turn, inv_turn)

For yearly prediction, features are aggregated as:
- Last value (`_last`)
- Mean (`_mean`)
- Standard deviation (`_std`) for ratios
- Slope (`_slope`) for trend indicators
- Observation count (`obs_count_in_year`)

## Data Split

- Training: First 30 years (by IPO date)
- Validation: Next 10 years
- Test: Last 10 years
- Company-level split ensures no data leakage
