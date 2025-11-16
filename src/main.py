from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure project root is in sys.path so `python src/main.py` can find `models`
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.logistic_regression import build_model_from_params


def _base_monthly_feature_names() -> List[str]:
    """Base monthly financial feature names (16 features)."""
    return [
        "revtq",
        "pe_op_basic",
        "pe_exi",
        "pe_inc",
        "ptb",
        "bm",
        "roa",
        "roe",
        "npm",
        "tobinq",
        "revenue_growth",
        "quick_ratio",
        "de_ratio",
        "curr_ratio",
        "at_turn",
        "inv_turn",
    ]


def infer_time_column(columns: Iterable[str]) -> str:
    candidates = ["public_year", "year", "fyear", "fiscal_year", "datadate", "year_end"]
    for c in candidates:
        if c in columns:
            return c
    raise ValueError("Could not find year column. Please specify via --time_col (e.g., year or fyear).")


def infer_label_column(columns: Iterable[str]) -> str:
    candidates = ["delist", "is_delist", "delisted", "label", "delist_flag"]
    for c in candidates:
        if c in columns:
            return c
    raise ValueError("Could not find label column (delist indicator). Please specify via --label_col (e.g., delist).")


def default_feature_list() -> List[str]:
    # Common financial indicator names inferred from screenshots; main program will intersect with actual column names
    return _base_monthly_feature_names()


def default_yearly_aggregated_feature_list() -> List[str]:
    """Default feature list after yearly aggregation (for "predict next year using current year").
    - last and mean for all base variables
    - volatility std (only for ratio/turnover types)
    - within-year trend slope (for profitability/scale representative variables)
    - observation count obs_count_in_year
    """
    base = _base_monthly_feature_names()
    base_last = [f"{c}_last" for c in base]
    base_mean = [f"{c}_mean" for c in base]
    std_cols = ["quick_ratio", "de_ratio", "curr_ratio", "at_turn", "inv_turn"]
    std_feats = [f"{c}_std" for c in std_cols]
    slope_cols = ["revtq", "roa", "roe", "npm", "tobinq"]
    slope_feats = [f"{c}_slope" for c in slope_cols]
    return base_last + base_mean + std_feats + slope_feats + ["obs_count_in_year"]


def build_yearly_from_monthly(
    df: pd.DataFrame, cutoff_month: Optional[int] = None
) -> pd.DataFrame:
    """Aggregate monthly data to yearly and generate next year label y_next_year.
    Only uses current year information to construct yearly features; optionally cutoff by cutoff_month to avoid potential information leakage at year-end.
    """
    assert "permno" in df.columns, "Data missing company identifier column permno"
    year_col = "public_year" if "public_year" in df.columns else infer_time_column(df.columns)
    month_col = "public_month"
    features = [c for c in _base_monthly_feature_names() if c in df.columns]

    dfw = df.copy()
    if month_col not in dfw.columns:
        # If no month info, treat as December (year-end), still can perform "last/mean" aggregation
        dfw[month_col] = 12
    # Keep only needed columns to avoid groupby overhead
    keep_cols = list(dict.fromkeys(["permno", year_col, month_col, "delist"] + features))
    dfw = dfw[keep_cols]

    # Convert to numeric and clean inf
    for c in features:
        dfw[c] = pd.to_numeric(dfw[c], errors="coerce")
    dfw = dfw.replace([np.inf, -np.inf], np.nan)

    # Optional: only keep data before cutoff month within the year
    if cutoff_month is not None:
        dfw = dfw[dfw[month_col] <= int(cutoff_month)]

    # Sort
    dfw = dfw.sort_values(["permno", year_col, month_col], kind="mergesort")

    # Yearly aggregation: last/mean/std/slope/observation count
    # last/mean
    last_vals = dfw.groupby(["permno", year_col], as_index=True)[features].last().add_suffix("_last")
    mean_vals = dfw.groupby(["permno", year_col], as_index=True)[features].mean().add_suffix("_mean")
    # std (only compute subset)
    std_cols = [c for c in ["quick_ratio", "de_ratio", "curr_ratio", "at_turn", "inv_turn"] if c in features]
    std_vals = (
        dfw.groupby(["permno", year_col], as_index=True)[std_cols].std().add_suffix("_std")
        if std_cols
        else pd.DataFrame(index=last_vals.index)
    )

    # Slope (least squares on month sequence, simple univariate regression after centering)
    slope_cols = [c for c in ["revtq", "roa", "roe", "npm", "tobinq"] if c in features]

    def slope_per_group(g: pd.DataFrame, cols: List[str]) -> pd.Series:
        x = g[month_col].to_numpy()
        x = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-6)
        out = {}
        for c in cols:
            y = g[c].to_numpy()
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() >= 2:
                xv, yv = x[mask], y[mask]
                slope = float(np.dot(xv, yv) / (np.dot(xv, xv) + 1e-6))
            else:
                slope = float("nan")
            out[f"{c}_slope"] = slope
        return pd.Series(out)

    if slope_cols:
        # Only select columns needed for computation to avoid pandas apply behavior change warnings on grouping columns
        slopes = (
            dfw.groupby(["permno", year_col])[[month_col] + slope_cols]
            .apply(lambda g: slope_per_group(g, slope_cols))
            .astype(float)
        )
    else:
        slopes = pd.DataFrame(index=last_vals.index)

    # Observation count
    obs = dfw.groupby(["permno", year_col]).size().rename("obs_count_in_year").to_frame()

    # Combine yearly features
    yearly = (
        last_vals.join(mean_vals, how="outer")
        .join(std_vals, how="outer")
        .join(slopes, how="outer")
        .join(obs, how="outer")
        .reset_index()
    )

    # Generate "next year" label: deduplicate by company->year, sort, then shift(-1)
    if "delist" not in dfw.columns:
        raise ValueError("Data missing delist column, cannot construct next year label y_next_year.")
    label_year = (
        dfw[["permno", year_col, "delist"]]
        .drop_duplicates(subset=["permno", year_col])
        .sort_values(["permno", year_col], kind="mergesort")
    )
    label_year["y_next_year"] = label_year.groupby("permno")["delist"].shift(-1)

    yearly = yearly.merge(
        label_year[["permno", year_col, "y_next_year"]],
        on=["permno", year_col],
        how="left",
    )

    # Only keep rows with next year labels
    yearly = yearly[yearly["y_next_year"].notna()].copy()
    yearly["y_next_year"] = yearly["y_next_year"].astype(int)

    # Standardize column name expectations
    if year_col != "public_year":
        yearly = yearly.rename(columns={year_col: "public_year"})
    return yearly


def company_level_split_by_ipo(
    df: pd.DataFrame,
    company_col: str = "permno",
    ipo_col: Optional[str] = "ipodate",
    year_col: str = "public_year",
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> pd.Series:
    """Company-level split: assign all years of each company to train/val/test based on IPO time (or first year if unavailable).
    Ensures "all years of the same company are in only one set", and stratifies by company chronological order.
    Returns pd.Series of same length as df with values train/val/test.
    """
    r_train, r_val, r_test = ratios
    if not np.isclose(r_train + r_val + r_test, 1.0):
        raise ValueError("ratios must sum to 1.0")
    if company_col not in df.columns:
        raise ValueError(f"Missing grouping column {company_col}")
    if year_col not in df.columns:
        raise ValueError(f"Missing year column {year_col}")

    # Calculate company-level sort key
    comp = df[[company_col, year_col]].drop_duplicates().copy()
    if ipo_col and ipo_col in df.columns:
        ipo_key = (
            df[[company_col, ipo_col]]
            .dropna(subset=[ipo_col])
            .sort_values([company_col, ipo_col], kind="mergesort")
            .drop_duplicates(subset=[company_col], keep="first")
        )
        comp = comp.merge(ipo_key, on=company_col, how="left")
        # If some companies lack ipodate, use January of their earliest year as approximation
        comp["_sort_key"] = comp[ipo_col]
        if comp["_sort_key"].isna().any():
            fallback = (
                df.groupby(company_col)[year_col].min().rename("_first_year").to_frame()
            )
            comp = comp.merge(fallback, on=company_col, how="left")
            # Construct a comparable date: _first_year-01-01
            comp.loc[comp["_sort_key"].isna(), "_sort_key"] = pd.to_datetime(
                comp.loc[comp["_sort_key"].isna(), "_first_year"].astype(int).astype(str) + "-01-01",
                errors="coerce",
            )
    else:
        # No ipodate, use company's earliest year as sort key
        comp_key = df.groupby(company_col)[year_col].min().rename("_first_year").to_frame()
        comp = comp.merge(comp_key, on=company_col, how="left")
        comp["_sort_key"] = pd.to_datetime(
            comp["_first_year"].astype(int).astype(str) + "-01-01", errors="coerce"
        )

    comp = comp.drop_duplicates(subset=[company_col]).sort_values(
        ["_sort_key", company_col], kind="mergesort"
    )
    n_companies = len(comp)
    i1 = int(np.floor(n_companies * r_train))
    i2 = int(np.floor(n_companies * (r_train + r_val)))
    i1 = min(max(i1, 0), max(n_companies - 2, 0))
    i2 = min(max(i2, i1 + 1), max(n_companies - 1, 1))
    comp["split"] = "train"
    comp.iloc[i1:i2, comp.columns.get_loc("split")] = "val"
    comp.iloc[i2:, comp.columns.get_loc("split")] = "test"

    # Map back to original df
    tag_map = comp[[company_col, "split"]]
    out = df[[company_col]].merge(tag_map, on=company_col, how="left")["split"]
    out.index = df.index
    return out


def chronological_group_split(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> pd.Series:
    """Return split tag for each row: train/val/test (split by chronological order within each group)."""
    r_train, r_val, r_test = ratios
    if not np.isclose(r_train + r_val + r_test, 1.0):
        raise ValueError("ratios must sum to 1.0")

    splits = []
    for _, g in df.groupby(group_col, sort=False):
        # If public_month exists and using public_year as time column, perform two-level sort
        if time_col == "public_year" and "public_month" in g.columns:
            g_sorted = g.sort_values([time_col, "public_month"], kind="mergesort")
        else:
            g_sorted = g.sort_values(time_col, kind="mergesort")  # Stable sort
        n = len(g_sorted)
        i1 = int(np.floor(n * r_train))
        i2 = int(np.floor(n * (r_train + r_val)))
        # Boundary protection: ensure at least the last record stays in test (if sample size sufficient)
        i1 = min(max(i1, 0), max(n - 2, 0))
        i2 = min(max(i2, i1 + 1), max(n - 1, 1))
        tags = np.array(["train"] * n, dtype=object)
        tags[i1:i2] = "val"
        tags[i2:] = "test"
        splits.append(pd.Series(tags, index=g_sorted.index))
    return pd.concat(splits).reindex(df.index)


def build_feature_matrix(
    df: pd.DataFrame, features: Sequence[str]
) -> Tuple[pd.DataFrame, List[str]]:
    present = [c for c in features if c in df.columns]
    missing = [c for c in features if c not in df.columns]
    if not present:
        raise ValueError(
            f"None of the specified features exist in the data. Missing examples: {missing[:5]}. Please specify correct column names via --features."
        )
    if missing:
        print(f"[Warning] The following features were not found in data and will be ignored: {', '.join(missing)}")
    return df[present], present


def sanitize_feature_df(
    X: pd.DataFrame, drop_zero_variance: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """Convert features to numeric, replace inf/-inf with NaN, and optionally remove all-NaN/zero-variance columns."""
    X_numeric = X.apply(pd.to_numeric, errors="coerce")
    X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)

    dropped: List[str] = []
    all_nan_cols = X_numeric.columns[X_numeric.isna().all()].tolist()
    if all_nan_cols:
        dropped.extend(all_nan_cols)
        X_numeric = X_numeric.drop(columns=all_nan_cols)
        print(f"[Warning] The following feature columns are all missing and have been removed: {', '.join(all_nan_cols)}")

    if drop_zero_variance:
        zero_var_cols = [c for c in X_numeric.columns if X_numeric[c].nunique(dropna=True) <= 1]
        if zero_var_cols:
            dropped.extend(zero_var_cols)
            X_numeric = X_numeric.drop(columns=zero_var_cols)
            print(f"[Info] The following zero-variance/constant feature columns have been removed: {', '.join(zero_var_cols)}")

    return X_numeric, dropped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict delist using financial statements.")
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join("data", "final_table_merged.xlsx"),
        help="Data file path (Excel)",
    )
    parser.add_argument("--model_name", type=str, default="logistic_regression", help="Model name")
    parser.add_argument("--label_col", type=str, default="delist", help="Label column name (0/1 delist indicator)")
    parser.add_argument("--group_col", type=str, default="permno", help="Grouping column name (company id)")
    parser.add_argument("--time_col", type=str, default="public_year", help="Year column name")
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Comma-separated feature column names; if not provided, use built-in defaults and intersect with data columns",
    )
    # Model-related tunable parameters
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse of regularization strength for LogisticRegression (larger = weaker regularization)",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=1000,
        help="Maximum iterations for LogisticRegression",
    )
    parser.add_argument(
        "--predict_next_year",
        action="store_true",
        help="Enable yearly aggregation: use current year (aggregated) features to predict next year delist (generates y_next_year)",
        default=True,
    )
    parser.add_argument(
        "--cutoff_month",
        type=int,
        default=None,
        help="Cutoff month for yearly aggregation (only keep monthly observations <= cutoff_month in current year; default None means use full year)",
    )
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test set ratio")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--save_model_path", type=str, default=None, help="Model save path")
    parser.add_argument(
        "--drop_zero_variance",
        action="store_true",
        help="Remove zero-variance feature columns (constant columns) to improve robustness",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Read data
    df = pd.read_excel(args.data_path)

    # If "predict next year using current year" is enabled, first aggregate monthly data to yearly and construct y_next_year
    if bool(args.predict_next_year):
        df = build_yearly_from_monthly(df, cutoff_month=args.cutoff_month)
        # Override column names for yearly task
        args.label_col = "y_next_year"
        args.time_col = "public_year"
    else:
        if args.time_col is None:
            args.time_col = infer_time_column(df.columns)
        if args.label_col is None:
            args.label_col = infer_label_column(df.columns)

    # Select features
    if args.features:
        feat_list = [x.strip() for x in args.features.split(",") if x.strip()]
    else:
        feat_list = (
            default_yearly_aggregated_feature_list()
            if bool(args.predict_next_year)
            else default_feature_list()
        )

    X_df, selected_features = build_feature_matrix(df, feat_list)
    # Clean features: replace inf/abnormal values with NaN, remove all-NaN/zero-variance columns
    X_df, dropped_cols = sanitize_feature_df(X_df, drop_zero_variance=bool(args.drop_zero_variance))
    if dropped_cols:
        selected_features = [c for c in selected_features if c not in dropped_cols]
    y = df[args.label_col].astype(int).values

    # Construct split
    splits = chronological_group_split(
        df=df,
        group_col=args.group_col,
        time_col=args.time_col,
        ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
    )

    # If yearly prediction enabled, prefer "company-level (by IPO/first year)" split to avoid same company across sets
    if bool(args.predict_next_year):
        splits = company_level_split_by_ipo(
            df=df,
            company_col=args.group_col,
            ipo_col="ipodate" if "ipodate" in df.columns else None,
            year_col=args.time_col,
            ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
        )

    # Split data
    train_idx = splits == "train"
    val_idx = splits == "val"
    test_idx = splits == "test"

    X_train = X_df.loc[train_idx].values
    y_train = y[train_idx.values]
    X_val = X_df.loc[val_idx].values
    y_val = y[val_idx.values]
    X_test = X_df.loc[test_idx].values
    y_test = y[test_idx.values]

    # Training set class check: if only single class, try to introduce companies with positive class from val/test
    def _has_two_classes(arr: np.ndarray) -> bool:
        u = np.unique(arr)
        return u.size >= 2

    if not _has_two_classes(y_train):
        # Find which companies in val/test have positive class
        company_col = args.group_col
        need_pos = 1 if (len(np.unique(y_train)) == 1 and np.unique(y_train)[0] == 0) else 0
        # Prefer searching from val
        candidate_val_companies = df.loc[val_idx & (df[args.label_col] == 1), company_col].unique().tolist()
        candidate_test_companies = df.loc[test_idx & (df[args.label_col] == 1), company_col].unique().tolist()
        moved = False
        for comp_list in [candidate_val_companies, candidate_test_companies]:
            if comp_list:
                comp_to_move = comp_list[0]
                # Mark all samples of this company as train
                splits.loc[(df[company_col] == comp_to_move)] = "train"
                train_idx = splits == "train"
                val_idx = splits == "val"
                test_idx = splits == "test"
                X_train = X_df.loc[train_idx].values
                y_train = y[train_idx.values]
                X_val = X_df.loc[val_idx].values
                y_val = y[val_idx.values]
                X_test = X_df.loc[test_idx].values
                y_test = y[test_idx.values]
                if _has_two_classes(y_train):
                    moved = True
                    print(f"[Info] Training set contains only single class, moved company {comp_to_move} from validation/test to training to ensure trainability.")
                    break
        if not moved and not _has_two_classes(y_train):
            raise ValueError("Training set contains only single class and no positive class companies found in validation/test sets for adjustment. Please check data or modify split ratios.")

    # Model construction
    if args.model_name != "logistic_regression":
        raise ValueError("Currently only model_name=logistic_regression is supported")
    model = build_model_from_params(
        {
            "random_state": args.random_state,
            "C": float(args.C),
            "max_iter": int(args.max_iter),
        }
    )

    # Pre-training info: sample size and class distribution, parameters and pipeline
    def _dist(name: str, y_arr: np.ndarray) -> str:
        if y_arr.size == 0:
            return f"{name}: 0"
        vals, cnts = np.unique(y_arr, return_counts=True)
        parts = [f"{int(v)}={int(c)}" for v, c in zip(vals, cnts)]
        return f"{name}: n={y_arr.size} ({', '.join(parts)})"

    print(_dist("Train", y_train))
    print(_dist("Val", y_val))
    print(_dist("Test", y_test))
    try:
        print("Model parameters:", model.get_params())
    except Exception:
        pass
    try:
        # Briefly print pipeline step names (components have many internal parameters, keep it simple)
        from pprint import pprint
        if getattr(model, "pipeline", None) is not None:
            print("Pipeline steps:", [name for name, _ in model.pipeline.steps])
    except Exception:
        pass

    # Training
    print("[Training] Starting model training...")
    model.fit(X_train, y_train)
    print("[Training] Training completed")

    # Evaluation
    val_metrics = model.evaluate(X_val, y_val)
    test_metrics = model.evaluate(X_test, y_test)

    print("Validation set metrics:", val_metrics)
    print("Test set metrics:", test_metrics)
    print(f"Features used ({len(selected_features)}): {', '.join(selected_features)}")

    # Display current model parameters (coefficients and intercept)
    try:
        clf = model.pipeline.named_steps["clf"]  # type: ignore
        coef = getattr(clf, "coef_", None)
        intercept = getattr(clf, "intercept_", None)
        if coef is not None:
            coef = np.asarray(coef).reshape(-1)
            # Align feature names
            feature_coefs = list(zip(selected_features, coef))
            # Top 15 by absolute value
            topk = sorted(feature_coefs, key=lambda x: abs(x[1]), reverse=True)[:15]
            print("Intercept:", float(intercept[0]) if intercept is not None else "N/A")
            print("Top-15 weights (sorted by |coef|):")
            for name, w in topk:
                print(f"  {name}: {w:.6f}")
    except Exception:
        pass

    # Save model
    if args.save_model_path:
        os.makedirs(os.path.dirname(args.save_model_path), exist_ok=True)
        model.save(args.save_model_path)
        print(f"Model saved to: {args.save_model_path}")


if __name__ == "__main__":
    main()


