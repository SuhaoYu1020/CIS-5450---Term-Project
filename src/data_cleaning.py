import argparse
import os
import pandas as pd


def clean_finance(finance_path: str, ipo_delist_path: str, output_path: str) -> None:
    finance_df = pd.read_excel(finance_path, sheet_name=0)
    ipo_delist_df = pd.read_excel(ipo_delist_path, sheet_name=0)

    finance_df["public_date"] = pd.to_datetime(finance_df["public_date"], errors="coerce")
    finance_df["public_year"] = finance_df["public_date"].dt.year
    finance_df["public_month"] = finance_df["public_date"].dt.month

    ipo_delist_df["ipo_date"] = pd.to_datetime(ipo_delist_df["ipo_date"], errors="coerce")
    ipo_delist_df["delist_date"] = pd.to_datetime(ipo_delist_df["delist_date"], errors="coerce")
    ipo_small = ipo_delist_df[["permno", "ipo_date", "delist_date"]].copy()

    finance_df = pd.merge(finance_df, ipo_small, how="left", on="permno")

    finance_df["delist"] = finance_df["ipo_date_y"].notna().astype(int)

    finance_df["delist_year"] = 0
    mask = finance_df["delist"] == 1
    days_diff = (finance_df.loc[mask, "delist_date"] - finance_df.loc[mask, "ipo_date_y"]).dt.days
    years = (days_diff / 365).round(0).astype(int)
    finance_df.loc[mask, "delist_year"] = years

    # Only mark delist=1 in the actual delist year (IPO year + delist_year), all other years are 0
    event_year = finance_df["ipo_date_y"].dt.year + finance_df["delist_year"].astype("Int64")
    finance_df["delist"] = 0
    finance_df.loc[(finance_df["ipo_date_y"].notna()) & (finance_df["public_year"] == event_year), "delist"] = 1

    finance_merged = finance_df.drop(columns=["ipo_date_y", "delist_date", "public_date"])

    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    finance_merged.to_excel(output_path, index=False)


def merge_final_tables(input_paths: list[str], output_path: str) -> None:
    # Read three cleaned sub-tables and merge
    dfs = [pd.read_excel(p) for p in input_paths]
    merged_df = pd.concat(dfs, ignore_index=True)

    # Remove all-empty rows and columns
    merged_df = merged_df.dropna(axis=0, how="all").dropna(axis=1, how="all")

    # Strip leading/trailing spaces from column names
    merged_df = merged_df.rename(columns=lambda x: x.strip())

    # Key field types
    if "permno" in merged_df.columns:
        merged_df["permno"] = merged_df["permno"].astype("Int64")
    if "public_year" in merged_df.columns:
        merged_df["public_year"] = merged_df["public_year"].astype("Int64")

    # Sort (ensure chronological order by permno -> public_year -> public_month)
    sort_cols = []
    if "permno" in merged_df.columns:
        sort_cols.append("permno")
    elif "gvkey" in merged_df.columns:
        sort_cols.append("gvkey")
    if "public_year" in merged_df.columns:
        sort_cols.append("public_year")
    if "public_month" in merged_df.columns:
        sort_cols.append("public_month")
    if sort_cols:
        merged_df = merged_df.sort_values(sort_cols)

    # Group-wise filling: group by entity (must have permno; otherwise fallback to global), bfill then ffill to ensure trailing missing values can use nearest previous value
    group_cols = []
    if "permno" in merged_df.columns:
        group_cols = ["permno"]
    elif "gvkey" in merged_df.columns:
        group_cols = ["gvkey"]
    if group_cols:
        merged_df = merged_df.groupby(group_cols, group_keys=False).apply(lambda g: g.bfill().ffill())
    else:
        merged_df = merged_df.bfill().ffill()

    # Remove redundant columns (if exist)
    drop_cols = [
        "prcc_c",
        "pstk",
        "pe_op_dil",
        "shrout",
        "total_assets",
        "total_liabilities",
        "total_asset_growth",
    ]
    existing_drop = [c for c in drop_cols if c in merged_df.columns]
    if existing_drop:
        merged_df = merged_df.drop(columns=existing_drop)

    # Remove rows with missing key fields (only remove missing values for columns that exist in data)
    needed_cols = [
        "tobinq",
        "revenue_growth",
        "quick_ratio",
        "curr_ratio",
        "inv_turn",
        "at_turn",
        "de_ratio",
        "pe_op_basic",
        "roa",
        "roe",
        "npm",
    ]
    subset_cols = [c for c in needed_cols if c in merged_df.columns]
    if subset_cols:
        merged_df = merged_df.dropna(subset=subset_cols)

    # Output
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    merged_df.to_excel(output_path, index=False)


def build_default_paths(index: int) -> tuple[str, str, str]:
    # Use current project root as default search location to avoid absolute paths from personal desktop
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(project_root, "data")
    finance_path = os.path.join(data_dir, f"finance_1974_2024_{index}_5y.xlsx")
    ipo_delist_path = os.path.join(data_dir, f"ipo_delist_within5y_1974_2024_{index}.xlsx")
    # Output to data directory, maintain consistency with existing file naming in repository
    output_path = os.path.join(data_dir, f"final_table_{index}.xlsx")
    return finance_path, ipo_delist_path, output_path


def build_merge_paths() -> tuple[list[str], str]:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(project_root, "data")
    inputs = [os.path.join(data_dir, f"final_table_{i}.xlsx") for i in (1, 2, 3)]
    output = os.path.join(data_dir, "final_table_merged.xlsx")
    return inputs, output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified data cleaning script: generates final_table_{i}.xlsx; supports --merge to combine 1/2/3 into final_table_merged.xlsx"
    )
    parser.add_argument(
        "--index",
        type=int,
        choices=[1, 2, 3],
        help="Choose 1/2/3 to use default input/output paths within project",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge data/final_table_1/2/3.xlsx and output data/final_table_merged.xlsx",
    )
    parser.add_argument(
        "--finance-path",
        type=str,
        help="Absolute path to financial data Excel (overrides --index default path)",
    )
    parser.add_argument(
        "--ipo-delist-path",
        type=str,
        help="Absolute path to IPO/delist mapping Excel (overrides --index default path)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Absolute path to output Excel (overrides --index default path)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # When only merging, skip required parameter validation for cleaning
    if not args.merge:
        if args.index is None and (not args.finance_path or not args.ipo_delist_path or not args.output_path):
            raise SystemExit(
                "Please provide --index 1|2|3, or provide --finance-path, --ipo-delist-path, and --output-path together; if only merging, use --merge."
            )

    if args.merge:
        input_paths, merge_output = build_merge_paths()
        merge_final_tables(input_paths, merge_output)
        return

    if args.index is not None:
        finance_path, ipo_delist_path, output_path = build_default_paths(args.index)
    else:
        finance_path, ipo_delist_path, output_path = args.finance_path, args.ipo_delist_path, args.output_path

    clean_finance(finance_path, ipo_delist_path, output_path)


if __name__ == "__main__":
    main()


