import pandas as pd

finance_path = "/Users/liyansong/Desktop/finance_1974_2024_3_5y.xlsx"
ipo_delist_path = "/Users/liyansong/Desktop/ipo_delist_within5y_1974_2024_3.xlsx"

finance_df = pd.read_excel(finance_path, sheet_name=0)
ipo_delist_df = pd.read_excel(ipo_delist_path, sheet_name=0)

finance_df["public_date"] = pd.to_datetime(finance_df["public_date"], errors="coerce")
finance_df["public_year"] = finance_df["public_date"].dt.year
finance_df["public_month"] = finance_df["public_date"].dt.month

ipo_delist_df["ipo_date"] = pd.to_datetime(ipo_delist_df["ipo_date"], errors="coerce")
ipo_delist_df["delist_date"] = pd.to_datetime(ipo_delist_df["delist_date"], errors="coerce")
ipo_small = ipo_delist_df[["permno", "ipo_date", "delist_date"]].copy()

finance_df = pd.merge(finance_df, ipo_small, how="left", on= "permno")

finance_df["delist"] = finance_df["ipo_date_y"].notna().astype(int)

finance_df["delist_year"] = 0
mask = finance_df["delist"] == 1
days_diff = (finance_df.loc[mask, "delist_date"] - finance_df.loc[mask, "ipo_date_y"]).dt.days
years = (days_diff / 365).round(0).astype(int)
finance_df.loc[mask, "delist_year"] = years

finance_merged = finance_df.drop(columns=["ipo_date_y", "delist_date", "public_date"])

output_path = "/Users/liyansong/Desktop/final_table_3.xlsx"
finance_merged.to_excel(output_path, index=False)