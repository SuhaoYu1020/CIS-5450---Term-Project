from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

def _read_excel_safely(path: str) -> pd.DataFrame:
    """
    读取 Excel 或 CSV（如果 Excel 失败则尝试 CSV）。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".xlsx":
        try:
            return pd.read_excel(path)
        except Exception:
            # 回退到同名 CSV
            csv_path = os.path.splitext(path)[0] + ".csv"
            if os.path.exists(csv_path):
                return pd.read_csv(csv_path)
            raise
    if ext == ".csv":
        return pd.read_csv(path)
    # 其他后缀也尝试用 pandas 读
    return pd.read_excel(path)


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.date


def _standardize_ipo_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    将 IPO 表标准化为列: permno, comnam, ticker, ipo_date, exchcd, shrcd
    兼容列名: {permno, comnam, ticker, namedt/ipo_date, exchcd, shrcd}
    """
    columns_lower = {c.lower(): c for c in df.columns}
    col = lambda name: columns_lower.get(name)

    out = pd.DataFrame()
    if col("permno") is None:
        raise ValueError("IPO 表缺少 permno 列")
    out["permno"] = df[col("permno")]

    # 公司名
    name_col = col("comnam") or col("comnamename") or col("companyname")
    out["comnam"] = df[name_col] if name_col else None

    # 代码
    ticker_col = col("ticker") or col("ncusip") or col("cusip8")
    out["ticker"] = df[ticker_col] if ticker_col else None

    # IPO 日期
    ipo_col = col("ipo_date") or col("namedt")
    if ipo_col is None:
        raise ValueError("IPO 表缺少 ipo_date/namedt 列")
    out["ipo_date"] = _to_datetime(df[ipo_col])

    # exchcd/shrcd（可选）
    if col("exchcd"):
        out["exchcd"] = df[col("exchcd")]
    if col("shrcd"):
        out["shrcd"] = df[col("shrcd")]

    return out


def _standardize_delist_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    将 Delist 表标准化为列: permno, comnam, ticker, delist_date, exchcd, shrcd
    兼容列名: {permno, comnam, ticker, nameendt/delist_date/dlstdt, exchcd, shrcd}
    """
    columns_lower = {c.lower(): c for c in df.columns}
    col = lambda name: columns_lower.get(name)

    out = pd.DataFrame()
    if col("permno") is None:
        raise ValueError("Delist 表缺少 permno 列")
    out["permno"] = df[col("permno")]

    # 公司名
    name_col = col("comnam") or col("comnamename") or col("companyname")
    out["comnam_delist"] = df[name_col] if name_col else None

    # 代码
    ticker_col = col("ticker") or col("ncusip") or col("cusip8")
    out["ticker_delist"] = df[ticker_col] if ticker_col else None

    # 退市日期优先顺序: delist_date > nameendt > dlstdt
    date_col = col("delist_date") or col("nameendt") or col("dlstdt")
    if date_col is None:
        raise ValueError("Delist 表缺少 delist_date/nameendt/dlstdt 列")
    out["delist_date"] = _to_datetime(df[date_col])

    # exchcd/shrcd（可选）
    if col("exchcd"):
        out["exchcd_delist"] = df[col("exchcd")]
    if col("shrcd"):
        out["shrcd_delist"] = df[col("shrcd")]

    return out


def _within_years(ipo: pd.Series, delist: pd.Series, years: int) -> pd.Series:
    ipo_dt = pd.to_datetime(ipo, errors="coerce")
    delist_dt = pd.to_datetime(delist, errors="coerce")
    return (delist_dt.notna()) & (ipo_dt.notna()) & (delist_dt <= ipo_dt + pd.DateOffset(years=years)) & (delist_dt >= ipo_dt)


def process(
    ipo_path: str = os.path.join(DATA_DIR, "ipo_1974_2024_2.xlsx"),
    delist_path: str = os.path.join(DATA_DIR, "delist_1974_2024_2.xlsx"),
    output_path: Optional[str] = None,
    years: int = 5,
) -> pd.DataFrame:
    """
    读取 IPO 与 Delist 数据，按 `permno` 合并，筛选 IPO 后 `years` 年内退市的公司。

    返回列包含：permno, comnam, ticker, ipo_date, delist_date, 以及可用的 exchcd/shrcd。
    若提供 output_path，保存为 .xlsx（若失败回退 .csv）。
    """
    ipo_df_raw = _read_excel_safely(ipo_path)
    delist_df_raw = _read_excel_safely(delist_path)

    ipo_df = _standardize_ipo_columns(ipo_df_raw)
    delist_df = _standardize_delist_columns(delist_df_raw)

    # 按 permno 合并，每个 permno 取一条 IPO（假设输入已去重）与一条 Delist（假设输入已去重）
    merged = pd.merge(ipo_df, delist_df, on="permno", how="inner")

    # 统一公司名/代码优先 IPO 端
    merged["comnam_final"] = merged["comnam"].fillna(merged.get("comnam_delist"))
    merged["ticker_final"] = merged["ticker"].fillna(merged.get("ticker_delist"))

    # 筛选 IPO 至多 years 年内退市
    mask = _within_years(merged["ipo_date"], merged["delist_date"], years)
    filtered = merged.loc[mask].copy()

    # 输出所需列
    cols = [
        "permno",
        "comnam_final",
        "ticker_final",
        "ipo_date",
        "delist_date",
    ]
    if "exchcd" in filtered.columns:
        cols.append("exchcd")
    if "shrcd" in filtered.columns:
        cols.append("shrcd")

    result = filtered[cols].rename(columns={
        "comnam_final": "comnam",
        "ticker_final": "ticker",
    }).sort_values(["ipo_date", "delist_date", "permno"]).reset_index(drop=True)

    if output_path:
        ext = os.path.splitext(output_path)[1].lower()
        try:
            if ext == ".xlsx":
                result.to_excel(output_path, index=False)
            else:
                result.to_csv(output_path, index=False)
        except Exception:
            # 回退到 CSV
            fallback = os.path.splitext(output_path)[0] + ".csv"
            result.to_csv(fallback, index=False)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="筛选 IPO 后 N 年内退市的公司")
    parser.add_argument(
        "--ipo-path",
        dest="ipo_path",
        type=str,
        default=os.path.join(DATA_DIR, "ipo_1974_2024_2.xlsx"),
        help="IPO 数据文件路径（.xlsx 或 .csv）",
    )
    parser.add_argument(
        "--delist-path",
        dest="delist_path",
        type=str,
        default=os.path.join(DATA_DIR, "delist_1974_2024_2.xlsx"),
        help="退市数据文件路径（.xlsx 或 .csv）",
    )
    parser.add_argument(
        "--output-path",
        dest="output_path",
        type=str,
        default=os.path.join(DATA_DIR, "ipo_delist_within5y_1974_2024_2.xlsx"),
        help="输出保存路径（.xlsx 或 .csv，可选）",
    )
    parser.add_argument(
        "--years",
        dest="years",
        type=int,
        default=5,
        help="IPO 后多少年内退市（默认 5）",
    )
    args = parser.parse_args()

    df_out = process(
        ipo_path=args.ipo_path,
        delist_path=args.delist_path,
        output_path=args.output_path,
        years=args.years,
    )
    print(f"筛选结果行数: {len(df_out)}")
    print(df_out.head(10))

