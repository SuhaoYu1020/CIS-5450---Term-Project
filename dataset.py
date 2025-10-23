"""
从 WRDS 抓取 CRSP IPO 或退市（Delist）数据，并保存到本地。

核心逻辑：
- IPO 模式：以每个 permno 的最早 namedt 作为 IPO 日期
- 退市模式（--delist）：基于 dsedelist/msedelist 的 dlstdt 作为退市日期，并在该日期
  区间连接 names 表族以获取 comnam/ticker/exchcd/shrcd（namedt<=dlstdt<=nameendt）
- 支持起止日期（--start 与 --end，YYYY-MM-DD）
- 支持 NYSE/AMEX/NASDAQ
- 自动在多个库表中回退
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Tuple

import pandas as pd

try:
    import wrds  # type: ignore
except Exception as import_error:  # pragma: no cover
    raise SystemExit(
        "未能导入 wrds 库。请先运行: pip install wrds pandas openpyxl"
    ) from import_error


EXCHANGE_NAME_TO_CODE = {
    "NYSE": 1,
    "AMEX": 2,
    "NASDAQ": 3,
}

TABLE_FALLBACKS: List[Tuple[str, str]] = [
    ("crspa", "dsenames"),
    ("crspa", "dse"),
    ("crsp", "dsenames"),
    ("crsp", "dse"),
    ("crsp", "stocknames"),
]

DELIST_FALLBACKS: List[Tuple[str, str]] = [
    ("crspa", "dsedelist"),
    ("crspa", "msedelist"),
    ("crsp", "dsedelist"),
    ("crsp", "msedelist"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 WRDS 抓取 CRSP IPO / Delist 数据")
    parser.add_argument("--start", required=True, help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="结束日期 YYYY-MM-DD")
    parser.add_argument("--exchange", type=str, default="NYSE", help="交易所（NYSE/AMEX/NASDAQ 或 1/2/3）")
    parser.add_argument("--limit", type=int, default=None, help="仅返回前 N 行")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")
    parser.add_argument("--username", type=str, default=os.environ.get("WRDS_USERNAME") or "suhaoyu")
    parser.add_argument("--password", type=str, default=os.environ.get("WRDS_PASSWORD") or "Yusuhao031020.")
    parser.add_argument("--delist", action="store_true", help="获取退市记录（按最晚日期）")
    return parser.parse_args()


def normalize_exchange(exchange: str) -> List[int]:
    if not exchange:
        return [1, 2, 3]
    exchange_clean = str(exchange).strip().upper()
    if exchange_clean.isdigit():
        return [int(exchange_clean)]
    if exchange_clean in EXCHANGE_NAME_TO_CODE:
        return [EXCHANGE_NAME_TO_CODE[exchange_clean]]
    if "," in exchange_clean:
        return [EXCHANGE_NAME_TO_CODE[x.strip()] for x in exchange_clean.split(",") if x.strip() in EXCHANGE_NAME_TO_CODE]
    raise ValueError(f"无法识别的交易所: {exchange}")


def resolve_date_range(start_date_str: Optional[str], end_date_str: Optional[str]) -> Tuple[str, str]:
    if not start_date_str or not end_date_str:
        raise SystemExit("必须提供 --start 和 --end，格式为 YYYY-MM-DD")
    return start_date_str, end_date_str


# === IPO 查询 ===
def build_ipo_sql(library: str, table: str, exch_codes: List[int], start_date: str, end_date: str, limit: Optional[int] = None) -> str:
    exch_str = ",".join(str(x) for x in exch_codes)
    sql = f"""
    SELECT DISTINCT ON (permno)
        permno,
        comnam,
        ticker,
        namedt AS ipo_date,
        exchcd,
        shrcd
    FROM {library}.{table}
    WHERE shrcd IN (10, 11)
      AND exchcd IN ({exch_str})
    ORDER BY permno, namedt
    """
    sql = f"""
    SELECT * FROM ({sql}) AS base
    WHERE ipo_date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY ipo_date
    """
    if limit:
        sql += f"\nLIMIT {int(limit)}"
    sql += ";"
    return sql


def build_delist_sql_dsedelist(
    delist_library: str,
    delist_table: str,
    names_library: str,
    names_table: str,
    exch_codes: List[int],
    start_date: str,
    end_date: str,
    limit: Optional[int] = None,
) -> str:
    exch_str = ",".join(str(x) for x in exch_codes)
    delist_filtered = f"""
        SELECT permno, dlstdt, dlstcd
        FROM {delist_library}.{delist_table}
        WHERE dlstdt BETWEEN '{start_date}' AND '{end_date}'
    """
    sql = f"""
    SELECT DISTINCT ON (d.permno)
        d.permno,
        n.comnam,
        n.ticker,
        d.dlstdt AS delist_date,
        n.exchcd,
        n.shrcd,
        d.dlstcd
    FROM ({delist_filtered}) AS d
    JOIN {names_library}.{names_table} AS n
      ON n.permno = d.permno
     AND (n.namedt IS NULL OR n.namedt <= d.dlstdt)
     AND (n.nameendt IS NULL OR n.nameendt >= d.dlstdt)
    WHERE n.shrcd IN (10, 11)
      AND n.exchcd IN ({exch_str})
    ORDER BY d.permno, d.dlstdt DESC
    """
    if limit:
        sql += f"\nLIMIT {int(limit)}"
    sql += ";"
    return sql


def try_query_with_fallbacks(db: "wrds.Connection", exch_codes: List[int], start_date: str, end_date: str, limit: Optional[int], delist: bool = False) -> Tuple[pd.DataFrame, Tuple[str, str]]:
    last_error = None
    if not delist:
        for library, table in TABLE_FALLBACKS:
            sql = build_ipo_sql(library, table, exch_codes, start_date, end_date, limit)
            try:
                print(f"尝试查询 IPO {library}.{table} ...")
                df = db.raw_sql(sql)
                if not df.empty:
                    return df, (library, table)
            except Exception as err:
                last_error = err
                continue
    else:
        for d_lib, d_tbl in DELIST_FALLBACKS:
            for n_lib, n_tbl in TABLE_FALLBACKS:
                sql = build_delist_sql_dsedelist(d_lib, d_tbl, n_lib, n_tbl, exch_codes, start_date, end_date, limit)
                try:
                    print(f"尝试查询 退市 {d_lib}.{d_tbl} JOIN {n_lib}.{n_tbl} ...")
                    df = db.raw_sql(sql)
                    if not df.empty:
                        return df, (f"{d_lib}.{d_tbl}", f"{n_lib}.{n_tbl}")
                except Exception as err:
                    last_error = err
                    continue
    if last_error:
        raise last_error
    raise RuntimeError("所有库表都查询失败。")


def save_dataframe(df: pd.DataFrame, output_path: str) -> str:
    output_ext = os.path.splitext(output_path)[1].lower()
    if output_ext == ".xlsx":
        try:
            df.to_excel(output_path, index=False)
            return output_path
        except Exception:
            csv_path = os.path.splitext(output_path)[0] + ".csv"
            df.to_csv(csv_path, index=False)
            return csv_path
    else:
        df.to_csv(output_path, index=False)
        return output_path


def main() -> None:
    args = parse_args()
    start_date, end_date = resolve_date_range(args.start, args.end)
    exch_codes = normalize_exchange(args.exchange)

    tag = "delist" if args.delist else "ipo"
    exch_tag = "-".join(str(x) for x in exch_codes)
    output_path = args.output or f"{tag}_{start_date[:4]}_{end_date[:4]}_{exch_tag}.xlsx"

    print(f"连接 WRDS ...")
    db = wrds.Connection(wrds_username=args.username, wrds_password=args.password)

    df, (library, table) = try_query_with_fallbacks(
        db, exch_codes, start_date, end_date, args.limit, delist=args.delist
    )

    db.close()
    print(f"已从 {library}.{table} 获得 {len(df)} 行结果")

    saved = save_dataframe(df, output_path)
    print(f"结果保存至：{saved}")


if __name__ == "__main__":
    main()
