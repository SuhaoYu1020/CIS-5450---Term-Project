"""
从 WRDS 抓取 CRSP IPO 或退市（Delist）数据，并保存到本地。

核心逻辑：
- IPO 模式：
  * 使用 Compustat 的 ipodate 字段（--use-compustat，默认）
  * 或使用 CRSP 的每个 permno 的最早 namedt 作为 IPO 日期（--use-crsp）
- 退市模式（--delist）：基于 dsedelist/msedelist 的 dlstdt 作为退市日期，并在该日期
  区间连接 names 表族以获取 comnam/ticker/exchcd/shrcd（namedt<=dlstdt<=nameendt）
- 支持起止日期（--start 与 --end，YYYY-MM-DD）
- 支持 NYSE/AMEX/NASDAQ
- 自动在多个库表中回退
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict, Any

import pandas as pd
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

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

COMPUSTAT_FALLBACKS: List[Tuple[str, str]] = [
    ("comp", "company"),
]

FINRATIO_FALLBACKS: List[Tuple[str, str]] = [
    ("wrdsapps", "firm_ratio")
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
    parser.add_argument("--use-crsp", action="store_true", help="使用 CRSP 最早记录而非 Compustat ipodate")
    parser.add_argument("--fetch-financials", action="store_true", help="根据 IPO 文件拉取财务报表数据（ROA/ROE）")
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


# === Compustat IPO 查询（使用真实的 ipodate 字段）===
def build_compustat_ipo_sql(library: str, table: str, exch_codes: List[int], start_date: str, end_date: str, limit: Optional[int] = None) -> str:
    """
    从 Compustat 查询真正有 ipodate 字段的公司
    主要从 comp.company 或 comp.funda 获取 ipodate
    然后 JOIN CRSP 表获取交易所和股票代码信息
    """
    exch_str = ",".join(str(x) for x in exch_codes)

    # 尝试从 comp.company 获取 ipodate
    if table == "company":
        sql = f"""
        SELECT
            c.gvkey,
            c.conm AS comnam,
            crsp.ticker AS ticker,
            c.ipodate AS ipo_date,
            crsp.exchcd,
            crsp.shrcd,
            crsp.permno
        FROM {library}.{table} AS c
        LEFT JOIN crsp.ccmxpf_lnkhist AS lnk
          ON c.gvkey = lnk.gvkey
         AND lnk.linktype IN ('LU', 'LC')
         AND lnk.linkprim IN ('P', 'C')
        LEFT JOIN crsp.stocknames AS crsp
          ON lnk.lpermno = crsp.permno
         AND crsp.namedt <= c.ipodate
         AND (crsp.nameenddt >= c.ipodate OR crsp.nameenddt IS NULL)
        WHERE c.ipodate IS NOT NULL
          AND c.ipodate BETWEEN '{start_date}' AND '{end_date}'
          AND crsp.shrcd IN (10, 11)
          AND crsp.exchcd IN ({exch_str})
        ORDER BY c.ipodate
        """
    else:
        # 从 funda 获取 ipodate（年度数据）
        sql = f"""
        SELECT DISTINCT ON (f.gvkey)
            f.gvkey,
            f.conm AS comnam,
            crsp.ticker AS ticker,
            f.ipodate AS ipo_date,
            crsp.exchcd,
            crsp.shrcd,
            crsp.permno
        FROM {library}.{table} AS f
        LEFT JOIN crsp.ccmxpf_lnkhist AS lnk
          ON f.gvkey = lnk.gvkey
         AND lnk.linktype IN ('LU', 'LC')
         AND lnk.linkprim IN ('P', 'C')
        LEFT JOIN crsp.stocknames AS crsp
          ON lnk.lpermno = crsp.permno
         AND crsp.namedt <= f.ipodate
         AND (crsp.nameenddt >= f.ipodate OR crsp.nameenddt IS NULL)
        WHERE f.ipodate IS NOT NULL
          AND f.ipodate BETWEEN '{start_date}' AND '{end_date}'
          AND crsp.shrcd IN (10, 11)
          AND crsp.exchcd IN ({exch_str})
        ORDER BY f.gvkey, f.ipodate
        """

    if limit:
        sql += f"\nLIMIT {int(limit)}"
    sql += ";"
    return sql


# === CRSP IPO 查询（使用最早 namedt）===
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
     AND (n.nameenddt IS NULL OR n.nameenddt >= d.dlstdt)
    WHERE n.shrcd IN (10, 11)
      AND n.exchcd IN ({exch_str})
    ORDER BY d.permno, d.dlstdt DESC
    """
    if limit:
        sql += f"\nLIMIT {int(limit)}"
    sql += ";"
    return sql


def try_query_with_fallbacks(db: "wrds.Connection", exch_codes: List[int], start_date: str, end_date: str, limit: Optional[int], delist: bool = False, use_crsp: bool = False) -> Tuple[pd.DataFrame, Tuple[str, str]]:
    last_error = None
    if not delist:
        # 默认使用 Compustat ipodate，除非指定 --use-crsp
        if not use_crsp:
            print("使用 Compustat ipodate 字段查询 IPO 数据...")
            for library, table in COMPUSTAT_FALLBACKS:
                sql = build_compustat_ipo_sql(library, table, exch_codes, start_date, end_date, limit)
                try:
                    print(f"尝试查询 Compustat IPO {library}.{table} ...")
                    df = db.raw_sql(sql)
                    if not df.empty:
                        print(f"成功！找到 {len(df)} 条有 ipodate 的记录")
                        return df, (library, table)
                except Exception as err:
                    last_error = err
                    print(f"失败: {err}")
                    continue
        else:
            print("使用 CRSP 最早记录日期查询 IPO 数据...")
            for library, table in TABLE_FALLBACKS:
                sql = build_ipo_sql(library, table, exch_codes, start_date, end_date, limit)
                try:
                    print(f"尝试查询 CRSP IPO {library}.{table} ...")
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


# === 财务报表数据拉取功能 ===
def read_ipo_file(start_date: str, end_date: str, exchange: str) -> pd.DataFrame:
    """
    根据参数读取对应的 IPO Excel 文件
    文件格式: ipo_{start_year}_{end_year}_{exchange_code}.xlsx
    """
    start_year = start_date[:4]
    end_year = end_date[:4]
    exch_codes = normalize_exchange(exchange)
    exch_tag = "-".join(str(x) for x in exch_codes)

    filename = f"ipo_{start_year}_{end_year}_{exch_tag}.xlsx"

    if not os.path.exists(filename):
        raise FileNotFoundError(f"未找到 IPO 文件: {filename}")

    print(f"读取 IPO 文件: {filename}")
    df = pd.read_excel(filename)

    # 确保有必需的列
    required_cols = ["permno", "ipo_date"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"IPO 文件缺少必需的列: {col}")

    return df


def build_financial_ratios_sql(
    library: str,
    table: str,
    permno: int,
    start_date: str,
    end_date: str
) -> str:
    """
    构建查询财务比率的 SQL（ROA 和 ROE）
    从 wrdsapps.firm_ratio 表查询
    """
    sql = f"""
    SELECT
        permno,
        public_date,
        roa,
        roe
    FROM {library}.{table}
    WHERE permno = {permno}
      AND public_date >= '{start_date}'
      AND public_date <= '{end_date}'
    ORDER BY public_date
    """
    return sql


def get_continuous_years(df: pd.DataFrame, max_years: int = 5) -> pd.DataFrame:
    """
    获取连续的年度数据，最多 max_years 年
    如果出现数据断裂（年份不连续），则在断裂处终止
    """
    if df.empty:
        return df

    # 确保按日期排序
    df = df.sort_values("public_date").copy()

    # 提取年份
    df["year"] = pd.to_datetime(df["public_date"]).dt.year

    # 获取唯一年份
    years = df["year"].unique()

    # 检查连续性
    continuous_years = [years[0]]
    for i in range(1, min(len(years), max_years)):
        if years[i] == continuous_years[-1] + 1:
            continuous_years.append(years[i])
        else:
            # 年份不连续，终止
            break

    # 只保留连续年份的数据
    result = df[df["year"].isin(continuous_years)].copy()
    result = result.drop(columns=["year"])

    return result


def fetch_financial_data_for_ipo(
    db: "wrds.Connection",
    ipo_df: pd.DataFrame
) -> pd.DataFrame:
    """
    为 IPO 文件中的每个公司拉取财务数据
    从 ipodate 开始，最多连续5年
    """
    all_results = []

    total_companies = len(ipo_df)
    print(f"开始为 {total_companies} 家公司拉取财务数据...")

    for idx, row in ipo_df.iterrows():
        permno = int(row["permno"])
        ipo_date = pd.to_datetime(row["ipo_date"])

        # 计算结束日期（IPO 日期后5年）
        end_date = ipo_date + timedelta(days=365 * 5)

        print(f"[{idx + 1}/{total_companies}] 处理 permno={permno}, IPO日期={ipo_date.date()}")

        # 尝试从不同的表查询
        df_financial = None
        for library, table in FINRATIO_FALLBACKS:
            try:
                sql = build_financial_ratios_sql(
                    library,
                    table,
                    permno,
                    ipo_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d")
                )
                df_temp = db.raw_sql(sql)

                if not df_temp.empty:
                    df_financial = df_temp
                    print(f"  从 {library}.{table} 找到 {len(df_temp)} 条记录")
                    break
            except Exception as err:
                print(f"  查询 {library}.{table} 失败: {err}")
                continue

        if df_financial is None or df_financial.empty:
            print(f"  未找到财务数据")
            continue

        # 获取连续年份的数据（最多5年）
        df_continuous = get_continuous_years(df_financial, max_years=5)

        if df_continuous.empty:
            print(f"  无连续年份数据")
            continue

        # 添加 IPO 相关信息
        df_continuous["ipo_date"] = ipo_date

        # 如果原始 IPO 文件有其他列（如 comnam, ticker 等），也添加进去
        for col in ipo_df.columns:
            if col not in df_continuous.columns and col != "permno":
                df_continuous[col] = row[col]

        print(f"  保留 {len(df_continuous)} 条连续数据")
        all_results.append(df_continuous)

    if not all_results:
        return pd.DataFrame()

    # 合并所有结果
    final_df = pd.concat(all_results, ignore_index=True)

    # 重新排列列顺序
    cols = ["permno", "ipo_date", "public_date", "roa", "roe"]
    # 添加其他列
    for col in final_df.columns:
        if col not in cols:
            cols.append(col)

    final_df = final_df[cols]

    return final_df


def main() -> None:
    args = parse_args()
    start_date, end_date = resolve_date_range(args.start, args.end)
    exch_codes = normalize_exchange(args.exchange)

    # 如果是拉取财务数据模式
    if args.fetch_financials:
        exch_tag = "-".join(str(x) for x in exch_codes)
        output_path = args.output or f"finance_{start_date[:4]}_{end_date[:4]}_{exch_tag}_5y.xlsx"

        print(f"财务数据拉取模式")

        # 读取 IPO 文件
        ipo_df = read_ipo_file(start_date, end_date, args.exchange)
        print(f"从 IPO 文件读取到 {len(ipo_df)} 家公司")

        # 连接 WRDS
        print(f"连接 WRDS ...")
        if args.password:
            os.environ["WRDS_PASSWORD"] = args.password
        db = wrds.Connection(wrds_username=args.username)

        # 拉取财务数据
        financial_df = fetch_financial_data_for_ipo(db, ipo_df)

        db.close()

        if financial_df.empty:
            print("未获取到任何财务数据")
            return

        print(f"共获取 {len(financial_df)} 条财务记录")

        # 保存结果
        saved = save_dataframe(financial_df, output_path)
        print(f"结果保存至：{saved}")
        return

    # 原有的 IPO/Delist 数据拉取模式
    tag = "delist" if args.delist else "ipo"
    exch_tag = "-".join(str(x) for x in exch_codes)
    output_path = args.output or f"{tag}_{start_date[:4]}_{end_date[:4]}_{exch_tag}.xlsx"

    print(f"连接 WRDS ...")
    # wrds.Connection 只接受 wrds_username，密码必须通过环境变量 WRDS_PASSWORD 提供
    if args.password:
        os.environ["WRDS_PASSWORD"] = args.password
    db = wrds.Connection(wrds_username=args.username)

    use_crsp = getattr(args, "use_crsp", False)
    df, (library, table) = try_query_with_fallbacks(
        db, exch_codes, start_date, end_date, args.limit, delist=args.delist, use_crsp=use_crsp
    )

    db.close()
    print(f"已从 {library}.{table} 获得 {len(df)} 行结果")

    saved = save_dataframe(df, output_path)
    print(f"结果保存至：{saved}")


if __name__ == "__main__":
    main()