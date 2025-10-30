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

COMPUSTAT_FUNDA_FALLBACKS: List[Tuple[str, str]] = [
    ("comp", "funda"),
    ("comp_na_annual_all", "funda")
]

# Compustat 季度表（fundq）回退顺序：优先使用 compd，再回退到 comp
COMPUSTAT_FUNDQ_FALLBACKS: List[Tuple[str, str]] = [
    ("compd", "fundq"),
    ("comp", "fundq")
]

CRSP_STOCK_FALLBACKS: List[Tuple[str, str]] = [
    ("crsp", "msf"),
    ("crspa", "msf")
]

# CCM + Compustat 融合表（用于 prcc_c / pstk）
CRSP_CCM_FUNDA_FALLBACKS: List[Tuple[str, str]] = [
    ("crspa", "ccmfunda"),
]

# Compustat 年度表，作为 prcc_c / pstk 的后备来源
COMPUSTAT_ANNUAL_FALLBACKS_FOR_PRICE: List[Tuple[str, str]] = [
    ("compd", "funda"),
    ("comp", "funda"),
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
        SELECT DISTINCT ON (crsp.permno)
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
        ORDER BY crsp.permno, c.ipodate
        """
    else:
        # 从 funda 获取 ipodate（年度数据）
        sql = f"""
        SELECT DISTINCT ON (crsp.permno)
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
        ORDER BY crsp.permno, f.ipodate
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
    end_date: str,
    include_optional: bool = True
) -> str:
    """
    构建查询财务比率的 SQL（多个财务指标）
    从 wrdsapps.firm_ratio 表查询
    包含: P/E, Market/Book, ROA, ROE, Net Profit Margin, Quick Ratio,
         Debt/Equity, Current Ratio, Asset Turnover, Inventory Turnover,
         以及计算 Tobin's Q 和增长率所需的字段

    参数:
        include_optional: 如果为 False，只查询核心字段
    """
    # 核心字段（这些字段在 firm_ratio 中更有可能存在）
    core_fields = """
        permno,
        public_date,
        pe_op_basic,
        pe_op_dil,
        pe_exi,
        pe_inc,
        ptb,
        bm,
        roa,
        roe,
        npm,
        quick_ratio,
        de_ratio,
        curr_ratio,
        at_turn,
        inv_turn
    """

    # 可选字段：为避免 firm_ratio 中不存在的列导致查询失败，这里不再请求
    # 收入将通过 comp/compd.fundq 的 revtq 获取；资产通过 comp.funda 回退获取
    optional_fields = ""

    fields = core_fields + ("," + optional_fields if include_optional and optional_fields.strip() else "")

    sql = f"""
    SELECT
        {fields}
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


def calculate_revenue_growth(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算 Revenue Growth
    公式: Revenue Growth = (Revenue_t - Revenue_{t-1}) / Revenue_{t-1}

    优先使用 Compustat 季度收入 revtq（来自 fundq）按季度计算；
    若 revtq 缺失，则回退到 firm_ratio 中的 revt_sale/sale_equity/sale_invcap。
    """
    if df.empty:
        return df

    df = df.copy()
    df = df.sort_values("public_date")

    # 确定使用哪个 revenue 字段（优先 revtq）
    revenue_col = None
    if "revtq" in df.columns and df["revtq"].notna().any():
        revenue_col = "revtq"
    elif "revt_sale" in df.columns and df["revt_sale"].notna().any():
        revenue_col = "revt_sale"
    elif "sale_equity" in df.columns and df["sale_equity"].notna().any():
        revenue_col = "sale_equity"
    elif "sale_invcap" in df.columns and df["sale_invcap"].notna().any():
        revenue_col = "sale_invcap"

    print(f"revenue_col: {revenue_col}")
    if revenue_col is None:
        # 没有可用的 revenue 字段，添加空列
        df["revenue_growth"] = None
        return df

    # 仅在 revenue 非空的行计算增长，相对上一个非空值
    growth = df.loc[df[revenue_col].notna(), revenue_col].pct_change()
    df["revenue_growth"] = None
    df.loc[growth.index, "revenue_growth"] = growth

    return df


def get_continuous_quarters(df: pd.DataFrame, max_quarters: int = 20) -> pd.DataFrame:
    """
    获取连续的季度数据，最多 max_quarters 个季度（默认 5 年 * 4 季度）。
    连续性基于 public_date 的季度周期（to_period('Q')）。
    """
    if df.empty:
        return df

    df = df.sort_values("public_date").copy()
    periods = pd.to_datetime(df["public_date"]).dt.to_period("Q")
    unique_periods = periods.drop_duplicates().sort_values().tolist()
    if not unique_periods:
        return df.iloc[0:0]

    continuous = [unique_periods[0]]
    for i in range(1, min(len(unique_periods), max_quarters)):
        if unique_periods[i] == continuous[-1] + 1:
            continuous.append(unique_periods[i])
        else:
            break

    mask = periods.isin(continuous)
    return df.loc[mask].copy()


def calculate_tobins_q(df: pd.DataFrame, db: "wrds.Connection", permno: int) -> pd.DataFrame:
    """
    处理 Tobin's Q 数据
    如果 tobins_q 字段已存在且有值，直接使用
    否则尝试手动计算: Tobin's Q = (Market Value of Equity + Debt) / Total Assets

    注意：手动计算 Tobin's Q 需要获取市值和债务的绝对值，这比较复杂
    因此优先使用数据库中已计算好的 tobins_q 字段
    """
    if df.empty:
        return df

    df = df.copy()

    # 如果 tobins_q 已经存在且有数据，直接返回
    if "tobins_q" in df.columns and df["tobins_q"].notna().any():
        return df

    # 如果没有 tobins_q，添加空列
    # 手动计算 Tobin's Q 需要额外的市值数据，这里暂不实现
    # 用户可以根据需要扩展此功能
    if "tobins_q" not in df.columns:
        df["tobins_q"] = None

    return df


def calculate_total_asset_growth(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算 Total Asset Growth
    公式: Total Asset Growth = (Total Assets_t - Total Assets_{t-1}) / Total Assets_{t-1}

    使用 total_assets 字段
    """
    if df.empty:
        return df

    df = df.copy()
    df = df.sort_values("public_date")

    if "total_assets" not in df.columns or not df["total_assets"].notna().any():
        # 没有可用的 total_assets 字段，添加空列
        df["total_asset_growth"] = None
        return df

    # 计算 total asset growth
    df["total_asset_growth"] = df["total_assets"].pct_change()

    return df


def build_ccmfunda_price_pstk_sql(
    library: str,
    table: str,
    permno: int,
    start_date: str,
    end_date: str
) -> str:
    """
    从 crspa.ccmfunda 获取 prcc_c（价格）和 pstk（优先股），按年度（datadate）
    通过 ccmxpf_lnkhist 将 gvkey 映射为 permno。
    """
    sql = f"""
    SELECT
        lnk.lpermno AS permno,
        f.datadate AS public_date,
        f.prcc_c,
        f.pstk
    FROM {library}.{table} AS f
    JOIN crsp.ccmxpf_lnkhist AS lnk
      ON f.gvkey = lnk.gvkey
     AND lnk.linktype IN ('LU', 'LC')
     AND lnk.linkprim IN ('P', 'C')
     AND f.datadate >= lnk.linkdt
     AND (f.datadate <= lnk.linkenddt OR lnk.linkenddt IS NULL)
    WHERE lnk.lpermno = {permno}
      AND f.datadate >= '{start_date}'
      AND f.datadate <= '{end_date}'
      -- 不限制 prcc_c 非空，以免过滤掉含 pstk 的记录
    ORDER BY f.datadate
    """
    return sql


def fetch_ccmfunda_price_pstk(
    db: "wrds.Connection",
    permno: int,
    start_date: str,
    end_date: str
) -> Optional[pd.DataFrame]:
    for library, table in CRSP_CCM_FUNDA_FALLBACKS:
        try:
            sql = build_ccmfunda_price_pstk_sql(library, table, permno, start_date, end_date)
            print(f"  → 尝试 {library}.{table} 获取 prcc_c/pstk ...")
            df = db.raw_sql(sql)
            if not df.empty:
                print(f"    ✓ {len(df)} 条 prcc_c/pstk 记录")
                return df
            else:
                print("    ✗ 无 prcc_c/pstk 记录")
        except Exception as err:
            print(f"    ✗ 查询失败: {err}")
            continue
    return None


def build_compustat_price_pstk_sql(
    library: str,
    table: str,
    permno: int,
    start_date: str,
    end_date: str
) -> str:
    """
    从 compd/comp.funda 获取 prcc_c 与 pstk（年度），通过 ccmxpf_lnkhist 关联 permno。
    """
    sql = f"""
    SELECT
        lnk.lpermno AS permno,
        f.datadate AS public_date,
        f.prcc_c,
        f.pstk
    FROM {library}.{table} AS f
    JOIN crsp.ccmxpf_lnkhist AS lnk
      ON f.gvkey = lnk.gvkey
     AND lnk.linktype IN ('LU', 'LC')
     AND lnk.linkprim IN ('P', 'C')
     AND f.datadate >= lnk.linkdt
     AND (f.datadate <= lnk.linkenddt OR lnk.linkenddt IS NULL)
    WHERE lnk.lpermno = {permno}
      AND f.datadate >= '{start_date}'
      AND f.datadate <= '{end_date}'
      AND f.indfmt = 'INDL'
      AND f.datafmt = 'STD'
      AND f.popsrc = 'D'
      AND f.consol = 'C'
    ORDER BY f.datadate
    """
    return sql


def fetch_compustat_price_pstk(
    db: "wrds.Connection",
    permno: int,
    start_date: str,
    end_date: str
) -> Optional[pd.DataFrame]:
    for library, table in COMPUSTAT_ANNUAL_FALLBACKS_FOR_PRICE:
        try:
            sql = build_compustat_price_pstk_sql(library, table, permno, start_date, end_date)
            print(f"  → 尝试 {library}.{table} 获取 prcc_c/pstk ...")
            df = db.raw_sql(sql)
            if not df.empty:
                print(f"    ✓ {len(df)} 条 prcc_c/pstk 记录")
                return df
            else:
                print("    ✗ 无 prcc_c/pstk 记录")
        except Exception as err:
            print(f"    ✗ 查询失败: {err}")
            continue
    return None


def build_msf_shrout_sql(
    library: str,
    table: str,
    permno: int,
    start_date: str,
    end_date: str
) -> str:
    """
    从 CRSP 月度收益表 msf 获取 shrout（月度，千股）。
    """
    sql = f"""
    SELECT
        permno,
        date AS public_date,
        shrout
    FROM {library}.{table}
    WHERE permno = {permno}
      AND date >= '{start_date}'
      AND date <= '{end_date}'
    ORDER BY date
    """
    return sql


def fetch_msf_shrout(
    db: "wrds.Connection",
    permno: int,
    start_date: str,
    end_date: str
) -> Optional[pd.DataFrame]:
    for library, table in CRSP_STOCK_FALLBACKS:
        try:
            sql = build_msf_shrout_sql(library, table, permno, start_date, end_date)
            print(f"  → 尝试 {library}.{table} 获取 shrout ...")
            df = db.raw_sql(sql)
            if not df.empty:
                print(f"    ✓ {len(df)} 条 shrout 记录")
                return df
            else:
                print("    ✗ 无 shrout 记录")
        except Exception as err:
            print(f"    ✗ 查询失败: {err}")
            continue
    return None


def build_compustat_total_assets_sql(
    library: str,
    table: str,
    permno: int,
    start_date: str,
    end_date: str
) -> str:
    """
    从 Compustat funda 表获取 Total Assets (at 字段)
    通过 ccmxpf_lnkhist 链接 permno 和 gvkey
    """
    sql = f"""
    SELECT
        lnk.lpermno AS permno,
        f.datadate AS public_date,
        f.at AS total_assets
    FROM {library}.{table} AS f
    JOIN crsp.ccmxpf_lnkhist AS lnk
      ON f.gvkey = lnk.gvkey
     AND lnk.linktype IN ('LU', 'LC')
     AND lnk.linkprim IN ('P', 'C')
     AND f.datadate >= lnk.linkdt
     AND (f.datadate <= lnk.linkenddt OR lnk.linkenddt IS NULL)
    WHERE lnk.lpermno = {permno}
      AND f.datadate >= '{start_date}'
      AND f.datadate <= '{end_date}'
      AND f.indfmt = 'INDL'
      AND f.datafmt = 'STD'
      AND f.popsrc = 'D'
      AND f.consol = 'C'
      AND f.at IS NOT NULL
    ORDER BY f.datadate
    """
    return sql


def fetch_total_assets_from_compustat(
    db: "wrds.Connection",
    permno: int,
    start_date: str,
    end_date: str
) -> Optional[pd.DataFrame]:
    """
    从 Compustat 获取 Total Assets 数据作为后备方案
    """
    for library, table in COMPUSTAT_FUNDA_FALLBACKS:
        try:
            sql = build_compustat_total_assets_sql(
                library, table, permno, start_date, end_date
            )
            print(f"    尝试 {library}.{table} ...")
            df = db.raw_sql(sql)
            if not df.empty:
                print(f"    ✓ 找到 {len(df)} 条记录")
                return df
            else:
                print(f"    ✗ 无记录")
        except Exception as err:
            print(f"    ✗ 查询失败: {err}")
            continue
    return None


def build_compustat_total_liabilities_sql(
    library: str,
    table: str,
    permno: int,
    start_date: str,
    end_date: str
) -> str:
    """
    从 Compustat 年度 funda 表获取 Total Liabilities (lt 字段，单位：百万美元)
    通过 ccmxpf_lnkhist 链接 permno 和 gvkey。
    """
    sql = f"""
    SELECT
        lnk.lpermno AS permno,
        f.datadate AS public_date,
        f.lt AS total_liabilities
    FROM {library}.{table} AS f
    JOIN crsp.ccmxpf_lnkhist AS lnk
      ON f.gvkey = lnk.gvkey
     AND lnk.linktype IN ('LU', 'LC')
     AND lnk.linkprim IN ('P', 'C')
     AND f.datadate >= lnk.linkdt
     AND (f.datadate <= lnk.linkenddt OR lnk.linkenddt IS NULL)
    WHERE lnk.lpermno = {permno}
      AND f.datadate >= '{start_date}'
      AND f.datadate <= '{end_date}'
      AND f.indfmt = 'INDL'
      AND f.datafmt = 'STD'
      AND f.popsrc = 'D'
      AND f.consol = 'C'
      AND f.lt IS NOT NULL
    ORDER BY f.datadate
    """
    return sql


def fetch_total_liabilities_from_compustat(
    db: "wrds.Connection",
    permno: int,
    start_date: str,
    end_date: str
) -> Optional[pd.DataFrame]:
    for library, table in [("compd", "funda"), ("comp", "funda")]:
        try:
            sql = build_compustat_total_liabilities_sql(
                library, table, permno, start_date, end_date
            )
            print(f"    尝试 {library}.{table} (lt) ...")
            df = db.raw_sql(sql)
            if not df.empty:
                print(f"    ✓ 找到 {len(df)} 条 total_liabilities 记录")
                return df
            else:
                print("    ✗ 无 total_liabilities 记录")
        except Exception as err:
            print(f"    ✗ 查询失败: {err}")
            continue
    return None


def calculate_tobinq(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算 Tobin's Q（年度口径，保留月度行；非年报月置为 NaN）
    公式：tobinq = (shrout*prcc_c + pstk + total_liabilities) / total_assets

    - shrout 来自 CRSP msf，单位通常为千股，这里转换为股（*1000）
    - prcc_c、pstk 来自 ccmfunda（Compustat/CRSP 合并表）
    - total_liabilities (lt) 与 total_assets (at) 为百万美元口径
    - 为保证口径一致，shrout*prcc_c 结果从美元转换为百万美元（/1e6）
    """
    if df.empty:
        return df

    df = df.copy()

    # 转换 shrout 为股，并将市值转换为百万美元
    if "shrout" in df.columns:
        df["shrout_shares"] = df["shrout"] * 1000.0
    else:
        df["shrout_shares"] = None

    # 仅在四个组成项齐备时计算
    def _calc_row(row: pd.Series) -> Optional[float]:
        try:
            prcc_c = row.get("prcc_c")
            pstk = row.get("pstk")
            total_liabilities = row.get("total_liabilities")
            total_assets = row.get("total_assets")
            shrout_shares = row.get("shrout_shares")
            if pd.isna(prcc_c) or pd.isna(total_assets) or pd.isna(shrout_shares):
                return None
            # 市值（百万美元）
            market_equity_m = (shrout_shares * prcc_c) / 1_000_000.0
            # pstk 与 lt 已是百万美元
            if pd.isna(pstk):
                pstk_val = 0.0
            else:
                pstk_val = pstk
            if pd.isna(total_liabilities):
                tl_val = 0.0
            else:
                tl_val = total_liabilities
            numerator = market_equity_m + pstk_val + tl_val
            if pd.isna(total_assets) or total_assets == 0:
                return None
            return float(numerator) / float(total_assets)
        except Exception:
            return None

    df["tobinq"] = df.apply(_calc_row, axis=1)
    # 清理临时列
    if "shrout_shares" in df.columns:
        df = df.drop(columns=["shrout_shares"])
    return df


def build_compustat_quarterly_revenue_sql(
    library: str,
    table: str,
    permno: int,
    start_date: str,
    end_date: str
) -> str:
    """
    从 Compustat fundq 表获取季度收入 revtq，
    通过 ccmxpf_lnkhist 将 gvkey 映射到 permno。
    使用 datadate 作为 public_date。
    """
    sql = f"""
    SELECT
        lnk.lpermno AS permno,
        q.datadate AS public_date,
        q.revtq AS revtq
    FROM {library}.{table} AS q
    JOIN crsp.ccmxpf_lnkhist AS lnk
      ON q.gvkey = lnk.gvkey
     AND lnk.linktype IN ('LU', 'LC')
     AND lnk.linkprim IN ('P', 'C')
     AND q.datadate >= lnk.linkdt
     AND (q.datadate <= lnk.linkenddt OR lnk.linkenddt IS NULL)
    WHERE lnk.lpermno = {permno}
      AND q.datadate >= '{start_date}'
      AND q.datadate <= '{end_date}'
      AND q.indfmt = 'INDL'
      AND q.datafmt = 'STD'
      AND q.popsrc = 'D'
      AND q.consol = 'C'
      AND q.revtq IS NOT NULL
    ORDER BY q.datadate
    """
    return sql


def fetch_quarterly_revenue_from_compustat(
    db: "wrds.Connection",
    permno: int,
    start_date: str,
    end_date: str
) -> Optional[pd.DataFrame]:
    """
    从 Compustat fundq 获取季度收入 revtq。
    """
    for library, table in COMPUSTAT_FUNDQ_FALLBACKS:
        try:
            sql = build_compustat_quarterly_revenue_sql(
                library, table, permno, start_date, end_date
            )
            print(f"    尝试 {library}.{table} (revtq) ...")
            df = db.raw_sql(sql)
            if not df.empty:
                print(f"    ✓ 找到 {len(df)} 条季度收入记录")
                return df
            else:
                print(f"    ✗ 无季度收入记录")
        except Exception as err:
            print(f"    ✗ 查询失败: {err}")
            continue
    return None


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
            # 查询核心字段和可选字段（用于后续计算）
            try:
                sql = build_financial_ratios_sql(
                    library,
                    table,
                    permno,
                    ipo_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d"),
                    include_optional=True
                )
                print(f"  尝试查询 {library}.{table}，日期范围: {ipo_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")
                df_temp = db.raw_sql(sql)

                if not df_temp.empty:
                    df_financial = df_temp
                    print(f"  ✓ 从 {library}.{table} 找到 {len(df_temp)} 条记录")
                    break
                else:
                    print(f"  ✗ {library}.{table} 返回0条记录")
            except Exception as err:
                print(f"  ✗ 查询 {library}.{table} 失败: {err}")
                continue

        if df_financial is None or df_financial.empty:
            print(f"  ⚠ 未找到财务数据，可能原因：")
            print(f"    - permno {permno} 在 {library}.{table} 中无记录")
            print(f"    - IPO 日期 {ipo_date.date()} 太早，数据库可能不包含该时期数据")
            continue

        # 如果查询结果中有 at 字段，重命名为 total_assets
        if "at" in df_financial.columns:
            df_financial = df_financial.rename(columns={"at": "total_assets"})

        # 添加缺失的计算字段为空列（这些字段需要后续计算）
        calculated_cols = ["revenue_growth", "total_asset_growth"]
        for col in calculated_cols:
            if col not in df_financial.columns:
                df_financial[col] = None

        # 检查是否有 total_assets 字段，如果没有则从 Compustat 获取
        if "total_assets" not in df_financial.columns or not df_financial["total_assets"].notna().any():
            print(f"  → firm_ratio 中无 total_assets，尝试从 Compustat 获取...")
            df_total_assets = fetch_total_assets_from_compustat(
                db,
                permno,
                ipo_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d")
            )

            if df_total_assets is not None and not df_total_assets.empty:
                print(f"  ✓ 从 Compustat 获取到 {len(df_total_assets)} 条 total_assets 记录")
                # 合并 total_assets 数据到 df_financial
                # 如果 total_assets 列已存在，先删除它
                if "total_assets" in df_financial.columns:
                    df_financial = df_financial.drop(columns=["total_assets"])

                # 使用 public_date 作为合并键
                df_financial = pd.merge(
                    df_financial,
                    df_total_assets[["public_date", "total_assets"]],
                    on="public_date",
                    how="left"
                )
            else:
                print(f"  ✗ 从 Compustat 也未获取到 total_assets")

        # 获取季度收入 revtq，并合并到 df_financial（作为季度增长的基础）
        print(f"  → 尝试从 Compustat fundq 获取季度收入 revtq ...")
        df_revenue_q = fetch_quarterly_revenue_from_compustat(
            db,
            permno,
            ipo_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )
        if df_revenue_q is not None and not df_revenue_q.empty:
            # 以月度财报为基表，左连接季度收入；非季度月份的 revtq 将为 NaN
            df_financial = pd.merge(
                df_financial,
                df_revenue_q,
                on=["permno", "public_date"],
                how="left"
            )
        else:
            print(f"  ⚠ 未能获取 revtq，将回退到 firm_ratio 提供的收入字段（若有）")

        # 获取连续年份范围内的月度数据（最多5年），保留月度频率
        df_continuous = get_continuous_years(df_financial, max_years=5)

        if df_continuous.empty:
            print(f"  ⚠ 无连续年份数据")
            continue

        print(f"  ✓ 获得 {len(df_continuous)} 条连续时间数据（按月度保留）")

        # 合并年度 prcc_c/pstk、总负债 lt（百万美元），以及月度 shrout（千股）
        # 统一到“月份末”以避免日期不对齐导致的空值
        df_continuous = df_continuous.copy()
        df_continuous["merge_month"] = pd.to_datetime(df_continuous["public_date"]).dt.to_period("M").dt.to_timestamp("M")
        # 先尝试 ccmfunda，其次回退 compd/comp.funda
        prices_df = fetch_ccmfunda_price_pstk(
            db,
            permno,
            ipo_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )
        if (prices_df is None) or prices_df.empty:
            prices_df = fetch_compustat_price_pstk(
                db,
                permno,
                ipo_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d")
            )
        if prices_df is not None and not prices_df.empty:
            prices_df = prices_df.copy()
            prices_df["merge_month"] = pd.to_datetime(prices_df["public_date"]).dt.to_period("M").dt.to_timestamp("M")
            df_continuous = pd.merge(
                df_continuous,
                prices_df[["merge_month", "prcc_c", "pstk"]],
                on="merge_month",
                how="left"
            )

        liabilities_df = fetch_total_liabilities_from_compustat(
            db,
            permno,
            ipo_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )
        if liabilities_df is not None and not liabilities_df.empty:
            liabilities_df = liabilities_df.copy()
            liabilities_df["merge_month"] = pd.to_datetime(liabilities_df["public_date"]).dt.to_period("M").dt.to_timestamp("M")
            df_continuous = pd.merge(
                df_continuous,
                liabilities_df[["merge_month", "total_liabilities"]],
                on="merge_month",
                how="left"
            )

        shrout_df = fetch_msf_shrout(
            db,
            permno,
            ipo_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )
        if shrout_df is not None and not shrout_df.empty:
            shrout_df = shrout_df.copy()
            shrout_df["merge_month"] = pd.to_datetime(shrout_df["public_date"]).dt.to_period("M").dt.to_timestamp("M")
            df_continuous = pd.merge(
                df_continuous,
                shrout_df[["merge_month", "shrout"]],
                on="merge_month",
                how="left"
            )

        # 确保 TobinQ 相关列存在（即使本公司未取到对应数据，后续也能安全重排列）
        for _col in ["prcc_c", "pstk", "shrout", "total_liabilities"]:
            if _col not in df_continuous.columns:
                df_continuous[_col] = pd.NA

        # 清理合并辅助列
        if "merge_month" in df_continuous.columns:
            df_continuous = df_continuous.drop(columns=["merge_month"])

        # 如果存在旧列 tobins_q，删除避免重复
        if "tobins_q" in df_continuous.columns:
            df_continuous = df_continuous.drop(columns=["tobins_q"])

        # 计算新的 TobinQ（年度计算、月度保留，非年报月为 NaN）
        df_continuous = calculate_tobinq(df_continuous)

        # 计算 Revenue Growth（仅在季度末非空处计算，对应上一季度非空值）
        df_continuous = calculate_revenue_growth(df_continuous)

        # 计算 Total Asset Growth
        df_continuous = calculate_total_asset_growth(df_continuous)

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
    cols = [
        "permno", "ipo_date", "public_date",
        "revtq",  # Quarterly revenue
        "prcc_c", "pstk", "shrout", "total_liabilities",  # TobinQ 组成项
        "pe_op_basic", "pe_op_dil", "pe_exi", "pe_inc",  # P/E ratios
        "ptb", "bm",  # Market/Book
        "roa", "roe",  # ROA and ROE
        "npm",  # Net Profit Margin
        "tobinq",  # 新的 Tobin's Q
        "revenue_growth",  # Revenue Growth
        "total_asset_growth",  # Total Asset Growth
        "quick_ratio", "de_ratio", "curr_ratio",  # Liquidity and leverage ratios
        "at_turn", "inv_turn"  # Turnover ratios
    ]
    # 添加其他列
    for col in final_df.columns:
        if col not in cols:
            cols.append(col)

    # 在重排列前，确保所有期望列都存在；不存在的补为 NaN，避免 KeyError
    for _col in cols:
        if _col not in final_df.columns:
            final_df[_col] = pd.NA
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