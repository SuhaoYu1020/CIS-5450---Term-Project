"""
Fetch CRSP IPO or Delist data from WRDS and save locally.

Core logic:
- IPO mode:
  * Use Compustat ipodate field (--use-compustat, default)
  * Or use CRSP's earliest namedt per permno as IPO date (--use-crsp)
- Delist mode (--delist): Based on dsedelist/msedelist dlstdt as delist date, and join names table family
  in that date range to get comnam/ticker/exchcd/shrcd (namedt<=dlstdt<=nameendt)
- Support start/end dates (--start and --end, YYYY-MM-DD)
- Support NYSE/AMEX/NASDAQ
- Automatically fallback across multiple library tables
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict, Any

import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

try:
    import wrds  # type: ignore
except Exception as import_error:  # pragma: no cover
    raise SystemExit(
        "Failed to import wrds library. Please run: pip install wrds pandas openpyxl"
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

# Compustat quarterly table (fundq) fallback order: prefer compd, then fallback to comp
COMPUSTAT_FUNDQ_FALLBACKS: List[Tuple[str, str]] = [
    ("compd", "fundq"),
    ("comp", "fundq")
]

CRSP_STOCK_FALLBACKS: List[Tuple[str, str]] = [
    ("crsp", "msf"),
    ("crspa", "msf")
]

# CCM + Compustat merged table (for prcc_c / pstk)
CRSP_CCM_FUNDA_FALLBACKS: List[Tuple[str, str]] = [
    ("crspa", "ccmfunda"),
]

# Compustat annual table, as backup source for prcc_c / pstk
COMPUSTAT_ANNUAL_FALLBACKS_FOR_PRICE: List[Tuple[str, str]] = [
    ("compd", "funda"),
    ("comp", "funda"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch CRSP IPO / Delist data from WRDS")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--exchange", type=str, default="NYSE", help="Exchange (NYSE/AMEX/NASDAQ or 1/2/3)")
    parser.add_argument("--limit", type=int, default=None, help="Only return first N rows")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--username", type=str, default=os.environ.get("WRDS_USERNAME") or "suhaoyu")
    parser.add_argument("--password", type=str, default=os.environ.get("WRDS_PASSWORD") or "Yusuhao031020.")
    parser.add_argument("--delist", action="store_true", help="Get delist records (by latest date)")
    parser.add_argument("--use-crsp", action="store_true", help="Use CRSP earliest record instead of Compustat ipodate")
    parser.add_argument("--fetch-financials", action="store_true", help="Fetch financial statement data (ROA/ROE) based on IPO file")
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
    raise ValueError(f"Unrecognized exchange: {exchange}")


def resolve_date_range(start_date_str: Optional[str], end_date_str: Optional[str]) -> Tuple[str, str]:
    if not start_date_str or not end_date_str:
        raise SystemExit("Must provide --start and --end, format: YYYY-MM-DD")
    return start_date_str, end_date_str


# === Compustat IPO query (using real ipodate field) ===
def build_compustat_ipo_sql(library: str, table: str, exch_codes: List[int], start_date: str, end_date: str, limit: Optional[int] = None) -> str:
    """
    Query companies with real ipodate field from Compustat
    Mainly get ipodate from comp.company or comp.funda
    Then JOIN CRSP tables to get exchange and ticker information
    """
    exch_str = ",".join(str(x) for x in exch_codes)

    # Try to get ipodate from comp.company
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
        # Get ipodate from funda (annual data)
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


# === CRSP IPO query (using earliest namedt) ===
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
        # Default to use Compustat ipodate unless --use-crsp is specified
        if not use_crsp:
            print("Querying IPO data using Compustat ipodate field...")
            for library, table in COMPUSTAT_FALLBACKS:
                sql = build_compustat_ipo_sql(library, table, exch_codes, start_date, end_date, limit)
                try:
                    print(f"Attempting to query Compustat IPO {library}.{table} ...")
                    df = db.raw_sql(sql)
                    if not df.empty:
                        print(f"Success! Found {len(df)} records with ipodate")
                        return df, (library, table)
                except Exception as err:
                    last_error = err
                    print(f"Failed: {err}")
                    continue
        else:
            print("Querying IPO data using CRSP earliest record date...")
            for library, table in TABLE_FALLBACKS:
                sql = build_ipo_sql(library, table, exch_codes, start_date, end_date, limit)
                try:
                    print(f"Attempting to query CRSP IPO {library}.{table} ...")
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
                    print(f"Attempting to query delist {d_lib}.{d_tbl} JOIN {n_lib}.{n_tbl} ...")
                    df = db.raw_sql(sql)
                    if not df.empty:
                        return df, (f"{d_lib}.{d_tbl}", f"{n_lib}.{n_tbl}")
                except Exception as err:
                    last_error = err
                    continue
    if last_error:
        raise last_error
    raise RuntimeError("All library tables query failed.")


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


# === Financial statement data fetching functionality ===
def read_ipo_file(start_date: str, end_date: str, exchange: str) -> pd.DataFrame:
    """
    Read corresponding IPO Excel file based on parameters
    File format: ipo_{start_year}_{end_year}_{exchange_code}.xlsx
    """
    start_year = start_date[:4]
    end_year = end_date[:4]
    exch_codes = normalize_exchange(exchange)
    exch_tag = "-".join(str(x) for x in exch_codes)

    filename = f"ipo_{start_year}_{end_year}_{exch_tag}.xlsx"
    filepath = os.path.join(DATA_DIR, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"IPO file not found: {filepath}")

    print(f"Reading IPO file: {filepath}")
    df = pd.read_excel(filepath)

    # Ensure required columns exist
    required_cols = ["permno", "ipo_date"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"IPO file missing required column: {col}")

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
    Build SQL to query financial ratios (multiple financial indicators)
    Query from wrdsapps.firm_ratio table
    Includes: P/E, Market/Book, ROA, ROE, Net Profit Margin, Quick Ratio,
         Debt/Equity, Current Ratio, Asset Turnover, Inventory Turnover,
         as well as fields needed to calculate Tobin's Q and growth rates

    Parameters:
        include_optional: If False, only query core fields
    """
    # Core fields (these fields are more likely to exist in firm_ratio)
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

    # Optional fields: to avoid query failure due to non-existent columns in firm_ratio, not requesting here
    # Revenue will be obtained from comp/compd.fundq revtq; assets will be obtained via comp.funda fallback
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
    Get continuous annual data, up to max_years years
    If data break occurs (non-consecutive years), terminate at break point
    """
    if df.empty:
        return df

    # Ensure sorted by date
    df = df.sort_values("public_date").copy()

    # Extract year
    df["year"] = pd.to_datetime(df["public_date"]).dt.year

    # Get unique years
    years = df["year"].unique()

    # Check continuity
    continuous_years = [years[0]]
    for i in range(1, min(len(years), max_years)):
        if years[i] == continuous_years[-1] + 1:
            continuous_years.append(years[i])
        else:
            # Years not consecutive, terminate
            break

    # Only keep data from consecutive years
    result = df[df["year"].isin(continuous_years)].copy()
    result = result.drop(columns=["year"])

    return result


def calculate_revenue_growth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Revenue Growth
    Formula: Revenue Growth = (Revenue_t - Revenue_{t-1}) / Revenue_{t-1}

    Prefer using Compustat quarterly revenue revtq (from fundq) calculated quarterly;
    If revtq missing, fallback to revt_sale/sale_equity/sale_invcap in firm_ratio.
    """
    if df.empty:
        return df

    df = df.copy()
    df = df.sort_values("public_date")

    # Determine which revenue field to use (prefer revtq)
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
        # No available revenue field, add empty column
        df["revenue_growth"] = None
        return df

    # Only calculate growth for rows where revenue is not null, relative to previous non-null value
    growth = df.loc[df[revenue_col].notna(), revenue_col].pct_change()
    df["revenue_growth"] = None
    df.loc[growth.index, "revenue_growth"] = growth

    return df


def get_continuous_quarters(df: pd.DataFrame, max_quarters: int = 20) -> pd.DataFrame:
    """
    Get continuous quarterly data, up to max_quarters quarters (default 5 years * 4 quarters).
    Continuity based on quarterly period of public_date (to_period('Q')).
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
    Process Tobin's Q data
    If tobins_q field already exists and has values, use directly
    Otherwise try manual calculation: Tobin's Q = (Market Value of Equity + Debt) / Total Assets

    Note: Manual calculation of Tobin's Q requires getting absolute values of market cap and debt, which is complex
    Therefore prefer using pre-calculated tobins_q field from database
    """
    if df.empty:
        return df

    df = df.copy()

    # If tobins_q already exists and has data, return directly
    if "tobins_q" in df.columns and df["tobins_q"].notna().any():
        return df

    # If no tobins_q, add empty column
    # Manual calculation of Tobin's Q requires additional market cap data, not implemented here
    # Users can extend this functionality as needed
    if "tobins_q" not in df.columns:
        df["tobins_q"] = None

    return df


def calculate_total_asset_growth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Total Asset Growth
    Formula: Total Asset Growth = (Total Assets_t - Total Assets_{t-1}) / Total Assets_{t-1}

    Use total_assets field
    """
    if df.empty:
        return df

    df = df.copy()
    df = df.sort_values("public_date")

    if "total_assets" not in df.columns or not df["total_assets"].notna().any():
        # No available total_assets field, add empty column
        df["total_asset_growth"] = None
        return df

    # Calculate total asset growth
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
    Get prcc_c (price) and pstk (preferred stock) from crspa.ccmfunda, by year (datadate)
    Map gvkey to permno via ccmxpf_lnkhist.
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
      -- Do not restrict prcc_c to be non-null to avoid filtering out records containing pstk
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
            print(f"  → Attempting {library}.{table} to get prcc_c/pstk ...")
            df = db.raw_sql(sql)
            if not df.empty:
                print(f"    ✓ Found {len(df)} prcc_c/pstk records")
                return df
            else:
                print("    ✗ No prcc_c/pstk records")
        except Exception as err:
            print(f"    ✗ Query failed: {err}")
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
    Get prcc_c and pstk (annual) from compd/comp.funda, link permno via ccmxpf_lnkhist.
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
            print(f"  → Attempting {library}.{table} to get prcc_c/pstk ...")
            df = db.raw_sql(sql)
            if not df.empty:
                print(f"    ✓ Found {len(df)} prcc_c/pstk records")
                return df
            else:
                print("    ✗ No prcc_c/pstk records")
        except Exception as err:
            print(f"    ✗ Query failed: {err}")
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
    Get shrout (monthly, thousands of shares) from CRSP monthly stock return table msf.
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
            print(f"  → Attempting {library}.{table} to get shrout ...")
            df = db.raw_sql(sql)
            if not df.empty:
                print(f"    ✓ Found {len(df)} shrout records")
                return df
            else:
                print("    ✗ No shrout records")
        except Exception as err:
            print(f"    ✗ Query failed: {err}")
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
    Get Total Assets (at field) from Compustat funda table
    Link permno and gvkey via ccmxpf_lnkhist
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
    Get Total Assets data from Compustat as backup solution
    """
    for library, table in COMPUSTAT_FUNDA_FALLBACKS:
        try:
            sql = build_compustat_total_assets_sql(
                library, table, permno, start_date, end_date
            )
            print(f"    Attempting {library}.{table} ...")
            df = db.raw_sql(sql)
            if not df.empty:
                print(f"    ✓ Found {len(df)} records")
                return df
            else:
                print(f"    ✗ No records")
        except Exception as err:
            print(f"    ✗ Query failed: {err}")
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
    Get Total Liabilities (lt field, unit: millions USD) from Compustat annual funda table
    Link permno and gvkey via ccmxpf_lnkhist.
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
            print(f"    Attempting {library}.{table} (lt) ...")
            df = db.raw_sql(sql)
            if not df.empty:
                print(f"    ✓ Found {len(df)} total_liabilities records")
                return df
            else:
                print("    ✗ No total_liabilities records")
        except Exception as err:
            print(f"    ✗ Query failed: {err}")
            continue
    return None


def calculate_tobinq(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Tobin's Q (annual basis, keep monthly rows; non-annual-report months set to NaN)
    Formula: tobinq = (shrout*prcc_c + pstk + total_liabilities) / total_assets

    - shrout from CRSP msf, unit usually thousands of shares, convert to shares here (*1000)
    - prcc_c, pstk from ccmfunda (Compustat/CRSP merged table)
    - total_liabilities (lt) and total_assets (at) are in millions USD
    - To ensure consistent units, convert shrout*prcc_c result from USD to millions USD (/1e6)
    """
    if df.empty:
        return df

    df = df.copy()

    # Convert shrout to shares and market cap to millions USD
    if "shrout" in df.columns:
        df["shrout_shares"] = df["shrout"] * 1000.0
    else:
        df["shrout_shares"] = None

    # Only calculate when all four components are available
    def _calc_row(row: pd.Series) -> Optional[float]:
        try:
            prcc_c = row.get("prcc_c")
            pstk = row.get("pstk")
            total_liabilities = row.get("total_liabilities")
            total_assets = row.get("total_assets")
            shrout_shares = row.get("shrout_shares")
            if pd.isna(prcc_c) or pd.isna(total_assets) or pd.isna(shrout_shares):
                return None
            # Market cap (millions USD)
            market_equity_m = (shrout_shares * prcc_c) / 1_000_000.0
            # pstk and lt are already in millions USD
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
    # Clean temporary columns
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
    Get quarterly revenue revtq from Compustat fundq table,
    Map gvkey to permno via ccmxpf_lnkhist.
    Use datadate as public_date.
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
    Get quarterly revenue revtq from Compustat fundq.
    """
    for library, table in COMPUSTAT_FUNDQ_FALLBACKS:
        try:
            sql = build_compustat_quarterly_revenue_sql(
                library, table, permno, start_date, end_date
            )
            print(f"    Attempting {library}.{table} (revtq) ...")
            df = db.raw_sql(sql)
            if not df.empty:
                print(f"    ✓ Found {len(df)} quarterly revenue records")
                return df
            else:
                print(f"    ✗ No quarterly revenue records")
        except Exception as err:
            print(f"    ✗ Query failed: {err}")
            continue
    return None


def fetch_financial_data_for_ipo(
    db: "wrds.Connection",
    ipo_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Fetch financial data for each company in IPO file
    Starting from ipodate, up to 5 consecutive years
    """
    all_results = []

    total_companies = len(ipo_df)
    print(f"Starting to fetch financial data for {total_companies} companies...")

    for idx, row in ipo_df.iterrows():
        permno = int(row["permno"])
        ipo_date = pd.to_datetime(row["ipo_date"])

        # Calculate end date (5 years after IPO date)
        end_date = ipo_date + timedelta(days=365 * 5)

        print(f"[{idx + 1}/{total_companies}] Processing permno={permno}, IPO date={ipo_date.date()}")

        # Try querying from different tables
        df_financial = None
        for library, table in FINRATIO_FALLBACKS:
            # Query core fields and optional fields (for subsequent calculations)
            try:
                sql = build_financial_ratios_sql(
                    library,
                    table,
                    permno,
                    ipo_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d"),
                    include_optional=True
                )
                print(f"  Attempting to query {library}.{table}, date range: {ipo_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
                df_temp = db.raw_sql(sql)

                if not df_temp.empty:
                    df_financial = df_temp
                    print(f"  ✓ Found {len(df_temp)} records from {library}.{table}")
                    break
                else:
                    print(f"  ✗ {library}.{table} returned 0 records")
            except Exception as err:
                print(f"  ✗ Query {library}.{table} failed: {err}")
                continue

        if df_financial is None or df_financial.empty:
            print(f"  ⚠ Financial data not found, possible reasons:")
            print(f"    - permno {permno} has no records in {library}.{table}")
            print(f"    - IPO date {ipo_date.date()} is too early, database may not contain data for this period")
            continue

        # If at field exists in query results, rename to total_assets
        if "at" in df_financial.columns:
            df_financial = df_financial.rename(columns={"at": "total_assets"})

        # Add missing calculated fields as empty columns (these fields need subsequent calculation)
        calculated_cols = ["revenue_growth", "total_asset_growth"]
        for col in calculated_cols:
            if col not in df_financial.columns:
                df_financial[col] = None

        # Check if total_assets field exists, if not get from Compustat
        if "total_assets" not in df_financial.columns or not df_financial["total_assets"].notna().any():
            print(f"  → No total_assets in firm_ratio, attempting to get from Compustat...")
            df_total_assets = fetch_total_assets_from_compustat(
                db,
                permno,
                ipo_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d")
            )

            if df_total_assets is not None and not df_total_assets.empty:
                print(f"  ✓ Got {len(df_total_assets)} total_assets records from Compustat")
                # Merge total_assets data into df_financial
                # If total_assets column already exists, delete it first
                if "total_assets" in df_financial.columns:
                    df_financial = df_financial.drop(columns=["total_assets"])

                # Use public_date as merge key
                df_financial = pd.merge(
                    df_financial,
                    df_total_assets[["public_date", "total_assets"]],
                    on="public_date",
                    how="left"
                )
            else:
                print(f"  ✗ Could not get total_assets from Compustat either")

        # Get quarterly revenue revtq and merge into df_financial (as basis for quarterly growth)
        print(f"  → Attempting to get quarterly revenue revtq from Compustat fundq ...")
        df_revenue_q = fetch_quarterly_revenue_from_compustat(
            db,
            permno,
            ipo_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )
        if df_revenue_q is not None and not df_revenue_q.empty:
            # Use monthly financial reports as base table, left join quarterly revenue; revtq for non-quarterly months will be NaN
            df_financial = pd.merge(
                df_financial,
                df_revenue_q,
                on=["permno", "public_date"],
                how="left"
            )
        else:
            print(f"  ⚠ Could not get revtq, will fallback to revenue fields provided by firm_ratio (if available)")

        # Get monthly data within continuous year range (up to 5 years), preserve monthly frequency
        df_continuous = get_continuous_years(df_financial, max_years=5)

        if df_continuous.empty:
            print(f"  ⚠ No continuous year data")
            continue

        print(f"  ✓ Got {len(df_continuous)} continuous time data records (preserved monthly)")

        # Merge annual prcc_c/pstk, total liabilities lt (millions USD), and monthly shrout (thousands of shares)
        # Unify to "month-end" to avoid null values due to date misalignment
        df_continuous = df_continuous.copy()
        df_continuous["merge_month"] = pd.to_datetime(df_continuous["public_date"]).dt.to_period("M").dt.to_timestamp("M")
        # First try ccmfunda, then fallback to compd/comp.funda
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

        # Ensure TobinQ related columns exist (even if this company didn't get corresponding data, can safely reorder columns later)
        for _col in ["prcc_c", "pstk", "shrout", "total_liabilities"]:
            if _col not in df_continuous.columns:
                df_continuous[_col] = pd.NA

        # Clean merge auxiliary columns
        if "merge_month" in df_continuous.columns:
            df_continuous = df_continuous.drop(columns=["merge_month"])

        # If old column tobins_q exists, delete to avoid duplication
        if "tobins_q" in df_continuous.columns:
            df_continuous = df_continuous.drop(columns=["tobins_q"])

        # Calculate new TobinQ (annual calculation, monthly preserved, non-annual-report months as NaN)
        df_continuous = calculate_tobinq(df_continuous)

        # Calculate Revenue Growth (only calculate at quarter-end non-null points, relative to previous quarter non-null value)
        df_continuous = calculate_revenue_growth(df_continuous)

        # Calculate Total Asset Growth
        df_continuous = calculate_total_asset_growth(df_continuous)

        # Add IPO related information
        df_continuous["ipo_date"] = ipo_date

        # If original IPO file has other columns (e.g., comnam, ticker, etc.), also add them
        for col in ipo_df.columns:
            if col not in df_continuous.columns and col != "permno":
                df_continuous[col] = row[col]

        print(f"  Kept {len(df_continuous)} continuous data records")
        all_results.append(df_continuous)

    if not all_results:
        return pd.DataFrame()

    # Merge all results
    final_df = pd.concat(all_results, ignore_index=True)

    # Reorder columns
    cols = [
        "permno", "ipo_date", "public_date",
        "revtq",  # Quarterly revenue
        "prcc_c", "pstk", "shrout", "total_liabilities",  # TobinQ components
        "pe_op_basic", "pe_op_dil", "pe_exi", "pe_inc",  # P/E ratios
        "ptb", "bm",  # Market/Book
        "roa", "roe",  # ROA and ROE
        "npm",  # Net Profit Margin
        "tobinq",  # New Tobin's Q
        "revenue_growth",  # Revenue Growth
        "total_asset_growth",  # Total Asset Growth
        "quick_ratio", "de_ratio", "curr_ratio",  # Liquidity and leverage ratios
        "at_turn", "inv_turn"  # Turnover ratios
    ]
    # Add other columns
    for col in final_df.columns:
        if col not in cols:
            cols.append(col)

    # Before reordering, ensure all expected columns exist; fill missing ones with NaN to avoid KeyError
    for _col in cols:
        if _col not in final_df.columns:
            final_df[_col] = pd.NA
    final_df = final_df[cols]

    return final_df


def main() -> None:
    args = parse_args()
    start_date, end_date = resolve_date_range(args.start, args.end)
    exch_codes = normalize_exchange(args.exchange)

    # If fetching financial data mode
    if args.fetch_financials:
        exch_tag = "-".join(str(x) for x in exch_codes)
        output_path = args.output or os.path.join(
            DATA_DIR, f"finance_{start_date[:4]}_{end_date[:4]}_{exch_tag}_5y.xlsx"
        )

        print(f"Financial data fetching mode")

        # Read IPO file
        ipo_df = read_ipo_file(start_date, end_date, args.exchange)
        print(f"Read {len(ipo_df)} companies from IPO file")

        # Connect to WRDS
        print(f"Connecting to WRDS ...")
        if args.password:
            os.environ["WRDS_PASSWORD"] = args.password
        db = wrds.Connection(wrds_username=args.username)

        # Fetch financial data
        financial_df = fetch_financial_data_for_ipo(db, ipo_df)

        db.close()

        if financial_df.empty:
            print("No financial data obtained")
            return

        print(f"Obtained {len(financial_df)} financial records in total")

        # Save results
        saved = save_dataframe(financial_df, output_path)
        print(f"Results saved to: {saved}")
        return

    # Original IPO/Delist data fetching mode
    tag = "delist" if args.delist else "ipo"
    exch_tag = "-".join(str(x) for x in exch_codes)
    output_path = args.output or os.path.join(
        DATA_DIR, f"{tag}_{start_date[:4]}_{end_date[:4]}_{exch_tag}.xlsx"
    )

    print(f"Connecting to WRDS ...")
    # wrds.Connection only accepts wrds_username, password must be provided via environment variable WRDS_PASSWORD
    if args.password:
        os.environ["WRDS_PASSWORD"] = args.password
    db = wrds.Connection(wrds_username=args.username)

    use_crsp = getattr(args, "use_crsp", False)
    df, (library, table) = try_query_with_fallbacks(
        db, exch_codes, start_date, end_date, args.limit, delist=args.delist, use_crsp=use_crsp
    )

    db.close()
    print(f"Obtained {len(df)} rows from {library}.{table}")

    saved = save_dataframe(df, output_path)
    print(f"Results saved to: {saved}")


if __name__ == "__main__":
    main()