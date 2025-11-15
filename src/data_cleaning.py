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

    # 仅在实际退市发生的年份（IPO 年份 + delist_year）标记 delist=1，其余年份为 0
    event_year = finance_df["ipo_date_y"].dt.year + finance_df["delist_year"].astype("Int64")
    finance_df["delist"] = 0
    finance_df.loc[(finance_df["ipo_date_y"].notna()) & (finance_df["public_year"] == event_year), "delist"] = 1

    finance_merged = finance_df.drop(columns=["ipo_date_y", "delist_date", "public_date"])

    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    finance_merged.to_excel(output_path, index=False)


def merge_final_tables(input_paths: list[str], output_path: str) -> None:
    # 读取三个清洗后的分表并合并
    dfs = [pd.read_excel(p) for p in input_paths]
    merged_df = pd.concat(dfs, ignore_index=True)

    # 删除全空行与全空列
    merged_df = merged_df.dropna(axis=0, how="all").dropna(axis=1, how="all")

    # 去除列名首尾空格
    merged_df = merged_df.rename(columns=lambda x: x.strip())

    # 关键字段类型
    if "permno" in merged_df.columns:
        merged_df["permno"] = merged_df["permno"].astype("Int64")
    if "public_year" in merged_df.columns:
        merged_df["public_year"] = merged_df["public_year"].astype("Int64")

    # 排序（确保按 permno -> public_year -> public_month 的时间顺序）
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

    # 组内填充：按实体分组（必须 permno；若无则退化为全局），先 bfill 再 ffill，保证尾部缺失也能用上方最近值填充
    group_cols = []
    if "permno" in merged_df.columns:
        group_cols = ["permno"]
    elif "gvkey" in merged_df.columns:
        group_cols = ["gvkey"]
    if group_cols:
        merged_df = merged_df.groupby(group_cols, group_keys=False).apply(lambda g: g.bfill().ffill())
    else:
        merged_df = merged_df.bfill().ffill()

    # 删除冗余列（若存在）
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

    # 关键字段缺失删除（仅删除存在于数据中的那些列的缺失）
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

    # 输出
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    merged_df.to_excel(output_path, index=False)


def build_default_paths(index: int) -> tuple[str, str, str]:
    # 以当前项目根目录为默认搜索位置，避免个人桌面的绝对路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(project_root, "data")
    finance_path = os.path.join(data_dir, f"finance_1974_2024_{index}_5y.xlsx")
    ipo_delist_path = os.path.join(data_dir, f"ipo_delist_within5y_1974_2024_{index}.xlsx")
    # 输出到 data 目录，保持与仓库中现有文件命名一致
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
        description="统一的数据清洗脚本：生成 final_table_{i}.xlsx；支持 --merge 合并 1/2/3 为 final_table_merged.xlsx"
    )
    parser.add_argument(
        "--index",
        type=int,
        choices=[1, 2, 3],
        help="选择 1/2/3 使用项目内的默认输入输出路径",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="合并 data/final_table_1/2/3.xlsx 并输出 data/final_table_merged.xlsx",
    )
    parser.add_argument(
        "--finance-path",
        type=str,
        help="财务数据 Excel 的绝对路径（覆盖 --index 的默认路径）",
    )
    parser.add_argument(
        "--ipo-delist-path",
        type=str,
        help="IPO/退市映射 Excel 的绝对路径（覆盖 --index 的默认路径）",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="输出 Excel 的绝对路径（覆盖 --index 的默认路径）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 仅做合并时，跳过清洗必需参数校验
    if not args.merge:
        if args.index is None and (not args.finance_path or not args.ipo_delist_path or not args.output_path):
            raise SystemExit(
                "请提供 --index 1|2|3，或同时提供 --finance-path、--ipo-delist-path 与 --output-path；若仅需合并请使用 --merge。"
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


