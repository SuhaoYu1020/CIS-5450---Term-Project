from __future__ import annotations

import argparse
import os
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from models.logistic_regression import build_model_from_params


def infer_time_column(columns: Iterable[str]) -> str:
    candidates = ["public_year", "year", "fyear", "fiscal_year", "datadate", "year_end"]
    for c in candidates:
        if c in columns:
            return c
    raise ValueError("未能找到年份列，请通过 --time_col 指定（例如 year 或 fyear）。")


def infer_label_column(columns: Iterable[str]) -> str:
    candidates = ["delist", "is_delist", "delisted", "label", "delist_flag"]
    for c in candidates:
        if c in columns:
            return c
    raise ValueError("未能找到标签列（是否退市）。请通过 --label_col 指定（如 delist）。")


def default_feature_list() -> List[str]:
    # 来自截图推断的常见财务指标名称；主程序会与实际列名求交集
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


def chronological_group_split(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> pd.Series:
    """为每行返回 split 标记：train/val/test（按每个 group 的时间先后顺序切分）。"""
    r_train, r_val, r_test = ratios
    if not np.isclose(r_train + r_val + r_test, 1.0):
        raise ValueError("ratios 之和必须为 1.0")

    splits = []
    for _, g in df.groupby(group_col, sort=False):
        # 若存在 public_month，且使用 public_year 作为时间列，则进行两级排序
        if time_col == "public_year" and "public_month" in g.columns:
            g_sorted = g.sort_values([time_col, "public_month"], kind="mergesort")
        else:
            g_sorted = g.sort_values(time_col, kind="mergesort")  # 稳定排序
        n = len(g_sorted)
        i1 = int(np.floor(n * r_train))
        i2 = int(np.floor(n * (r_train + r_val)))
        # 边界保护：确保至少把最后一条留在 test（若样本数足够）
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
            f"指定的特征均不存在于数据中。缺失示例：{missing[:5]}。请用 --features 指定正确列名。"
        )
    if missing:
        print(f"[警告] 下列特征在数据中未找到，将被忽略：{', '.join(missing)}")
    return df[present], present


def sanitize_feature_df(
    X: pd.DataFrame, drop_zero_variance: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """将特征转为数值、替换 inf/-inf 为 NaN，并可选移除全 NaN/零方差列。"""
    X_numeric = X.apply(pd.to_numeric, errors="coerce")
    X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)

    dropped: List[str] = []
    all_nan_cols = X_numeric.columns[X_numeric.isna().all()].tolist()
    if all_nan_cols:
        dropped.extend(all_nan_cols)
        X_numeric = X_numeric.drop(columns=all_nan_cols)
        print(f"[警告] 下列特征列全为缺失，已移除：{', '.join(all_nan_cols)}")

    if drop_zero_variance:
        zero_var_cols = [c for c in X_numeric.columns if X_numeric[c].nunique(dropna=True) <= 1]
        if zero_var_cols:
            dropped.extend(zero_var_cols)
            X_numeric = X_numeric.drop(columns=zero_var_cols)
            print(f"[提示] 下列特征列零方差/常数列，已移除：{', '.join(zero_var_cols)}")

    return X_numeric, dropped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict delist using financial statements.")
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join("data", "final_table_merged.xlsx"),
        help="数据文件路径（Excel）",
    )
    parser.add_argument("--model_name", type=str, default="logistic_regression", help="模型名称")
    parser.add_argument("--label_col", type=str, default="delist", help="标签列名（0/1 是否退市）")
    parser.add_argument("--group_col", type=str, default="permno", help="分组列名（公司 id）")
    parser.add_argument("--time_col", type=str, default="public_year", help="年份列名")
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="逗号分隔的特征列名；若不提供则使用内置默认并与数据列求交集",
    )
    parser.add_argument("--train_ratio", type=float, default=0.7, help="训练集比例")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="验证集比例")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="测试集比例")
    parser.add_argument("--random_state", type=int, default=42, help="随机种子")
    parser.add_argument("--save_model_path", type=str, default=None, help="模型保存路径")
    parser.add_argument(
        "--drop_zero_variance",
        action="store_true",
        help="移除零方差特征列（常数列）以提升鲁棒性",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 读取数据
    df = pd.read_excel(args.data_path)
    if args.time_col is None:
        args.time_col = infer_time_column(df.columns)
    if args.label_col is None:
        args.label_col = infer_label_column(df.columns)

    # 选择特征
    if args.features:
        feat_list = [x.strip() for x in args.features.split(",") if x.strip()]
    else:
        feat_list = default_feature_list()

    X_df, selected_features = build_feature_matrix(df, feat_list)
    # 清洗特征，替换无穷/异常为 NaN，删除全 NaN/零方差列
    X_df, dropped_cols = sanitize_feature_df(X_df, drop_zero_variance=bool(args.drop_zero_variance))
    if dropped_cols:
        selected_features = [c for c in selected_features if c not in dropped_cols]
    y = df[args.label_col].astype(int).values

    # 构造 split
    splits = chronological_group_split(
        df=df,
        group_col=args.group_col,
        time_col=args.time_col,
        ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
    )

    # 切分数据
    train_idx = splits == "train"
    val_idx = splits == "val"
    test_idx = splits == "test"

    X_train = X_df.loc[train_idx].values
    y_train = y[train_idx.values]
    X_val = X_df.loc[val_idx].values
    y_val = y[val_idx.values]
    X_test = X_df.loc[test_idx].values
    y_test = y[test_idx.values]

    # 模型构建
    if args.model_name != "logistic_regression":
        raise ValueError("目前仅支持 model_name=logistic_regression")
    model = build_model_from_params({"random_state": args.random_state})

    # 训练
    model.fit(X_train, y_train)

    # 评估
    val_metrics = model.evaluate(X_val, y_val)
    test_metrics = model.evaluate(X_test, y_test)

    print("验证集指标:", val_metrics)
    print("测试集指标:", test_metrics)
    print(f"使用特征（{len(selected_features)}）: {', '.join(selected_features)}")

    # 保存模型
    if args.save_model_path:
        os.makedirs(os.path.dirname(args.save_model_path), exist_ok=True)
        model.save(args.save_model_path)
        print(f"模型已保存到: {args.save_model_path}")


if __name__ == "__main__":
    main()


