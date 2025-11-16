from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 确保项目根目录在 sys.path 中，便于 `python src/main.py` 方式运行能找到 `models`
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.logistic_regression import build_model_from_params


def _base_monthly_feature_names() -> List[str]:
    """原始月度层面的基础财务字段名（16 个）。"""
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
    raise ValueError("未能找到年份列，请通过 --time_col 指定（例如 year 或 fyear）。")


def infer_label_column(columns: Iterable[str]) -> str:
    candidates = ["delist", "is_delist", "delisted", "label", "delist_flag"]
    for c in candidates:
        if c in columns:
            return c
    raise ValueError("未能找到标签列（是否退市）。请通过 --label_col 指定（如 delist）。")


def default_feature_list() -> List[str]:
    # 来自截图推断的常见财务指标名称；主程序会与实际列名求交集
    return _base_monthly_feature_names()


def default_yearly_aggregated_feature_list() -> List[str]:
    """年度聚合后的默认特征列表（用于“用今年预测下一年”）。
    - 所有基础变量的 last 与 mean
    - 波动性 std（仅用于几个比率/周转类）
    - 年内趋势 slope（用于盈利/规模代表性变量）
    - 披露次数 obs_count_in_year
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
    """从月度数据聚合到年度，并生成下一年标签 y_next_year。
    仅使用当年信息构造年度特征；可选按 cutoff_month 截止，避免“年末”潜在信息泄漏。
    """
    assert "permno" in df.columns, "数据缺少公司标识列 permno"
    year_col = "public_year" if "public_year" in df.columns else infer_time_column(df.columns)
    month_col = "public_month"
    features = [c for c in _base_monthly_feature_names() if c in df.columns]

    dfw = df.copy()
    if month_col not in dfw.columns:
        # 无月份信息则视作 12 月（年末），依然可进行“last/mean”等聚合
        dfw[month_col] = 12
    # 只保留需要的列，避免 groupby 额外负担
    keep_cols = list(dict.fromkeys(["permno", year_col, month_col, "delist"] + features))
    dfw = dfw[keep_cols]

    # 数值化并清理 inf
    for c in features:
        dfw[c] = pd.to_numeric(dfw[c], errors="coerce")
    dfw = dfw.replace([np.inf, -np.inf], np.nan)

    # 可选：年度内仅保留截止月份之前的数据
    if cutoff_month is not None:
        dfw = dfw[dfw[month_col] <= int(cutoff_month)]

    # 排序
    dfw = dfw.sort_values(["permno", year_col, month_col], kind="mergesort")

    # 年度聚合：last/mean/std/斜率/观测次数
    # last/mean
    last_vals = dfw.groupby(["permno", year_col], as_index=True)[features].last().add_suffix("_last")
    mean_vals = dfw.groupby(["permno", year_col], as_index=True)[features].mean().add_suffix("_mean")
    # std（只计算子集）
    std_cols = [c for c in ["quick_ratio", "de_ratio", "curr_ratio", "at_turn", "inv_turn"] if c in features]
    std_vals = (
        dfw.groupby(["permno", year_col], as_index=True)[std_cols].std().add_suffix("_std")
        if std_cols
        else pd.DataFrame(index=last_vals.index)
    )

    # 斜率（对月份序列做最小二乘，中心化后的简单一元回归）
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
        # 仅选择计算所需列，避免 pandas 对分组列的 apply 行为变更带来的警告
        slopes = (
            dfw.groupby(["permno", year_col])[[month_col] + slope_cols]
            .apply(lambda g: slope_per_group(g, slope_cols))
            .astype(float)
        )
    else:
        slopes = pd.DataFrame(index=last_vals.index)

    # 披露次数
    obs = dfw.groupby(["permno", year_col]).size().rename("obs_count_in_year").to_frame()

    # 组合年度特征
    yearly = (
        last_vals.join(mean_vals, how="outer")
        .join(std_vals, how="outer")
        .join(slopes, how="outer")
        .join(obs, how="outer")
        .reset_index()
    )

    # 生成“下一年”标签：按公司→年去重排序后 shift(-1)
    if "delist" not in dfw.columns:
        raise ValueError("数据缺少 delist 列，无法构造下一年标签 y_next_year。")
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

    # 仅保留存在下一年标签的行
    yearly = yearly[yearly["y_next_year"].notna()].copy()
    yearly["y_next_year"] = yearly["y_next_year"].astype(int)

    # 统一列名期望
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
    """公司级切分：按公司 IPO 时间（若无则用首个年份）全量分配到 train/val/test。
    满足“同一公司的所有年份仅位于一个集合”，并按公司时间先后顺序分层。
    返回与 df 等长的 pd.Series，值为 train/val/test。
    """
    r_train, r_val, r_test = ratios
    if not np.isclose(r_train + r_val + r_test, 1.0):
        raise ValueError("ratios 之和必须为 1.0")
    if company_col not in df.columns:
        raise ValueError(f"缺少分组列 {company_col}")
    if year_col not in df.columns:
        raise ValueError(f"缺少年份列 {year_col}")

    # 计算公司级排序键
    comp = df[[company_col, year_col]].drop_duplicates().copy()
    if ipo_col and ipo_col in df.columns:
        ipo_key = (
            df[[company_col, ipo_col]]
            .dropna(subset=[ipo_col])
            .sort_values([company_col, ipo_col], kind="mergesort")
            .drop_duplicates(subset=[company_col], keep="first")
        )
        comp = comp.merge(ipo_key, on=company_col, how="left")
        # 若少数公司无 ipodate，用其最早年份的 1 月作为近似
        comp["_sort_key"] = comp[ipo_col]
        if comp["_sort_key"].isna().any():
            fallback = (
                df.groupby(company_col)[year_col].min().rename("_first_year").to_frame()
            )
            comp = comp.merge(fallback, on=company_col, how="left")
            # 构造一个可比较的日期：_first_year-01-01
            comp.loc[comp["_sort_key"].isna(), "_sort_key"] = pd.to_datetime(
                comp.loc[comp["_sort_key"].isna(), "_first_year"].astype(int).astype(str) + "-01-01",
                errors="coerce",
            )
    else:
        # 没有 ipodate，就用公司最早年份作为排序键
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

    # 映射回原 df
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
    # 模型相关可调参数
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="LogisticRegression 的正则强度的倒数（越大越弱正则）",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=1000,
        help="LogisticRegression 的最大迭代次数",
    )
    parser.add_argument(
        "--predict_next_year",
        action="store_true",
        help="启用年度聚合：使用当年（聚合后）特征预测下一年是否退市（生成 y_next_year）",
        default=True,
    )
    parser.add_argument(
        "--cutoff_month",
        type=int,
        default=None,
        help="年度聚合的截止月份（仅保留当年 <= cutoff_month 的月度观测；默认 None 表示使用全年信息）",
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

    # 若启用“用今年预测下一年”，先将月度数据聚合为年度并构造 y_next_year
    if bool(args.predict_next_year):
        df = build_yearly_from_monthly(df, cutoff_month=args.cutoff_month)
        # 覆盖为年度任务的列名
        args.label_col = "y_next_year"
        args.time_col = "public_year"
    else:
        if args.time_col is None:
            args.time_col = infer_time_column(df.columns)
        if args.label_col is None:
            args.label_col = infer_label_column(df.columns)

    # 选择特征
    if args.features:
        feat_list = [x.strip() for x in args.features.split(",") if x.strip()]
    else:
        feat_list = (
            default_yearly_aggregated_feature_list()
            if bool(args.predict_next_year)
            else default_feature_list()
        )

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

    # 若启用年度预测下一年，优先采用“公司级（按 IPO/首年）”切分，避免同一公司跨集合
    if bool(args.predict_next_year):
        splits = company_level_split_by_ipo(
            df=df,
            company_col=args.group_col,
            ipo_col="ipodate" if "ipodate" in df.columns else None,
            year_col=args.time_col,
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

    # 训练集类别检查：若仅有单一类别，尝试从 val/test 引入含正类的公司
    def _has_two_classes(arr: np.ndarray) -> bool:
        u = np.unique(arr)
        return u.size >= 2

    if not _has_two_classes(y_train):
        # 找出 val/test 中有哪些公司带正类
        company_col = args.group_col
        need_pos = 1 if (len(np.unique(y_train)) == 1 and np.unique(y_train)[0] == 0) else 0
        # 从 val 优先寻找
        candidate_val_companies = df.loc[val_idx & (df[args.label_col] == 1), company_col].unique().tolist()
        candidate_test_companies = df.loc[test_idx & (df[args.label_col] == 1), company_col].unique().tolist()
        moved = False
        for comp_list in [candidate_val_companies, candidate_test_companies]:
            if comp_list:
                comp_to_move = comp_list[0]
                # 将该公司的全部样本标记为 train
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
                    print(f"[提示] 训练集仅含单一类别，已将公司 {comp_to_move} 从验证/测试移动到训练以确保可训练。")
                    break
        if not moved and not _has_two_classes(y_train):
            raise ValueError("训练集仅含单一类别，且在验证/测试集中找不到正类公司用于调整。请检查数据或修改切分比例。")

    # 模型构建
    if args.model_name != "logistic_regression":
        raise ValueError("目前仅支持 model_name=logistic_regression")
    model = build_model_from_params(
        {
            "random_state": args.random_state,
            "C": float(args.C),
            "max_iter": int(args.max_iter),
        }
    )

    # 训练前信息：样本规模与类别分布、参数与流水线
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
        print("模型参数:", model.get_params())
    except Exception:
        pass
    try:
        # 简要打印流水线步骤名（组件内部参数较多，保持简洁）
        from pprint import pprint
        if getattr(model, "pipeline", None) is not None:
            print("流水线步骤:", [name for name, _ in model.pipeline.steps])
    except Exception:
        pass

    # 训练
    print("[训练] 开始训练模型...")
    model.fit(X_train, y_train)
    print("[训练] 训练完成")

    # 评估
    val_metrics = model.evaluate(X_val, y_val)
    test_metrics = model.evaluate(X_test, y_test)

    print("验证集指标:", val_metrics)
    print("测试集指标:", test_metrics)
    print(f"使用特征（{len(selected_features)}）: {', '.join(selected_features)}")

    # 显示当前模型参数（系数与截距）
    try:
        clf = model.pipeline.named_steps["clf"]  # type: ignore
        coef = getattr(clf, "coef_", None)
        intercept = getattr(clf, "intercept_", None)
        if coef is not None:
            coef = np.asarray(coef).reshape(-1)
            # 对齐特征名
            feature_coefs = list(zip(selected_features, coef))
            # 取绝对值前 15 个
            topk = sorted(feature_coefs, key=lambda x: abs(x[1]), reverse=True)[:15]
            print("截距(intercept):", float(intercept[0]) if intercept is not None else "N/A")
            print("权重 Top-15 (按|coef|排序):")
            for name, w in topk:
                print(f"  {name}: {w:.6f}")
    except Exception:
        pass

    # 保存模型
    if args.save_model_path:
        os.makedirs(os.path.dirname(args.save_model_path), exist_ok=True)
        model.save(args.save_model_path)
        print(f"模型已保存到: {args.save_model_path}")


if __name__ == "__main__":
    main()


