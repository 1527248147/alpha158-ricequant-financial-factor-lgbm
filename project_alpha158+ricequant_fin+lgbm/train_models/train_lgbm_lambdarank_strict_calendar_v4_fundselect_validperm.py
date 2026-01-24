# train_lgbm_lambdarank_strict_calendar_v4_fundselect_validperm.py
# -*- coding: utf-8 -*-


# 运行指令：
# python train_models\train_lgbm_lambdarank_strict_calendar_v4_fundselect_validperm.py ^
#   --data_dir "labeled\_calendar\labeled_yearly_parquet" ^
#   --out_dir  "train_models\_train_lambdarank_v4_alpha_plus_fundTop15_seed42" ^
#   --train_years 2021-2023 --valid_years 2024 --test_years 2025 ^
#   --label_col label__ret_1d_qlib ^
#   --missing_drop_thresh 0.98 ^
#   --cast_float32 ^
#   --exclude_regex "(?i)dividend" ^
#   --fund_perm_csv "train_models\feature_importance_valid_perm.csv" ^
#   --fund_perm_topk 15 ^
#   --fund_perm_min_drop 1e-4 ^
#   --clip_y_abs 0 ^
#   --relevance_bins 10 ^
#   --min_group_size 30 ^
#   --truncation_level 50 ^
#   --num_boost_round 5000 ^
#   --early_stopping_rounds 400 ^
#   --save_test_pred ^
#   --seed 42 ^
#   --fund_perm_min_drop 0

# -*- coding: utf-8 -*-
"""
本脚本用途（给其他开发者看的总览）
=================================

这是一个用于 A 股日频截面排序的 LightGBM LambdaRank 训练脚本，核心特点：

1) 数据来源
   - 输入数据目录 data_dir 下按年存储的 parquet（year=YYYY.parquet）
   - 每条样本对应 (datetime, order_book_id) 一只股票在某个交易日 t 的特征
   - 数据中包含严格可交易语义的 label：
       label__ret_1d_qlib = Close(t+2) / Close(t+1) - 1
     注意：t+1 买入、t+2 卖出，如果 t+1 或 t+2 在 OHLCV 缺行（停牌/缺数据），label=NaN

2) 严格过滤（防止“不可交易样本”污染训练）
   - 本脚本会在读取数据后立即删除 label 为 NaN 的样本：
       df = df[df[label_col].notna()]
     这样训练/验证/测试使用的样本都严格满足可交易语义。

3) 特征处理策略
   - 所有特征列从 parquet schema 自动推断（排除 label/date/id）
   - 支持通过 exclude_regex 排除某类特征（这里默认只排除 dividend 相关）
   - fund__ 特征可选：从 valid_perm（验证集 permutation importance）文件里挑 TopK
     评估指标是 valid_rankic_drop_mean（打乱该特征后验证集 RankIC 的下降幅度）
     只有 drop >= min_drop 的 fund 特征才会被选入（可能导致实际选到的 fund 少于 topk）

4) 训练目标（LambdaRank）
   - LightGBM objective = "lambdarank"
   - 标签并不是连续收益，而是“每天内按收益分位离散”的 relevance（0..n_bins-1）
     这样模型学到的是“同一天股票的相对排序”，更贴近 TopK long-only 策略。

5) 评估指标（RankIC）
   - 训练与验证的评估使用 Spearman RankIC：每天算一次“预测分数与真实收益”的秩相关
   - 对所有天取均值/标准差/IR，作为训练质量的判断依据
   - RankIC 使用的是原始收益 raw_y（未离散），更贴近真实交易目标

6) 输出产物（回测/复现用）
   - model.txt：训练后的 LightGBM 模型
   - used_features.txt/csv：本次训练真正使用的特征列表（回测推理必须用同样列顺序）
   - excluded_features.csv：被 regex 排除的特征
   - selected_fund_features.csv（若启用 valid_perm 选 fund）：被挑选的 fund 特征
   - summary.json：本次训练的所有关键配置与指标摘要
   - test_predictions.csv（可选）：测试集预测分数与真实收益，用于离线分析
"""

import argparse
import gc
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb


# -----------------------------
# IO + schema
# -----------------------------
def safe_makedirs(p: Path):
    """创建输出目录（若不存在则创建）。"""
    p.mkdir(parents=True, exist_ok=True)


def parse_years(spec: str) -> List[int]:
    """
    解析年份参数字符串为 int 列表。
    支持三种写法：
      - "2021-2023" -> [2021, 2022, 2023]
      - "2021,2023,2025" -> [2021, 2023, 2025]
      - "2024" -> [2024]
    """
    spec = (spec or "").strip()
    if not spec:
        return []
    if "," in spec:
        ys = [int(x.strip()) for x in spec.split(",") if x.strip()]
        return sorted(set(ys))
    m = re.match(r"^\s*(\d{4})\s*-\s*(\d{4})\s*$", spec)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if a > b:
            a, b = b, a
        return list(range(a, b + 1))
    if re.match(r"^\d{4}$", spec):
        return [int(spec)]
    raise ValueError(f"Invalid years spec: {spec}")


def find_year_files(data_dir: Path) -> Dict[int, Path]:
    """
    在 data_dir 下查找按年 parquet 文件。
    约定：文件名中包含 20xx（比如 year=2024.parquet 或 2024.parquet 都可以）
    返回：{year: file_path}，按 year 排序。
    """
    out = {}
    for fp in data_dir.glob("*.parquet"):
        m = re.search(r"(20\d{2})", fp.name)
        if not m:
            continue
        out[int(m.group(1))] = fp
    if not out:
        raise FileNotFoundError(f"No year parquet detected under: {data_dir}")
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def schema_columns(sample_parquet: Path) -> List[str]:
    """读取 parquet 文件 schema，返回所有列名。"""
    pf = pq.ParquetFile(sample_parquet)
    return [f.name for f in pf.schema_arrow]


def load_parquet(path: Path, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    读取 parquet 为 pandas DataFrame。
    - columns：只读指定列（减少 IO 和内存占用）
    - split_blocks/self_destruct：更节省内存（对大数据集很关键）
    """
    table = pq.read_table(path, columns=columns)
    return table.to_pandas(split_blocks=True, self_destruct=True)


def load_years_concat_strict(
    year_files: Dict[int, Path],
    years: List[int],
    columns_to_read: List[str],
    label_col: str,
    cast_float32: bool,
) -> pd.DataFrame:
    """
    严格读取指定年份数据并拼接：
    - 读取 parquet 的 columns_to_read
    - 强制删除 label 为 NaN 的行（不可交易样本：t+1 或 t+2 缺行，或年末尾部 t+2 不存在）
    - 可选：把数值特征 cast 成 float32（显著降低内存）
    """
    dfs = []
    for y in years:
        fp = year_files.get(y)
        if fp is None:
            print(f"[WARN] Missing parquet for year={y}, skip.")
            continue

        df = load_parquet(fp, columns=columns_to_read)
        if label_col not in df.columns:
            raise KeyError(f"Label col '{label_col}' not found in {fp}")

        before = len(df)
        # ✅ 关键：严格过滤不可交易样本（label NaN）
        df = df[df[label_col].notna()].copy()
        after = len(df)
        print(f"[INFO] year={y} drop NaN label: {before:,} -> {after:,}")

        # 可选：把数值列转 float32，节省内存/加速训练
        if cast_float32:
            for c in df.columns:
                if c == label_col:
                    continue
                if pd.api.types.is_numeric_dtype(df[c].dtype):
                    df[c] = df[c].astype(np.float32)

        dfs.append(df)
        print(f"[INFO] Loaded year={y}: rows={len(df):,}, cols={len(df.columns)}")

    if not dfs:
        return pd.DataFrame(columns=columns_to_read)
    return pd.concat(dfs, ignore_index=True)


# -----------------------------
# Feature selection helpers
# -----------------------------
def build_feature_candidates(schema_cols: List[str], label_col: str, date_col: str, id_col: str) -> List[str]:
    """
    从 schema 中推断所有“候选特征列”：
    - 排除 label 列、date 列、id 列
    - 排除所有以 label__ 开头的列（防止未来信息泄露）
    """
    drop_cols = {label_col, date_col, id_col}
    feats = []
    for c in schema_cols:
        if c.startswith("label__"):
            continue
        if c in drop_cols:
            continue
        feats.append(c)
    return feats


def filter_by_regex_exclude(feat_cols: List[str], exclude_regex: str) -> Tuple[List[str], List[str]]:
    """
    按 regex 排除特征：
    - exclude_regex 默认 (?i)dividend：排除所有名字里带 dividend（大小写不敏感）的特征
    返回：(保留的特征列表, 被排除的特征列表)
    """
    rx = re.compile(exclude_regex) if exclude_regex else None
    if rx is None:
        return feat_cols, []
    kept, excluded = [], []
    for c in feat_cols:
        if rx.search(str(c)):
            excluded.append(c)
        else:
            kept.append(c)
    return kept, excluded


def select_fund_from_valid_perm(
    perm_csv: Path,
    topk: int,
    min_drop: float,
    exclude_regex: str,
) -> pd.DataFrame:
    """
    从验证集 permutation importance 文件（valid_perm）里挑 fund__ 特征：
    - valid_perm.csv 必须包含列：feature, valid_rankic_drop_mean
    - 先取 feature 以 fund__ 开头的行
    - 再排除 exclude_regex（例如 dividend）
    - 按 valid_rankic_drop_mean 降序排序
    - 再过滤 drop >= min_drop
    - 最后取 TopK

    注意：
    - 若满足 drop>=min_drop 的 fund 特征少于 topk，则最终选择数会 < topk
    """
    df = pd.read_csv(perm_csv)
    df["feature"] = df["feature"].astype(str)
    df["valid_rankic_drop_mean"] = pd.to_numeric(df["valid_rankic_drop_mean"], errors="coerce").fillna(0.0)

    fund = df[df["feature"].str.startswith("fund__")].copy()

    if exclude_regex:
        rx = re.compile(exclude_regex)
        fund = fund[~fund["feature"].apply(lambda x: bool(rx.search(x)))]

    fund = fund.sort_values("valid_rankic_drop_mean", ascending=False)

    if min_drop is not None:
        fund = fund[fund["valid_rankic_drop_mean"] >= float(min_drop)]

    fund = fund.head(int(topk)).reset_index(drop=True)
    return fund


def filter_features_on_train(
    df_train: pd.DataFrame,
    feat_cand: List[str],
    missing_drop_thresh: Optional[float],
) -> List[str]:
    """
    只基于训练集做特征清洗（避免泄露验证/测试信息）：
    1) 只保留在 df_train 中存在且为数值 dtype 的列
    2) 删除全 NaN 或方差为 0 的列（无法提供有效信息）
    3) 可选：删除缺失率 >= missing_drop_thresh 的列（例如 0.98 表示缺 98% 以上删掉）
    4) 安全检查：任何 label__ 开头的列都不能作为特征
    """
    # 仅保留数值列
    feat_cols = [c for c in feat_cand if c in df_train.columns and pd.api.types.is_numeric_dtype(df_train[c].dtype)]

    # 删除零方差/全 NaN
    var = df_train[feat_cols].var(axis=0, skipna=True)
    feat_cols = [c for c in feat_cols if pd.notna(var[c]) and var[c] > 0]
    print(f"[INFO] After drop zero-var/all-NaN: {len(feat_cols)} features")

    # 删除缺失率过高的特征（越大越宽松）
    if missing_drop_thresh is not None:
        miss = df_train[feat_cols].isna().mean()
        feat_cols2 = [c for c in feat_cols if miss[c] < missing_drop_thresh]
        print(f"[INFO] After missing_drop_thresh={missing_drop_thresh}: {len(feat_cols2)} features (from {len(feat_cols)})")
        feat_cols = feat_cols2

    # 防泄露：label 列绝不能进入特征
    leaked = [c for c in feat_cols if str(c).startswith("label__")]
    if leaked:
        raise RuntimeError(f"LEAK DETECTED: label__* appeared in features: {leaked[:20]}")
    return feat_cols


def save_used_features(out_dir: Path, feat_cols: List[str], excluded_cols: List[str], selected_fund_df: Optional[pd.DataFrame]):
    """
    保存本次训练真正使用的特征列表（回测推理必须一致）：
    - used_features.txt：按行保存（顺序 = 训练时特征顺序）
    - used_features.csv：同内容的 csv
    - excluded_features.csv：被 regex 排除的特征
    - selected_fund_features.csv：从 valid_perm 里挑出的 fund（如果启用）
    """
    out_dir = Path(out_dir)
    (out_dir / "used_features.txt").write_text("\n".join(map(str, feat_cols)) + "\n", encoding="utf-8")
    pd.DataFrame({"feature": feat_cols}).to_csv(out_dir / "used_features.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"excluded_feature": excluded_cols}).to_csv(out_dir / "excluded_features.csv", index=False, encoding="utf-8-sig")

    if selected_fund_df is not None:
        selected_fund_df.to_csv(out_dir / "selected_fund_features.csv", index=False, encoding="utf-8-sig")

    print(f"[INFO] Saved used_features + excluded_features (+ selected_fund_features if any).")


# -----------------------------
# Grouping + RankIC
# -----------------------------
def sort_and_group(df: pd.DataFrame, date_col: str, id_col: str) -> Tuple[pd.DataFrame, List[int]]:
    """
    对数据按“交易日 + 股票代码”排序，并构造 LightGBM ranker 所需的 group：
    - group 的含义：每一天是一组（同一天所有股票一起排序）
    - 返回 (排序后的 df, groups)
      groups 是一个 list，每个元素是该天包含的样本数，例如 [3800, 3795, 3810, ...]
    """
    dt = pd.to_datetime(df[date_col], errors="coerce")
    if dt.isna().any():
        raise ValueError(f"{date_col} contains unparsable values.")
    # 只保留日期到“天”级别，忽略时分秒
    day = dt.values.astype("datetime64[D]")

    df2 = df.copy()
    df2["_day"] = day
    # 稳定排序（mergesort）保证相同 key 时顺序一致，利于复现
    if id_col in df2.columns:
        df2 = df2.sort_values(["_day", id_col], kind="mergesort")
    else:
        df2 = df2.sort_values(["_day"], kind="mergesort")

    # 计算每一天的样本数（group size）
    day_sorted = df2["_day"].to_numpy()
    groups = []
    n = len(day_sorted)
    i = 0
    while i < n:
        j = i + 1
        while j < n and day_sorted[j] == day_sorted[i]:
            j += 1
        groups.append(int(j - i))
        i = j

    df2 = df2.drop(columns=["_day"])
    return df2, groups


def drop_small_groups(df_s: pd.DataFrame, g: List[int], min_g: int) -> Tuple[pd.DataFrame, List[int]]:
    """
    丢弃“样本数太少的天”，避免某些天只有很少股票导致 rank loss / rankic 不稳定。
    - min_g=30 表示：某天少于 30 只股票则整天不要。
    """
    if min_g <= 1:
        return df_s, g
    keep_mask = np.zeros(len(df_s), dtype=bool)
    new_g = []
    start = 0
    for size in g:
        end = start + size
        if size >= min_g:
            keep_mask[start:end] = True
            new_g.append(size)
        start = end
    return df_s[keep_mask].copy(), new_g


def spearman_rankic_by_group(y: np.ndarray, p: np.ndarray, group_sizes: List[int]) -> Tuple[float, float, float, int]:
    """
    计算 RankIC（Spearman 秩相关）：
    - 对每一天：
        计算 真实收益 y 与 预测分数 p 的秩相关系数
    - 返回：
        mean：所有天 RankIC 平均值
        std ：所有天 RankIC 标准差
        ir  ：Information Ratio = mean/std
        n_days：参与统计的天数
    """
    ics = []
    start = 0
    for g in group_sizes:
        end = start + g
        yy = y[start:end]
        pp = p[start:end]
        start = end

        # 样本太少没意义
        if yy.size < 3:
            continue

        # 去掉 NaN/inf（理论上 y 已经过滤，但 pred 也可能出问题）
        m = np.isfinite(yy) & np.isfinite(pp)
        yy = yy[m]
        pp = pp[m]
        if yy.size < 3:
            continue

        # Spearman = Pearson(corr(rank(y), rank(p)))
        ry = pd.Series(yy).rank(method="average").to_numpy()
        rp = pd.Series(pp).rank(method="average").to_numpy()
        if np.std(ry) == 0 or np.std(rp) == 0:
            continue

        ic = float(np.corrcoef(ry, rp)[0, 1])
        if np.isfinite(ic):
            ics.append(ic)

    if not ics:
        return np.nan, np.nan, np.nan, 0

    ics = np.asarray(ics, dtype=np.float64)
    mean = float(np.mean(ics))
    std = float(np.std(ics, ddof=1)) if ics.size > 1 else np.nan
    ir = float(mean / std) if np.isfinite(std) and std > 0 else np.nan
    return mean, std, ir, int(ics.size)


def make_relevance_per_day(raw_y: np.ndarray, group_sizes: List[int], n_bins: int) -> np.ndarray:
    """
    将连续收益 raw_y 转成 LambdaRank 所需的“离散 relevance 标签”（0..n_bins-1）：
    - 对每一天单独处理（只在当天截面内离散）
    - 做法：按收益 rank -> 映射到分位区间 -> 得到 0..n_bins-1

    直觉：
    - 最高收益分位得到最大 relevance（如 9）
    - 最低收益分位得到最小 relevance（如 0）
    """
    rel = np.empty_like(raw_y, dtype=np.float32)
    start = 0
    for g in group_sizes:
        end = start + g
        y = raw_y[start:end]

        # 极端情况：只有1个样本或全 NaN
        if g <= 1 or np.all(~np.isfinite(y)):
            rel[start:end] = 0
            start = end
            continue

        m = np.isfinite(y)
        y2 = y[m]
        if y2.size <= 1 or np.all(y2 == y2[0]):
            rel[start:end] = 0
            start = end
            continue

        # rank -> percent -> bin
        r = pd.Series(y2).rank(method="average").to_numpy()
        pct = (r - 1.0) / max(1.0, (y2.size - 1.0))
        b = np.floor(pct * n_bins).astype(np.int32)
        b = np.clip(b, 0, n_bins - 1)

        out = np.zeros(g, dtype=np.float32)
        out[m] = b.astype(np.float32)
        rel[start:end] = out
        start = end

    return rel


def clip_y(y: np.ndarray, clip_abs: float) -> np.ndarray:
    """
    对收益做截断（可选）：
    - clip_abs<=0 表示不截断
    - 目的：减少极端收益造成的分位边界抖动（离散 relevance 时更稳定）
    注意：这里截断只用于“构造 relevance”，RankIC 评估仍使用原始收益。
    """
    if clip_abs is None or clip_abs <= 0:
        return y.astype(np.float32, copy=True)
    return np.clip(y, -clip_abs, clip_abs).astype(np.float32, copy=True)


def make_feval_rankic(dset_to_info: dict):
    """
    自定义 LightGBM feval：在 valid 集上实时输出 rank_ic_mean，用于：
    - 观察训练曲线
    - early stopping（如果启用）

    注意：
    - LambdaRank 默认 metric 不一定等同你关心的 RankIC，所以用自定义 feval 更贴近策略目标。
    - feval 的输入 preds 是模型输出的排序分数（不是概率）。
    """
    def _feval(preds: np.ndarray, dataset: lgb.Dataset):
        info = dset_to_info.get(id(dataset))
        if info is None:
            return ("rank_ic_mean", np.nan, True)
        raw_y, group_sizes = info
        mean, std, ir, n_days = spearman_rankic_by_group(raw_y, preds, group_sizes)
        # True 表示“越大越好”
        return ("rank_ic_mean", mean, True)
    return _feval


# -----------------------------
# Params
# -----------------------------
def default_rank_params(seed: int, threads: int, n_bins: int, trunc_level: int) -> dict:
    """
    LambdaRank 的默认超参（相对保守稳定）：
    - learning_rate=0.02，num_boost_round=5000（配合 early stopping）
    - lambdarank_truncation_level：更关注 TopK 排序（例如 50）
    - label_gain = [0,1,2,...,n_bins-1]：relevance 的增益设为线性
    """
    label_gain = list(range(int(n_bins)))
    return {
        "objective": "lambdarank",
        "metric": "None",  # 不用内置 metric，改用自定义 RankIC
        "learning_rate": 0.02,

        "max_depth": 8,
        "num_leaves": 127,
        "min_data_in_leaf": 2000,

        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,

        "lambda_l1": 0.0,
        "lambda_l2": 10.0,
        "min_gain_to_split": 0.02,

        # 重点：TopK 友好（只对前 trunc_level 的位置更敏感）
        "lambdarank_truncation_level": int(trunc_level),

        # relevance 的 gain 映射（0..n_bins-1）
        "label_gain": label_gain,

        "verbosity": -1,
        "seed": int(seed),
        "num_threads": int(threads),

        # 大量特征时 col-wise 更稳
        "force_col_wise": True,
    }


def main():
    # -----------------------------
    # 1) 解析命令行参数
    # -----------------------------
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", type=str, required=True, help="按年 parquet 的输入目录（带 label 的 labeled_yearly_parquet）")
    ap.add_argument("--out_dir", type=str, required=True, help="训练输出目录")

    ap.add_argument("--train_years", type=str, default="2021-2023")
    ap.add_argument("--valid_years", type=str, default="2024")
    ap.add_argument("--test_years", type=str, default="2025")

    ap.add_argument("--label_col", type=str, default="label__ret_1d_qlib")
    ap.add_argument("--date_col", type=str, default="datetime")
    ap.add_argument("--id_col", type=str, default="order_book_id")

    ap.add_argument("--missing_drop_thresh", type=float, default=0.98, help="特征缺失率过滤阈值；<=0 表示不启用")
    ap.add_argument("--cast_float32", action="store_true", help="将数值特征转为 float32 以节省内存")

    # 排除特征（默认只排除 dividend）
    ap.add_argument("--exclude_regex", type=str, default=r"(?i)dividend")

    # fund 特征选择（从 valid_perm 里挑）
    ap.add_argument("--fund_perm_csv", type=str, default="", help="feature_importance_valid_perm.csv")
    ap.add_argument("--fund_perm_topk", type=int, default=9, help="最多保留多少个 fund__ 特征")
    ap.add_argument("--fund_perm_min_drop", type=float, default=1e-4, help="perm drop >= 该阈值才保留 fund")

    # label -> relevance 的处理
    ap.add_argument("--clip_y_abs", type=float, default=0.0, help="对收益截断（仅用于构造 relevance）；<=0 表示不截断")
    ap.add_argument("--relevance_bins", type=int, default=10, help="把每天收益离散成多少档（例如 10 档：0..9）")
    ap.add_argument("--min_group_size", type=int, default=30, help="丢弃样本数少于该值的交易日（每天是一组）")

    # TopK 友好
    ap.add_argument("--truncation_level", type=int, default=50, help="lambdarank_truncation_level，越小越关注TopK")

    # 训练控制
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--num_boost_round", type=int, default=5000)
    ap.add_argument("--early_stopping_rounds", type=int, default=300, help="0 表示不早停")
    ap.add_argument("--save_test_pred", action="store_true", help="输出测试集预测到 csv（便于分析）")

    args = ap.parse_args()

    # -----------------------------
    # 2) 准备路径与 schema
    # -----------------------------
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    safe_makedirs(out_dir)

    year_files = find_year_files(data_dir)
    sample_fp = next(iter(year_files.values()))
    schema_cols = schema_columns(sample_fp)

    # 必要列检查
    for required in [args.label_col, args.date_col]:
        if required not in schema_cols:
            raise KeyError(f"Required col not in schema: {required}")

    # split 年份
    train_years = parse_years(args.train_years)
    valid_years = parse_years(args.valid_years)
    test_years = parse_years(args.test_years)

    missing_drop_thresh = args.missing_drop_thresh if (args.missing_drop_thresh and args.missing_drop_thresh > 0) else None
    clip_abs = args.clip_y_abs if (args.clip_y_abs and args.clip_y_abs > 0) else 0.0
    n_bins = int(args.relevance_bins)
    if n_bins < 2:
        raise ValueError("relevance_bins must be >=2")

    # -----------------------------
    # 3) 构造候选特征集合
    # -----------------------------
    # 所有候选特征（从 schema 推断）
    feat_all = build_feature_candidates(schema_cols, args.label_col, args.date_col, args.id_col)

    # 只排除 dividend（或你传入的 regex）
    feat_kept, feat_excluded = filter_by_regex_exclude(feat_all, args.exclude_regex)

    # 这里把 “fund__” 与 “非 fund__” 分开
    alpha_like = [c for c in feat_kept if not str(c).startswith("fund__")]   # Alpha158 + 其它非 fund
    fund_all = [c for c in feat_kept if str(c).startswith("fund__")]         # 所有 fund__

    selected_fund_df = None
    chosen_fund = fund_all

    # -----------------------------
    # 4) 如果提供 valid_perm，则从里面挑 fund__（更可靠：直接看对 valid RankIC 的贡献）
    # -----------------------------
    if args.fund_perm_csv.strip():
        selected_fund_df = select_fund_from_valid_perm(
            perm_csv=Path(args.fund_perm_csv),
            topk=int(args.fund_perm_topk),
            min_drop=float(args.fund_perm_min_drop),
            exclude_regex=str(args.exclude_regex or "").strip(),
        )
        # 只保留 schema 中存在的列（防止 perm 文件里有不存在的列名）
        chosen_fund = [f for f in selected_fund_df["feature"].astype(str).tolist() if f in schema_cols]
        print(f"[INFO] Selected fund from valid_perm: {len(chosen_fund)} (topk={args.fund_perm_topk}, min_drop={args.fund_perm_min_drop})")
    else:
        print(f"[INFO] No fund_perm_csv provided => keep ALL fund__ (minus dividend): {len(chosen_fund)}")

    # 最终候选特征：Alpha-like 全保留 + 选中的 fund
    feat_cand = alpha_like + chosen_fund

    # -----------------------------
    # 5) 只读需要的列（减少 IO）
    # -----------------------------
    cols_to_read = [args.label_col, args.date_col]
    if args.id_col in schema_cols:
        cols_to_read.append(args.id_col)
    cols_to_read += feat_cand

    # 去重保持顺序
    seen = set()
    cols_to_read = [c for c in cols_to_read if not (c in seen or seen.add(c))]

    # -----------------------------
    # 6) 严格读取 train/valid/test（删除 label NaN）
    # -----------------------------
    df_train = load_years_concat_strict(year_files, train_years, cols_to_read, args.label_col, bool(args.cast_float32))
    df_valid = load_years_concat_strict(year_files, valid_years, cols_to_read, args.label_col, bool(args.cast_float32))
    if df_train.empty or df_valid.empty:
        raise RuntimeError(f"Empty train/valid after strict filtering. train={len(df_train)}, valid={len(df_valid)}")

    df_test = None
    if test_years:
        df_test = load_years_concat_strict(year_files, test_years, cols_to_read, args.label_col, bool(args.cast_float32))
        if df_test.empty:
            print("[WARN] Empty test after strict filtering.")
            df_test = None

    # -----------------------------
    # 7) 基于 train 做特征清洗（零方差 / 缺失率过滤）
    # -----------------------------
    feat_cols = filter_features_on_train(df_train, feat_cand, missing_drop_thresh)

    # 保存本次训练真正使用的特征（回测推理必须严格一致）
    save_used_features(out_dir, feat_cols, excluded_cols=feat_excluded, selected_fund_df=selected_fund_df)

    # -----------------------------
    # 8) 排序 + 分组（每天一组）
    # -----------------------------
    df_train_s, g_train = sort_and_group(df_train, args.date_col, args.id_col)
    df_valid_s, g_valid = sort_and_group(df_valid, args.date_col, args.id_col)

    # 丢弃股票数太少的天
    df_train_s, g_train = drop_small_groups(df_train_s, g_train, int(args.min_group_size))
    df_valid_s, g_valid = drop_small_groups(df_valid_s, g_valid, int(args.min_group_size))
    if len(g_train) == 0 or len(g_valid) == 0:
        raise RuntimeError("No groups left after min_group_size filtering.")

    df_test_s, g_test = (None, None)
    if df_test is not None:
        df_test_s, g_test = sort_and_group(df_test, args.date_col, args.id_col)
        df_test_s, g_test = drop_small_groups(df_test_s, g_test, int(args.min_group_size))
        if len(g_test) == 0:
            df_test_s, g_test = None, None

    # -----------------------------
    # 9) 构造训练数组 X 与 y
    # -----------------------------
    X_tr = df_train_s[feat_cols].to_numpy(dtype=np.float32, copy=True)
    X_va = df_valid_s[feat_cols].to_numpy(dtype=np.float32, copy=True)

    # raw_y 是连续收益（用于 RankIC 评估）
    raw_y_tr = df_train_s[args.label_col].to_numpy(dtype=np.float32, copy=True)
    raw_y_va = df_valid_s[args.label_col].to_numpy(dtype=np.float32, copy=True)

    # y_rel 是离散 relevance（用于 LambdaRank 训练目标）
    y_tr_rel = make_relevance_per_day(clip_y(raw_y_tr, clip_abs), g_train, n_bins=n_bins)
    y_va_rel = make_relevance_per_day(clip_y(raw_y_va, clip_abs), g_valid, n_bins=n_bins)

    # LightGBM ranker 数据集：必须传 group
    dtrain = lgb.Dataset(X_tr, label=y_tr_rel, group=g_train, feature_name=feat_cols, free_raw_data=False)
    dvalid = lgb.Dataset(X_va, label=y_va_rel, group=g_valid, feature_name=feat_cols, free_raw_data=False)

    # 让自定义 feval 能拿到 raw_y 与 group（用 id(dataset) 作为 key）
    dset_info = {id(dtrain): (raw_y_tr, g_train), id(dvalid): (raw_y_va, g_valid)}

    # -----------------------------
    # 10) 训练 LambdaRank（仅用 valid 做早停）
    # -----------------------------
    params = default_rank_params(args.seed, args.threads, n_bins=n_bins, trunc_level=int(args.truncation_level))
    print("[INFO] Params:", params)

    callbacks = [lgb.log_evaluation(period=50)]
    if int(args.early_stopping_rounds) > 0:
        callbacks.append(
            lgb.early_stopping(
                stopping_rounds=int(args.early_stopping_rounds),
                first_metric_only=True,
                verbose=True
            )
        )

    # ✅ 关键点：valid_sets 只传 dvalid，早停只依据 valid 的自定义 RankIC
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=int(args.num_boost_round),
        valid_sets=[dvalid],
        valid_names=["valid"],
        feval=make_feval_rankic(dset_info),
        callbacks=callbacks,
    )

    best_iter = model.best_iteration or int(args.num_boost_round)
    print(f"[INFO] best_iteration={best_iter}")

    # -----------------------------
    # 11) 训练完成后：计算 train/valid/test 的 RankIC（用 raw_y）
    # -----------------------------
    p_tr = model.predict(X_tr, num_iteration=best_iter)
    p_va = model.predict(X_va, num_iteration=best_iter)

    tr_mean, tr_std, tr_ir, tr_days = spearman_rankic_by_group(raw_y_tr, p_tr, g_train)
    va_mean, va_std, va_ir, va_days = spearman_rankic_by_group(raw_y_va, p_va, g_valid)

    metrics = {
        "train_rank_ic_mean": tr_mean, "train_rank_ic_std": tr_std, "train_rank_ic_ir": tr_ir, "train_rank_ic_n_days": tr_days,
        "valid_rank_ic_mean": va_mean, "valid_rank_ic_std": va_std, "valid_rank_ic_ir": va_ir, "valid_rank_ic_n_days": va_days,
    }

    # 测试集只用于最终评估（不参与训练/早停/调参）
    if df_test_s is not None and g_test is not None:
        X_te = df_test_s[feat_cols].to_numpy(dtype=np.float32, copy=True)
        raw_y_te = df_test_s[args.label_col].to_numpy(dtype=np.float32, copy=True)

        p_te = model.predict(X_te, num_iteration=best_iter)
        te_mean, te_std, te_ir, te_days = spearman_rankic_by_group(raw_y_te, p_te, g_test)

        metrics.update({
            "test_rank_ic_mean": te_mean, "test_rank_ic_std": te_std, "test_rank_ic_ir": te_ir, "test_rank_ic_n_days": te_days,
        })

        # 可选：保存测试集的 (date,id,真实收益,预测分数) 用于离线分析
        if args.save_test_pred:
            out_cols = []
            if args.date_col in df_test_s.columns:
                out_cols.append(args.date_col)
            if args.id_col in df_test_s.columns:
                out_cols.append(args.id_col)

            out_pred = df_test_s[out_cols].copy() if out_cols else pd.DataFrame(index=np.arange(len(df_test_s)))
            out_pred["y_raw"] = raw_y_te
            out_pred["pred"] = p_te
            out_pred.to_csv(out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

        del X_te, raw_y_te, p_te

    # -----------------------------
    # 12) 保存模型与 summary.json（可复现实验）
    # -----------------------------
    model_path = out_dir / "model.txt"
    model.save_model(str(model_path))

    summary = {
        "objective": "lambdarank",
        "data_dir": str(data_dir),
        "train_years": train_years,
        "valid_years": valid_years,
        "test_years": test_years,
        "label_col": args.label_col,
        "exclude_regex": args.exclude_regex,
        "fund_perm_csv": args.fund_perm_csv,
        "fund_perm_topk": int(args.fund_perm_topk),
        "fund_perm_min_drop": float(args.fund_perm_min_drop),
        "n_features": int(len(feat_cols)),
        "best_iteration": int(best_iter),
        "params": params,
        "metrics": metrics,
        "outputs": {
            "model": str(model_path),
            "used_features_txt": str(out_dir / "used_features.txt"),
            "used_features_csv": str(out_dir / "used_features.csv"),
            "excluded_features_csv": str(out_dir / "excluded_features.csv"),
            "selected_fund_features_csv": str(out_dir / "selected_fund_features.csv") if args.fund_perm_csv.strip() else "",
            "test_predictions_csv": str(out_dir / "test_predictions.csv") if args.save_test_pred else "",
        },
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved: {model_path}")
    print(f"[INFO] Saved: {out_dir / 'summary.json'}")

    # -----------------------------
    # 13) 清理内存（大数据训练非常必要）
    # -----------------------------
    del df_train, df_valid, df_test, df_train_s, df_valid_s, df_test_s
    del X_tr, X_va, y_tr_rel, y_va_rel
    del raw_y_tr, raw_y_va
    del dtrain, dvalid, model, p_tr, p_va
    gc.collect()


if __name__ == "__main__":
    main()

