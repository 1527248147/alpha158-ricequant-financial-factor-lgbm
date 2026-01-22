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
    p.mkdir(parents=True, exist_ok=True)


def parse_years(spec: str) -> List[int]:
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
    pf = pq.ParquetFile(sample_parquet)
    return [f.name for f in pf.schema_arrow]


def load_parquet(path: Path, columns: Optional[List[str]] = None) -> pd.DataFrame:
    table = pq.read_table(path, columns=columns)
    return table.to_pandas(split_blocks=True, self_destruct=True)


def load_years_concat_strict(
    year_files: Dict[int, Path],
    years: List[int],
    columns_to_read: List[str],
    label_col: str,
    cast_float32: bool,
) -> pd.DataFrame:
    """Strict: drop NaN label rows (t+1/t+2 missing or year tail)."""
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
        df = df[df[label_col].notna()].copy()  # ✅ label NaN 不参与训练/验证/测试
        after = len(df)
        print(f"[INFO] year={y} drop NaN label: {before:,} -> {after:,}")

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
    """All feature candidates = schema minus {label/date/id} minus label__*."""
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
    Read feature_importance_valid_perm.csv and pick fund__ features by valid_rankic_drop_mean.
    Keep drop >= min_drop.
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
    feat_cols = [c for c in feat_cand if c in df_train.columns and pd.api.types.is_numeric_dtype(df_train[c].dtype)]

    var = df_train[feat_cols].var(axis=0, skipna=True)
    feat_cols = [c for c in feat_cols if pd.notna(var[c]) and var[c] > 0]
    print(f"[INFO] After drop zero-var/all-NaN: {len(feat_cols)} features")

    if missing_drop_thresh is not None:
        miss = df_train[feat_cols].isna().mean()
        feat_cols2 = [c for c in feat_cols if miss[c] < missing_drop_thresh]
        print(f"[INFO] After missing_drop_thresh={missing_drop_thresh}: {len(feat_cols2)} features (from {len(feat_cols)})")
        feat_cols = feat_cols2

    leaked = [c for c in feat_cols if str(c).startswith("label__")]
    if leaked:
        raise RuntimeError(f"LEAK DETECTED: label__* appeared in features: {leaked[:20]}")
    return feat_cols


def save_used_features(out_dir: Path, feat_cols: List[str], excluded_cols: List[str], selected_fund_df: Optional[pd.DataFrame]):
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
    dt = pd.to_datetime(df[date_col], errors="coerce")
    if dt.isna().any():
        raise ValueError(f"{date_col} contains unparsable values.")
    day = dt.values.astype("datetime64[D]")

    df2 = df.copy()
    df2["_day"] = day
    if id_col in df2.columns:
        df2 = df2.sort_values(["_day", id_col], kind="mergesort")
    else:
        df2 = df2.sort_values(["_day"], kind="mergesort")

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
    ics = []
    start = 0
    for g in group_sizes:
        end = start + g
        yy = y[start:end]
        pp = p[start:end]
        start = end
        if yy.size < 3:
            continue
        m = np.isfinite(yy) & np.isfinite(pp)
        yy = yy[m]
        pp = pp[m]
        if yy.size < 3:
            continue
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
    rel = np.empty_like(raw_y, dtype=np.float32)
    start = 0
    for g in group_sizes:
        end = start + g
        y = raw_y[start:end]
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
    if clip_abs is None or clip_abs <= 0:
        return y.astype(np.float32, copy=True)
    return np.clip(y, -clip_abs, clip_abs).astype(np.float32, copy=True)


def make_feval_rankic(dset_to_info: dict):
    def _feval(preds: np.ndarray, dataset: lgb.Dataset):
        info = dset_to_info.get(id(dataset))
        if info is None:
            return ("rank_ic_mean", np.nan, True)
        raw_y, group_sizes = info
        mean, std, ir, n_days = spearman_rankic_by_group(raw_y, preds, group_sizes)
        return ("rank_ic_mean", mean, True)
    return _feval


# -----------------------------
# Params
# -----------------------------
def default_rank_params(seed: int, threads: int, n_bins: int, trunc_level: int) -> dict:
    label_gain = list(range(int(n_bins)))
    return {
        "objective": "lambdarank",
        "metric": "None",
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
        "lambdarank_truncation_level": int(trunc_level),
        "label_gain": label_gain,
        "verbosity": -1,
        "seed": int(seed),
        "num_threads": int(threads),
        "force_col_wise": True,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--train_years", type=str, default="2021-2023")
    ap.add_argument("--valid_years", type=str, default="2024")
    ap.add_argument("--test_years", type=str, default="2025")

    ap.add_argument("--label_col", type=str, default="label__ret_1d_qlib")
    ap.add_argument("--date_col", type=str, default="datetime")
    ap.add_argument("--id_col", type=str, default="order_book_id")

    ap.add_argument("--missing_drop_thresh", type=float, default=0.98)
    ap.add_argument("--cast_float32", action="store_true")

    # exclude ONLY dividend-like
    ap.add_argument("--exclude_regex", type=str, default=r"(?i)dividend")

    # fund selection from valid_perm
    ap.add_argument("--fund_perm_csv", type=str, default="", help="feature_importance_valid_perm.csv from previous run")
    ap.add_argument("--fund_perm_topk", type=int, default=9, help="how many fund__ to keep")
    ap.add_argument("--fund_perm_min_drop", type=float, default=1e-4, help="keep fund__ with perm drop >= this")

    # label processing
    ap.add_argument("--clip_y_abs", type=float, default=0.0)
    ap.add_argument("--relevance_bins", type=int, default=10)
    ap.add_argument("--min_group_size", type=int, default=30)

    # TopK focus
    ap.add_argument("--truncation_level", type=int, default=50)

    # training
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--num_boost_round", type=int, default=5000)
    ap.add_argument("--early_stopping_rounds", type=int, default=300)
    ap.add_argument("--save_test_pred", action="store_true")

    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    safe_makedirs(out_dir)

    year_files = find_year_files(data_dir)
    sample_fp = next(iter(year_files.values()))
    schema_cols = schema_columns(sample_fp)

    for required in [args.label_col, args.date_col]:
        if required not in schema_cols:
            raise KeyError(f"Required col not in schema: {required}")

    train_years = parse_years(args.train_years)
    valid_years = parse_years(args.valid_years)
    test_years = parse_years(args.test_years)

    missing_drop_thresh = args.missing_drop_thresh if (args.missing_drop_thresh and args.missing_drop_thresh > 0) else None
    clip_abs = args.clip_y_abs if (args.clip_y_abs and args.clip_y_abs > 0) else 0.0
    n_bins = int(args.relevance_bins)
    if n_bins < 2:
        raise ValueError("relevance_bins must be >=2")

    # ---- all feature candidates (minus label/date/id) ----
    feat_all = build_feature_candidates(schema_cols, args.label_col, args.date_col, args.id_col)

    # ---- exclude dividend only ----
    feat_kept, feat_excluded = filter_by_regex_exclude(feat_all, args.exclude_regex)

    # ---- split alpha vs fund ----
    alpha_like = [c for c in feat_kept if not str(c).startswith("fund__")]
    fund_all = [c for c in feat_kept if str(c).startswith("fund__")]

    selected_fund_df = None
    chosen_fund = fund_all

    # ---- choose important fund features from valid_perm_csv ----
    if args.fund_perm_csv.strip():
        selected_fund_df = select_fund_from_valid_perm(
            perm_csv=Path(args.fund_perm_csv),
            topk=int(args.fund_perm_topk),
            min_drop=float(args.fund_perm_min_drop),
            exclude_regex=str(args.exclude_regex or "").strip(),
        )
        chosen_fund = [f for f in selected_fund_df["feature"].astype(str).tolist() if f in schema_cols]
        print(f"[INFO] Selected fund from valid_perm: {len(chosen_fund)} (topk={args.fund_perm_topk}, min_drop={args.fund_perm_min_drop})")
    else:
        print(f"[INFO] No fund_perm_csv provided => keep ALL fund__ (minus dividend): {len(chosen_fund)}")

    # final feature candidates
    feat_cand = alpha_like + chosen_fund

    # ---- columns to read ----
    cols_to_read = [args.label_col, args.date_col]
    if args.id_col in schema_cols:
        cols_to_read.append(args.id_col)
    cols_to_read += feat_cand
    # unique
    seen = set()
    cols_to_read = [c for c in cols_to_read if not (c in seen or seen.add(c))]

    # ---- load splits strictly ----
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

    # ---- train-based cleaning (zero-var / missing thresh) ----
    feat_cols = filter_features_on_train(df_train, feat_cand, missing_drop_thresh)
    save_used_features(out_dir, feat_cols, excluded_cols=feat_excluded, selected_fund_df=selected_fund_df)

    # ---- sort & group ----
    df_train_s, g_train = sort_and_group(df_train, args.date_col, args.id_col)
    df_valid_s, g_valid = sort_and_group(df_valid, args.date_col, args.id_col)

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

    # ---- arrays ----
    X_tr = df_train_s[feat_cols].to_numpy(dtype=np.float32, copy=True)
    X_va = df_valid_s[feat_cols].to_numpy(dtype=np.float32, copy=True)
    raw_y_tr = df_train_s[args.label_col].to_numpy(dtype=np.float32, copy=True)
    raw_y_va = df_valid_s[args.label_col].to_numpy(dtype=np.float32, copy=True)

    y_tr_rel = make_relevance_per_day(clip_y(raw_y_tr, clip_abs), g_train, n_bins=n_bins)
    y_va_rel = make_relevance_per_day(clip_y(raw_y_va, clip_abs), g_valid, n_bins=n_bins)

    dtrain = lgb.Dataset(X_tr, label=y_tr_rel, group=g_train, feature_name=feat_cols, free_raw_data=False)
    dvalid = lgb.Dataset(X_va, label=y_va_rel, group=g_valid, feature_name=feat_cols, free_raw_data=False)

    dset_info = {id(dtrain): (raw_y_tr, g_train), id(dvalid): (raw_y_va, g_valid)}

    params = default_rank_params(args.seed, args.threads, n_bins=n_bins, trunc_level=int(args.truncation_level))
    print("[INFO] Params:", params)

    callbacks = [lgb.log_evaluation(period=50)]
    if int(args.early_stopping_rounds) > 0:
        callbacks.append(lgb.early_stopping(stopping_rounds=int(args.early_stopping_rounds),
                                           first_metric_only=True, verbose=True))

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=int(args.num_boost_round),
        valid_sets=[dvalid],          # ✅ 只用 valid 早停
        valid_names=["valid"],
        feval=make_feval_rankic(dset_info),
        callbacks=callbacks,
    )

    best_iter = model.best_iteration or int(args.num_boost_round)
    print(f"[INFO] best_iteration={best_iter}")

    # ---- metrics ----
    p_tr = model.predict(X_tr, num_iteration=best_iter)
    p_va = model.predict(X_va, num_iteration=best_iter)
    tr_mean, tr_std, tr_ir, tr_days = spearman_rankic_by_group(raw_y_tr, p_tr, g_train)
    va_mean, va_std, va_ir, va_days = spearman_rankic_by_group(raw_y_va, p_va, g_valid)

    metrics = {
        "train_rank_ic_mean": tr_mean, "train_rank_ic_std": tr_std, "train_rank_ic_ir": tr_ir, "train_rank_ic_n_days": tr_days,
        "valid_rank_ic_mean": va_mean, "valid_rank_ic_std": va_std, "valid_rank_ic_ir": va_ir, "valid_rank_ic_n_days": va_days,
    }

    if df_test_s is not None and g_test is not None:
        X_te = df_test_s[feat_cols].to_numpy(dtype=np.float32, copy=True)
        raw_y_te = df_test_s[args.label_col].to_numpy(dtype=np.float32, copy=True)
        p_te = model.predict(X_te, num_iteration=best_iter)
        te_mean, te_std, te_ir, te_days = spearman_rankic_by_group(raw_y_te, p_te, g_test)
        metrics.update({
            "test_rank_ic_mean": te_mean, "test_rank_ic_std": te_std, "test_rank_ic_ir": te_ir, "test_rank_ic_n_days": te_days,
        })
        if args.save_test_pred:
            out_cols = []
            if args.date_col in df_test_s.columns: out_cols.append(args.date_col)
            if args.id_col in df_test_s.columns: out_cols.append(args.id_col)
            out_pred = df_test_s[out_cols].copy() if out_cols else pd.DataFrame(index=np.arange(len(df_test_s)))
            out_pred["y_raw"] = raw_y_te
            out_pred["pred"] = p_te
            out_pred.to_csv(out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
        del X_te, raw_y_te, p_te

    # ---- save ----
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

    # cleanup
    del df_train, df_valid, df_test, df_train_s, df_valid_s, df_test_s
    del X_tr, X_va, y_tr_rel, y_va_rel
    del raw_y_tr, raw_y_va
    del dtrain, dvalid, model, p_tr, p_va
    gc.collect()


if __name__ == "__main__":
    main()



