# -*- coding: utf-8 -*-
"""
Compute strict Qlib-style labels from OHLCV yearly parquet using an explicit trading calendar,
then merge labels back to feature yearly parquet.

Label (Qlib CN semantics):
  label__ret_1d_qlib = P(t+2)/P(t+1) - 1
Strictness:
  t+1 and t+2 are defined by trading calendar, and MUST exist as rows for that stock in OHLCV.
If missing (often suspension/no-trade), label becomes NaN.

Key alignment:
  Prefer feature key column: order_book_id if exists else instrument.
  OHLCV key could be symbol/order_book_id/instrument/etc.
  Script normalizes OHLCV key to feature key format automatically (common CN formats).

OHLCV date column:
  Could be 'date' (your case) or 'datetime' etc. Script auto-detects.

Output:
  out_dir/_labels_only/year=YYYY.parquet
  out_dir/labeled_yearly_parquet/year=YYYY.parquet   (if mode append_full)
"""

# 运行命令如下：
# python labeled\build_labels_strict_by_calendar_from_ohlcv.py ^
#   --feature_dir "dataset\alpha158_plus_fund_yearly_parquet" ^
#   --ohlcv_dir   "dataset\rq_ohlcv_yearly_parquet" ^
#   --calendar_csv "dataset\trading_calendar_from_merged.csv" ^
#   --out_dir "labeled\_calendar" ^
#   --price_col close ^
#   --mode append_full



import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

YEAR_RE = re.compile(r"year=(\d{4})\.parquet$", re.IGNORECASE)


# -----------------------------
# Helpers: list files / schema
# -----------------------------
def list_year_files(data_dir: Path) -> List[Tuple[int, Path]]:
    files = []
    for p in data_dir.glob("*.parquet"):
        m = YEAR_RE.search(p.name)
        if m:
            files.append((int(m.group(1)), p))
    files.sort(key=lambda x: x[0])
    if not files:
        raise FileNotFoundError(f"No yearly parquet like year=2005.parquet in: {data_dir}")
    return files


def schema_cols(fp: Path) -> List[str]:
    return pq.read_schema(fp).names


def detect_feature_key_col(feature_fp: Path) -> str:
    cols = set(schema_cols(feature_fp))
    if "order_book_id" in cols:
        return "order_book_id"
    if "instrument" in cols:
        return "instrument"
    raise ValueError("Feature parquet has neither 'order_book_id' nor 'instrument'.")


def detect_date_col(fp: Path, user_date_col: Optional[str] = None) -> str:
    cols = set(schema_cols(fp))
    if user_date_col:
        if user_date_col not in cols:
            raise ValueError(f"--date_col '{user_date_col}' not in {fp.name}. Example cols: {schema_cols(fp)[:30]}")
        return user_date_col
    # common names
    for c in ["datetime", "date", "trade_date", "trading_date", "dt"]:
        if c in cols:
            return c
    raise ValueError(f"Cannot detect date column in {fp.name}. Please set --ohlcv_date_col / --feature_date_col.")


def detect_ohlcv_key_col(ohlcv_fp: Path, prefer: str) -> str:
    cols = set(schema_cols(ohlcv_fp))
    if prefer in cols:
        return prefer
    for c in ["symbol", "order_book_id", "instrument", "code", "sec_code", "ticker"]:
        if c in cols:
            return c
    raise ValueError(f"Cannot detect key column in OHLCV {ohlcv_fp.name}. Example cols: {schema_cols(ohlcv_fp)[:30]}")


def detect_ohlcv_price_col(ohlcv_fp: Path, user_price_col: Optional[str]) -> str:
    cols = set(schema_cols(ohlcv_fp))
    if user_price_col:
        if user_price_col not in cols:
            raise ValueError(f"--price_col '{user_price_col}' not in OHLCV file. Example cols: {schema_cols(ohlcv_fp)[:30]}")
        return user_price_col
    for c in ["close", "open", "vwap", "Close", "Open", "VWAP"]:
        if c in cols:
            return c
    raise ValueError("Cannot auto-detect price col in OHLCV; set --price_col (e.g., close/open).")


def load_calendar(calendar_csv: Path):
    cal = pd.read_csv(calendar_csv)
    if "date" not in cal.columns:
        raise ValueError(f"Calendar csv must have column 'date'. Got: {list(cal.columns)}")
    cal_dates = pd.to_datetime(cal["date"]).dt.date.to_list()
    date_to_idx = {d: i for i, d in enumerate(cal_dates)}
    return cal_dates, date_to_idx


def write_parquet(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out_path)


# -----------------------------
# Code normalization
# -----------------------------
def infer_code_format(sample: str) -> str:
    """
    Return one of: 'rq', 'qlib', 'ts', 'raw'
      rq:    000001.XSHE / 600000.XSHG
      qlib:  SZ000001 / SH600000
      ts:    000001.SZ / 600000.SH
    """
    if not isinstance(sample, str) or len(sample) < 6:
        return "raw"
    s = sample.strip()

    if s.endswith(".XSHE") or s.endswith(".XSHG"):
        return "rq"
    if (s.startswith("SZ") or s.startswith("SH")) and len(s) >= 8:
        return "qlib"
    if s.endswith(".SZ") or s.endswith(".SH"):
        return "ts"
    return "raw"


def to_rq(code: str) -> str:
    s = str(code).strip()
    if s.endswith(".XSHE") or s.endswith(".XSHG"):
        return s
    if s.endswith(".SZ"):
        return s.replace(".SZ", ".XSHE")
    if s.endswith(".SH"):
        return s.replace(".SH", ".XSHG")
    if s.startswith("SZ") and len(s) >= 8:
        return s[2:] + ".XSHE"
    if s.startswith("SH") and len(s) >= 8:
        return s[2:] + ".XSHG"
    return s


def to_qlib(code: str) -> str:
    s = str(code).strip()
    if (s.startswith("SZ") or s.startswith("SH")) and len(s) >= 8:
        return s
    if s.endswith(".XSHE"):
        return "SZ" + s.split(".")[0]
    if s.endswith(".XSHG"):
        return "SH" + s.split(".")[0]
    if s.endswith(".SZ"):
        return "SZ" + s.split(".")[0]
    if s.endswith(".SH"):
        return "SH" + s.split(".")[0]
    return s


def to_ts(code: str) -> str:
    s = str(code).strip()
    if s.endswith(".SZ") or s.endswith(".SH"):
        return s
    if s.endswith(".XSHE"):
        return s.replace(".XSHE", ".SZ")
    if s.endswith(".XSHG"):
        return s.replace(".XSHG", ".SH")
    if s.startswith("SZ") and len(s) >= 8:
        return s[2:] + ".SZ"
    if s.startswith("SH") and len(s) >= 8:
        return s[2:] + ".SH"
    return s


def normalize_to_target(series: pd.Series, target_format: str) -> pd.Series:
    if target_format == "rq":
        return series.astype(str).map(to_rq)
    if target_format == "qlib":
        return series.astype(str).map(to_qlib)
    if target_format == "ts":
        return series.astype(str).map(to_ts)
    return series.astype(str)


# -----------------------------
# Build strict labels from OHLCV
# -----------------------------
def read_minimal_ohlcv_year(fp: Path, date_col: str, key_col: str, price_col: str) -> pd.DataFrame:
    tbl = pq.read_table(fp, columns=[date_col, key_col, price_col])
    df = tbl.to_pandas()
    df.rename(columns={date_col: "datetime"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.date
    df[key_col] = df[key_col].astype(str)
    df.rename(columns={price_col: "price"}, inplace=True)
    return df


def build_future(df: pd.DataFrame, key_col_norm: str) -> pd.DataFrame:
    df = df.sort_values([key_col_norm, "datetime"], kind="mergesort").reset_index(drop=True)
    g = df.groupby(key_col_norm, sort=False)
    df["next_dt1"] = g["datetime"].shift(-1)
    df["next_p1"] = g["price"].shift(-1)
    df["next_dt2"] = g["datetime"].shift(-2)
    df["next_p2"] = g["price"].shift(-2)
    return df


def patch_cross_year(prev_df: pd.DataFrame, cur_df: pd.DataFrame, key_col_norm: str) -> pd.DataFrame:
    cur_sorted = cur_df.sort_values([key_col_norm, "datetime"], kind="mergesort")
    head2 = cur_sorted.groupby(key_col_norm, sort=False).head(2).copy()
    head2["rk"] = head2.groupby(key_col_norm, sort=False).cumcount()

    head1 = head2[head2["rk"] == 0][[key_col_norm, "datetime", "price"]].rename(
        columns={"datetime": "h1_dt", "price": "h1_p"}
    )
    head2b = head2[head2["rk"] == 1][[key_col_norm, "datetime", "price"]].rename(
        columns={"datetime": "h2_dt", "price": "h2_p"}
    )

    prev_sorted = prev_df.sort_values([key_col_norm, "datetime"], kind="mergesort")
    tail2 = prev_sorted.groupby(key_col_norm, sort=False).tail(2).copy()
    tail2["tk"] = tail2.groupby(key_col_norm, sort=False).cumcount()
    second_last = tail2[tail2["tk"] == 0][[key_col_norm, "datetime"]]
    last = tail2[tail2["tk"] == 1][[key_col_norm, "datetime"]]

    prev_df = prev_df.set_index([key_col_norm, "datetime"], drop=False)

    p_last = last.merge(head1, on=key_col_norm, how="left").merge(head2b, on=key_col_norm, how="left")
    if len(p_last):
        p_last = p_last.set_index([key_col_norm, "datetime"])
        idx = p_last.index
        prev_df.loc[idx, "next_dt1"] = p_last["h1_dt"]
        prev_df.loc[idx, "next_p1"] = p_last["h1_p"]
        prev_df.loc[idx, "next_dt2"] = p_last["h2_dt"]
        prev_df.loc[idx, "next_p2"] = p_last["h2_p"]

    p_second = second_last.merge(head1, on=key_col_norm, how="left")
    if len(p_second):
        p_second = p_second.set_index([key_col_norm, "datetime"])
        idx2 = p_second.index
        prev_df.loc[idx2, "next_dt2"] = p_second["h1_dt"]
        prev_df.loc[idx2, "next_p2"] = p_second["h1_p"]

    return prev_df.reset_index(drop=True)


def recompute_strict_qlib_label(df: pd.DataFrame,
                                key_col_norm: str,
                                cal_dates: List,
                                date_to_idx: Dict,
                                ipo_min_bars: Optional[int]) -> pd.DataFrame:
    dt_idx = df["datetime"].map(date_to_idx).astype("Int32")
    df["dt_idx"] = dt_idx

    max_i = len(cal_dates) - 1
    exp1 = dt_idx + 1
    exp2 = dt_idx + 2
    exp_dt1 = pd.Series([cal_dates[i] if (pd.notna(i) and i <= max_i) else None for i in exp1], index=df.index)
    exp_dt2 = pd.Series([cal_dates[i] if (pd.notna(i) and i <= max_i) else None for i in exp2], index=df.index)

    ok_t1 = (df["next_dt1"] == exp_dt1)
    ok_t2 = (df["next_dt2"] == exp_dt2)
    ok = ok_t1 & ok_t2

    # IPO filter (approx): require entry day t+1 has >= N bars since first appearance in OHLCV
    if ipo_min_bars:
        df.sort_values([key_col_norm, "datetime"], kind="mergesort", inplace=True)
        g = df.groupby(key_col_norm, sort=False)
        first_idx = g["dt_idx"].transform("min")
        entry_idx = df["dt_idx"] + 1
        ok_ipo = (entry_idx - first_idx + 1) >= ipo_min_bars
        ok = ok & ok_ipo

    p1 = np.where(ok, df["next_p1"], np.nan)
    p2 = np.where(ok, df["next_p2"], np.nan)

    df["label__ret_1d_qlib"] = p2 / p1 - 1.0
    df["label__up_1d_qlib"] = np.where(pd.notna(df["label__ret_1d_qlib"]),
                                       (df["label__ret_1d_qlib"] > 0).astype(np.int8),
                                       np.nan)
    return df


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_dir", type=str, required=True, help="Feature yearly parquet dir (year=YYYY.parquet)")
    ap.add_argument("--ohlcv_dir", type=str, required=True, help="OHLCV yearly parquet dir (year=YYYY.parquet)")
    ap.add_argument("--calendar_csv", type=str, required=True, help="Trading calendar csv with column 'date'")
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--price_col", type=str, default="close", help="Price col in OHLCV (close/open/vwap...)")
    ap.add_argument("--mode", type=str, default="append_full", choices=["labels_only", "append_full"])
    ap.add_argument("--ipo_min_bars", type=int, default=None)

    ap.add_argument("--feature_date_col", type=str, default=None, help="Optional: feature date col name")
    ap.add_argument("--ohlcv_date_col", type=str, default=None, help="Optional: OHLCV date col name")
    ap.add_argument("--force_target_code_format", type=str, default=None,
                    choices=[None, "rq", "qlib", "ts", "raw"],
                    help="Optional: force code normalization target; default inferred from feature key sample")
    args = ap.parse_args()

    feature_dir = Path(args.feature_dir)
    ohlcv_dir = Path(args.ohlcv_dir)
    out_dir = Path(args.out_dir)
    out_labels_dir = out_dir / "_labels_only"
    out_full_dir = out_dir / "labeled_yearly_parquet"

    feat_years = dict(list_year_files(feature_dir))
    ohlcv_years = dict(list_year_files(ohlcv_dir))
    common_years = sorted(set(feat_years.keys()) & set(ohlcv_years.keys()))
    if not common_years:
        raise ValueError("No common years between feature_dir and ohlcv_dir.")

    # detect columns
    feat_fp0 = feat_years[common_years[0]]
    ohlcv_fp0 = ohlcv_years[common_years[0]]

    feat_key = detect_feature_key_col(feat_fp0)
    feat_date_col = detect_date_col(feat_fp0, args.feature_date_col)

    ohlcv_key_raw = detect_ohlcv_key_col(ohlcv_fp0, prefer=feat_key)
    ohlcv_date_col = detect_date_col(ohlcv_fp0, args.ohlcv_date_col)
    price_col = detect_ohlcv_price_col(ohlcv_fp0, args.price_col)

    # infer target code format from feature key sample
    feat_sample_tbl = pq.read_table(feat_fp0, columns=[feat_key]).to_pandas()
    feat_sample_val = str(feat_sample_tbl[feat_key].dropna().astype(str).iloc[0])
    target_fmt = args.force_target_code_format or infer_code_format(feat_sample_val)

    print(f"[INFO] feature key={feat_key}, feature date={feat_date_col}")
    print(f"[INFO] ohlcv key(raw)={ohlcv_key_raw}, ohlcv date={ohlcv_date_col}, price_col={price_col}")
    print(f"[INFO] target code format inferred from feature sample '{feat_sample_val}' => {target_fmt}")

    cal_dates, date_to_idx = load_calendar(Path(args.calendar_csv))
    print(f"[INFO] calendar size={len(cal_dates)} first={cal_dates[0]} last={cal_dates[-1]}")

    # A) Build labels-only from OHLCV
    prev_year = None
    prev_min = None

    for y in common_years:
        print(f"[A] Read OHLCV year {y}")
        cur = read_minimal_ohlcv_year(ohlcv_years[y], ohlcv_date_col, ohlcv_key_raw, price_col)

        # normalize OHLCV key to feature key format
        cur["key_norm"] = normalize_to_target(cur[ohlcv_key_raw], target_fmt)

        # build future on normalized key
        cur = build_future(cur, "key_norm")
        cur = recompute_strict_qlib_label(cur, "key_norm", cal_dates, date_to_idx, args.ipo_min_bars)

        if prev_min is not None:
            print(f"[A] Patch boundary {prev_year} <- {y}")
            prev_min = patch_cross_year(prev_min, cur, "key_norm")
            prev_min = recompute_strict_qlib_label(prev_min, "key_norm", cal_dates, date_to_idx, args.ipo_min_bars)

            out_fp = out_labels_dir / f"year={prev_year}.parquet"
            lab = prev_min[["datetime", "key_norm", "label__ret_1d_qlib", "label__up_1d_qlib"]].copy()
            lab.rename(columns={"key_norm": feat_key}, inplace=True)
            write_parquet(lab, out_fp)
            print(f"[A] Wrote labels: {out_fp}")

        prev_year = y
        prev_min = cur

    if prev_min is not None:
        out_fp = out_labels_dir / f"year={prev_year}.parquet"
        lab = prev_min[["datetime", "key_norm", "label__ret_1d_qlib", "label__up_1d_qlib"]].copy()
        lab.rename(columns={"key_norm": feat_key}, inplace=True)
        write_parquet(lab, out_fp)
        print(f"[A] Wrote labels(last): {out_fp}")

    if args.mode == "labels_only":
        print(f"[DONE] labels_only => {out_labels_dir}")
        return

    # B) Merge labels back to feature parquet
    out_full_dir.mkdir(parents=True, exist_ok=True)

    for y in common_years:
        print(f"[B] Merge year {y}")
        feat = pq.read_table(feat_years[y]).to_pandas()

        # normalize feature date + key
        feat.rename(columns={feat_date_col: "datetime"}, inplace=True)
        feat["datetime"] = pd.to_datetime(feat["datetime"]).dt.date
        feat[feat_key] = feat[feat_key].astype(str)

        lab = pq.read_table(out_labels_dir / f"year={y}.parquet").to_pandas()
        lab["datetime"] = pd.to_datetime(lab["datetime"]).dt.date
        lab[feat_key] = lab[feat_key].astype(str)

        merged = feat.merge(lab, on=["datetime", feat_key], how="left", validate="many_to_one")

        out_fp = out_full_dir / f"year={y}.parquet"
        write_parquet(merged, out_fp)
        print(f"[B] Wrote full labeled: {out_fp}")

    print(f"[DONE] append_full => {out_full_dir}")


if __name__ == "__main__":
    main()
