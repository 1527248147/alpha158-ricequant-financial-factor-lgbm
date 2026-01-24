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



# -*- coding: utf-8 -*-
"""
【用途】
用“显式交易日历”从 OHLCV 年度 parquet 生成严格版 Qlib 风格 label，
然后把 label 合并回 feature 年度 parquet。

【Qlib CN 常见 label 语义】
  label__ret_1d_qlib = P(t+2) / P(t+1) - 1

【严格性（Strictness）】
- t+1 / t+2 不是“下一条数据行”，而是“交易日历上的下一/下下个交易日”。
- 且该股票必须在 OHLCV 中存在 t+1 和 t+2 的记录（否则：停牌/无交易 -> label=NaN）。
- 训练时可以删掉 label 为 NaN 的样本。

【Key 对齐】
- feature 表主键通常是 order_book_id（否则 instrument）。
- OHLCV 表主键可能叫 symbol/order_book_id/instrument/...。
- 脚本会把 OHLCV 的 key 统一转换成 feature 的 key 格式（常见 A 股格式）。

【日期列自动识别】
- feature/ohlcv 的日期列可能叫 datetime/date/trade_date 等，脚本会自动探测。

【输出】
- out_dir/_labels_only/year=YYYY.parquet      （只包含 key/date + label）
- out_dir/labeled_yearly_parquet/year=YYYY.parquet  （features + 新 label，若 mode=append_full）
"""

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# 年文件名匹配：year=2005.parquet
YEAR_RE = re.compile(r"year=(\d{4})\.parquet$", re.IGNORECASE)


# -----------------------------
# 工具函数：列出 year=YYYY.parquet 文件 / 读取 schema
# -----------------------------
def list_year_files(data_dir: Path) -> List[Tuple[int, Path]]:
    """
    在 data_dir 下找所有 year=YYYY.parquet，返回 [(year, path), ...] 按 year 排序
    """
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
    """只读 schema，不把 parquet 全量读进内存（更快）"""
    return pq.read_schema(fp).names


def detect_feature_key_col(feature_fp: Path) -> str:
    """
    feature 主键列探测：优先 order_book_id，其次 instrument
    """
    cols = set(schema_cols(feature_fp))
    if "order_book_id" in cols:
        return "order_book_id"
    if "instrument" in cols:
        return "instrument"
    raise ValueError("Feature parquet has neither 'order_book_id' nor 'instrument'.")


def detect_date_col(fp: Path, user_date_col: Optional[str] = None) -> str:
    """
    日期列探测：如果用户指定就用用户指定，否则按常见候选列名查找
    """
    cols = set(schema_cols(fp))
    if user_date_col:
        if user_date_col not in cols:
            raise ValueError(
                f"--date_col '{user_date_col}' not in {fp.name}. Example cols: {schema_cols(fp)[:30]}"
            )
        return user_date_col
    # 常见日期列名
    for c in ["datetime", "date", "trade_date", "trading_date", "dt"]:
        if c in cols:
            return c
    raise ValueError(f"Cannot detect date column in {fp.name}. Please set --ohlcv_date_col / --feature_date_col.")


def detect_ohlcv_key_col(ohlcv_fp: Path, prefer: str) -> str:
    """
    OHLCV 主键列探测：
    - 先尝试与 feature 相同的列名（prefer）
    - 否则从常见候选里找
    """
    cols = set(schema_cols(ohlcv_fp))
    if prefer in cols:
        return prefer
    for c in ["symbol", "order_book_id", "instrument", "code", "sec_code", "ticker"]:
        if c in cols:
            return c
    raise ValueError(f"Cannot detect key column in OHLCV {ohlcv_fp.name}. Example cols: {schema_cols(ohlcv_fp)[:30]}")


def detect_ohlcv_price_col(ohlcv_fp: Path, user_price_col: Optional[str]) -> str:
    """
    OHLCV 价格列探测：默认 close，也支持 open/vwap 等
    """
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
    """
    读取交易日历 csv（必须包含 date 列），返回：
    - cal_dates: List[date] 交易日列表
    - date_to_idx: dict {date -> index} 用于快速定位 t 的交易日索引
    """
    cal = pd.read_csv(calendar_csv)
    if "date" not in cal.columns:
        raise ValueError(f"Calendar csv must have column 'date'. Got: {list(cal.columns)}")
    cal_dates = pd.to_datetime(cal["date"]).dt.date.to_list()
    date_to_idx = {d: i for i, d in enumerate(cal_dates)}
    return cal_dates, date_to_idx


def write_parquet(df: pd.DataFrame, out_path: Path):
    """统一写 parquet，确保输出目录存在"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out_path)


# -----------------------------
# 股票代码格式归一化
# -----------------------------
def infer_code_format(sample: str) -> str:
    """
    根据 feature 里拿到的样例 key，推断目标格式（target_format）
    返回：'rq' / 'qlib' / 'ts' / 'raw'

    - rq:   000001.XSHE / 600000.XSHG   （米筐/RQ 常见）
    - qlib: SZ000001 / SH600000         （Qlib 常见）
    - ts:   000001.SZ / 600000.SH       （Tushare 常见）
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
    """把各种格式转为 rq 格式：000001.XSHE / 600000.XSHG"""
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
    """把各种格式转为 qlib 格式：SZ000001 / SH600000"""
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
    """把各种格式转为 tushare 格式：000001.SZ / 600000.SH"""
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
    """
    把一个 series 的股票代码统一成 target_format
    """
    if target_format == "rq":
        return series.astype(str).map(to_rq)
    if target_format == "qlib":
        return series.astype(str).map(to_qlib)
    if target_format == "ts":
        return series.astype(str).map(to_ts)
    return series.astype(str)


# -----------------------------
# 从 OHLCV 构建严格 label 的核心逻辑
# -----------------------------
def read_minimal_ohlcv_year(fp: Path, date_col: str, key_col: str, price_col: str) -> pd.DataFrame:
    """
    只读取 OHLCV 必要列：
    - date_col: 日期列
    - key_col : 股票代码列
    - price_col: 用于算 label 的价格（默认 close）
    """
    tbl = pq.read_table(fp, columns=[date_col, key_col, price_col])
    df = tbl.to_pandas()
    df.rename(columns={date_col: "datetime"}, inplace=True)
    # 统一成 python date，方便与交易日历 cal_dates 对齐
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.date
    df[key_col] = df[key_col].astype(str)
    df.rename(columns={price_col: "price"}, inplace=True)
    return df


def build_future(df: pd.DataFrame, key_col_norm: str) -> pd.DataFrame:
    """
    对每只股票，按日期排序后用 shift(-1)/shift(-2) 得到“下一条 OHLCV 记录”的日期/价格。
    注意：这一步得到的是“下一条可用数据”，不是交易所意义上的 t+1/t+2。
    严格性由后面的 trading calendar 校验保证。
    """
    df = df.sort_values([key_col_norm, "datetime"], kind="mergesort").reset_index(drop=True)
    g = df.groupby(key_col_norm, sort=False)
    df["next_dt1"] = g["datetime"].shift(-1)
    df["next_p1"] = g["price"].shift(-1)
    df["next_dt2"] = g["datetime"].shift(-2)
    df["next_p2"] = g["price"].shift(-2)
    return df


def patch_cross_year(prev_df: pd.DataFrame, cur_df: pd.DataFrame, key_col_norm: str) -> pd.DataFrame:
    """
    处理跨年边界：year=YYYY 的最后一两天，其 t+1/t+2 可能在 year=YYYY+1 里。
    这里把 prev_df 的最后两条记录的 next_dt1/next_dt2 指向 cur_df 的前两条记录（同一只股票）。
    """
    cur_sorted = cur_df.sort_values([key_col_norm, "datetime"], kind="mergesort")
    # 取每只股票在新一年最前面两天（若存在）
    head2 = cur_sorted.groupby(key_col_norm, sort=False).head(2).copy()
    head2["rk"] = head2.groupby(key_col_norm, sort=False).cumcount()

    head1 = head2[head2["rk"] == 0][[key_col_norm, "datetime", "price"]].rename(
        columns={"datetime": "h1_dt", "price": "h1_p"}
    )
    head2b = head2[head2["rk"] == 1][[key_col_norm, "datetime", "price"]].rename(
        columns={"datetime": "h2_dt", "price": "h2_p"}
    )

    prev_sorted = prev_df.sort_values([key_col_norm, "datetime"], kind="mergesort")
    # 取每只股票在旧一年最后两天
    tail2 = prev_sorted.groupby(key_col_norm, sort=False).tail(2).copy()
    tail2["tk"] = tail2.groupby(key_col_norm, sort=False).cumcount()
    second_last = tail2[tail2["tk"] == 0][[key_col_norm, "datetime"]]
    last = tail2[tail2["tk"] == 1][[key_col_norm, "datetime"]]

    # 用 MultiIndex 定位需要 patch 的行
    prev_df = prev_df.set_index([key_col_norm, "datetime"], drop=False)

    # patch “最后一天”：它的 next1/next2 应该接到新年的 head1/head2
    p_last = last.merge(head1, on=key_col_norm, how="left").merge(head2b, on=key_col_norm, how="left")
    if len(p_last):
        p_last = p_last.set_index([key_col_norm, "datetime"])
        idx = p_last.index
        prev_df.loc[idx, "next_dt1"] = p_last["h1_dt"]
        prev_df.loc[idx, "next_p1"] = p_last["h1_p"]
        prev_df.loc[idx, "next_dt2"] = p_last["h2_dt"]
        prev_df.loc[idx, "next_p2"] = p_last["h2_p"]

    # patch “倒数第二天”：它的 next2 可能接到新年的 head1
    p_second = second_last.merge(head1, on=key_col_norm, how="left")
    if len(p_second):
        p_second = p_second.set_index([key_col_norm, "datetime"])
        idx2 = p_second.index
        prev_df.loc[idx2, "next_dt2"] = p_second["h1_dt"]
        prev_df.loc[idx2, "next_p2"] = p_second["h1_p"]

    return prev_df.reset_index(drop=True)


def recompute_strict_qlib_label(
    df: pd.DataFrame,
    key_col_norm: str,
    cal_dates: List,
    date_to_idx: Dict,
    ipo_min_bars: Optional[int],
) -> pd.DataFrame:
    """
    通过交易日历实现“严格 t+1/t+2”校验，并生成 Qlib 风格 label：

    1) dt_idx = date_to_idx[datetime]，把日期映射成交易日序号
    2) 期望的 t+1 日期 = cal_dates[dt_idx + 1]
       期望的 t+2 日期 = cal_dates[dt_idx + 2]
    3) 判断 OHLCV 的 next_dt1/next_dt2 是否严格等于期望日期
       - 若不等：说明中间有停牌/无交易/缺行 -> label=NaN
    4) 通过 ok 掩码计算 ret：next_p2/next_p1 - 1
    """

    # 每一行日期在交易日历中的位置（可能有缺失：不在日历中 -> NaN）
    dt_idx = df["datetime"].map(date_to_idx).astype("Int32")
    df["dt_idx"] = dt_idx

    max_i = len(cal_dates) - 1

    # 期望的 t+1/t+2 的交易日索引
    exp1 = dt_idx + 1
    exp2 = dt_idx + 2

    # 把索引转回期望的日期（越界则 None）
    exp_dt1 = pd.Series([cal_dates[i] if (pd.notna(i) and i <= max_i) else None for i in exp1], index=df.index)
    exp_dt2 = pd.Series([cal_dates[i] if (pd.notna(i) and i <= max_i) else None for i in exp2], index=df.index)

    # 严格性校验：OHLCV 的“下一条记录日期”必须等于交易所意义上的 t+1
    ok_t1 = (df["next_dt1"] == exp_dt1)
    ok_t2 = (df["next_dt2"] == exp_dt2)
    ok = ok_t1 & ok_t2

    # 可选 IPO 过滤：要求进入日(t+1)距首次出现 >= N 个交易日
    # 注意：这只是近似（用 OHLCV 首次出现当作上市起点），但在缺少官方上市日时很实用
    if ipo_min_bars:
        df.sort_values([key_col_norm, "datetime"], kind="mergesort", inplace=True)
        g = df.groupby(key_col_norm, sort=False)
        first_idx = g["dt_idx"].transform("min")      # 每只股票首次出现的交易日索引
        entry_idx = df["dt_idx"] + 1                  # 进入日 = t+1
        ok_ipo = (entry_idx - first_idx + 1) >= ipo_min_bars
        ok = ok & ok_ipo

    # 只有 ok 的样本才允许有 p1/p2，否则置 NaN
    p1 = np.where(ok, df["next_p1"], np.nan)
    p2 = np.where(ok, df["next_p2"], np.nan)

    # 生成回归 label
    df["label__ret_1d_qlib"] = p2 / p1 - 1.0

    # 生成分类 label：ret>0 为 1，否则 0；ret=NaN 时分类也应 NaN（避免误导）
    df["label__up_1d_qlib"] = np.where(
        pd.notna(df["label__ret_1d_qlib"]),
        (df["label__ret_1d_qlib"] > 0).astype(np.int8),
        np.nan
    )
    return df


# -----------------------------
# 主流程
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

    # 可选：当列名不标准时手动指定
    ap.add_argument("--feature_date_col", type=str, default=None, help="Optional: feature date col name")
    ap.add_argument("--ohlcv_date_col", type=str, default=None, help="Optional: OHLCV date col name")
    ap.add_argument(
        "--force_target_code_format",
        type=str,
        default=None,
        choices=[None, "rq", "qlib", "ts", "raw"],
        help="Optional: force code normalization target; default inferred from feature key sample"
    )
    args = ap.parse_args()

    feature_dir = Path(args.feature_dir)
    ohlcv_dir = Path(args.ohlcv_dir)
    out_dir = Path(args.out_dir)
    out_labels_dir = out_dir / "_labels_only"
    out_full_dir = out_dir / "labeled_yearly_parquet"

    # 找到 feature/ohlcv 中共同存在的年份（只处理交集，避免缺年）
    feat_years = dict(list_year_files(feature_dir))
    ohlcv_years = dict(list_year_files(ohlcv_dir))
    common_years = sorted(set(feat_years.keys()) & set(ohlcv_years.keys()))
    if not common_years:
        raise ValueError("No common years between feature_dir and ohlcv_dir.")

    # 用第一年的文件探测列名
    feat_fp0 = feat_years[common_years[0]]
    ohlcv_fp0 = ohlcv_years[common_years[0]]

    # feature: key/date 列探测
    feat_key = detect_feature_key_col(feat_fp0)
    feat_date_col = detect_date_col(feat_fp0, args.feature_date_col)

    # ohlcv: key/date/price 列探测
    ohlcv_key_raw = detect_ohlcv_key_col(ohlcv_fp0, prefer=feat_key)
    ohlcv_date_col = detect_date_col(ohlcv_fp0, args.ohlcv_date_col)
    price_col = detect_ohlcv_price_col(ohlcv_fp0, args.price_col)

    # 从 feature 里抽一个 key 的样例值，用于推断统一后的股票代码格式
    feat_sample_tbl = pq.read_table(feat_fp0, columns=[feat_key]).to_pandas()
    feat_sample_val = str(feat_sample_tbl[feat_key].dropna().astype(str).iloc[0])
    target_fmt = args.force_target_code_format or infer_code_format(feat_sample_val)

    print(f"[INFO] feature key={feat_key}, feature date={feat_date_col}")
    print(f"[INFO] ohlcv key(raw)={ohlcv_key_raw}, ohlcv date={ohlcv_date_col}, price_col={price_col}")
    print(f"[INFO] target code format inferred from feature sample '{feat_sample_val}' => {target_fmt}")

    # 读取交易日历：用于严格定义 t+1 / t+2
    cal_dates, date_to_idx = load_calendar(Path(args.calendar_csv))
    print(f"[INFO] calendar size={len(cal_dates)} first={cal_dates[0]} last={cal_dates[-1]}")

    # -------------------------
    # A) 先生成 labels_only（从 OHLCV 算）
    # 重点：跨年处理
    # - 旧一年最后两天的 t+1/t+2 可能落在下一年
    # - patch_cross_year 会把 prev_year 的末尾 next_dt1/next_dt2 补上
    # -------------------------
    prev_year = None
    prev_min = None

    for y in common_years:
        print(f"[A] Read OHLCV year {y}")

        # 读取该年 OHLCV 最小必要列
        cur = read_minimal_ohlcv_year(ohlcv_years[y], ohlcv_date_col, ohlcv_key_raw, price_col)

        # 把 OHLCV 的 key 转成与 feature 相同的格式（避免 join 对不上）
        cur["key_norm"] = normalize_to_target(cur[ohlcv_key_raw], target_fmt)

        # 先得到“下一条数据记录”的 next_dt1/next_dt2
        cur = build_future(cur, "key_norm")

        # 用交易日历校验 next_dt1/next_dt2 是否等于真正的 t+1/t+2，生成严格 label
        cur = recompute_strict_qlib_label(cur, "key_norm", cal_dates, date_to_idx, args.ipo_min_bars)

        # 从第二年开始，我们可以把 prev_year 写出（因为已经看到了当前年的 head 信息，可以做跨年 patch）
        if prev_min is not None:
            print(f"[A] Patch boundary {prev_year} <- {y}")

            # 把 prev_year 末尾两天的 next 指向新年的前两天（同股票）
            prev_min = patch_cross_year(prev_min, cur, "key_norm")

            # patch 后需要重新基于严格规则生成 label（因为 next_dt1/next_dt2 被更新了）
            prev_min = recompute_strict_qlib_label(prev_min, "key_norm", cal_dates, date_to_idx, args.ipo_min_bars)

            # 写出 prev_year 的 labels_only
            out_fp = out_labels_dir / f"year={prev_year}.parquet"
            lab = prev_min[["datetime", "key_norm", "label__ret_1d_qlib", "label__up_1d_qlib"]].copy()
            # labels_only 的 key 列名改回与 feature 一样（order_book_id 或 instrument）
            lab.rename(columns={"key_norm": feat_key}, inplace=True)
            write_parquet(lab, out_fp)
            print(f"[A] Wrote labels: {out_fp}")

        prev_year = y
        prev_min = cur

    # 最后一年的 labels_only 直接写出（它的跨年 t+1/t+2 无法完整补齐，尾部会自然 NaN）
    if prev_min is not None:
        out_fp = out_labels_dir / f"year={prev_year}.parquet"
        lab = prev_min[["datetime", "key_norm", "label__ret_1d_qlib", "label__up_1d_qlib"]].copy()
        lab.rename(columns={"key_norm": feat_key}, inplace=True)
        write_parquet(lab, out_fp)
        print(f"[A] Wrote labels(last): {out_fp}")

    # 只生成 labels_only 就结束
    if args.mode == "labels_only":
        print(f"[DONE] labels_only => {out_labels_dir}")
        return

    # -------------------------
    # B) 把 labels_only 合并回 feature（以 feature 为基准 left join）
    # 说明：
    # - 用 how='left'：保留 feature 的所有行
    # - label 对不上或严格规则导致 NaN：label 列就是 NaN
    # - 训练时应 drop label 为 NaN 的样本
    # -------------------------
    out_full_dir.mkdir(parents=True, exist_ok=True)

    for y in common_years:
        print(f"[B] Merge year {y}")

        # 读入当年的 feature 全表（注意：如果 feature 原本带旧 label，这里不会删，会原样保留）
        feat = pq.read_table(feat_years[y]).to_pandas()

        # 统一 feature 的 key/date 格式
        feat.rename(columns={feat_date_col: "datetime"}, inplace=True)
        feat["datetime"] = pd.to_datetime(feat["datetime"]).dt.date
        feat[feat_key] = feat[feat_key].astype(str)

        # 读入当年的 labels_only
        lab = pq.read_table(out_labels_dir / f"year={y}.parquet").to_pandas()
        lab["datetime"] = pd.to_datetime(lab["datetime"]).dt.date
        lab[feat_key] = lab[feat_key].astype(str)

        # 以 feature 为基准拼接 label（feature 行可能没有 label -> NaN）
        merged = feat.merge(lab, on=["datetime", feat_key], how="left", validate="many_to_one")

        # 输出带 label 的全量数据
        out_fp = out_full_dir / f"year={y}.parquet"
        write_parquet(merged, out_fp)
        print(f"[B] Wrote full labeled: {out_fp}")

    print(f"[DONE] append_full => {out_full_dir}")


if __name__ == "__main__":
    main()
