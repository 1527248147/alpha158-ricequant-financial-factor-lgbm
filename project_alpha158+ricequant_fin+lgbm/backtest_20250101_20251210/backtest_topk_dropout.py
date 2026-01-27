# backtest_topk_dropout.py
# Qlib-style TopkDropout backtest with conservative A-share constraints
# - topk=50, n_drop=5
# - shift=1 (t signal -> t+1 execution)
# - deal_price=close
# - strict tradability checks, limit-locked detection (FIXED prev_close), IPO filter, forced liquidation
# - tail sell-only to 2025-12-15 and terminal liquidation on data end


# cd /d C:\AI_STOCK\project_alpha158+ricequant_fin+lgbm

# python backtest_20250101_20251210\backtest_topk_dropout.py ^
#   --model_dir "train_models\_train_lambdarank_v4_alpha_plus_fundTop15_seed42" ^
#   --features_dir "labeled\_calendar\labeled_yearly_parquet" ^
#   --ohlcv_dir "dataset\rq_ohlcv_yearly_parquet" ^
#   --calendar_csv "dataset\trading_calendar_from_merged.csv" ^
#   --out_dir "backtest_20250101_20251210\out_topk_dropout_2025" ^
#   --topk 50 --n_drop 5 --init_cash 1000000 --settlement_t1
#   --enable_benchmark --benchmark_csv "backtest_20250101_20251210\benchmark_000985.csv" ^
#   --compute_ic






# backtest_topk_dropout.py
# Qlib-style TopkDropout backtest with conservative A-share constraints + Benchmark(000985) + IC/RankIC
#
# 核心语义：
# - 每天收盘 EOD 用当天特征算分数 score(d)
# - shift=1：score(d) 在下一个交易日 d+1 才允许执行买卖（这里执行价用 close，代表“收盘调仓”语义）
# - TopkDropout：持仓目标 topk；每天尽量保持 topk；卖出 n_drop 个（优先卖出不在 topk 的、或分数最低的）
#
# 交易约束（保守）：
# - 必须有 OHLCV 行且 open/high/low/close>0 且 volume>0 才视为“可交易”
# - 涨跌停“锁死”判断：用 OHLCV + prev_close 推断（保守），主板同时检查 10% 和 5%(覆盖ST)，300/688=20%，8开头(BJ)=30%
# - 买入日若锁死则跳过；卖出日若锁死则顺延（must_sell=True，blocked_sell_days++）
# - 长期缺行/停牌/退市：blocked_sell_days 达到 max_blocked_days 强平（按最后可得价格再 haircut）
# - data_end_date 终止日：对剩余持仓做 terminal liquidation（再 haircut）
#
# Benchmark：
# - 使用本地 CSV（你已用 RiceQuant 拉好）：backtest_20250101_20251210\benchmark_000985.csv
# - 需要至少包含：date + close（列名可不同，会自动猜）
#
# IC / RankIC：
# - 对“信号日 t”的分数 score(t)，用严格交易日历的 t+1 与 t+2 close 计算 label：
#   ret = close(t+2)/close(t+1) - 1
# - 在 t+1 当天，我们已经拿到了 close(t+1)；再去取下一交易日 close(t+2) 即可计算当天的 IC/RIC
# - 输出到 ic_daily.csv，并在 diagnostics_summary.json 里给出均值等统计

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except Exception as e:
    raise RuntimeError("lightgbm is required. Please `pip install lightgbm`.") from e

try:
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
except Exception as e:
    raise RuntimeError("pyarrow is required. Please `pip install pyarrow`.") from e


# -----------------------------
# Utils
# -----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_date(s: str) -> pd.Timestamp:
    return pd.to_datetime(s).normalize()


def fmt_date(d: pd.Timestamp) -> str:
    return d.strftime("%Y-%m-%d")


def year_of(d: pd.Timestamp) -> int:
    return int(d.strftime("%Y"))


def read_calendar_dates(calendar_csv: str) -> List[pd.Timestamp]:
    df = pd.read_csv(calendar_csv)
    col = "date" if "date" in df.columns else df.columns[0]
    s = pd.to_datetime(df[col], errors="coerce").dropna().dt.normalize()
    cal = sorted(pd.Series(s.unique()).tolist())
    if not cal:
        raise ValueError(f"Empty trading calendar: {calendar_csv}")
    return cal


def prev_trading_day(cal: List[pd.Timestamp], d: pd.Timestamp) -> Optional[pd.Timestamp]:
    arr = np.array(cal, dtype="datetime64[ns]")
    idx = np.searchsorted(arr, np.datetime64(d), side="left") - 1
    if idx < 0:
        return None
    return cal[idx]


def next_trading_day(cal: List[pd.Timestamp], d: pd.Timestamp) -> Optional[pd.Timestamp]:
    arr = np.array(cal, dtype="datetime64[ns]")
    idx = np.searchsorted(arr, np.datetime64(d), side="right")
    if idx >= len(cal):
        return None
    return cal[idx]


def load_used_features(model_dir: str) -> List[str]:
    p = Path(model_dir) / "used_features.txt"
    if not p.exists():
        raise FileNotFoundError(f"used_features.txt not found: {p}")
    feats = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    bad = [c for c in feats if c.startswith("label__")]
    if bad:
        raise ValueError(f"used_features.txt contains label columns (leakage): {bad[:5]} ...")
    return feats


def find_year_parquet(dir_path: str, year: int) -> Path:
    d = Path(dir_path)
    patterns = [
        d / f"year={year}.parquet",
        d / f"{year}.parquet",
        d / f"rq_ohlcv_{year}.parquet",
        d / f"ohlcv_{year}.parquet",
        d / f"features_{year}.parquet",
    ]
    for p in patterns:
        if p.exists():
            return p
    hits = sorted(d.glob(f"*{year}*.parquet"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"Cannot find parquet for year={year} in {dir_path}")


def parquet_existing_columns(p: Path) -> List[str]:
    return pq.ParquetFile(str(p)).schema_arrow.names


def read_parquet_safe(p: Path, want_cols: List[str]) -> pd.DataFrame:
    exist = set(parquet_existing_columns(p))
    cols = [c for c in want_cols if c in exist]
    if not cols:
        return pd.read_parquet(p)
    return pd.read_parquet(p, columns=cols)


# -----------------------------
# Benchmark loader (local CSV)
# -----------------------------
def _pick_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def load_benchmark_csv(path: str) -> pd.DataFrame:
    """
    读取本地 benchmark CSV（000985 中证全指）
    需要至少有：date + close
    列名可能不同：会自动猜
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"benchmark_csv not found: {p}")

    df = pd.read_csv(p)
    if df.empty:
        raise ValueError(f"benchmark_csv is empty: {p}")

    date_col = _pick_col(df.columns.tolist(), ["date", "datetime", "trade_date", "time"])
    close_col = _pick_col(df.columns.tolist(), ["close", "Close", "收盘", "收盘价", "price", "index_close"])

    if date_col is None or close_col is None:
        raise ValueError(
            f"benchmark_csv must contain date+close. "
            f"Detected date_col={date_col}, close_col={close_col}, columns={df.columns.tolist()}"
        )

    out = df[[date_col, close_col]].copy()
    out = out.rename(columns={date_col: "date", close_col: "close"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["date", "close"])
    out = out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return out


# -----------------------------
# OHLCV cache
# -----------------------------
class OHLCVYearCache:
    def __init__(self, ohlcv_dir: str):
        self.ohlcv_dir = ohlcv_dir
        self._cache: Dict[int, pd.DataFrame] = {}

    def _load_year(self, year: int) -> None:
        p = find_year_parquet(self.ohlcv_dir, year)
        df = pd.read_parquet(p)

        # 统一字段名
        if "symbol" not in df.columns and "order_book_id" in df.columns:
            df = df.rename(columns={"order_book_id": "symbol"})
        if "date" not in df.columns and "datetime" in df.columns:
            df = df.rename(columns={"datetime": "date"})

        # 找 volume 列
        vol_col = None
        for c in ["volume", "total_volume", "vol", "volume_traded"]:
            if c in df.columns:
                vol_col = c
                break
        if vol_col is None:
            df["volume"] = np.nan
            vol_col = "volume"

        miss = [c for c in ["date", "symbol", "open", "high", "low", "close"] if c not in df.columns]
        if miss:
            raise ValueError(f"OHLCV missing columns {miss} in {p}")

        df = df[["date", "symbol", "open", "high", "low", "close", vol_col]].copy()
        if vol_col != "volume":
            df = df.rename(columns={vol_col: "volume"})

        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df = df.dropna(subset=["date", "symbol"])
        df["symbol"] = df["symbol"].astype(str)

        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.set_index(["date", "symbol"]).sort_index()
        self._cache[year] = df

    def get_day(self, d: pd.Timestamp) -> pd.DataFrame:
        y = year_of(d)
        if y not in self._cache:
            self._load_year(y)
        dfy = self._cache[y]
        try:
            return dfy.xs(d, level=0, drop_level=False)
        except KeyError:
            return pd.DataFrame(columns=dfy.columns).set_index(["date", "symbol"])


# -----------------------------
# Features day loader
# -----------------------------
class FeaturesDayLoader:
    def __init__(self, features_dir: str, used_features: List[str]):
        self.features_dir = features_dir
        self.used_features = used_features
        self._ds_cache: Dict[int, ds.Dataset] = {}
        self._dt_field_type: Dict[int, str] = {}

    def _get_dataset(self, year: int) -> ds.Dataset:
        if year in self._ds_cache:
            return self._ds_cache[year]
        p = find_year_parquet(self.features_dir, year)
        dset = ds.dataset(str(p), format="parquet")
        self._ds_cache[year] = dset

        schema = dset.schema
        if "datetime" not in schema.names:
            raise ValueError(f"Features parquet missing 'datetime' column: {p}")

        dt_type = schema.field("datetime").type
        if pa.types.is_timestamp(dt_type) or pa.types.is_date(dt_type):
            self._dt_field_type[year] = "timestamp"
        else:
            self._dt_field_type[year] = "string"
        return dset

    def load_day(self, d: pd.Timestamp) -> pd.DataFrame:
        y = year_of(d)
        dset = self._get_dataset(y)

        cols = ["datetime", "order_book_id"] + self.used_features
        missing = [c for c in (["order_book_id"] + self.used_features) if c not in dset.schema.names]
        if missing:
            raise ValueError(f"Features parquet year {y} missing columns: {missing[:10]} ...")

        if self._dt_field_type[y] == "timestamp":
            filt = (ds.field("datetime") == np.datetime64(d))
        else:
            filt = (ds.field("datetime") == fmt_date(d))

        table = dset.to_table(filter=filt, columns=cols)
        df = table.to_pandas()
        df["order_book_id"] = df["order_book_id"].astype(str)
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce").dt.normalize()
        df = df.dropna(subset=["datetime", "order_book_id"])
        return df


# -----------------------------
# IPO first-tradeable-day builder
# -----------------------------
def build_first_tradeable_day(
    ohlcv_dir: str,
    symbols_needed: List[str],
    cache_path: Path,
) -> Dict[str, pd.Timestamp]:
    """
    用 OHLCV 扫描每只股票第一次出现“close>0 & volume>0”的交易日，作为 first_tradeable_day
    用于过滤新股（ipo_min_days）
    """
    if cache_path.exists():
        df = pd.read_csv(cache_path)
        df["first_day"] = pd.to_datetime(df["first_day"], errors="coerce").dt.normalize()
        out = {r["symbol"]: r["first_day"] for _, r in df.dropna(subset=["symbol", "first_day"]).iterrows()}
        return {s: out[s] for s in symbols_needed if s in out}

    needed = set(symbols_needed)
    first: Dict[str, pd.Timestamp] = {}

    files = sorted(Path(ohlcv_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in ohlcv_dir: {ohlcv_dir}")

    def extract_year(p: Path) -> int:
        digits = "".join([c if c.isdigit() else " " for c in p.stem]).split()
        yrs = [int(x) for x in digits if len(x) == 4 and 1990 <= int(x) <= 2100]
        return yrs[0] if yrs else 9999

    files = sorted(files, key=extract_year)

    for p in files:
        if not needed:
            break

        df = read_parquet_safe(p, ["date", "symbol", "order_book_id", "close", "volume", "total_volume"])

        if "symbol" not in df.columns and "order_book_id" in df.columns:
            df = df.rename(columns={"order_book_id": "symbol"})
        if "date" not in df.columns and "datetime" in df.columns:
            df = df.rename(columns={"datetime": "date"})

        vol_col = "volume" if "volume" in df.columns else ("total_volume" if "total_volume" in df.columns else None)
        if vol_col is None:
            continue
        if not {"date", "symbol", "close"}.issubset(df.columns):
            continue

        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df["symbol"] = df["symbol"].astype(str)
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df[vol_col] = pd.to_numeric(df[vol_col], errors="coerce")

        df = df[df["symbol"].isin(needed)]
        df = df.dropna(subset=["date", "symbol", "close", vol_col])
        df = df[(df["close"] > 0) & (df[vol_col] > 0)]
        if df.empty:
            continue

        g = df.groupby("symbol")["date"].min()
        for sym, dt in g.items():
            if sym not in first:
                first[sym] = dt
        needed -= set(first.keys())

    out_df = pd.DataFrame({"symbol": list(first.keys()), "first_day": [fmt_date(v) for v in first.values()]})
    out_df.to_csv(cache_path, index=False, encoding="utf-8")
    return first


# -----------------------------
# Trading rules
# -----------------------------
def is_valid_price_row(row: pd.Series) -> bool:
    """
    保守：没有 open/high/low/close 或者 <=0，或者 volume<=0，都视为不可交易
    """
    for c in ["open", "high", "low", "close"]:
        v = row.get(c, np.nan)
        if not (isinstance(v, (int, float, np.floating)) and np.isfinite(v) and v > 0):
            return False
    vvol = row.get("volume", np.nan)
    if not (isinstance(vvol, (int, float, np.floating)) and np.isfinite(vvol) and vvol > 0):
        return False
    return True


def infer_limit_rates(sym: str) -> List[float]:
    """
    推断涨跌停比例（保守）

    - 688xxx 科创板：20%
    - 300xxx 创业板：20%
    - 8xxxx  北交所：30%（若你不想考虑北交所，可删掉）
    - 其它主板：同时检查 10% 和 5%（覆盖 ST 5% 但不需要 ST 标签）
    """
    code = sym.split(".")[0] if "." in sym else sym
    if code.startswith("688"):
        return [0.20]
    if code.startswith("300"):
        return [0.20]
    if code.startswith("8"):
        return [0.30]
    # 主板：10% + 5%(ST)
    return [0.10, 0.05]


def is_limit_locked(row: pd.Series, prev_close: float, eps: float, tiny: float, sym: str) -> bool:
    """
    用 OHLCV + prev_close 做“涨跌停锁死”判定（保守）

    LOCKED 条件：
      - 当天几乎没有价格波动：(high-low)/prev_close <= tiny
      - 且 close 接近 涨停价 或 跌停价（按推断的 limit_rate）

    说明：
    - 没有 ST 标记时，主板同时检查 10% 和 5%，可覆盖 ST 5% 情况（更保守，可能略多跳过）
    - 300/688 检查 20%，8 开头检查 30%
    """
    if not np.isfinite(prev_close) or prev_close <= 0:
        return True  # 更保守：不知道昨收，直接当锁死/不可交易

    high = float(row["high"])
    low = float(row["low"])
    close = float(row["close"])
    if not (np.isfinite(high) and np.isfinite(low) and np.isfinite(close)):
        return True

    # 如果日内有明显波动，就不算“锁死”
    if (high - low) / prev_close > tiny:
        return False

    # 贴近涨停/跌停价
    for L in infer_limit_rates(sym):
        up = prev_close * (1.0 + L)
        dn = prev_close * (1.0 - L)
        if abs(close - up) / up < eps:
            return True
        if abs(close - dn) / dn < eps:
            return True
    return False


def calc_fee(value: float, bps: float, min_cost: float) -> float:
    """
    bps: 万分之 bps
    """
    fee = value * (bps / 10000.0)
    return max(fee, min_cost) if value > 0 else 0.0


@dataclass
class Position:
    symbol: str
    shares: int
    buy_date: str
    buy_price: float
    must_sell: bool = False              # 是否“必须卖”（例如不在 topk 或 tail 强制清仓）
    blocked_sell_days: int = 0           # 卖不掉累计天数（涨跌停/停牌/缺行/无量）
    last_mark_price: float = 0.0         # 最近一次能拿到的 close 用于 MTM 和强平价格兜底


# -----------------------------
# Performance metrics
# -----------------------------
def _max_drawdown_from_nav(nav: pd.Series) -> float:
    nav = nav.dropna()
    if nav.empty:
        return np.nan
    peak = nav.cummax()
    dd = (nav / peak) - 1.0
    return float(dd.min())


def compute_basic_metrics(returns: pd.Series, benchmark_returns: Optional[pd.Series] = None) -> Dict:
    """
    不依赖 scipy 的基础绩效指标（够用且稳）
    - annual_return, annual_vol, sharpe, max_drawdown, hit_rate
    - 若有基准：excess_return, info_ratio, beta, alpha(简化)
    """
    r = returns.dropna()
    out: Dict = {}
    out["n_obs"] = int(len(r))
    if len(r) < 30:
        out["error"] = "数据不足（需要30个以上观测值）"
        return out

    ann_ret = float(r.mean() * 252.0)
    ann_vol = float(r.std(ddof=1) * np.sqrt(252.0))
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0

    nav = (1 + r).cumprod()
    mdd = _max_drawdown_from_nav(nav)

    out.update({
        "annual_return": ann_ret,
        "annual_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "cumulative_return": float(nav.iloc[-1] - 1.0),
        "max_drawdown": float(mdd),
        "hit_rate": float((r > 0).mean()),
        "mean_daily_return": float(r.mean()),
        "std_daily_return": float(r.std(ddof=1)),
    })

    if benchmark_returns is not None:
        bm = benchmark_returns.reindex(r.index).dropna()
        rr = r.reindex(bm.index).dropna()
        bm = bm.reindex(rr.index).dropna()
        if len(rr) >= 30 and len(bm) >= 30:
            excess = rr - bm
            te = float(excess.std(ddof=1) * np.sqrt(252.0))
            ir = float(excess.mean() * 252.0 / te) if te > 0 else 0.0

            cov = float(np.cov(rr.values, bm.values, ddof=1)[0, 1])
            var_bm = float(np.var(bm.values, ddof=1))
            beta = cov / var_bm if var_bm > 0 else np.nan
            alpha = float((rr.mean() - beta * bm.mean()) * 252.0) if np.isfinite(beta) else np.nan

            out.update({
                "annual_excess_return": float(excess.mean() * 252.0),
                "tracking_error": te,
                "information_ratio": ir,
                "beta": float(beta) if np.isfinite(beta) else np.nan,
                "alpha": alpha,
            })
    return out


# -----------------------------
# Backtest core
# -----------------------------
def run_backtest(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # 交易日历
    cal = read_calendar_dates(args.calendar_csv)
    cal_set = set(cal)
    cal_idx = {d: i for i, d in enumerate(cal)}

    # 时间窗口
    start = parse_date(args.start_date)
    end = parse_date(args.end_date)
    data_end = parse_date(args.data_end_date)
    tail_start = parse_date(args.tail_start_date)

    # 对齐到交易日
    if start not in cal_set:
        start = next_trading_day(cal, start) or start
    if end not in cal_set:
        end = prev_trading_day(cal, end) or end
    if data_end not in cal_set:
        data_end = prev_trading_day(cal, data_end) or data_end

    if start > end:
        raise ValueError(f"start_date > end_date after alignment: {start} > {end}")
    if end > data_end:
        raise ValueError(f"end_date {end} is after data_end_date {data_end}")

    # preheat：为了在 start 当天能拿到 prev_close（昨收），至少要多读一天
    preheat = prev_trading_day(cal, start)
    if preheat is None:
        raise ValueError("Cannot find preheat trading day before start.")

    # 载入模型和特征列
    used_features = load_used_features(args.model_dir)
    model_path = Path(args.model_dir) / "model.txt"
    if not model_path.exists():
        raise FileNotFoundError(f"model.txt not found: {model_path}")
    booster = lgb.Booster(model_file=str(model_path))

    feat_loader = FeaturesDayLoader(args.features_dir, used_features)
    ohlcv_cache = OHLCVYearCache(args.ohlcv_dir)

    # Universe：从 [preheat, end] 的 features 里出现过的股票集合（避免扫全市场过慢）
    universe_syms: List[str] = []
    for d in cal:
        if d < preheat:
            continue
        if d > end:
            break
        df_day = feat_loader.load_day(d)
        universe_syms.extend(df_day["order_book_id"].unique().tolist())
    universe_syms = sorted(set(universe_syms))
    if not universe_syms:
        raise RuntimeError("No symbols found in features for the selected window.")

    # IPO 首个可交易日缓存
    ipo_cache_path = out_dir / "first_tradeable_day_cache.csv"
    first_day_map = build_first_tradeable_day(args.ohlcv_dir, universe_syms, ipo_cache_path)

    # 账户状态
    cash_available = float(args.init_cash)  # 可用现金
    cash_settling = 0.0                     # 卖出待结算（T+1 可用）
    positions: Dict[str, Position] = {}     # 当前持仓

    # last_close：记录“昨天 close”，用于推断涨跌停价
    # 注意：我们在每天循环开始时 snapshot 到 prev_close_map，再做交易；每天结束后才更新 last_close
    last_close: Dict[str, float] = {}

    # scores_pending：上一交易日 EOD 生成的 score（shift=1），用于本日执行
    scores_pending: Optional[pd.Series] = None
    pending_signal_day: Optional[pd.Timestamp] = None  # scores_pending 对应的信号日（t）

    # 输出表
    nav_rows = []
    diag_rows = []
    trade_rows = []
    ic_rows = []

    # 诊断统计
    buy_skip_totals = {
        "missing_row": 0,
        "invalid_row": 0,
        "limit_locked": 0,
        "ipo_too_new": 0,
        "lot_too_small": 0,
        "insufficient_cash": 0,
    }
    sell_block_totals = {"missing_row": 0, "invalid_row": 0, "limit_locked": 0}
    forced_liq_total = 0
    terminal_liq_total = 0
    cum_fee = 0.0

    # 主循环：从 preheat 到 data_end
    for d in cal:
        if d < preheat:
            continue
        if d > data_end:
            break

        d_str = fmt_date(d)

        # 交易开始前，snapshot 昨收（防止当天更新 close 影响涨跌停推断）
        prev_close_map = last_close.copy()

        # T+1 结算：把昨天卖出的钱转入可用现金
        if args.settlement_t1 and cash_settling != 0.0:
            cash_available += cash_settling
            cash_settling = 0.0

        # 读取当天 OHLCV（用于交易和 MTM）
        day_ohlcv = ohlcv_cache.get_day(d)
        day_index = day_ohlcv.index

        def get_ohlcv(day_df: pd.DataFrame, day_idx, day: pd.Timestamp, sym: str) -> Optional[pd.Series]:
            key = (day, sym)
            if key in day_idx:
                return day_df.loc[key]
            return None

        # 先对已有持仓做 MTM（用当日 close）
        for sym, pos in positions.items():
            row = get_ohlcv(day_ohlcv, day_index, d, sym)
            if row is not None:
                c = row.get("close", np.nan)
                if np.isfinite(c) and c > 0:
                    pos.last_mark_price = float(c)

        # 窗口定义
        in_main = (start <= d <= end)
        in_tail = (d > end) and (tail_start <= d <= data_end)

        # 今天可执行的分数（来自上一日 EOD）
        exec_scores = scores_pending if (scores_pending is not None and pending_signal_day is not None) else None
        sig_day = pending_signal_day  # exec_scores 对应的信号日 t（执行日是 t+1）

        # ---------------- IC / RankIC（可选） ----------------
        # 在执行日 d（=t+1）时，去取下一交易日 d2（=t+2）close，计算 label=close(d2)/close(d)-1
        # 再与 score(t) 相关，得到 IC / RankIC（Spearman）
        if args.compute_ic and exec_scores is not None and sig_day is not None:
            d2 = next_trading_day(cal, d)
            if d2 is not None and d2 <= data_end:
                day2_ohlcv = ohlcv_cache.get_day(d2)
                day2_idx = day2_ohlcv.index

                # 构建 label
                labels = {}
                for sym, sc in exec_scores.items():
                    # 只在“严格可得”时算：d 与 d2 都要有 valid row
                    row1 = get_ohlcv(day_ohlcv, day_index, d, sym)
                    row2 = get_ohlcv(day2_ohlcv, day2_idx, d2, sym)
                    if row1 is None or row2 is None:
                        continue
                    if (not is_valid_price_row(row1)) or (not is_valid_price_row(row2)):
                        continue
                    c1 = float(row1["close"])
                    c2 = float(row2["close"])
                    if not (np.isfinite(c1) and np.isfinite(c2) and c1 > 0 and c2 > 0):
                        continue
                    labels[sym] = (c2 / c1) - 1.0

                if len(labels) >= 20:
                    s_label = pd.Series(labels, dtype=float)
                    s_score = exec_scores.reindex(s_label.index).astype(float)

                    # Pearson IC
                    ic = float(s_score.corr(s_label, method="pearson"))
                    # Spearman RankIC
                    ric = float(s_score.corr(s_label, method="spearman"))

                    ic_rows.append({
                        "date": fmt_date(sig_day),         # 信号日 t
                        "exec_date": fmt_date(d),          # 执行日 t+1
                        "label_date": fmt_date(d2),        # t+2
                        "n": int(len(s_label)),
                        "ic": ic,
                        "rank_ic": ric,
                    })
                else:
                    ic_rows.append({
                        "date": fmt_date(sig_day),
                        "exec_date": fmt_date(d),
                        "label_date": fmt_date(d2),
                        "n": int(len(labels)),
                        "ic": np.nan,
                        "rank_ic": np.nan,
                    })

        # ---------------- 日内统计变量 ----------------
        buy_filled = 0
        sell_filled = 0
        sell_blocked = 0
        forced_liq_today = 0
        terminal_liq_today = 0

        buy_skip = {k: 0 for k in buy_skip_totals.keys()}
        sell_block = {k: 0 for k in sell_block_totals.keys()}

        traded_value_buy = 0.0
        traded_value_sell = 0.0
        fee_today = 0.0

        # ---------------- SELL ----------------
        sell_list: List[str] = []

        # tail：只卖不买，强制清仓
        if in_tail:
            sell_list = list(positions.keys())
            for sym in sell_list:
                positions[sym].must_sell = True

        # main：TopkDropout 决定要卖哪些
        elif in_main and exec_scores is not None:
            held = list(positions.keys())
            if held:
                held_scores = {sym: float(exec_scores.get(sym, -1e18)) for sym in held}
                held_sorted_low = sorted(held, key=lambda s: held_scores[s])

                topk_syms = set(exec_scores.sort_values(ascending=False).head(args.topk).index.tolist())
                not_in_topk = [s for s in held_sorted_low if s not in topk_syms]

                planned = []
                planned.extend(not_in_topk[: args.n_drop])
                if len(planned) < args.n_drop:
                    remain = [s for s in held_sorted_low if s in topk_syms and s not in planned]
                    planned.extend(remain[: (args.n_drop - len(planned))])

                # must_sell 的优先
                must = [s for s in held if positions[s].must_sell]
                seen = set()
                for s in must + planned:
                    if s not in seen:
                        sell_list.append(s)
                        seen.add(s)

        for sym in sell_list:
            if sym not in positions:
                continue
            pos = positions[sym]

            row = get_ohlcv(day_ohlcv, day_index, d, sym)

            # 1) 缺行：卖不掉，blocked+1；到 max_blocked_days 强平
            if row is None:
                pos.must_sell = True
                pos.blocked_sell_days += 1
                sell_blocked += 1
                sell_block["missing_row"] += 1
                sell_block_totals["missing_row"] += 1

                if pos.blocked_sell_days >= args.max_blocked_days:
                    px = (pos.last_mark_price if pos.last_mark_price > 0 else pos.buy_price) * (1.0 - args.haircut)
                    value = px * pos.shares
                    fee = calc_fee(value, args.sell_cost_bps + args.sell_stamp_bps, args.min_cost)
                    proceeds = max(0.0, value - fee)

                    if args.settlement_t1:
                        cash_settling += proceeds
                    else:
                        cash_available += proceeds

                    traded_value_sell += value
                    fee_today += fee
                    cum_fee += fee
                    forced_liq_today += 1
                    forced_liq_total += 1

                    trade_rows.append({
                        "action": "FORCE_SELL",
                        "symbol": sym,
                        "buy_date": pos.buy_date,
                        "sell_date": d_str,
                        "buy_price": pos.buy_price,
                        "sell_price": px,
                        "shares": pos.shares,
                        "reason": "forced_liquidation_missing_row",
                        "sell_delay_days": pos.blocked_sell_days,
                        "fee": fee,
                    })
                    del positions[sym]
                continue

            # 2) 行不合法：卖不掉，blocked+1；到 max_blocked_days 强平
            if not is_valid_price_row(row):
                pos.must_sell = True
                pos.blocked_sell_days += 1
                sell_blocked += 1
                sell_block["invalid_row"] += 1
                sell_block_totals["invalid_row"] += 1

                if pos.blocked_sell_days >= args.max_blocked_days:
                    px = (pos.last_mark_price if pos.last_mark_price > 0 else pos.buy_price) * (1.0 - args.haircut)
                    value = px * pos.shares
                    fee = calc_fee(value, args.sell_cost_bps + args.sell_stamp_bps, args.min_cost)
                    proceeds = max(0.0, value - fee)

                    if args.settlement_t1:
                        cash_settling += proceeds
                    else:
                        cash_available += proceeds

                    traded_value_sell += value
                    fee_today += fee
                    cum_fee += fee
                    forced_liq_today += 1
                    forced_liq_total += 1

                    trade_rows.append({
                        "action": "FORCE_SELL",
                        "symbol": sym,
                        "buy_date": pos.buy_date,
                        "sell_date": d_str,
                        "buy_price": pos.buy_price,
                        "sell_price": px,
                        "shares": pos.shares,
                        "reason": "forced_liquidation_invalid_row",
                        "sell_delay_days": pos.blocked_sell_days,
                        "fee": fee,
                    })
                    del positions[sym]
                continue

            # 3) 涨跌停锁死：卖不掉，blocked+1；到 max_blocked_days 强平（保守）
            prevc = prev_close_map.get(sym, np.nan)
            if is_limit_locked(row, prevc, args.limit_eps, args.limit_tiny, sym):
                pos.must_sell = True
                pos.blocked_sell_days += 1
                sell_blocked += 1
                sell_block["limit_locked"] += 1
                sell_block_totals["limit_locked"] += 1

                if pos.blocked_sell_days >= args.max_blocked_days:
                    px = float(row["close"]) * (1.0 - args.haircut)
                    value = px * pos.shares
                    fee = calc_fee(value, args.sell_cost_bps + args.sell_stamp_bps, args.min_cost)
                    proceeds = max(0.0, value - fee)

                    if args.settlement_t1:
                        cash_settling += proceeds
                    else:
                        cash_available += proceeds

                    traded_value_sell += value
                    fee_today += fee
                    cum_fee += fee
                    forced_liq_today += 1
                    forced_liq_total += 1

                    trade_rows.append({
                        "action": "FORCE_SELL",
                        "symbol": sym,
                        "buy_date": pos.buy_date,
                        "sell_date": d_str,
                        "buy_price": pos.buy_price,
                        "sell_price": px,
                        "shares": pos.shares,
                        "reason": "forced_liquidation_limit_locked",
                        "sell_delay_days": pos.blocked_sell_days,
                        "fee": fee,
                    })
                    del positions[sym]
                continue

            # 4) 正常卖出（close 成交）
            px = float(row["close"])
            value = px * pos.shares
            fee = calc_fee(value, args.sell_cost_bps + args.sell_stamp_bps, args.min_cost)
            proceeds = max(0.0, value - fee)

            if args.settlement_t1:
                cash_settling += proceeds
            else:
                cash_available += proceeds

            traded_value_sell += value
            fee_today += fee
            cum_fee += fee
            sell_filled += 1

            trade_rows.append({
                "action": "SELL",
                "symbol": sym,
                "buy_date": pos.buy_date,
                "sell_date": d_str,
                "buy_price": pos.buy_price,
                "sell_price": px,
                "shares": pos.shares,
                "reason": "sold",
                "sell_delay_days": pos.blocked_sell_days,
                "fee": fee,
            })
            del positions[sym]

        # ---------------- BUY（tail 不买） ----------------
        if in_main and (not in_tail) and exec_scores is not None:
            scores_sorted = exec_scores.sort_values(ascending=False)

            slots = max(0, args.topk - len(positions))
            if slots > 0:
                for sym, _sc in scores_sorted.items():
                    if slots <= 0:
                        break
                    if sym in positions:
                        continue

                    row = get_ohlcv(day_ohlcv, day_index, d, sym)
                    if row is None:
                        buy_skip["missing_row"] += 1
                        buy_skip_totals["missing_row"] += 1
                        continue
                    if not is_valid_price_row(row):
                        buy_skip["invalid_row"] += 1
                        buy_skip_totals["invalid_row"] += 1
                        continue

                    prevc = prev_close_map.get(sym, np.nan)
                    if is_limit_locked(row, prevc, args.limit_eps, args.limit_tiny, sym):
                        buy_skip["limit_locked"] += 1
                        buy_skip_totals["limit_locked"] += 1
                        continue

                    # 新股过滤：距离 first_tradeable_day 需 >= ipo_min_days
                    fd = first_day_map.get(sym, None)
                    if fd is None or fd not in cal_idx:
                        buy_skip["ipo_too_new"] += 1
                        buy_skip_totals["ipo_too_new"] += 1
                        continue
                    age = cal_idx[d] - cal_idx[fd]
                    if age < args.ipo_min_days:
                        buy_skip["ipo_too_new"] += 1
                        buy_skip_totals["ipo_too_new"] += 1
                        continue

                    px = float(row["close"])
                    if cash_available <= 0:
                        buy_skip["insufficient_cash"] += 1
                        buy_skip_totals["insufficient_cash"] += 1
                        break

                    # 简单等权：剩余现金 / 剩余槽位
                    budget = cash_available / float(slots)
                    shares = int(budget / px / args.lot_size) * args.lot_size
                    if shares <= 0:
                        buy_skip["lot_too_small"] += 1
                        buy_skip_totals["lot_too_small"] += 1
                        continue

                    value = px * shares
                    fee = calc_fee(value, args.buy_cost_bps, args.min_cost)
                    total_cost = value + fee

                    # 不够钱则收缩 shares
                    if total_cost > cash_available + 1e-9:
                        shares2 = int((cash_available / (px * (1 + args.buy_cost_bps / 10000.0))) / args.lot_size) * args.lot_size
                        if shares2 <= 0:
                            buy_skip["insufficient_cash"] += 1
                            buy_skip_totals["insufficient_cash"] += 1
                            continue
                        shares = shares2
                        value = px * shares
                        fee = calc_fee(value, args.buy_cost_bps, args.min_cost)
                        total_cost = value + fee
                        if total_cost > cash_available + 1e-9:
                            buy_skip["insufficient_cash"] += 1
                            buy_skip_totals["insufficient_cash"] += 1
                            continue

                    # 成交
                    cash_available -= total_cost
                    traded_value_buy += value
                    fee_today += fee
                    cum_fee += fee
                    buy_filled += 1

                    positions[sym] = Position(
                        symbol=sym,
                        shares=shares,
                        buy_date=d_str,
                        buy_price=px,
                        must_sell=False,
                        blocked_sell_days=0,
                        last_mark_price=px,
                    )

                    trade_rows.append({
                        "action": "BUY",
                        "symbol": sym,
                        "buy_date": d_str,
                        "sell_date": "",
                        "buy_price": px,
                        "sell_price": "",
                        "shares": shares,
                        "reason": "enter_topk",
                        "sell_delay_days": "",
                        "fee": fee,
                    })
                    slots -= 1

        # ---------------- data_end 终止强平 ----------------
        if d == data_end:
            for sym in list(positions.keys()):
                pos = positions[sym]
                px = (pos.last_mark_price if pos.last_mark_price > 0 else pos.buy_price) * (1.0 - args.haircut)
                value = px * pos.shares
                fee = calc_fee(value, args.sell_cost_bps + args.sell_stamp_bps, args.min_cost)
                proceeds = max(0.0, value - fee)

                cash_available += proceeds
                traded_value_sell += value
                fee_today += fee
                cum_fee += fee
                terminal_liq_today += 1
                terminal_liq_total += 1

                trade_rows.append({
                    "action": "TERMINAL_FORCE_SELL",
                    "symbol": sym,
                    "buy_date": pos.buy_date,
                    "sell_date": d_str,
                    "buy_price": pos.buy_price,
                    "sell_price": px,
                    "shares": pos.shares,
                    "reason": "terminal_liquidation",
                    "sell_delay_days": pos.blocked_sell_days,
                    "fee": fee,
                })
                del positions[sym]

        # ---------------- NAV & Diagnostics ----------------
        pos_value = 0.0
        for sym, pos in positions.items():
            mp = pos.last_mark_price if pos.last_mark_price > 0 else pos.buy_price
            pos_value += mp * pos.shares

        nav_total = cash_available + cash_settling + pos_value
        cash_ratio = (cash_available + cash_settling) / nav_total if nav_total > 0 else 1.0

        nav_rows.append({
            "date": d_str,
            "nav_total": nav_total,
            "cash_available": cash_available,
            "cash_settling": cash_settling,
            "positions_value": pos_value,
            "holdings": len(positions),
            "fee_today": fee_today,
            "cum_fee": cum_fee,
        })

        overdue = [p for p in positions.values() if p.must_sell]
        max_blocked = max([p.blocked_sell_days for p in overdue], default=0)

        diag_rows.append({
            "date": d_str,
            "in_main_window": int(in_main),
            "in_tail_window": int(in_tail),
            "holdings_target": args.topk,
            "holdings_actual": len(positions),
            "buy_filled": buy_filled,
            "sell_filled": sell_filled,
            "sell_blocked": sell_blocked,
            "forced_liquidations_today": forced_liq_today,
            "terminal_liquidations_today": terminal_liq_today,
            "overdue_positions": len(overdue),
            "max_blocked_sell_days": max_blocked,
            "traded_value_buy": traded_value_buy,
            "traded_value_sell": traded_value_sell,
            "fee_today": fee_today,
            "cash_ratio": cash_ratio,
            **{f"buy_skip_{k}": v for k, v in buy_skip.items()},
            **{f"sell_block_{k}": v for k, v in sell_block.items()},
        })

        # ---------------- EOD 生成下一天执行的 score ----------------
        if d <= end:
            df_feat = feat_loader.load_day(d)
            if df_feat.empty:
                scores_today = pd.Series(dtype=float)
            else:
                X = df_feat[used_features].copy()
                yhat = booster.predict(X, num_iteration=booster.best_iteration or -1)
                scores_today = pd.Series(yhat, index=df_feat["order_book_id"].values, dtype=float)
            scores_pending = scores_today
            pending_signal_day = d
        else:
            scores_pending = None
            pending_signal_day = None

        # ---------------- EOD 更新 last_close 给下一天当 prev_close ----------------
        if not day_ohlcv.empty:
            tmp = day_ohlcv.reset_index()
            for _, r in tmp.iterrows():
                sym = str(r["symbol"])
                c = r["close"]
                if np.isfinite(c) and c > 0:
                    last_close[sym] = float(c)

    # ===================== 回测结束：汇总输出 =====================
    nav_df = pd.DataFrame(nav_rows)
    diag_df = pd.DataFrame(diag_rows)
    trades_df = pd.DataFrame(trade_rows)
    ic_df = pd.DataFrame(ic_rows)

    # 方便对比：把手续费加回去，得到“零成本”的 NAV
    nav_df["nav_zero_cost_fixed_trades"] = nav_df["nav_total"] + nav_df["cum_fee"]

    nav_df["date_ts"] = pd.to_datetime(nav_df["date"])
    nav_df = nav_df.sort_values("date_ts").reset_index(drop=True)

    # 策略日收益
    nav_df["ret"] = nav_df["nav_total"].pct_change().fillna(0.0)

    # ---------------- Benchmark 对齐 ----------------
    bm_metrics = None
    if args.enable_benchmark:
        if not args.benchmark_csv:
            raise ValueError("--enable_benchmark requires --benchmark_csv")
        bm_raw = load_benchmark_csv(args.benchmark_csv)

        bm_raw["bm_ret"] = bm_raw["close"].pct_change()
        bm_raw["bm_nav"] = (1.0 + bm_raw["bm_ret"].fillna(0.0)).cumprod()

        # 对齐到 nav_df 的日期
        bm_aligned = bm_raw.set_index("date").reindex(nav_df["date_ts"]).copy()
        # benchmark 缺失：这里不强行 forward-fill，缺了就 NaN（更真实）
        nav_df["bm_close"] = bm_aligned["close"].values
        nav_df["bm_ret"] = bm_aligned["bm_ret"].values
        nav_df["bm_nav"] = bm_aligned["bm_nav"].values

        # 归一化：与策略起始日对齐
        base = nav_df["bm_nav"].dropna()
        if not base.empty:
            base0 = float(base.iloc[0])
            nav_df["bm_nav_norm"] = nav_df["bm_nav"] / base0
        else:
            nav_df["bm_nav_norm"] = np.nan

        # 策略 NAV 归一化（便于画图）
        nav_df["nav_norm"] = nav_df["nav_total"] / float(nav_df["nav_total"].iloc[0])

        # 基准指标（与策略同日期交集）
        strat_r = nav_df.set_index("date_ts")["ret"]
        bm_r = nav_df.set_index("date_ts")["bm_ret"]
        bm_metrics = compute_basic_metrics(strat_r, bm_r)

    # ---------------- IC / RankIC 汇总 ----------------
    ic_summary = {}
    if not ic_df.empty:
        ic_valid = ic_df.dropna(subset=["ic", "rank_ic"])
        ic_summary = {
            "ic_mean": float(ic_valid["ic"].mean()) if not ic_valid.empty else np.nan,
            "ic_std": float(ic_valid["ic"].std(ddof=1)) if len(ic_valid) > 1 else np.nan,
            "rank_ic_mean": float(ic_valid["rank_ic"].mean()) if not ic_valid.empty else np.nan,
            "rank_ic_std": float(ic_valid["rank_ic"].std(ddof=1)) if len(ic_valid) > 1 else np.nan,
            "ic_obs_days": int(len(ic_valid)),
        }

    # ---------------- 策略自身指标 ----------------
    strat_metrics = compute_basic_metrics(nav_df.set_index("date_ts")["ret"], None)

    def nav_at(date_str: str) -> Optional[float]:
        m = nav_df[nav_df["date"] == date_str]
        if m.empty:
            return None
        return float(m.iloc[0]["nav_total"])

    summary = {
        "window": {
            "start": fmt_date(start),
            "end": fmt_date(end),
            "data_end": fmt_date(data_end),
            "preheat": fmt_date(preheat),
            "tail_start": fmt_date(tail_start),
        },
        "strategy": {
            "name": "TopkDropout (qlib-style)",
            "topk": args.topk,
            "n_drop": args.n_drop,
            "shift": 1,
            "deal_price": "close",
        },
        "rules": {
            "settlement_t1": bool(args.settlement_t1),
            "ipo_min_days": args.ipo_min_days,
            "max_blocked_days": args.max_blocked_days,
            "haircut": args.haircut,
            "limit_eps": args.limit_eps,
            "limit_tiny": args.limit_tiny,
            "lot_size": args.lot_size,
            "limit_inference": "mainboard(10%+5% ST), 300/688(20%), BJ(30%)",
            "costs_bps": {
                "buy_cost_bps": args.buy_cost_bps,
                "sell_cost_bps": args.sell_cost_bps,
                "sell_stamp_bps": args.sell_stamp_bps,
                "min_cost": args.min_cost,
            },
        },
        "result": {
            "nav_start": float(nav_df.iloc[0]["nav_total"]),
            "nav_end_mtm_at_window_end": nav_at(fmt_date(end)),
            "nav_end_at_data_end": nav_at(fmt_date(data_end)),
            "nav_end_terminal": float(nav_df.iloc[-1]["nav_total"]),
            "total_fee_paid": float(nav_df["cum_fee"].iloc[-1]),
            "forced_liquidations_total": int(forced_liq_total),
            "terminal_liquidations_total": int(terminal_liq_total),
            "trades_total": int(len(trades_df)),
            "buys": int((trades_df["action"] == "BUY").sum()) if not trades_df.empty else 0,
            "sells": int((trades_df["action"] == "SELL").sum()) if not trades_df.empty else 0,
        },
        "buy_skip_reason": buy_skip_totals,
        "sell_block_reason": sell_block_totals,
        "metrics_strategy": strat_metrics,
        "metrics_vs_benchmark": bm_metrics if bm_metrics is not None else None,
        "ic_rankic": ic_summary if ic_summary else None,
    }

    # ---------------- 落盘 ----------------
    ensure_dir(out_dir)

    # main csv
    nav_df.drop(columns=["date_ts"]).to_csv(out_dir / "daily_nav.csv", index=False, encoding="utf-8")
    diag_df.to_csv(out_dir / "diagnostics_daily.csv", index=False, encoding="utf-8")
    trades_df.to_csv(out_dir / "trades.csv", index=False, encoding="utf-8")

    # ic daily
    if not ic_df.empty:
        ic_df.to_csv(out_dir / "ic_daily.csv", index=False, encoding="utf-8")

    # summary json
    with open(out_dir / "diagnostics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # net vs zero cost
    nav_df[["date", "nav_total", "nav_zero_cost_fixed_trades", "fee_today", "cum_fee"]].to_csv(
        out_dir / "nav_net_vs_zero_cost.csv", index=False, encoding="utf-8"
    )

    # plot: strategy vs benchmark
    if args.enable_benchmark:
        try:
            import matplotlib.pyplot as plt

            plot_df = nav_df.dropna(subset=["nav_norm"]).copy()
            plt.figure()
            plt.plot(plot_df["date"], plot_df["nav_norm"], label="Strategy (NAV normalized)")
            if "bm_nav_norm" in plot_df.columns:
                plt.plot(plot_df["date"], plot_df["bm_nav_norm"], label="Benchmark 000985 (normalized)")
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "nav_compare.png", dpi=150)
            plt.close()
        except Exception as e:
            print(f"[WARN] Failed to plot nav_compare.png: {e}")

    print("[OK] Backtest finished.")
    print(f"Output dir: {out_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    # IO
    p.add_argument("--model_dir", required=True)
    p.add_argument("--features_dir", required=True)
    p.add_argument("--ohlcv_dir", required=True)
    p.add_argument("--calendar_csv", required=True)
    p.add_argument("--out_dir", required=True)

    # window
    p.add_argument("--start_date", default="2025-01-01")
    p.add_argument("--end_date", default="2025-12-10")
    p.add_argument("--data_end_date", default="2025-12-15")
    p.add_argument("--tail_start_date", default="2025-12-11")

    # strategy params
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--n_drop", type=int, default=5)
    p.add_argument("--init_cash", type=float, default=1_000_000.0)
    p.add_argument("--settlement_t1", action="store_true",
                   help="If set, sell proceeds become available next trading day (T+1).")

    # constraints
    p.add_argument("--ipo_min_days", type=int, default=60)
    p.add_argument("--max_blocked_days", type=int, default=60)
    p.add_argument("--haircut", type=float, default=0.05)          # 强平/终止强平额外冲击成本（保守）

    # limit lock detector
    p.add_argument("--limit_eps", type=float, default=0.0015)       # 接近涨跌停价的相对误差阈值
    p.add_argument("--limit_tiny", type=float, default=0.001)       # 日内振幅阈值（越小越保守）

    # trade lot
    p.add_argument("--lot_size", type=int, default=100)

    # costs
    p.add_argument("--buy_cost_bps", type=float, default=15.0)      # 双边佣金示例：15bp
    p.add_argument("--sell_cost_bps", type=float, default=15.0)
    p.add_argument("--sell_stamp_bps", type=float, default=10.0)    # 印花税示例：10bp（按你的设定）
    p.add_argument("--min_cost", type=float, default=0.0)

    # benchmark & ic
    p.add_argument("--enable_benchmark", action="store_true")
    p.add_argument("--benchmark_csv", default="", help="Local benchmark csv (000985).")
    p.add_argument("--compute_ic", action="store_true", help="Compute daily IC & RankIC and output ic_daily.csv")

    return p


def main():
    args = build_arg_parser().parse_args()
    run_backtest(args)


if __name__ == "__main__":
    main()
