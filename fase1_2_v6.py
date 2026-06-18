#!/usr/bin/env python3
"""
fase1_2_v6.py — EUR/USD three-layer market-structure classifier.

Architecture (v6 spec):
    Layer 1: Direction      — Bull / Bear / Neutral
    Layer 2: Environment    — Trend / Range / Compression / Expansion
    Layer 3: Local          — Impulse / Pullback / ReversalCandidate / None

Each layer carries an independent confidence score (0-100).
Layers are NEVER combined into a single master label.

Key fixes vs v5:
- Three independent pipelines; no shared confidence bleeding across layers.
- compression_score rebuilt: uses ATR percentile rank vs rolling window.
  (v5 compression_score was stuck at 100 for 90%+ of bars due to range_ratio saturation.)
- Direction uses swing structure + slope; stabilized with per-layer hysteresis.
- Environment uses ATR rank (volatility regime) + efficiency.
- Local uses recent price behaviour relative to Direction.
- safe_div() always returns pd.Series to avoid numpy .fillna() crashes.

Usage:
    pip install pandas numpy matplotlib
    python fase1_2_v6.py fetched_data_eurusd_401697501.csv

Outputs:
    v6_features.csv
    v6_classified.csv
    v6_segments.csv
    v6_summary.csv
    visual_v6/*.png
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Patch
except Exception:
    plt = None

# ---------------------------------------------------------------------------
# Session constants
# ---------------------------------------------------------------------------
LONDON_START_UTC = 7
LONDON_END_UTC   = 16
NY_START_UTC     = 12
NY_END_UTC       = 21

DIR_COLORS = {"Bull": "#2ca02c", "Bear": "#d62728", "Neutral": "#aec7e8"}
ENV_COLORS = {
    "Trend":       "#1f77b4",
    "Range":       "#9aa0a6",
    "Compression": "#bdbdbd",
    "Expansion":   "#ff7f0e",
}
LOC_COLORS = {
    "Impulse":           "#1f77b4",
    "Pullback":          "#98df8a",
    "ReversalCandidate": "#9467bd",
    "None":              "#eeeeee",
}


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
@dataclass
class Params:
    # swing detection
    pivot_left:    int   = 3
    pivot_right:   int   = 3
    min_swing_atr: float = 0.5

    # feature windows (candles)
    atr_window:        int = 14
    slope_fast:        int = 12
    slope_slow:        int = 36
    efficiency_window: int = 24
    atr_rank_window:   int = 100   # rolling window for ATR percentile rank

    # Layer 1 – Direction
    dir_bull_threshold:  float = 20.0
    dir_bear_threshold:  float = -20.0
    dir_hysteresis:      float = 10.0
    dir_min_segment:     int   = 20   # raised from 8: direction must hold for 20 bars
    dir_lookback_votes:  int   = 6

    # Layer 2 – Environment
    expansion_atr_rank:   float = 75.0
    compression_atr_rank: float = 35.0
    trend_efficiency_min: float = 25.0
    trend_score_min:      float = 25.0
    env_min_segment:      int   = 5

    # Layer 3 – Local
    impulse_efficiency:   float = 28.0  # lowered from 40: easier to activate
    impulse_atr_rank:     float = 50.0  # lowered from 60
    reversal_swing_count: int   = 2
    local_min_segment:    int   = 3

    # BOS cooldown
    bos_cooldown:         int   = 5     # min bars between BOS signals


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def normalize_col(c: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(c).strip().lower())


def find_input_csv(explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")
        return p
    candidates = [
        p for p in Path.cwd().glob("*.csv")
        if not any(x in p.name.lower() for x in ["v6_", "phase", "regime", "segment", "summary"])
    ]
    candidates.sort(key=lambda x: (0 if "fetched" in x.name.lower() or "eurusd" in x.name.lower() else 1, x.name))
    if not candidates:
        raise FileNotFoundError("No candidate OHLC CSV found.")
    return candidates[0]


def parse_datetime_series(s: pd.Series) -> pd.Series:
    if s.astype(str).str.contains(r"/Date\(", regex=True, na=False).any():
        nums = s.astype(str).str.extract(r"/Date\((\d+)")[0].astype("int64")
        return pd.to_datetime(nums, unit="ms", utc=True)
    return pd.to_datetime(s, utc=True, errors="coerce")


def load_ohlc(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    col_map = {normalize_col(c): c for c in df.columns}

    def pick(cands):
        for x in cands:
            if normalize_col(x) in col_map:
                return col_map[normalize_col(x)]
        return None

    dtc = pick(["bardate","datetime","date","time","timestamp","barstart","open time","opentime"])
    oc  = pick(["open","o"])
    hc  = pick(["high","h"])
    lc  = pick(["low","l"])
    cc  = pick(["close","c","last"])
    vc  = pick(["volume","vol","tickvolume"])

    missing = [n for n, c in [("datetime",dtc),("open",oc),("high",hc),("low",lc),("close",cc)] if c is None]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")

    out = pd.DataFrame({
        "datetime": parse_datetime_series(df[dtc]),
        "open":     pd.to_numeric(df[oc], errors="coerce"),
        "high":     pd.to_numeric(df[hc], errors="coerce"),
        "low":      pd.to_numeric(df[lc], errors="coerce"),
        "close":    pd.to_numeric(df[cc], errors="coerce"),
        "volume":   pd.to_numeric(df[vc], errors="coerce") if vc else np.nan,
    })
    out = (out.dropna(subset=["datetime","open","high","low","close"])
              .sort_values("datetime").drop_duplicates("datetime")
              .reset_index(drop=True))
    return out


def filter_london_ny(df: pd.DataFrame) -> pd.DataFrame:
    hour = df["datetime"].dt.hour
    mask = ((hour >= LONDON_START_UTC) & (hour < LONDON_END_UTC)) | \
           ((hour >= NY_START_UTC)     & (hour < NY_END_UTC))
    out = df[mask].copy().reset_index(drop=True)
    h   = out["datetime"].dt.hour
    lon = (h >= LONDON_START_UTC) & (h < LONDON_END_UTC)
    ny  = (h >= NY_START_UTC)     & (h < NY_END_UTC)
    out["session"]  = np.select([lon & ny, lon, ny], ["London+NY","London","NewYork"], default="Other")
    out["date_utc"] = out["datetime"].dt.date.astype(str)
    return out


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def safe_div(a, b, default=0.0) -> pd.Series:
    """Always returns pd.Series — never a raw numpy array."""
    idx = getattr(a, "index", getattr(b, "index", None))
    with np.errstate(divide="ignore", invalid="ignore"):
        arr = np.where(np.abs(np.asarray(b, dtype=float)) > 1e-12,
                       np.asarray(a, dtype=float) / np.asarray(b, dtype=float),
                       default)
    return pd.Series(arr, index=idx)


def rolling_slope(y: pd.Series, window: int) -> pd.Series:
    x  = np.arange(window, dtype=float)
    xm = x.mean()
    dn = ((x - xm) ** 2).sum()
    def _calc(v):
        if np.any(np.isnan(v)):
            return np.nan
        return ((x - xm) * (v - v.mean())).sum() / dn
    return y.rolling(window, min_periods=window).apply(_calc, raw=True)


def atr_percentile_rank(atr: pd.Series, window: int) -> pd.Series:
    """Rolling percentile rank of current ATR value within last `window` bars (0–100)."""
    def _rank(v):
        cur = v[-1]
        if np.isnan(cur):
            return np.nan
        return float(np.sum(v[:-1] <= cur)) / max(len(v) - 1, 1) * 100
    return atr.rolling(window, min_periods=10).apply(_rank, raw=True)


def compute_features(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    out = df.copy()

    # ATR
    pc  = out["close"].shift(1)
    tr  = pd.concat([out["high"] - out["low"],
                     (out["high"] - pc).abs(),
                     (out["low"]  - pc).abs()], axis=1).max(axis=1)
    out["tr"]  = tr
    out["atr"] = tr.rolling(p.atr_window, min_periods=3).mean()

    # Slopes (price units / candle)
    out["slope_fast"] = rolling_slope(out["close"], p.slope_fast)
    out["slope_slow"] = rolling_slope(out["close"], p.slope_slow)
    # Normalise by ATR so values are comparable across volatility regimes
    out["slope_fast_norm"] = safe_div(out["slope_fast"], out["atr"]) * 1000
    out["slope_slow_norm"] = safe_div(out["slope_slow"], out["atr"]) * 1000

    # Continuous trend score (-100..100)
    raw = (0.55 * out["slope_slow_norm"].fillna(0) +
           0.45 * out["slope_fast_norm"].fillna(0))
    out["trend_score"] = raw.clip(-100, 100)

    # Directional efficiency (0-100): how straight the price path is
    move = out["close"].diff(p.efficiency_window).abs()
    path = out["close"].diff().abs().rolling(p.efficiency_window, min_periods=5).sum()
    out["efficiency"] = safe_div(move, path).clip(0, 1) * 100

    # ATR percentile rank — the correct volatility regime signal.
    # v5 used range_ratio which saturated at 100 for 90%+ of bars.
    out["atr_rank"] = atr_percentile_rank(out["atr"], p.atr_rank_window)

    return out


# ---------------------------------------------------------------------------
# Swing detection
# ---------------------------------------------------------------------------
def detect_swings(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    out = df.copy()
    n   = len(out)
    out["swing_high"]      = False
    out["swing_low"]       = False
    out["structure_label"] = ""

    highs = out["high"].values
    lows  = out["low"].values
    atr_v = out["atr"].fillna(out["tr"].rolling(10, min_periods=1).mean()).values
    med_a = float(np.nanmedian(atr_v[atr_v > 0])) if np.any(atr_v > 0) else 1e-5

    L, R = p.pivot_left, p.pivot_right
    raw: List[Tuple[int, str, float]] = []
    for i in range(L, n - R):
        hs = highs[i-L:i+R+1]; ls = lows[i-L:i+R+1]
        if highs[i] == np.nanmax(hs) and np.sum(hs == highs[i]) == 1:
            raw.append((i, "H", highs[i]))
        if lows[i]  == np.nanmin(ls) and np.sum(ls == lows[i])  == 1:
            raw.append((i, "L", lows[i]))
    raw.sort(key=lambda x: x[0])

    # Alternate H/L, keep more extreme of same-type neighbours, enforce ATR distance
    swings: List[Tuple[int, str, float]] = []
    for i, typ, price in raw:
        if not swings:
            swings.append((i, typ, price)); continue
        li, lt, lp = swings[-1]
        if typ == lt:
            if (typ == "H" and price > lp) or (typ == "L" and price < lp):
                swings[-1] = (i, typ, price)
            continue
        a = atr_v[i] if np.isfinite(atr_v[i]) and atr_v[i] > 0 else med_a
        if abs(price - lp) >= p.min_swing_atr * a:
            swings.append((i, typ, price))

    last_h = last_l = None
    for i, typ, price in swings:
        if typ == "H":
            out.loc[i, "swing_high"] = True
            lab = "H?" if last_h is None else ("HH" if price > last_h else "LH")
            last_h = price
        else:
            out.loc[i, "swing_low"] = True
            lab = "L?" if last_l is None else ("HL" if price > last_l else "LL")
            last_l = price
        out.loc[i, "structure_label"] = lab

    out["last_structure"] = out["structure_label"].replace("", np.nan).ffill().fillna("")

    # Swing structure score: rolling balance of bullish vs bearish swings (-100..100)
    sw_score = np.zeros(n)
    idx_labs = [(i, out.loc[i, "structure_label"]) for i in range(n) if out.loc[i, "structure_label"]]
    for pos, (idx, _) in enumerate(idx_labs):
        recent = [x[1] for x in idx_labs[max(0, pos-5):pos+1]]
        s = sum(1 if r in ("HH","HL") else -1 if r in ("LH","LL") else 0 for r in recent)
        sw_score[idx] = s / max(len(recent), 1) * 100
    out["swing_score"] = (pd.Series(sw_score, index=out.index)
                            .replace(0, np.nan).ffill().fillna(0).clip(-100, 100))

    # Combined directional score used by Layer 1
    out["dir_score"] = (0.55 * out["trend_score"].fillna(0) +
                        0.45 * out["swing_score"].fillna(0)).clip(-100, 100)

    # BOS flags with cooldown to prevent dense clusters of signals
    out["bos_up"] = False; out["bos_down"] = False
    ch: List[float] = []; cl: List[float] = []
    last_bos_up = -999; last_bos_down = -999
    cooldown = p.bos_cooldown
    for i in range(n):
        if out.loc[i, "swing_high"]: ch.append(float(out.loc[i, "high"]))
        if out.loc[i, "swing_low"]:  cl.append(float(out.loc[i, "low"]))
        if len(ch) >= 2 and out.loc[i, "close"] > ch[-2] and (i - last_bos_up) >= cooldown:
            out.loc[i, "bos_up"] = True;   last_bos_up   = i
        if len(cl) >= 2 and out.loc[i, "close"] < cl[-2] and (i - last_bos_down) >= cooldown:
            out.loc[i, "bos_down"] = True; last_bos_down = i

    return out


# ---------------------------------------------------------------------------
# Stabilisation helper
# ---------------------------------------------------------------------------
def merge_short_segments(labels: List[str], min_len: int, neutral: str) -> List[str]:
    """Iteratively merge segments shorter than min_len into adjacent neighbours."""
    if not labels:
        return labels
    out = labels[:]

    def build_segs(arr):
        segs = []
        s = 0
        for i in range(1, len(arr) + 1):
            if i == len(arr) or arr[i] != arr[s]:
                segs.append((s, i - 1, arr[s]))
                s = i
        return segs

    changed = True
    while changed:
        changed = False
        segs = build_segs(out)
        for idx, (ss, se, sl) in enumerate(segs):
            if (se - ss + 1) < min_len:
                prev = segs[idx-1][2] if idx > 0        else None
                nxt  = segs[idx+1][2] if idx+1 < len(segs) else None
                if prev and nxt and prev == nxt:
                    rep = prev
                elif prev:
                    rep = prev
                elif nxt:
                    rep = nxt
                else:
                    rep = neutral
                for j in range(ss, se + 1):
                    out[j] = rep
                changed = True
                break   # rebuild segs after each change

    return out


# ---------------------------------------------------------------------------
# Layer 1: Direction  (Bull / Bear / Neutral)
# ---------------------------------------------------------------------------
def classify_direction(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    out  = df.copy()
    ds   = out["dir_score"].values
    n    = len(ds)

    # Rolling majority vote
    raw = []
    for i in range(n):
        w    = ds[max(0, i - p.dir_lookback_votes + 1):i + 1]
        mean = float(np.nanmean(w)) if len(w) else 0.0
        if   mean >= p.dir_bull_threshold: raw.append("Bull")
        elif mean <= p.dir_bear_threshold: raw.append("Bear")
        else:                              raw.append("Neutral")

    # Hysteresis: Bull→Bear or Bear→Bull must pass through Neutral
    stable = raw[:]
    for i in range(1, n):
        prev, cur = stable[i-1], stable[i]
        if prev == "Bull" and cur == "Bear":
            if ds[i] > p.dir_bear_threshold - p.dir_hysteresis:
                stable[i] = "Neutral"
        elif prev == "Bear" and cur == "Bull":
            if ds[i] < p.dir_bull_threshold + p.dir_hysteresis:
                stable[i] = "Neutral"

    stable = merge_short_segments(stable, min_len=p.dir_min_segment, neutral="Neutral")
    out["direction"] = stable

    # Confidence: how far dir_score is from the neutral band
    conf = pd.Series(0.0, index=out.index)
    bull = out["direction"] == "Bull"
    bear = out["direction"] == "Bear"
    neut = out["direction"] == "Neutral"
    span_b = 100.0 - p.dir_bull_threshold
    span_n = 100.0 - abs(p.dir_bear_threshold)
    conf[bull] = ((out.loc[bull, "dir_score"] - p.dir_bull_threshold).clip(0) / span_b * 100)
    conf[bear] = ((-out.loc[bear, "dir_score"] - abs(p.dir_bear_threshold)).clip(0) / span_n * 100)
    conf[neut] = ((p.dir_bull_threshold - out.loc[neut, "dir_score"].abs()).clip(0) / p.dir_bull_threshold * 100)
    out["direction_conf"] = conf.clip(0, 100).round(1)
    return out


# ---------------------------------------------------------------------------
# Layer 2: Environment  (Trend / Range / Compression / Expansion)
# ---------------------------------------------------------------------------
def classify_environment(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    out      = df.copy()
    atr_rank = out["atr_rank"].fillna(50).values
    eff      = out["efficiency"].fillna(0).values
    ts_abs   = out["trend_score"].fillna(0).abs().values
    n        = len(out)

    raw_env = []
    for i in range(n):
        ar = atr_rank[i]; ef = eff[i]; ta = ts_abs[i]
        if   ar >= p.expansion_atr_rank:                              raw_env.append("Expansion")
        elif ar <= p.compression_atr_rank:                            raw_env.append("Compression")
        elif ef >= p.trend_efficiency_min and ta >= p.trend_score_min: raw_env.append("Trend")
        else:                                                          raw_env.append("Range")

    env = merge_short_segments(raw_env, min_len=p.env_min_segment, neutral="Range")
    out["environment"] = env

    # Confidence
    conf = np.zeros(n)
    for i in range(n):
        e = env[i]; ar = atr_rank[i]; ef = eff[i]; ta = ts_abs[i]
        if e == "Expansion":
            conf[i] = (ar - p.expansion_atr_rank) / max(100 - p.expansion_atr_rank, 1) * 100
        elif e == "Compression":
            conf[i] = (p.compression_atr_rank - ar) / max(p.compression_atr_rank, 1) * 100
        elif e == "Trend":
            conf[i] = min(ef / 100, 1.0) * 50 + min(ta / 100, 1.0) * 50
        else:  # Range
            conf[i] = (max(0, p.trend_efficiency_min - ef) / p.trend_efficiency_min * 50 +
                       max(0, p.trend_score_min      - ta) / p.trend_score_min      * 50)
    out["environment_conf"] = pd.Series(conf, index=out.index).clip(0, 100).round(1)
    return out


# ---------------------------------------------------------------------------
# Layer 3: Local  (Impulse / Pullback / ReversalCandidate / None)
# ---------------------------------------------------------------------------
def classify_local(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    out       = df.copy()
    direction = out["direction"].values
    ts        = out["trend_score"].fillna(0).values
    atr_rank  = out["atr_rank"].fillna(50).values
    eff       = out["efficiency"].fillna(0).values
    last_str  = out["last_structure"].values
    n         = len(out)

    # Pre-build list of (index, label) for swing lookups
    sw_labs = [(i, out.loc[i, "structure_label"]) for i in range(n) if out.loc[i, "structure_label"]]

    def counter_count(i: int, d: str, lookback: int = 8) -> int:
        recent = [lab for idx, lab in sw_labs if i - lookback <= idx <= i]
        if d == "Bull": return sum(1 for r in recent if r in ("LH","LL"))
        if d == "Bear": return sum(1 for r in recent if r in ("HH","HL"))
        return 0

    raw_local = []
    for i in range(n):
        d  = direction[i]; t = ts[i]; ar = atr_rank[i]; ef = eff[i]; ls = last_str[i]

        # Impulse: strong aligned move with elevated volatility
        if d == "Bull" and t > 30 and ef >= p.impulse_efficiency and ar >= p.impulse_atr_rank:
            raw_local.append("Impulse")
        elif d == "Bear" and t < -30 and ef >= p.impulse_efficiency and ar >= p.impulse_atr_rank:
            raw_local.append("Impulse")
        # ReversalCandidate: counter-structure swings accumulating
        elif counter_count(i, d) >= p.reversal_swing_count:
            raw_local.append("ReversalCandidate")
        # Pullback: price moving against direction, direction still intact
        elif d == "Bull" and ls in ("LH","LL") and t > -20:
            raw_local.append("Pullback")
        elif d == "Bear" and ls in ("HH","HL") and t < 20:
            raw_local.append("Pullback")
        else:
            raw_local.append("None")

    local = merge_short_segments(raw_local, min_len=p.local_min_segment, neutral="None")
    out["local"] = local

    # Confidence
    conf = np.zeros(n)
    for i in range(n):
        loc = local[i]; d = direction[i]; t = ts[i]; ar = atr_rank[i]; ef = eff[i]
        if loc == "Impulse":
            conf[i] = min(ef / 100, 1.0) * 50 + min(ar / 100, 1.0) * 50
        elif loc == "Pullback":
            if d == "Bull":   conf[i] = min(abs(min(t, 0)) / 50, 1.0) * 100
            elif d == "Bear": conf[i] = min(abs(max(t, 0)) / 50, 1.0) * 100
            else:             conf[i] = 30.0
        elif loc == "ReversalCandidate":
            conf[i] = min(counter_count(i, d) / 4, 1.0) * 100
        else:
            conf[i] = 50.0
    out["local_conf"] = pd.Series(conf, index=out.index).clip(0, 100).round(1)
    return out


# ---------------------------------------------------------------------------
# Segment builder
# ---------------------------------------------------------------------------
def make_segments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()
    rows = []
    start = 0
    for i in range(1, len(df) + 1):
        eod = i == len(df)
        new = not eod and (
            df.loc[i, "direction"]   != df.loc[start, "direction"]   or
            df.loc[i, "environment"] != df.loc[start, "environment"] or
            df.loc[i, "local"]       != df.loc[start, "local"]       or
            df.loc[i, "date_utc"]    != df.loc[start, "date_utc"]
        )
        if eod or new:
            part = df.iloc[start:i]
            rows.append({
                "start":          part["datetime"].iloc[0],
                "end":            part["datetime"].iloc[-1],
                "date_utc":       part["date_utc"].iloc[0],
                "direction":      part["direction"].iloc[0],
                "environment":    part["environment"].iloc[0],
                "local":          part["local"].iloc[0],
                "bars":           len(part),
                "start_close":    part["close"].iloc[0],
                "end_close":      part["close"].iloc[-1],
                "return_pips":    (part["close"].iloc[-1] - part["close"].iloc[0]) * 10000,
                "avg_dir_score":  part["dir_score"].mean(),
                "avg_dir_conf":   part["direction_conf"].mean(),
                "avg_env_conf":   part["environment_conf"].mean(),
                "avg_local_conf": part["local_conf"].mean(),
            })
            start = i
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def choose_days(df: pd.DataFrame, n: int = 10) -> List[str]:
    days = sorted(df["date_utc"].unique())
    if len(days) <= n:
        return days
    return [days[i] for i in np.linspace(0, len(days)-1, n).round().astype(int)]


def add_band(ax, d: pd.DataFrame, field: str, color_map: Dict[str, str],
             alpha: float, label: bool = False):
    if d.empty:
        return
    vals  = d[field].tolist()
    times = d["datetime"].tolist()
    s = 0
    for i in range(1, len(d) + 1):
        if i == len(d) or vals[i] != vals[s]:
            x0 = times[s]
            x1 = times[i] if i < len(d) else times[-1]
            c  = color_map.get(vals[s], "#cccccc")
            ax.axvspan(x0, x1, color=c, alpha=alpha, linewidth=0)
            if label:
                mid = x0 + (x1 - x0) / 2
                ax.text(mid, 0.5, vals[s][:18], ha="center", va="center", fontsize=7)
            s = i


def plot_day(df: pd.DataFrame, day: str, outdir: Path, p: Params):
    if plt is None:
        return
    d = df[df["date_utc"] == day].copy().reset_index(drop=True)
    if d.empty:
        return

    fig = plt.figure(figsize=(20, 12))
    gs  = fig.add_gridspec(6, 1, height_ratios=[0.4, 0.4, 0.4, 3.0, 0.8, 0.8], hspace=0.06)
    ax_dir = fig.add_subplot(gs[0])
    ax_env = fig.add_subplot(gs[1], sharex=ax_dir)
    ax_loc = fig.add_subplot(gs[2], sharex=ax_dir)
    ax_px  = fig.add_subplot(gs[3], sharex=ax_dir)
    ax_ds  = fig.add_subplot(gs[4], sharex=ax_dir)
    ax_ar  = fig.add_subplot(gs[5], sharex=ax_dir)

    x = d["datetime"]
    add_band(ax_dir, d, "direction",   DIR_COLORS, alpha=0.85, label=True)
    add_band(ax_env, d, "environment", ENV_COLORS, alpha=0.85, label=True)
    add_band(ax_loc, d, "local",       LOC_COLORS, alpha=0.85, label=True)
    for ax, lbl in [(ax_dir,"Dir"),(ax_env,"Env"),(ax_loc,"Local")]:
        ax.set_yticks([]); ax.set_ylim(0,1)
        ax.spines[["left","right","top","bottom"]].set_visible(False)
        ax.set_ylabel(lbl, rotation=0, labelpad=32, fontsize=8)

    add_band(ax_px, d, "direction", DIR_COLORS, alpha=0.08)
    ax_px.plot(x, d["close"], linewidth=1.8, color="#333333")
    ax_px.vlines(x, d["low"], d["high"], alpha=0.3, linewidth=0.8, color="#555555")
    for _, r in d[d["bos_up"]].iterrows():
        ax_px.axvline(r["datetime"], color="#2ca02c", alpha=0.25, linewidth=1)
    for _, r in d[d["bos_down"]].iterrows():
        ax_px.axvline(r["datetime"], color="#d62728", alpha=0.25, linewidth=1)
    for _, r in d[d["structure_label"] != ""].iterrows():
        lab = r["structure_label"]
        if lab in ("HH","LH","H?"):
            y   = r["high"]
            col = "green" if lab == "HH" else "red" if lab == "LH" else "#1f77b4"
            ax_px.scatter(r["datetime"], y, marker="^", s=70, color=col, edgecolor="k", zorder=5)
            ax_px.text(r["datetime"], y, lab, color=col, ha="center", va="bottom", fontsize=8, fontweight="bold")
        else:
            y   = r["low"]
            col = "green" if lab == "HL" else "red" if lab == "LL" else "#1f77b4"
            ax_px.scatter(r["datetime"], y, marker="v", s=70, color=col, edgecolor="k", zorder=5)
            ax_px.text(r["datetime"], y, lab, color=col, ha="center", va="top", fontsize=8, fontweight="bold")
    ax_px.set_title(f"EUR/USD market structure v6 — {day} UTC  (London + NY)", fontsize=11)
    ax_px.set_ylabel("Price"); ax_px.grid(True, alpha=0.2)

    ax_ds.plot(x, d["dir_score"], linewidth=1.5, color="#1f77b4")
    ax_ds.axhline(0,                      color="k",        linewidth=0.7, alpha=0.4)
    ax_ds.axhline( p.dir_bull_threshold,  color="#2ca02c",  linewidth=0.6, linestyle="--", alpha=0.5)
    ax_ds.axhline( p.dir_bear_threshold,  color="#d62728",  linewidth=0.6, linestyle="--", alpha=0.5)
    ax_ds.set_ylim(-105, 105); ax_ds.set_ylabel("Dir\nscore", fontsize=8); ax_ds.grid(True, alpha=0.2)

    ax_ar.plot(x, d["atr_rank"], linewidth=1.5, color="#ff7f0e")
    ax_ar.axhline(p.expansion_atr_rank,   color="#ff7f0e", linewidth=0.7, linestyle="--", alpha=0.6)
    ax_ar.axhline(p.compression_atr_rank, color="#bdbdbd", linewidth=0.7, linestyle="--", alpha=0.6)
    ax_ar.set_ylim(-5, 105); ax_ar.set_ylabel("ATR\nrank", fontsize=8); ax_ar.grid(True, alpha=0.2)
    ax_ar.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax_ar.get_xticklabels(), rotation=30, ha="right")
    for ax in [ax_dir, ax_env, ax_loc, ax_px, ax_ds]:
        plt.setp(ax.get_xticklabels(), visible=False)
    ax_ar.set_xlabel("UTC time")

    handles = (
        [Patch(facecolor=c, label=k, alpha=0.7) for k,c in DIR_COLORS.items() if k in set(d["direction"])] +
        [Patch(facecolor=c, label=k, alpha=0.7) for k,c in ENV_COLORS.items() if k in set(d["environment"])]
    )
    ax_px.legend(handles=handles, loc="upper right", fontsize=8, ncol=2)

    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"v6_{day}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input",    nargs="?", help="OHLC CSV file")
    parser.add_argument("--days",   type=int,  default=10)
    parser.add_argument("--outdir",            default="visual_v6")
    parser.add_argument("--pivot",  type=int,  default=3)
    args = parser.parse_args()

    p    = Params(pivot_left=args.pivot, pivot_right=args.pivot)
    path = find_input_csv(args.input)
    print(f"Input: {path}")

    raw  = load_ohlc(path)
    print(f"Rows loaded: {len(raw):,}")
    print(f"Date range:  {raw['datetime'].min()} -> {raw['datetime'].max()}")

    sess = filter_london_ny(raw)
    print(f"After session filter: {len(sess):,}")

    feat = compute_features(sess, p)
    sw   = detect_swings(feat, p)
    d1   = classify_direction(sw, p)
    d2   = classify_environment(d1, p)
    d3   = classify_local(d2, p)
    d3   = d3.reset_index(drop=True)
    seg  = make_segments(d3)

    base_cols = [
        "datetime","date_utc","session","open","high","low","close","volume",
        "tr","atr","atr_rank","slope_fast","slope_slow","trend_score","efficiency",
        "swing_score","dir_score","swing_high","swing_low","structure_label",
        "last_structure","bos_up","bos_down",
    ]
    layer_cols = [
        "direction","direction_conf",
        "environment","environment_conf",
        "local","local_conf",
    ]

    d3[base_cols].to_csv("v6_features.csv", index=False)
    d3[base_cols + layer_cols].to_csv("v6_classified.csv", index=False)
    seg.to_csv("v6_segments.csv", index=False)

    summary = (d3.groupby(["direction","environment","local"])
                 .agg(bars=("close","size"),
                      avg_dir_score=("dir_score","mean"),
                      avg_dir_conf=("direction_conf","mean"),
                      avg_env_conf=("environment_conf","mean"),
                      avg_local_conf=("local_conf","mean"))
                 .reset_index()
                 .sort_values("bars", ascending=False))
    summary["pct"] = (summary["bars"] / summary["bars"].sum() * 100).round(2)
    summary.to_csv("v6_summary.csv", index=False)

    print("\n=== LAYER DISTRIBUTIONS ===")
    print("Direction:\n",   d3["direction"].value_counts().to_string())
    print("\nEnvironment:\n", d3["environment"].value_counts().to_string())
    print("\nLocal:\n",       d3["local"].value_counts().to_string())
    print("\n=== TOP COMBINATIONS (direction × environment × local) ===")
    print(summary[["direction","environment","local","bars","pct",
                   "avg_dir_conf","avg_env_conf","avg_local_conf"]].head(15).to_string(index=False))

    sw_dir = (d3["direction"]   != d3["direction"].shift()).sum()
    sw_env = (d3["environment"] != d3["environment"].shift()).sum()
    sw_loc = (d3["local"]       != d3["local"].shift()).sum()
    total  = len(d3)
    print(f"\nSwitch rates — Direction: {sw_dir/total*100:.2f}%  "
          f"Environment: {sw_env/total*100:.2f}%  Local: {sw_loc/total*100:.2f}%")

    if plt is not None:
        outdir = Path(args.outdir)
        for day in choose_days(d3, args.days):
            plot_day(d3, day, outdir, p)
        print(f"\nCharts -> {outdir}/")
    else:
        print("matplotlib not available; charts skipped.")

    print("\nDone.")


if __name__ == "__main__":
    main()
