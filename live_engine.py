#!/usr/bin/env python3
"""
live_engine.py — Live trading engine for EUR/USD market structure signals.

Uses the v6 three-layer classifier (Direction/Environment/Local) to detect
entry signals and manages exits via trailing stop + label-change monitoring.

Architecture:
  Every 10 minutes (on bar close):
    1. Fetch last 250 bars of EUR/USD from CityIndex API
    2. Run full v6 classification pipeline
    3. Detect entry signals (aligned rules from backtest)
    4. Check open trades for exit conditions
    5. Execute orders / close positions
    6. Update dashboard + log

Entry rules (validated in fase2/fase3):
  - Direction = Bull or Bear (not Neutral)
  - Environment = Trend or Expansion (Range has no edge)
  - Daily bias must match direction
  - No Bear in NY-only session (17-21 UTC)
  - One trade per combo_key (dir_env_local) simultaneously

Exit rules:
  - Trailing stop (activated after trigger_pips, trails by trail_dist)
  - Hard SL at 1.5 x ATR
  - Label-change exit: Direction flips OR Environment goes to Range
  - Max hold time (tier-specific)

Usage:
  python live_engine.py [--paper]   # --paper = log only, no real orders
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# ── All files are local to this folder — self-contained ──────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from session_manager import session_manager
from config import username, tradingAccountID
from lightstreamer_receiver import LightstreamerReceiver
from trade_execution import TradeExecutor
from scalp_position_api import PositionCloser
from open_orders_feed_module import PositionManagementModule
from risk_controls import (
    advance_bar as advance_risk_bar,
    ensure_risk_state,
    entry_allowed as risk_entry_allowed,
    record_close as record_risk_close,
    status as risk_status,
)

# ── Import v6 pipeline (same folder as live_engine.py) ──────────────────────
from fase1_2_v6 import (
    Params, compute_features, detect_swings,
    classify_direction, classify_environment, classify_local,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MARKET_ID   = "403897186"       # EUR/USD at CityIndex (scalp account)
MARKET_NAME = "EUR/USD"
CURRENCY    = "USD"
BARS_NEEDED = 250               # enough for all rolling windows
BAR_MINUTES = 10                # 10-min bars

LONDON_START = 7
LONDON_END   = 16
NY_START     = 12
NY_END       = 21

_HERE        = Path(__file__).parent          # always v6_engine folder, regardless of cwd
LOG_FILE     = _HERE / "live_engine_log.csv"
STATS_FILE   = _HERE / "live_stats.json"
STATE_FILE   = _HERE / "live_trades_state.json"
_LOG_PATH    = _HERE / "live_engine.log"

# Force our handlers onto the root logger — bypasses basicConfig already called by config.py
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
_root = logging.getLogger()
_root.setLevel(logging.DEBUG)
# Remove any handlers already attached (e.g. from config.py basicConfig)
_root.handlers.clear()
_root.addHandler(logging.StreamHandler(sys.stdout))
_fh = logging.FileHandler(str(_LOG_PATH), encoding="utf-8")
_fh.setFormatter(_fmt)
_root.addHandler(_fh)
for _h in _root.handlers:
    _h.setFormatter(_fmt)
# Suppress noisy third-party libraries
for _lib in ("urllib3", "requests", "lightstreamer", "websocket", "http.client"):
    logging.getLogger(_lib).setLevel(logging.WARNING)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Exit profiles (same as backtest)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ExitProfile:
    name:          str
    tp_pips:       float
    sl_atr_mult:   float
    trail_trigger: float   # pips profit before trail activates
    trail_dist:    float   # pips to trail behind best price
    max_bars:      int

PROFILES: Dict[str, ExitProfile] = {
    # Backtest 2026-06-12/16 (16 mnd): SL adaptive by ATR rank + bounce exit
    # -> 3-tier adaptive + bounce: +5114p, avg +1.41p, win 78.7%, DD -173p
    # trail_trigger lowered 2026-06-17: 7→5p (T1/T2), 6.5→4.5p (T3), 5.5→4p (T4)
    # EUR/USD rarely moves 7p in a single 10-min bar; trail was arming too late.
    "T1_TrendRC":     ExitProfile("Trend+RC",          8.0, 2.5, 5.0, 4.0, 999),
    "T2_Trend":       ExitProfile("Trend",             6.0, 2.5, 5.0, 4.0, 999),
    "T3_Expansion":   ExitProfile("Expansion+Impulse", 5.0, 2.5, 4.5, 4.0, 999),
    "T4_Compression": ExitProfile("Compression",       4.0, 2.5, 4.0, 3.0, 999),
}
# Minimum pips profit that must be locked in before trail stop can fire.
MIN_TRAIL_LOCK_PIPS = 1.5

# ── Adaptive SL by ATR rank (backtest 2026-06-16) ────────────────────────────
# High-vol bars hit SL more often, but with wider SL they complete bigger moves.
# ATR rank >75% -> 4.0x ATR; otherwise keep 2.5x from profile.
# Result: fewer SL hits, more TrailSL wins, avg +0.88p vs +0.41p (SL-only).
# Combined with bounce: +1.41p avg, DD -173p (vs +0.41p, DD -590p baseline).
def get_adaptive_sl_mult(profile_key: str, atr_rank: float) -> float:
    if atr_rank >= 75.0:
        return 4.0
    return PROFILES[profile_key].sl_atr_mult   # 2.5 from profile


def _update_reversal_strike(state: Dict, direction: str, pnl: float):
    """Set/clear a reversal strike after each trade close.

    A strike is set when a trade fails AND its direction differs from the
    prior closed trade — i.e. a direction-change attempt just failed.
    While active, a second reversal entry in that direction requires both
    daily_bias == direction AND local != ReversalCandidate.
    A profitable close in any direction clears the strike.
    """
    last = state.get("last_trade_direction")
    if pnl < 0 and last is not None and last != direction:
        state["reversal_strike"] = {"direction": direction}
        log.info(f"[STRIKE] Reversal strike SET on {direction} "
                 f"(direction change from {last} failed, pnl={pnl:+.1f}p)")
    elif pnl >= 0 and state.get("reversal_strike"):
        log.info(f"[STRIKE] Strike CLEARED — {direction} trade profitable "
                 f"(pnl={pnl:+.1f}p)")
        state["reversal_strike"] = None
    state["last_trade_direction"] = direction

# ── Bounce exit config (backtest 2026-06-16) ─────────────────────────────────
# After a hard SL triggers: hold position up to 2 bars, wait for price to
# recover. Safety floor at SL+6p adverse closes immediately if move continues.
# Both Bull and Bear bounce equally well on 10-min bars (contrary to swing).
# Best config: wait=2b, safety=6p, min_bounce=1p, trail=2p.
BOUNCE_MAX_BARS   = 2     # bars to wait after SL trigger
BOUNCE_SAFETY_P   = 6.0  # pips further adverse -> close immediately
BOUNCE_MIN_PIPS   = 1.0  # pips of recovery needed before trail activates
BOUNCE_TRAIL_PIPS = 2.0  # trail distance once bounce is detected

def get_profile_key(direction: str, environment: str, local: str) -> Optional[str]:
    if environment == "Trend" and local == "ReversalCandidate": return "T1_TrendRC"
    if environment == "Trend":                                   return "T2_Trend"
    if environment == "Expansion" and local == "Impulse":       return "T3_Expansion"
    if environment == "Expansion":                              return "T2_Trend"
    if environment == "Compression":                            return "T4_Compression"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Market data: fetch from CityIndex API
# ─────────────────────────────────────────────────────────────────────────────
def fetch_bars(n: int = BARS_NEEDED) -> pd.DataFrame:
    token   = session_manager.get_session_token()
    now_ts  = int(time.time())
    url = (
        f"https://ciapi.cityindex.com/TradingAPI/market/{MARKET_ID}/barhistorybefore"
        f"?interval=MINUTE&span={BAR_MINUTES}&toTimestampUTC={now_ts}"
        f"&maxResults={n}&priceType=MID"
    )
    headers = {"Session": token, "UserName": username}
    resp    = requests.get(url, headers=headers, timeout=15)
    if resp.status_code == 401:
        headers["Session"] = session_manager.refresh_session_token()
        resp = requests.get(url, headers=headers, timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(f"Fetch failed: {resp.status_code} {resp.text[:200]}")

    bars = resp.json().get("PriceBars", [])
    if not bars:
        raise RuntimeError("No bars returned from API")

    rows = []
    for b in bars:
        m   = re.search(r"\d+", b["BarDate"])
        ts  = int(m.group(0)) if m else 0
        dt  = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        rows.append({
            "datetime": dt,
            "open":  float(b["Open"]),
            "high":  float(b["High"]),
            "low":   float(b["Low"]),
            "close": float(b["Close"]),
            "volume": np.nan,
        })

    df = (pd.DataFrame(rows)
            .sort_values("datetime")
            .drop_duplicates("datetime")
            .reset_index(drop=True))
    df["date_utc"] = df["datetime"].dt.date.astype(str)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Run v6 classifier on fetched bars
# ─────────────────────────────────────────────────────────────────────────────
def classify_bars(df: pd.DataFrame) -> pd.DataFrame:
    p   = Params()
    out = compute_features(df, p)
    out = detect_swings(out, p)
    out = classify_direction(out, p)
    out = classify_environment(out, p)
    out = classify_local(out, p)
    return out.reset_index(drop=True)


def get_daily_bias(df: pd.DataFrame) -> str:
    today = df["date_utc"].iloc[-1]
    day   = df[df["date_utc"] == today].sort_values("datetime")
    if len(day) < 2:
        return "Mixed"
    net = day["close"].iloc[-1] - day["close"].iloc[0]
    dom = day["direction"].value_counts().index[0] if len(day) > 0 else "Neutral"
    if net > 0.0003 and dom == "Bull":   return "Bull"
    if net < -0.0003 and dom == "Bear":  return "Bear"
    return "Mixed"


def get_session_label(hour: int) -> str:
    if 7  <= hour < 12: return "London"
    if 12 <= hour < 16: return "Overlap"
    if 16 <= hour < 21: return "NY"
    return "Other"


# ─────────────────────────────────────────────────────────────────────────────
# Entry filter
# ─────────────────────────────────────────────────────────────────────────────
def entry_allowed(direction: str, environment: str, session: str, hour: int = 12) -> bool:
    if direction not in ("Bull", "Bear"):        return False
    if environment == "Range":                   return False
    # Compression blocked: backtest Feb 2025-Jun 2026 showed -1786p over 3841
    # trades (-0.46p avg). All edge is in Trend/Expansion.
    if environment == "Compression":             return False
    if direction == "Bear" and session == "NY":  return False
    # Block 1h before midnight and 1h after midnight UTC (day-flip buffer).
    # Classifier uses daily-bar stats that reset at 00:00 UTC, causing noisy
    # label flips around the day boundary. Avoid trading 23:00-00:59 UTC.
    if hour == 23 or hour == 0:                  return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# State: open trades persisted to JSON between runs
# ─────────────────────────────────────────────────────────────────────────────
def load_state() -> Dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                state = json.load(f)
                ensure_risk_state(state)
                return state
        except Exception:
            pass
    state = {
        "open_trades": {},
        "bounce_pending": {},
        "pending_signal": None,
        "reversal_strike": None,
        "last_trade_direction": None,
    }
    ensure_risk_state(state)
    return state


def save_state(state: Dict):
    with _state_save_lock:
        with _state_lock:
            payload = json.dumps(state, indent=2, default=str)
        temp_path = STATE_FILE.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, STATE_FILE)


# ─────────────────────────────────────────────────────────────────────────────
# Lightstreamer — persistent receiver (same as scalp_sys T2-TICK thread)
# ─────────────────────────────────────────────────────────────────────────────
_ls_receiver: Optional[LightstreamerReceiver] = None

# Shared lock: acquired by TickExitMonitor on each tick poll (~1ms),
# and by the main bar loop while touching state (~2s per 10-min cycle).
_state_lock = threading.Lock()
_state_save_lock = threading.Lock()
_stats_lock = threading.Lock()

# Tick-based exit monitor (started after ls_connect)
_tick_monitor = None


def ls_connect() -> bool:
    """Connect Lightstreamer and publish the new receiver to all consumers."""
    global _ls_receiver
    if _ls_receiver is not None:
        try:
            _ls_receiver.disconnect()
        except Exception:
            pass
        _ls_receiver = None

    for attempt in range(2):
        candidate = None
        try:
            token = (
                session_manager.get_session_token()
                if attempt == 0
                else session_manager.refresh_session_token()
            )
            log.info("[LS] Connecting to Lightstreamer (push.cityindex.com)...")
            candidate = LightstreamerReceiver(
                initial_market_ids=[int(MARKET_ID)],
                session_token=token,
                max_connect_attempts=3,
            )
            tick = candidate.fetch_market_data_one(MARKET_ID, timeout_secs=8)
            if tick:
                _ls_receiver = candidate
                if _tick_monitor is not None:
                    _tick_monitor.set_ls_receiver(candidate)
                log.info(f"[LS] Connected OK — Bid={float(tick['Bid']):.5f}  "
                         f"Offer={float(tick['Offer']):.5f}  "
                         f"AuditId={tick.get('AuditId','')}")
                return True
            log.warning("[LS] Connected but no tick within 8s.")
        except Exception as e:
            log.error(f"[LS] Connection attempt {attempt + 1} failed: {e}")
        finally:
            if candidate is not None and candidate is not _ls_receiver:
                try:
                    candidate.disconnect()
                except Exception:
                    pass

        if attempt == 0:
            log.warning("[LS] Refreshing CIAPI token and retrying once.")

    return False


def ls_get_tick() -> Optional[Dict]:
    """Return latest cached tick from the persistent receiver."""
    if _ls_receiver is None:
        return None
    try:
        return _ls_receiver.latest_data.get(str(MARKET_ID))
    except Exception:
        return None


def get_current_price(paper: bool = False) -> Tuple[float, float, str]:
    """Returns (bid, offer, audit_id) from live Lightstreamer cache."""
    if paper:
        return 0.0, 0.0, "PAPER"

    tick = ls_get_tick()
    if tick:
        bid, offer = float(tick["Bid"]), float(tick["Offer"])
        audit_id   = str(tick.get("AuditId", ""))
        log.info(f"[LS] Live tick — Bid={bid:.5f}  Offer={offer:.5f}  AuditId={audit_id}")
        return bid, offer, audit_id

    # No tick — reconnect once
    log.warning("[LS] No cached tick — reconnecting...")
    if ls_connect():
        tick = ls_get_tick()
        if tick:
            return float(tick["Bid"]), float(tick["Offer"]), str(tick.get("AuditId", ""))

    # Fallback
    log.warning("[PRICE] LS unavailable — using bar close estimate (FALLBACK).")
    df = fetch_bars(n=5)
    price = df["close"].iloc[-1]
    return price - 0.00005, price + 0.00005, "FALLBACK"


# ─────────────────────────────────────────────────────────────────────────────
# Order execution — uses EXACT same classes as scalp_sys, zero deviation
# ─────────────────────────────────────────────────────────────────────────────
QUANTITY = 1000
COMMISSION_PER_1K_EUR = 0.10   # EUR per 1000-unit round-trip (CityIndex: 0.05 open + 0.05 close)


def open_order(direction: str, bid: float, offer: float, audit_id: str,
               paper: bool = False) -> Tuple[Optional[str], float]:
    """Open via TradeExecutor.place_order() — identical to scalp_sys."""
    entry_price = offer if direction == "buy" else bid

    if paper:
        fake_id = f"PAPER_{int(time.time())}"
        log.info(f"  [PAPER] OPEN {direction.upper()} @ {entry_price:.5f}  "
                 f"Bid={bid:.5f}  Offer={offer:.5f}")
        return fake_id, entry_price

    try:
        executor = TradeExecutor()
        order_id, opening_price = executor.place_order(
            direction   = direction,
            bid         = bid,
            offer       = offer,
            audit_id    = audit_id,
            market_id   = int(MARKET_ID),
            market_name = MARKET_NAME,
            quantity_multiplier = 1.0,
        )
        if not order_id or int(order_id) <= 0:
            log.error(f"  [OPEN] Invalid OrderId returned: {order_id}")
            return None, 0.0
        log.info(f"  [OPEN] {direction.upper()} @ {opening_price:.5f} | OrderId={order_id}")
        return str(order_id), float(opening_price)
    except Exception as e:
        log.error(f"  [OPEN] Exception: {e}")
        return None, 0.0


def close_order(order_id: str, entry_direction: str,
                bid: float, offer: float, audit_id: str,
                quantity: float, entry_price: float,
                paper: bool = False) -> bool:
    """Close via PositionCloser.close_position() — identical to scalp_sys."""
    close_price = bid if entry_direction == "buy" else offer

    if paper:
        log.info(f"  [PAPER] CLOSE {entry_direction.upper()} @ {close_price:.5f}  "
                 f"OrderId={order_id}")
        return True

    try:
        tok    = session_manager.get_session_token()
        closer = PositionCloser(
            tok,
            tradingAccountID,
            username,
            perform_last_check=False,
        )
        success = closer.close_position(
            order_id    = int(order_id),
            direction   = entry_direction,
            price       = close_price,
            audit_id    = audit_id,
            quantity    = quantity,
            bid         = bid,
            ask         = offer,
            entry_price = entry_price,
            market_id   = int(MARKET_ID),
            reason      = "live_engine_exit",
        )
        return success
    except Exception as e:
        log.error(f"  [CLOSE] Exception: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Trade exit logic (mirrors backtest)
# ─────────────────────────────────────────────────────────────────────────────
def check_exit(trade: Dict, current_bar: pd.Series) -> Optional[str]:
    """
    Check if trade should exit. Returns exit reason string or None.
    Updates trade dict in-place (trail stop, best price).
    """
    p         = PROFILES[trade["profile_key"]]
    direction = trade["direction"]
    cur_dir   = current_bar["direction"]
    cur_env   = current_bar["environment"]
    high      = float(current_bar["high"])
    low       = float(current_bar["low"])
    close     = float(current_bar["close"])
    bars_held = trade["bars_held"]

    # Label-change exit — require >=2 bars held before acting on a flip.
    # A 1-bar flip is usually classifier noise, not a real reversal.
    # EnvRange exit removed 2026-06-12: backtest shows it net-negative
    # (-3.30p avg over 310 trades); trail/SL handle those cases better.
    if cur_dir != direction and bars_held >= 2:
        return "DirectionFlip"

    # Max bars
    if bars_held >= p.max_bars:
        return "MaxBars"

    # Price-based exits
    entry      = trade["entry_price"]
    sl         = trade["sl_price"]
    tp         = trade["tp_price"]
    trail_on   = trade["trail_active"]
    trail_stop = trade["trail_stop"]
    best       = trade["best_price"]

    if direction == "Bull":
        if high > best:
            trade["best_price"] = high
            best = high
        profit_pips = (best - entry) * 10000
        if profit_pips >= p.trail_trigger:
            trade["trail_active"] = True
            new_trail = best - (p.trail_dist / 10000)
            # Floor: trail stop must lock in at least MIN_TRAIL_LOCK_PIPS above entry
            min_lock = entry + (MIN_TRAIL_LOCK_PIPS / 10000)
            new_trail = max(new_trail, min_lock)
            trade["trail_stop"] = max(trail_stop, new_trail) if trail_stop > 0 else new_trail
        active_sl = max(sl, trade["trail_stop"]) if trade["trail_active"] else sl
        if low <= active_sl:
            return "TrailSL" if trade["trail_active"] else "SL"
        if high >= tp:
            return "TP"
    else:  # Bear
        if low < best or best == entry:
            trade["best_price"] = low
            best = low
        profit_pips = (entry - best) * 10000
        if profit_pips >= p.trail_trigger:
            trade["trail_active"] = True
            new_trail = best + (p.trail_dist / 10000)
            # Floor: trail stop must lock in at least MIN_TRAIL_LOCK_PIPS below entry
            max_lock = entry - (MIN_TRAIL_LOCK_PIPS / 10000)
            new_trail = min(new_trail, max_lock)
            trade["trail_stop"] = min(trail_stop, new_trail) if trail_stop > 0 else new_trail
        active_sl = min(sl, trade["trail_stop"]) if trade["trail_active"] else sl
        if high >= active_sl:
            return "TrailSL" if trade["trail_active"] else "SL"
        if low <= tp:
            return "TP"

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
def log_trade(record: Dict):
    df = pd.DataFrame([record])
    write_header = not LOG_FILE.exists()
    df.to_csv(LOG_FILE, mode="a", header=write_header, index=False)


def update_stats_json(pnl_pips: float, quantity: float, close_price: float,
                      ts: datetime):
    """Update live_stats.json with daily/weekly/monthly/yearly net P&L.

    commission_eur is deducted here (two fixed charges per round-trip).
    gross_eur is converted from pips using the EURUSD close price.
    net_eur = gross_eur - commission.
    """
    commission_eur = -abs((quantity / 1000.0) * COMMISSION_PER_1K_EUR)
    gross_eur      = pnl_pips * quantity / 10000.0 / close_price if close_price else 0.0
    net_eur        = gross_eur + commission_eur

    day_key   = ts.strftime("%Y-%m-%d")
    iso_cal   = ts.isocalendar()
    week_key  = f"{iso_cal[0]}-W{iso_cal[1]:02d}"
    month_key = ts.strftime("%Y-%m")
    year_key  = ts.strftime("%Y")

    with _stats_lock:
        if STATS_FILE.exists():
            try:
                with open(STATS_FILE, encoding="utf-8") as f:
                    stats = json.load(f)
            except Exception:
                stats = {}
        else:
            stats = {}

        for section, key in [
            ("daily", day_key), ("weekly", week_key),
            ("monthly", month_key), ("yearly", year_key),
        ]:
            bucket = stats.setdefault(section, {})
            entry  = bucket.setdefault(key, {
                "trades": 0, "gross_pips": 0.0,
                "commission_eur": 0.0, "net_eur": 0.0,
            })
            entry["trades"]         += 1
            entry["gross_pips"]     = round(entry.get("gross_pips", 0.0)     + pnl_pips,       2)
            entry["gross_eur"]      = round(entry.get("gross_eur",  0.0)     + gross_eur,      4)
            entry["commission_eur"] = round(entry.get("commission_eur", 0.0) + commission_eur, 4)
            entry["net_eur"]        = round(entry.get("net_eur",  0.0)       + net_eur,        4)

        stats["last_updated"] = ts.isoformat()

        temp = STATS_FILE.with_suffix(".tmp")
        with open(temp, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, STATS_FILE)

    log.debug("[STATS] day=%s  gross=%.2fp  comm=%.4f€  net=%.4f€",
              day_key, pnl_pips, commission_eur, net_eur)


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard
# ─────────────────────────────────────────────────────────────────────────────
def print_dashboard(classified: pd.DataFrame, open_trades: Dict,
                    daily_bias: str, session: str, bar_time: datetime,
                    bounce_pending: Optional[Dict] = None,
                    state: Optional[Dict] = None):
    last    = classified.iloc[-1]
    now_utc = datetime.now(timezone.utc)

    # Live tick from Lightstreamer
    tick = ls_get_tick()
    if tick:
        live_bid   = float(tick["Bid"])
        live_offer = float(tick["Offer"])
        live_mid   = (live_bid + live_offer) / 2
        live_str   = f"Bid={live_bid:.5f}  Ask={live_offer:.5f}  Mid={live_mid:.5f}"
        ls_status  = "LIVE (Lightstreamer)"
    else:
        live_str  = f"Bar close={last['close']:.5f}  (no LS tick)"
        ls_status = "OFFLINE / no tick yet"

    os.system("cls" if os.name == "nt" else "clear")
    print("=" * 65)
    print(f"  EUR/USD LIVE ENGINE")
    print(f"  Bar  : {bar_time.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"  Now  : {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 65)
    print(f"  Live price  : {live_str}")
    print(f"  LS feed     : {ls_status}")
    print(f"  Bar price   : {last['close']:.5f}  (H:{last['high']:.5f}  L:{last['low']:.5f})")
    print(f"  ATR         : {last['atr']*10000:.1f} pips  |  ATR rank: {last['atr_rank']:.0f}%")
    print()
    print(f"  DIRECTION   : {last['direction']:<10}  conf={last['direction_conf']:.0f}")
    print(f"  ENVIRONMENT : {last['environment']:<10}  conf={last['environment_conf']:.0f}")
    print(f"  LOCAL       : {last['local']:<10}  conf={last['local_conf']:.0f}")
    print()
    print(f"  Daily bias  : {daily_bias}  |  Session: {session}")
    print(f"  Dir score   : {last['dir_score']:.1f}  |  Efficiency: {last['efficiency']:.1f}")
    strike = state.get("reversal_strike") if state else None
    if strike:
        print(f"  *** REVERSAL STRIKE : {strike['direction']} — need bias={strike['direction']}+no RC ***")
    print()

    # Entry signal check — show each rule pass/fail
    d   = str(last["direction"])
    e   = str(last["environment"])
    loc = str(last["local"])
    combo   = f"{d}_{e}_{loc}"
    profile = get_profile_key(d, e, loc)
    allowed = entry_allowed(d, e, session, bar_time.hour)
    risk_ok, risk_reason = (
        risk_entry_allowed(state, d, last, now_utc)
        if state is not None and d in ("Bull", "Bear")
        else (True, "not evaluated")
    )
    signal  = allowed and profile is not None and risk_ok

    print(f"  SIGNAL      : {'>>> ACTIVE <<<' if signal else 'none'}", end="")
    if signal:
        print(f"  [{PROFILES[profile].name}]  {combo}")
    else:
        print()

    # Show exactly which rule blocked entry
    print(f"  Signal rules:")
    _h = bar_time.hour
    print(f"    Dir not Neutral  : {'PASS' if d in ('Bull','Bear') else 'FAIL  <- direction is Neutral'}")
    print(f"    Env Trend/Expans : {'PASS' if e not in ('Range','Compression') else 'FAIL  <- environment is ' + e}")
    print(f"    Bear+NY allowed  : {'PASS' if not (d=='Bear' and session=='NY') else 'FAIL  <- Bear blocked in NY session'}")
    print(f"    Midnight buffer  : {'FAIL  <- 23:00-01:00 UTC blocked' if _h in (23,0) else 'PASS'}")
    print(f"    Profile exists   : {'PASS  profile=' + profile if profile else 'FAIL  <- no profile for this combo'}")
    print(f"    Daily bias       : {daily_bias}  (info only)")
    _ds = float(last["dir_score"]) if pd.notna(last.get("dir_score")) else 0.0
    _loc = str(last["local"])
    if _loc in ("Pullback", "ReversalCandidate") and abs(_ds) < 20.0:
        print(f"    Dir-score gate   : FAIL  <- {_loc} blocked (dir_score={_ds:.1f}, need |score|>=20)")
    else:
        print(f"    Dir-score gate   : PASS  dir_score={_ds:.1f}")
    print(f"    Risk controls    : {'PASS' if risk_ok else 'FAIL'}  {risk_reason}")

    if state is not None:
        rs = risk_status(state)
        print(
            f"    Day realized     : "
            f"{rs['daily_realized_pips']:+.1f}p / "
            f"{rs['daily_realized_r']:+.2f}R"
        )

    print()
    print(f"  OPEN TRADES ({len(open_trades)}):")
    if not open_trades:
        print("    None")
    else:
        print(f"  {'Combo':<40} {'Entry':>8} {'Bars':>5} {'Trail':>6} {'SLx':>5} {'PnL*':>7}")
        for key, t in open_trades.items():
            entry = t["entry_price"]
            bars  = t["bars_held"]
            close = float(last["close"])
            pnl   = (close - entry) * 10000 if t["direction"] == "Bull" else (entry - close) * 10000
            trail = "ON " if t["trail_active"] else "off"
            slx   = f"{t.get('sl_mult', 2.5):.1f}x"
            print(f"  {key:<40} {entry:.5f} {bars:>5} {trail:>6} {slx:>5} {pnl:>+7.1f}")

    if bounce_pending:
        print(f"\n  BOUNCE PENDING ({len(bounce_pending)}) — SL held, waiting for recovery:")
        print(f"  {'Combo':<40} {'Entry':>8} {'Bars':>5} {'SafetyFl':>10} {'BncBars':>7}")
        close = float(last["close"])
        for key, bs in bounce_pending.items():
            entry = bs["entry_price"]
            bars  = bs.get("bars_held", 0)
            bnc   = bs["bars_in_bounce"]
            sf    = bs["safety_floor"]
            print(f"  {key:<40} {entry:.5f} {bars:>5} {sf:.5f} {bnc:>7}")

    print()

    # ── Net P&L stats (from live_stats.json) ─────────────────────────────
    try:
        if STATS_FILE.exists():
            with open(STATS_FILE, encoding="utf-8") as _sf:
                _st = json.load(_sf)
            now_utc2   = datetime.now(timezone.utc)
            _day_key   = now_utc2.strftime("%Y-%m-%d")
            _iso       = now_utc2.isocalendar()
            _week_key  = f"{_iso[0]}-W{_iso[1]:02d}"
            _month_key = now_utc2.strftime("%Y-%m")
            _year_key  = now_utc2.strftime("%Y")
            def _fmt(section, key):
                e = _st.get(section, {}).get(key)
                if not e:
                    return "n/a"
                return (f"{e['net_eur']:+.2f}€ net  "
                        f"({e['trades']}tr  gross={e.get('gross_eur', 0.0):+.2f}€  "
                        f"comm={e['commission_eur']:+.2f}€)")
            print(f"  NET P&L")
            print(f"    Today  {_day_key:<12}: {_fmt('daily',   _day_key)}")
            print(f"    Week   {_week_key:<12}: {_fmt('weekly',  _week_key)}")
            print(f"    Month  {_month_key:<12}: {_fmt('monthly', _month_key)}")
            print(f"    Year   {_year_key:<12}: {_fmt('yearly',  _year_key)}")
            print()
    except Exception:
        pass

    print("  * PnL = unrealised pips at current close price")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
def run(paper: bool = False):
    log.info(f"{'='*60}")
    log.info(f"EUR/USD V6 LIVE ENGINE STARTING")
    log.info(f"  paper={paper}  market={MARKET_NAME}  market_id={MARKET_ID}")
    log.info(f"  tradingAccountID={tradingAccountID}  username={username}")
    log.info(f"  folder={_HERE}")
    log.info(f"  log={_LOG_PATH}")
    log.info(f"  state={STATE_FILE}")
    log.info(f"{'='*60}")
    state = load_state()

    # ── Restore open trades from broker API (same as scalp_sys main()) ──────
    log.info("[STARTUP] Syncing open positions from CityIndex API...")
    try:
        existing = PositionManagementModule.get_open_positions() or []
        for p in existing:
            if str(p.get("MarketId", "")) != str(MARKET_ID):
                continue
            oid   = str(p.get("OrderId", ""))
            dirn  = str(p.get("Direction", "")).lower()   # 'buy' or 'sell'
            entry = float(p.get("Price", 0))
            if not oid or not entry:
                continue
            direction_label = "Bull" if dirn == "buy" else "Bear"
            combo_key = f"{direction_label}_restored_{oid}"
            local_trade = next(
                (
                    trade
                    for bucket in ("open_trades", "bounce_pending")
                    for trade in state.get(bucket, {}).values()
                    if str(trade.get("order_id", "")) == oid
                ),
                None,
            )
            if local_trade is not None:
                old_entry = float(local_trade.get("entry_price", entry))
                delta = entry - old_entry
                if abs(delta) > 0.0000001:
                    local_trade["entry_price"] = entry
                    for field_name in (
                        "sl_price",
                        "tp_price",
                        "best_price",
                        "trail_stop",
                        "sl_trigger",
                        "safety_floor",
                        "bounce_trail",
                        "best_recovery",
                    ):
                        value = local_trade.get(field_name)
                        if value not in (None, 0, 0.0):
                            local_trade[field_name] = float(value) + delta
                    log.warning(
                        f"  Corrected order={oid} to broker fill "
                        f"{old_entry:.5f} -> {entry:.5f}"
                    )
            else:
                state["open_trades"][combo_key] = {
                    "order_id":      oid,
                    "direction":     direction_label,
                    "environment":   "restored",
                    "local":         "restored",
                    "profile_key":   "T2_Trend",
                    "entry_price":   entry,
                    "entry_time":    datetime.now(timezone.utc).isoformat(),
                    "atr_entry":     0.0006,
                    "sl_price":      entry - 0.0009 if direction_label=="Bull" else entry + 0.0009,
                    "tp_price":      entry + 0.0006 if direction_label=="Bull" else entry - 0.0006,
                    "trail_dist":    0.0003,
                    "trail_trigger": 0.0004,
                    "trail_active":  False,
                    "trail_stop":    0.0,
                    "best_price":    entry,
                    "bars_held":     0,
                }
                log.info(f"  Restored: {direction_label} order={oid} entry={entry:.5f}")
        if not existing:
            log.info("  No open positions at broker.")
    except Exception as e:
        log.warning(f"  Position restore failed (non-fatal): {e}")

    # ── Start Lightstreamer feed (always — even in paper mode to verify feed) ─
    ls_ok = ls_connect()
    if not ls_ok:
        log.warning("[LS] Starting without live feed — will retry on first order.")

    # ── Start real-time tick-based exit monitor ───────────────────────────────
    global _tick_monitor
    from tick_exit_monitor import TickExitMonitor
    _tick_monitor = TickExitMonitor(
        state=state,
        lock=_state_lock,
        ls_receiver=_ls_receiver,
        market_id=MARKET_ID,
        close_order_fn=close_order,
        log_trade_fn=log_trade,
        quantity=float(QUANTITY),
        paper=paper,
        bounce_safety_p=BOUNCE_SAFETY_P,
        bounce_min_pips=BOUNCE_MIN_PIPS,
        bounce_trail_pips=BOUNCE_TRAIL_PIPS,
        min_trail_lock_p=MIN_TRAIL_LOCK_PIPS,
        state_changed_fn=lambda: save_state(state),
        on_trade_close_fn=lambda direction, pnl: _update_reversal_strike(state, direction, pnl),
        update_stats_fn=lambda pnl_pips, qty, close_px, ts: update_stats_json(pnl_pips, qty, close_px, ts),
    )
    _tick_monitor._reconnect_fn = ls_connect  # watchdog calls this on stale feed
    _tick_monitor.start()

    cycle = 0
    while True:
        cycle += 1
        try:
            now_utc = datetime.now(timezone.utc)
            log.debug(f"")
            log.debug(f"{'='*60}")
            log.debug(f"CYCLE {cycle}  wall={now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            log.debug(f"{'='*60}")

            # ── 1. Fetch bars ──────────────────────────────────────────────
            log.debug("STEP 1: Fetching bars from CityIndex API...")
            try:
                raw = fetch_bars(BARS_NEEDED)
            except Exception as e:
                log.error(f"STEP 1 FAILED — fetch_bars raised: {e}")
                raise

            last_bar_dt = raw['datetime'].iloc[-1]
            if last_bar_dt.tzinfo is None:
                last_bar_dt = last_bar_dt.replace(tzinfo=timezone.utc)
            bar_age_min = (now_utc - last_bar_dt).total_seconds() / 60
            log.debug(f"  bars={len(raw)}  first={raw['datetime'].iloc[0].strftime('%H:%M')}  "
                      f"last={raw['datetime'].iloc[-1].strftime('%Y-%m-%d %H:%M')} UTC  "
                      f"age={bar_age_min:.1f}min")
            if bar_age_min > 15:
                log.warning(f"  *** STALE BARS — {bar_age_min:.0f} min old! API may be lagging.")

            # ── 2. Classify ────────────────────────────────────────────────
            log.debug("STEP 2: Running v6 classifier...")
            try:
                classified = classify_bars(raw)
            except Exception as e:
                log.error(f"STEP 2 FAILED — classify_bars raised: {e}")
                raise

            last       = classified.iloc[-1]
            bar_time   = last["datetime"]
            hour       = bar_time.hour
            session    = get_session_label(hour)
            daily_bias = get_daily_bias(classified)

            log.debug(f"  bar={bar_time.strftime('%Y-%m-%d %H:%M')} UTC")
            log.debug(f"  DIRECTION  : {last['direction']}  conf={last['direction_conf']:.0f}  score={last['dir_score']:.1f}")
            log.debug(f"  ENVIRONMENT: {last['environment']}  conf={last['environment_conf']:.0f}")
            log.debug(f"  LOCAL      : {last['local']}  conf={last['local_conf']:.0f}")
            log.debug(f"  ATR={last['atr']*10000:.1f}p  atr_rank={last['atr_rank']:.0f}%  efficiency={last['efficiency']:.1f}")
            log.debug(f"  session={session}  daily_bias={daily_bias}  open_trades={len(state['open_trades'])}")
            if len(classified) >= 2:
                with _state_lock:
                    advance_risk_bar(state, last, classified.iloc[-2])

            # ── 3a. Bounce-pending: OHLC timeout only ─────────────────────────
            # Tick monitor handles BounceSafety/BounceProfit/BounceTrail in
            # real-time. OHLC loop only increments bars_in_bounce and fires
            # the BounceTimeout when max bars is reached.
            # Thread safety: list() snapshots prevent RuntimeError on concurrent
            # dict modification; guards before pops handle tick-monitor races.
            if "bounce_pending" not in state:
                state["bounce_pending"] = {}

            # Tick monitor handles real-time bounce exits; OHLC loop only
            # increments bar counter and fires BounceTimeout.
            bounce_to_close = []
            for bkey, bs in list(state["bounce_pending"].items()):
                bs["bars_in_bounce"] += 1
                if bs["bars_in_bounce"] >= BOUNCE_MAX_BARS:
                    bounce_to_close.append((bkey, "BounceTimeout"))
                    log.info(f"BOUNCE_TIMEOUT {bkey}  bars={bs['bars_in_bounce']}")

            def _do_close(order_id_str, direction, entry_price, reason_label, paper):
                bid, offer, audit_id = get_current_price(paper=paper)
                log.info(f"  [CLOSE] Live price Bid={bid:.5f}  Offer={offer:.5f}  AuditId={audit_id}")
                api_dir = "buy" if direction == "Bull" else "sell"
                success = close_order(
                    order_id=order_id_str, entry_direction=api_dir,
                    bid=bid, offer=offer, audit_id=audit_id,
                    quantity=float(QUANTITY), entry_price=float(entry_price), paper=paper,
                )
                close_price = bid if direction == "Bull" else offer
                pnl = (close_price - entry_price) * 10000 if direction == "Bull" \
                      else (entry_price - close_price) * 10000
                log.info(f"  {'OK' if success else 'FAILED'}  PnL={pnl:+.1f}p  "
                         f"entry={entry_price:.5f}  close={close_price:.5f}")
                return success, close_price, pnl

            for bkey, close_reason in bounce_to_close:
                if bkey not in state["bounce_pending"]:
                    continue  # tick monitor already closed it
                bs = state["bounce_pending"][bkey]
                success, close_price, pnl = _do_close(
                    str(bs.get("order_id", "0")), bs["direction"], bs["entry_price"],
                    close_reason, paper)
                if success:
                    with _state_lock:
                        if state["bounce_pending"].get(bkey) is bs:
                            risk_result = record_risk_close(
                                state,
                                bs,
                                pnl,
                                close_price,
                                datetime.now(timezone.utc),
                            )
                            state["bounce_pending"].pop(bkey, None)
                            _update_reversal_strike(state, bs["direction"], pnl)
                        else:
                            risk_result = None
                    if risk_result:
                        log.info(
                            f"  [RISK] Realized={risk_result['realized_r']:+.2f}R  "
                            f"day={risk_result['daily_realized_r']:+.2f}R  "
                            f"gate={'ARMED' if risk_result['gate_armed'] else 'clear'}"
                        )
                    update_stats_json(pnl, float(QUANTITY), close_price,
                                      datetime.now(timezone.utc))
                _comm = -abs((QUANTITY / 1000.0) * COMMISSION_PER_1K_EUR)
                _gross = round(pnl * QUANTITY / 10000.0 / close_price, 4) if close_price else 0.0
                log_trade({
                    "type": "CLOSE",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "combo_key": bkey,
                    "direction": bs["direction"],
                    "environment": bs.get("environment", ""),
                    "local": bs.get("local", ""),
                    "profile": bs.get("profile_key", ""),
                    "entry_price": bs["entry_price"],
                    "close_price": round(close_price, 5),
                    "pnl_pips": round(pnl, 2),
                    "commission_eur": round(_comm, 4),
                    "gross_eur": _gross,
                    "net_eur": round(_gross + _comm, 4),
                    "bars_held": bs.get("bars_held", 0) + bs["bars_in_bounce"],
                    "exit_reason": close_reason,
                    "entry_order_id": bs.get("order_id"),
                    "close_success": success,
                    "paper": paper,
                })

            # ── 3b. Update bars_held + DirectionFlip exit (OHLC-only exit) ──
            # SL / TP / TrailSL / Bounce are now handled in real-time by the
            # tick monitor. OHLC loop only handles DirectionFlip (needs
            # classifier label) and increments bars_held.
            log.debug(f"STEP 3b: bars_held update + DirectionFlip check "
                      f"({len(state['open_trades'])} open trade(s))...")
            for key in list(state["open_trades"]):
                if key in state["open_trades"]:
                    state["open_trades"][key]["bars_held"] += 1

            to_close = []
            for key, trade in list(state["open_trades"].items()):
                reason = check_exit(trade, last)
                if reason and reason != "DirectionFlip":
                    # Tick monitor already owns SL/TP/TrailSL — skip OHLC fires
                    log.debug(f"  OHLC exit [{key}] reason={reason} "
                              f"(ignored — tick monitor owns this)")
                    continue
                log.debug(f"  exit_check [{key}] bars={trade['bars_held']} "
                          f"entry={trade['entry_price']:.5f} -> {reason or 'hold'}")
                if reason == "DirectionFlip":
                    to_close.append((key, reason))

            for key, reason in to_close:
                if key not in state["open_trades"]:
                    continue  # tick monitor already closed it

                trade = state["open_trades"][key]

                # Verify position still exists at broker before attempting close
                try:
                    live_pos = PositionManagementModule.get_open_positions() or []
                    order_id_str = str(trade.get("order_id", ""))
                    still_open = any(str(p.get("OrderId","")) == order_id_str for p in live_pos)
                    if not still_open:
                        log.warning(f"  [EXIT] OrderId {order_id_str} not found at broker "
                                    f"(already closed?) — removing from local state")
                        state["open_trades"].pop(key, None)
                        continue
                except Exception as e:
                    log.warning(f"  [EXIT] Could not verify position at broker: {e} — proceeding anyway")

                log.info(f"EXIT  {key}  reason={reason}")
                success, close_price, pnl = _do_close(
                    str(trade.get("order_id", "0")), trade["direction"],
                    trade["entry_price"], reason, paper)

                if success:
                    with _state_lock:
                        if state["open_trades"].get(key) is trade:
                            risk_result = record_risk_close(
                                state,
                                trade,
                                pnl,
                                close_price,
                                datetime.now(timezone.utc),
                            )
                            state["open_trades"].pop(key, None)
                            _update_reversal_strike(state, trade["direction"], pnl)
                        else:
                            risk_result = None
                    if risk_result:
                        log.info(
                            f"  [RISK] Realized={risk_result['realized_r']:+.2f}R  "
                            f"day={risk_result['daily_realized_r']:+.2f}R  "
                            f"gate={'ARMED' if risk_result['gate_armed'] else 'clear'}"
                        )
                    update_stats_json(pnl, float(QUANTITY), close_price,
                                      datetime.now(timezone.utc))
                else:
                    log.warning(f"  [EXIT] Close failed — keeping {key} in state")

                _comm = -abs((QUANTITY / 1000.0) * COMMISSION_PER_1K_EUR)
                _gross = round(pnl * QUANTITY / 10000.0 / close_price, 4) if close_price else 0.0
                log_trade({
                    "type": "CLOSE",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "combo_key": key,
                    "direction": trade["direction"],
                    "environment": trade["environment"],
                    "local": trade["local"],
                    "profile": trade["profile_key"],
                    "entry_price": trade["entry_price"],
                    "close_price": round(close_price, 5),
                    "pnl_pips": round(pnl, 2),
                    "commission_eur": round(_comm, 4),
                    "gross_eur": _gross,
                    "net_eur": round(_gross + _comm, 4),
                    "bars_held": trade["bars_held"],
                    "exit_reason": reason,
                    "entry_order_id": trade.get("order_id"),
                    "close_success":  success,
                    "paper": paper,
                })

            # ── 4. Check for new entries (1-bar confirmation delay) ────────────
            # Backtest (16 mnd, 2026-06-17): delay=1 bar gives +5866p avg+1.45p
            # vs immediate +1903p avg+0.46p. Combined with adaptive SL + bounce:
            # +8704p avg+2.45p win=84.8% DD=-172p.
            # Logic: signal on bar N → save as pending_signal → bar N+1 must
            # confirm same direction + valid environment → enter at bar N+1 close
            # (i.e., market order placed at the start of bar N+2).
            log.debug("STEP 4: Entry check (1-bar confirmation)...")
            direction   = str(last["direction"])
            environment = str(last["environment"])
            local_lbl   = str(last["local"])
            combo_key   = f"{direction}_{environment}_{local_lbl}"
            profile_key = get_profile_key(direction, environment, local_lbl)

            MAX_OPEN = 1

            if "pending_signal" not in state:
                state["pending_signal"] = None

            # Ask the broker how many positions are open
            try:
                live_positions = PositionManagementModule.get_open_positions() or []
                n_open_broker  = sum(1 for p in live_positions
                                     if str(p.get("MarketId","")) == str(MARKET_ID))
                log.info(f"  Broker reports {n_open_broker} open position(s) on {MARKET_NAME}")
            except Exception as e:
                log.warning(f"  Could not query broker positions: {e} — using local state count")
                n_open_broker = len(state["open_trades"])
            n_open = n_open_broker

            trade_key = f"{combo_key}_{bar_time.strftime('%H%M')}"

            do_entry = False
            entry_direction = direction
            entry_environment = environment
            entry_local = local_lbl
            entry_profile_key = profile_key

            if n_open >= MAX_OPEN:
                # Can't enter — clear any pending signal
                if state["pending_signal"]:
                    log.info(f"  -> Clearing pending signal (position already open)")
                    state["pending_signal"] = None
            else:
                ps = state["pending_signal"]

                if ps is not None:
                    # We have a pending signal from last bar — check if it still holds
                    ps_dir = ps["direction"]
                    ps_env = ps["environment"]
                    ps_pk  = ps["profile_key"]
                    with _state_lock:
                        risk_ok, risk_reason = risk_entry_allowed(
                            state, ps_dir, last, now_utc
                        )
                    confirmed = (
                        direction == ps_dir
                        and environment not in ("Range", "Compression")
                        and entry_allowed(direction, environment, session, hour)
                        and profile_key is not None
                        and risk_ok
                    )
                    # Dir-score gate: Pullback/RC in a near-zero-conviction trend fail
                    # consistently. Backtest (16 mnd): |dir_score|<20 on Pullback/RC
                    # = avg -2.56p, -138p total. Filter saves +142p, DD -110p.
                    if confirmed and ps["local"] in ("Pullback", "ReversalCandidate"):
                        ds = ps.get("dir_score", 0.0)
                        if abs(ds) < 20.0:
                            log.info(
                                f"[DIR-GATE] Entry blocked — {ps['local']} in weak trend: "
                                f"dir_score={ds:.1f} (|score|<20 threshold)"
                            )
                            confirmed = False
                            state["pending_signal"] = None

                    # Reversal strike: if a direction-change trade just failed,
                    # require daily_bias to explicitly confirm AND local != RC
                    if confirmed:
                        strike = state.get("reversal_strike")
                        if strike and ps_dir == strike["direction"]:
                            bias_ok  = daily_bias == ps_dir
                            local_ok = ps["local"] != "ReversalCandidate"
                            if not bias_ok or not local_ok:
                                log.info(
                                    f"[STRIKE] Entry blocked — reversal strike on {ps_dir}: "
                                    f"daily_bias={daily_bias} (need {ps_dir}), "
                                    f"local={ps['local']} ({'ok' if local_ok else 'RC blocked'})"
                                )
                                confirmed = False
                                state["pending_signal"] = None
                    if confirmed:
                        log.info(f"ENTRY CONFIRMED: {ps_dir}_{ps_env} held for 2 bars "
                                 f"→ entering now  [{PROFILES[ps_pk].name}]")
                        do_entry = True
                        entry_direction   = ps_dir
                        entry_environment = ps_env
                        entry_local       = ps["local"]
                        entry_profile_key = ps_pk
                        state["pending_signal"] = None
                    else:
                        log.info(f"  -> Pending signal {ps_dir}_{ps_env} NOT confirmed "
                                 f"(now: {direction}_{environment}) — discarded")
                        if not risk_ok:
                            log.info(f"  -> Risk control: {risk_reason}")
                        state["pending_signal"] = None
                        # Fall through: check if current bar is a fresh signal
                        ps = None

                if ps is None and not do_entry:
                    # No pending signal — check if current bar has a valid signal to pend
                    log.info(f"Entry check: dir={direction} env={environment} loc={local_lbl} "
                             f"bias={daily_bias} sess={session} profile={profile_key} open={n_open}/{MAX_OPEN}")
                    if not entry_allowed(direction, environment, session, hour):
                        reasons = []
                        if direction not in ("Bull","Bear"):       reasons.append("dir=Neutral")
                        if environment in ("Range","Compression"): reasons.append(f"env={environment}")
                        if direction=="Bear" and session=="NY":    reasons.append("Bear+NY blocked")
                        if hour in (23, 0):                        reasons.append("midnight buffer")
                        log.info(f"  -> No signal: {', '.join(reasons)}")
                    elif profile_key is None:
                        log.info(f"  -> No signal: no profile for {combo_key}")
                    else:
                        with _state_lock:
                            risk_ok, risk_reason = risk_entry_allowed(
                                state, direction, last, now_utc
                            )
                        if not risk_ok:
                            log.info(
                                f"  -> No signal: risk control blocked "
                                f"{direction}: {risk_reason}"
                            )
                        else:
                            log.info(f"  -> SIGNAL PENDING (waiting 1 bar to confirm): "
                                     f"{combo_key}  [{PROFILES[profile_key].name}]")
                            state["pending_signal"] = {
                                "direction":   direction,
                                "environment": environment,
                                "local":       local_lbl,
                                "profile_key": profile_key,
                                "bar_time":    bar_time.isoformat(),
                                "dir_score":   float(last["dir_score"]) if pd.notna(last.get("dir_score")) else 0.0,
                            }

            if do_entry:
                # Use the signal's direction/profile (may differ from current bar's)
                direction   = entry_direction
                environment = entry_environment
                local_lbl   = entry_local
                profile_key = entry_profile_key
                combo_key   = f"{direction}_{environment}_{local_lbl}"
                trade_key   = f"{combo_key}_{bar_time.strftime('%H%M')}"

                log.info(f"ENTRY CONFIRMED: {combo_key}  [{PROFILES[profile_key].name}]")

                bid, offer, audit_id = get_current_price(paper=paper)
                log.info(f"  [ENTRY] Live price — Bid={bid:.5f}  Offer={offer:.5f}  AuditId={audit_id}")
                api_dir = "buy" if direction == "Bull" else "sell"
                order_id, entry_price = open_order(api_dir, bid, offer, audit_id, paper=paper)

                if order_id:
                    atr      = float(last["atr"])      if pd.notna(last["atr"])      and last["atr"]      > 0 else 0.0006
                    atr_rank = float(last["atr_rank"]) if pd.notna(last["atr_rank"]) else 50.0
                    p        = PROFILES[profile_key]
                    sl_mult  = get_adaptive_sl_mult(profile_key, atr_rank)
                    sl_dist  = atr * sl_mult
                    tp_dist  = p.tp_pips / 10000
                    log.info(f"  [ENTRY] ATR={atr*10000:.1f}p  ATRrank={atr_rank:.0f}%  "
                             f"SL_mult={sl_mult:.1f}x  SL={sl_dist*10000:.1f}p")

                    state["open_trades"][trade_key] = {
                        "order_id":     order_id,
                        "direction":    direction,
                        "environment":  environment,
                        "local":        local_lbl,
                        "profile_key":  profile_key,
                        "entry_price":  entry_price,
                        "entry_time":   bar_time.isoformat(),
                        "atr_entry":    atr,
                        "atr_rank":     atr_rank,
                        "sl_mult":      sl_mult,
                        "sl_price":     entry_price - sl_dist if direction == "Bull" else entry_price + sl_dist,
                        "tp_price":     entry_price + tp_dist if direction == "Bull" else entry_price - tp_dist,
                        "trail_dist":   p.trail_dist / 10000,
                        "trail_trigger":p.trail_trigger / 10000,
                        "trail_active": False,
                        "trail_stop":   0.0,
                        "best_price":   entry_price,
                        "bars_held":    0,
                    }

                    log_trade({
                        "type": "OPEN",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "combo_key": trade_key,
                        "direction": direction,
                        "environment": environment,
                        "local": local_lbl,
                        "profile": profile_key,
                        "entry_price": round(entry_price, 5),
                        "sl_price": round(state["open_trades"][trade_key]["sl_price"], 5),
                        "tp_price": round(state["open_trades"][trade_key]["tp_price"], 5),
                        "atr_pips": round(atr * 10000, 2),
                        "atr_rank": round(atr_rank, 1),
                        "sl_mult": sl_mult,
                        "daily_bias": daily_bias,
                        "session": session,
                        "order_id": order_id,
                        "paper": paper,
                        "dir_score": round(float(last["dir_score"]), 1) if pd.notna(last.get("dir_score")) else None,
                        "efficiency": round(float(last["efficiency"]), 1) if pd.notna(last.get("efficiency")) else None,
                        "dir_conf": round(float(last["direction_conf"]), 0) if pd.notna(last.get("direction_conf")) else None,
                    })

            # ── 5. Save state ──────────────────────────────────────────────
            log.debug(f"STEP 5: Saving state ({len(state['open_trades'])} open trades) to {STATE_FILE}")
            save_state(state)

            # ── 6. Dashboard ───────────────────────────────────────────────
            print_dashboard(classified, state["open_trades"], daily_bias, session, bar_time,
                            bounce_pending=state.get("bounce_pending", {}),
                            state=state)

            # ── 7. Sleep until next 10-min bar, refreshing price every 5s ──
            now = datetime.now(timezone.utc)
            total_seconds_in_hour = now.minute * 60 + now.second
            secs_into_block  = total_seconds_in_hour % (BAR_MINUTES * 60)
            secs_to_next_bar = (BAR_MINUTES * 60) - secs_into_block
            wait_seconds     = secs_to_next_bar + 30
            next_wake = datetime.fromtimestamp(now.timestamp() + wait_seconds, tz=timezone.utc)
            log.info(f"Sleeping {wait_seconds}s — next bar at {next_wake.strftime('%H:%M:%S')} UTC")

            # Tick display while waiting — update every 5s showing tick monitor status
            deadline = time.time() + wait_seconds
            while time.time() < deadline:
                tick = ls_get_tick()
                remaining = int(deadline - time.time())
                if tick:
                    bid_t   = float(tick["Bid"])
                    offer_t = float(tick["Offer"])
                    spread  = (offer_t - bid_t) * 10000
                    n_open  = len(state["open_trades"])
                    n_bounce = len(state.get("bounce_pending", {}))

                    # Show trail stop level for open trades (if armed)
                    trail_info = ""
                    for t in state["open_trades"].values():
                        if t.get("trail_active") and t.get("trail_stop"):
                            pips_from_stop = ((bid_t - t["trail_stop"]) * 10000
                                              if t["direction"] == "Bull"
                                              else (t["trail_stop"] - offer_t) * 10000)
                            trail_info = f"  Trail={t['trail_stop']:.5f}({pips_from_stop:+.1f}p)"
                        elif n_open:
                            for t2 in state["open_trades"].values():
                                trig = t2["trail_trigger"]
                                ep   = t2["entry_price"]
                                cur  = bid_t if t2["direction"]=="Bull" else offer_t
                                pips_to_arm = ((ep + trig - cur) * 10000
                                               if t2["direction"]=="Bull"
                                               else (cur - ep + trig) * 10000)
                                trail_info = f"  Trail=off(arm in {pips_to_arm:.1f}p)"

                    # Show staleness if tick monitor hasn't seen a new price recently
                    if (_tick_monitor and _tick_monitor._thread and
                            _tick_monitor._thread.is_alive()):
                        last_ts = _tick_monitor._last_tick_ts
                        if last_ts is None:
                            tm_status = "TICK-MON:WAITING"
                        else:
                            stale = (datetime.now(timezone.utc) - last_ts).total_seconds()
                            if stale >= 90:
                                tm_status = f"TICK-MON:RECONNECTING({stale:.0f}s)"
                            elif stale >= 60:
                                tm_status = f"TICK-MON:STALE({stale:.0f}s)"
                            else:
                                tm_status = "TICK-MON:ON"
                    else:
                        tm_status = "TICK-MON:OFF"
                    bounce_str = f"  Bounce={n_bounce}" if n_bounce else ""
                    print(f"\r  [{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC]  "
                          f"Bid={bid_t:.5f}  Ask={offer_t:.5f}  Spread={spread:.1f}p  "
                          f"Open={n_open}{bounce_str}{trail_info}  "
                          f"{tm_status}  Next bar in {remaining}s   ",
                          end="", flush=True)
                time.sleep(5)
            print()  # newline before next dashboard draw

        except KeyboardInterrupt:
            log.info("Shutdown requested. Saving state...")
            save_state(state)
            print("\n[Stopped by user]")
            break

        except Exception as e:
            log.error(f"CYCLE {cycle} FAILED with exception: {e}", exc_info=True)
            log.error(f"Full traceback above. Retrying in 60 seconds...")
            time.sleep(60)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", action="store_true",
                        help="Paper trade mode: log signals but do not place real orders")
    args = parser.parse_args()

    if args.paper:
        print("\n" + "="*50)
        print("  PAPER TRADE MODE — no real orders will be placed")
        print("="*50 + "\n")
    else:
        print("\n" + "="*50)
        print("  LIVE TRADE MODE — REAL ORDERS WILL BE PLACED")
        print("  Ctrl+C to stop cleanly")
        print("="*50 + "\n")
        confirm = input("  Type YES to confirm: ")
        if confirm.strip() != "YES":
            print("Aborted.")
            return

    run(paper=args.paper)


if __name__ == "__main__":
    main()
