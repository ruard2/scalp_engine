#!/usr/bin/env python3
"""
backtest_v6.py — Replay the live_engine logic against historical 5-min OHLC.

Imports the REAL functions from live_engine.py (check_exit, entry_allowed,
PROFILES, get_profile_key) and the v6 classifier, so what is tested is
exactly what runs live. 5-min bars are resampled to the 10-min bars the
engine trades on.

Simulation model (mirrors the live loop):
  - Bar i closes -> classify -> exits checked on NEXT bar's high/low
  - Entry on signal: fill at next bar open + half spread
  - One position at a time (MAX_OPEN=1), same midnight buffer, same rules
  - Cost: SPREAD_PIPS round trip per trade

Usage:  python backtest_v6.py [--days N]   (default: full history)
"""
import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

# Real live-engine logic — no copies
from live_engine import (
    PROFILES, MIN_TRAIL_LOCK_PIPS, get_profile_key, entry_allowed,
    check_exit, get_session_label, get_daily_bias, classify_bars,
)

DATA_FILE   = Path(r"C:\Users\Ruard\testingversion2\5_min_data\fetched_data_eurusd_401697501.csv")
SPREAD_PIPS = 1.0          # round-trip cost in pips (≈ $0.10 per 1000)
QUANTITY    = 1000
MAX_OPEN    = 1


def load_10min_bars() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    ts = df["BarDate"].str.extract(r"(\d+)")[0].astype("int64")
    df["datetime"] = pd.to_datetime(ts, unit="ms", utc=True)
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                            "Close": "close", "Volume": "volume"})
    df = df[["datetime", "open", "high", "low", "close", "volume"]]
    df = df.set_index("datetime").sort_index()
    r = df.resample("10min").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"), volume=("volume", "sum")
    ).dropna(subset=["open"]).reset_index()
    r["date_utc"] = r["datetime"].dt.date.astype(str)
    return r


def run_backtest(df: pd.DataFrame) -> pd.DataFrame:
    print(f"Classifying {len(df)} 10-min bars "
          f"({df['datetime'].iloc[0]:%Y-%m-%d} -> {df['datetime'].iloc[-1]:%Y-%m-%d})...")
    cl = classify_bars(df)

    # Pre-compute daily bias per day (same formula as live, evaluated on day-so-far
    # would be slow; bias is info-only in entry rules so we skip it)
    trades = []
    open_trades: dict = {}

    half_spread = (SPREAD_PIPS / 2) / 10000

    for i in range(200, len(cl) - 1):
        bar  = cl.iloc[i]
        nxt  = cl.iloc[i + 1]
        hour = bar["datetime"].hour
        session = get_session_label(hour)

        # ── exits: evaluate current bar against open trades (live: next cycle) ──
        for key in list(open_trades):
            t = open_trades[key]
            t["bars_held"] += 1
            reason = check_exit(t, bar)
            if reason:
                # exit price approximation per reason
                if reason == "TP":
                    px = t["tp_price"]
                elif reason in ("SL",):
                    px = t["sl_price"]
                elif reason == "TrailSL":
                    px = t["trail_stop"] if t["trail_active"] and t["trail_stop"] > 0 else t["sl_price"]
                else:  # DirectionFlip / EnvRange / MaxBars -> close at bar close
                    px = float(bar["close"])
                if t["direction"] == "Bull":
                    px -= half_spread
                    pnl = (px - t["entry_price"]) * 10000
                else:
                    px += half_spread
                    pnl = (t["entry_price"] - px) * 10000
                trades.append({
                    "entry_time": t["entry_time"], "exit_time": bar["datetime"],
                    "direction": t["direction"], "profile": t["profile_key"],
                    "combo": t["combo"], "session": t["session"],
                    "entry": t["entry_price"], "exit": round(px, 5),
                    "pnl_pips": round(pnl, 2), "bars": t["bars_held"],
                    "reason": reason,
                })
                del open_trades[key]

        # ── entries: signal on this bar, fill at next bar open ──
        d, e, loc = str(bar["direction"]), str(bar["environment"]), str(bar["local"])
        pk = get_profile_key(d, e, loc)
        if (len(open_trades) < MAX_OPEN
                and entry_allowed(d, e, session, hour)
                and pk is not None):
            entry_px = float(nxt["open"]) + (half_spread if d == "Bull" else -half_spread)
            atr = float(bar["atr"]) if pd.notna(bar["atr"]) and bar["atr"] > 0 else 0.0006
            p = PROFILES[pk]
            sl_dist, tp_dist = atr * p.sl_atr_mult, p.tp_pips / 10000
            key = f"{d}_{e}_{loc}_{bar['datetime']:%m%d%H%M}"
            open_trades[key] = {
                "combo": f"{d}_{e}_{loc}", "direction": d, "profile_key": pk,
                "session": session, "entry_price": entry_px,
                "entry_time": nxt["datetime"], "atr_entry": atr,
                "sl_price": entry_px - sl_dist if d == "Bull" else entry_px + sl_dist,
                "tp_price": entry_px + tp_dist if d == "Bull" else entry_px - tp_dist,
                "trail_dist": p.trail_dist / 10000, "trail_trigger": p.trail_trigger / 10000,
                "trail_active": False, "trail_stop": 0.0,
                "best_price": entry_px, "bars_held": 0,
            }

    return pd.DataFrame(trades)


def report(tr: pd.DataFrame):
    if tr.empty:
        print("No trades.")
        return
    tr["pnl_eur"] = tr["pnl_pips"] * 0.10   # 1 pip = $0.10 at qty 1000
    wins = tr[tr["pnl_pips"] > 0]
    total = tr["pnl_pips"].sum()
    days = (tr["exit_time"].max() - tr["entry_time"].min()).days or 1

    print("\n" + "=" * 64)
    print("  V6 LIVE-ENGINE BACKTEST RESULT")
    print("=" * 64)
    print(f"  Period      : {tr['entry_time'].min():%Y-%m-%d} -> {tr['exit_time'].max():%Y-%m-%d}  ({days} days)")
    print(f"  Trades      : {len(tr)}  ({len(tr)/days*7:.1f}/week)")
    print(f"  Win rate    : {len(wins)/len(tr)*100:.1f}%")
    print(f"  Total P&L   : {total:+.1f} pips  =  ${total*0.10:+.2f} at qty {QUANTITY}")
    print(f"  Avg/trade   : {tr['pnl_pips'].mean():+.2f} pips")
    print(f"  Best/Worst  : {tr['pnl_pips'].max():+.1f} / {tr['pnl_pips'].min():+.1f} pips")
    print(f"  Max drawdown: {(tr['pnl_pips'].cumsum() - tr['pnl_pips'].cumsum().cummax()).min():.1f} pips")
    print()
    print("  By exit reason:")
    g = tr.groupby("reason")["pnl_pips"].agg(["count", "sum", "mean"])
    for r, row in g.iterrows():
        print(f"    {r:<14} n={int(row['count']):>4}  sum={row['sum']:>+8.1f}p  avg={row['mean']:>+6.2f}p")
    print()
    print("  By profile:")
    g = tr.groupby("profile")["pnl_pips"].agg(["count", "sum", "mean"])
    for r, row in g.iterrows():
        print(f"    {r:<16} n={int(row['count']):>4}  sum={row['sum']:>+8.1f}p  avg={row['mean']:>+6.2f}p")
    print()
    print("  By session:")
    g = tr.groupby("session")["pnl_pips"].agg(["count", "sum", "mean"])
    for r, row in g.iterrows():
        print(f"    {r:<10} n={int(row['count']):>4}  sum={row['sum']:>+8.1f}p  avg={row['mean']:>+6.2f}p")
    print()
    print("  By combo (top 12 by |sum|):")
    g = tr.groupby("combo")["pnl_pips"].agg(["count", "sum", "mean"])
    for r, row in g.reindex(g["sum"].abs().sort_values(ascending=False).index).head(12).iterrows():
        print(f"    {r:<36} n={int(row['count']):>4}  sum={row['sum']:>+8.1f}p  avg={row['mean']:>+6.2f}p")
    print("=" * 64)

    out = Path(__file__).parent / "backtest_trades.csv"
    tr.to_csv(out, index=False)
    print(f"  Trades saved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=0, help="only last N days")
    args = ap.parse_args()

    df = load_10min_bars()
    if args.days:
        cutoff = df["datetime"].max() - pd.Timedelta(days=args.days)
        df = df[df["datetime"] >= cutoff].reset_index(drop=True)
    tr = run_backtest(df)
    report(tr)


if __name__ == "__main__":
    main()
