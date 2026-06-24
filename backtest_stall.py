#!/usr/bin/env python3
"""
backtest_stall.py — Test the "stall gate" rule against historical data.

Rule: if the last N closed trades in the same direction exited via TrailSL
at < STALL_PIPS profit, restrict next entry in that direction to
Impulse/None locals only (block Pullback and ReversalCandidate).

Tests three variants:
  baseline  — no stall gate
  stall-1   — block Pullback/RC after 1 consecutive tiny TrailSL
  stall-2   — block Pullback/RC after 2 consecutive tiny TrailSL

Includes adaptive SL (ATR rank >=75% -> 4.0x) and 1-bar entry delay,
matching the current live engine exactly.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from live_engine import (
    PROFILES, MIN_TRAIL_LOCK_PIPS, get_profile_key, entry_allowed,
    check_exit, get_session_label, classify_bars, get_adaptive_sl_mult,
)

DATA_FILE   = Path(r"C:\Users\Ruard\testingversion2\5_min_data\fetched_data_eurusd_401697501.csv")
SPREAD_PIPS = 1.0
HALF_SPREAD = (SPREAD_PIPS / 2) / 10000
MAX_OPEN    = 1
STALL_PIPS  = 2.0   # TrailSL exit below this counts as "tiny" / stalling


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


def run(cl: pd.DataFrame, stall_threshold: int) -> pd.DataFrame:
    """Run a single backtest variant.

    stall_threshold: 0=disabled, 1=block after 1 tiny trail, 2=after 2.
    """
    trades = []
    open_trades: dict = {}
    pending_signal = None

    # Track consecutive tiny TrailSL count per direction
    stall_count = {"Bull": 0, "Bear": 0}

    for i in range(200, len(cl) - 1):
        bar = cl.iloc[i]
        nxt = cl.iloc[i + 1]
        hour    = bar["datetime"].hour
        session = get_session_label(hour)

        # ── exits ────────────────────────────────────────────────────────────
        for key in list(open_trades):
            t = open_trades[key]
            t["bars_held"] += 1
            reason = check_exit(t, bar)
            if not reason:
                continue

            if reason == "TP":
                px = t["tp_price"]
            elif reason == "SL":
                px = t["sl_price"]
            elif reason == "TrailSL":
                px = t["trail_stop"] if t["trail_active"] and t["trail_stop"] > 0 else t["sl_price"]
            else:
                px = float(bar["close"])

            d = t["direction"]
            if d == "Bull":
                px  -= HALF_SPREAD
                pnl  = (px - t["entry_price"]) * 10000
            else:
                px  += HALF_SPREAD
                pnl  = (t["entry_price"] - px) * 10000

            # Update stall counter
            if stall_threshold > 0:
                if reason == "TrailSL" and pnl < STALL_PIPS:
                    stall_count[d] += 1
                else:
                    stall_count[d] = 0   # any other exit (TP, SL, flip) resets

            trades.append({
                "entry_time":  t["entry_time"],
                "exit_time":   bar["datetime"],
                "direction":   d,
                "profile":     t["profile_key"],
                "local":       t["local"],
                "combo":       t["combo"],
                "session":     t["session"],
                "entry":       t["entry_price"],
                "exit":        round(px, 5),
                "pnl_pips":    round(pnl, 2),
                "bars":        t["bars_held"],
                "reason":      reason,
                "stall_at_entry": t.get("stall_at_entry", 0),
            })
            del open_trades[key]

        # ── 1-bar entry delay (pending_signal pattern) ────────────────────
        d   = str(bar["direction"])
        e   = str(bar["environment"])
        loc = str(bar["local"])
        pk  = get_profile_key(d, e, loc)

        do_entry = False
        entry_d = entry_e = entry_loc = entry_pk = None

        if len(open_trades) >= MAX_OPEN:
            pending_signal = None
        else:
            if pending_signal is not None:
                ps = pending_signal
                confirmed = (
                    d == ps["d"]
                    and e not in ("Range", "Compression")
                    and entry_allowed(d, e, session, hour)
                    and pk is not None
                )
                # Stall gate check at confirmation
                if confirmed and stall_threshold > 0:
                    if stall_count[ps["d"]] >= stall_threshold:
                        if ps["loc"] in ("Pullback", "ReversalCandidate"):
                            confirmed = False  # blocked by stall gate

                if confirmed:
                    do_entry  = True
                    entry_d   = ps["d"]
                    entry_e   = ps["e"]
                    entry_loc = ps["loc"]
                    entry_pk  = ps["pk"]
                pending_signal = None

            if not do_entry:
                if (entry_allowed(d, e, session, hour) and pk is not None):
                    pending_signal = {"d": d, "e": e, "loc": loc, "pk": pk}

        if do_entry:
            entry_px = float(nxt["open"]) + (HALF_SPREAD if entry_d == "Bull" else -HALF_SPREAD)
            atr      = float(bar["atr"]) if pd.notna(bar["atr"]) and bar["atr"] > 0 else 0.0006
            atr_rank = float(bar["atr_rank"]) if pd.notna(bar["atr_rank"]) else 50.0
            p        = PROFILES[entry_pk]
            sl_mult  = get_adaptive_sl_mult(entry_pk, atr_rank)
            sl_dist  = atr * sl_mult
            tp_dist  = p.tp_pips / 10000
            key      = f"{entry_d}_{entry_e}_{entry_loc}_{bar['datetime']:%m%d%H%M}"
            open_trades[key] = {
                "combo":        f"{entry_d}_{entry_e}_{entry_loc}",
                "direction":    entry_d,
                "local":        entry_loc,
                "profile_key":  entry_pk,
                "session":      session,
                "entry_price":  entry_px,
                "entry_time":   nxt["datetime"],
                "atr_entry":    atr,
                "sl_price":     entry_px - sl_dist if entry_d == "Bull" else entry_px + sl_dist,
                "tp_price":     entry_px + tp_dist if entry_d == "Bull" else entry_px - tp_dist,
                "trail_dist":   p.trail_dist / 10000,
                "trail_trigger":p.trail_trigger / 10000,
                "trail_active": False,
                "trail_stop":   0.0,
                "best_price":   entry_px,
                "bars_held":    0,
                "stall_at_entry": stall_count.get(entry_d, 0),
            }

    return pd.DataFrame(trades)


def report(label: str, tr: pd.DataFrame):
    if tr.empty:
        print(f"\n{label}: no trades.")
        return
    wins = tr[tr["pnl_pips"] > 0]
    total = tr["pnl_pips"].sum()
    days  = (tr["exit_time"].max() - tr["entry_time"].min()).days or 1
    dd    = (tr["pnl_pips"].cumsum() - tr["pnl_pips"].cumsum().cummax()).min()

    print(f"\n{'='*64}")
    print(f"  {label}")
    print(f"{'='*64}")
    print(f"  Trades      : {len(tr)}  ({len(tr)/days*7:.1f}/week)")
    print(f"  Win rate    : {len(wins)/len(tr)*100:.1f}%")
    print(f"  Total P&L   : {total:+.1f}p")
    print(f"  Avg/trade   : {tr['pnl_pips'].mean():+.2f}p")
    print(f"  Max drawdown: {dd:.1f}p")
    print()
    print("  By local (Pullback / RC most relevant):")
    g = tr.groupby("local")["pnl_pips"].agg(["count", "sum", "mean"])
    for loc, row in g.iterrows():
        print(f"    {loc:<22} n={int(row['count']):>4}  sum={row['sum']:>+8.1f}p  avg={row['mean']:>+6.2f}p")
    print()
    print("  By exit reason:")
    g = tr.groupby("reason")["pnl_pips"].agg(["count", "sum", "mean"])
    for r, row in g.iterrows():
        print(f"    {r:<16} n={int(row['count']):>4}  sum={row['sum']:>+8.1f}p  avg={row['mean']:>+6.2f}p")

    # Trades that were taken while stall was active
    if "stall_at_entry" in tr.columns:
        stalled = tr[tr["stall_at_entry"] > 0]
        if not stalled.empty:
            print()
            print(f"  Trades entered WHILE stall active (n={len(stalled)}):")
            g2 = stalled.groupby("local")["pnl_pips"].agg(["count", "sum", "mean"])
            for loc, row in g2.iterrows():
                print(f"    {loc:<22} n={int(row['count']):>4}  sum={row['sum']:>+8.1f}p  avg={row['mean']:>+6.2f}p")


def main():
    print("Loading and classifying bars...")
    df = load_10min_bars()
    cl = classify_bars(df)
    print(f"  {len(cl)} bars  ({cl['datetime'].iloc[0]:%Y-%m-%d} -> {cl['datetime'].iloc[-1]:%Y-%m-%d})")

    tr0 = run(cl, stall_threshold=0)
    tr1 = run(cl, stall_threshold=1)
    tr2 = run(cl, stall_threshold=2)

    report("BASELINE  (no stall gate)", tr0)
    report("STALL-1   (block Pullback/RC after 1 tiny TrailSL)", tr1)
    report("STALL-2   (block Pullback/RC after 2 tiny TrailSL)", tr2)

    # Summary comparison
    print(f"\n{'='*64}")
    print("  SUMMARY COMPARISON")
    print(f"{'='*64}")
    print(f"  {'Variant':<44} {'Trades':>6} {'Total':>8} {'Avg':>7} {'Win%':>6} {'DD':>8}")
    for label, tr in [("Baseline", tr0), ("Stall-1", tr1), ("Stall-2", tr2)]:
        wins = tr[tr["pnl_pips"] > 0]
        dd   = (tr["pnl_pips"].cumsum() - tr["pnl_pips"].cumsum().cummax()).min()
        print(f"  {label:<44} {len(tr):>6} {tr['pnl_pips'].sum():>+8.1f}p"
              f" {tr['pnl_pips'].mean():>+6.2f}p {len(wins)/len(tr)*100:>5.1f}%"
              f" {dd:>7.1f}p")

    # Show what stall-1 blocked vs kept (Pullback and RC breakdown)
    print(f"\n  Blocked trades (in Baseline but not Stall-1):")
    blocked = tr0[~tr0["entry_time"].isin(tr1["entry_time"])]
    if not blocked.empty:
        g = blocked.groupby("local")["pnl_pips"].agg(["count", "sum", "mean"])
        for loc, row in g.iterrows():
            print(f"    {loc:<22} n={int(row['count']):>4}  sum={row['sum']:>+8.1f}p  avg={row['mean']:>+6.2f}p")
    else:
        print("    (none)")


if __name__ == "__main__":
    main()
