#!/usr/bin/env python3
"""
backtest_pullback_filter.py — Find what signal identifies bad Pullback/RC entries.

Phase 1: Run baseline, record every Pullback/RC trade with its entry-bar
         conditions (efficiency, atr_rank, dir_score, session, direction_conf).
         Show performance split by each dimension to find which correlates
         with losing Pullback/RC trades.

Phase 2: Test the top candidate filters as entry gates and compare P&L.

Includes adaptive SL + 1-bar delay, matching the live engine exactly.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from live_engine import (
    PROFILES, get_profile_key, entry_allowed, check_exit,
    get_session_label, classify_bars, get_adaptive_sl_mult,
)

DATA_FILE   = Path(r"C:\Users\Ruard\testingversion2\5_min_data\fetched_data_eurusd_401697501.csv")
SPREAD_PIPS = 1.0
HALF_SPREAD = (SPREAD_PIPS / 2) / 10000
MAX_OPEN    = 1
LOCALS_OF_INTEREST = {"Pullback", "ReversalCandidate"}


def load_10min_bars() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    ts = df["BarDate"].str.extract(r"(\d+)")[0].astype("int64")
    df["datetime"] = pd.to_datetime(ts, unit="ms", utc=True)
    df = df.rename(columns={"Open": "open", "High": "high",
                             "Low": "low", "Close": "close", "Volume": "volume"})
    df = df[["datetime", "open", "high", "low", "close", "volume"]]
    df = df.set_index("datetime").sort_index()
    r = df.resample("10min").agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"), close=("close","last"), volume=("volume","sum")
    ).dropna(subset=["open"]).reset_index()
    r["date_utc"] = r["datetime"].dt.date.astype(str)
    return r


def run(cl: pd.DataFrame, block_fn=None) -> pd.DataFrame:
    """
    block_fn(bar_series, local) -> bool
    Returns True if this Pullback/RC entry should be blocked.
    None = no blocking (baseline).
    """
    trades = []
    open_trades: dict = {}
    pending_signal = None

    for i in range(200, len(cl) - 1):
        bar = cl.iloc[i]
        nxt = cl.iloc[i + 1]
        hour    = bar["datetime"].hour
        session = get_session_label(hour)

        # exits
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
                px -= HALF_SPREAD; pnl = (px - t["entry_price"]) * 10000
            else:
                px += HALF_SPREAD; pnl = (t["entry_price"] - px) * 10000
            trades.append({
                "entry_time":   t["entry_time"],
                "exit_time":    bar["datetime"],
                "direction":    d,
                "local":        t["local"],
                "session":      t["session"],
                "pnl_pips":     round(pnl, 2),
                "bars":         t["bars_held"],
                "reason":       reason,
                "efficiency":   t["eff_at_entry"],
                "atr_rank":     t["atr_rank_at_entry"],
                "dir_score":    t["dir_score_at_entry"],
                "dir_conf":     t["dir_conf_at_entry"],
                "hour":         t["hour_at_entry"],
            })
            del open_trades[key]

        # 1-bar entry delay
        d   = str(bar["direction"])
        e   = str(bar["environment"])
        loc = str(bar["local"])
        pk  = get_profile_key(d, e, loc)

        do_entry = False
        entry_d = entry_e = entry_loc = entry_pk = None
        entry_bar = None

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
                if confirmed and block_fn is not None:
                    if ps["loc"] in LOCALS_OF_INTEREST:
                        if block_fn(ps["bar"], ps["loc"]):
                            confirmed = False
                if confirmed:
                    do_entry  = True
                    entry_d   = ps["d"]
                    entry_e   = ps["e"]
                    entry_loc = ps["loc"]
                    entry_pk  = ps["pk"]
                    entry_bar = ps["bar"]
                pending_signal = None

            if not do_entry:
                if entry_allowed(d, e, session, hour) and pk is not None:
                    pending_signal = {"d": d, "e": e, "loc": loc, "pk": pk,
                                      "bar": bar}

        if do_entry:
            entry_px = float(nxt["open"]) + (HALF_SPREAD if entry_d == "Bull" else -HALF_SPREAD)
            b        = entry_bar if entry_bar is not None else bar
            atr      = float(b["atr"])      if pd.notna(b["atr"])      and b["atr"]      > 0 else 0.0006
            atr_rank = float(b["atr_rank"]) if pd.notna(b["atr_rank"]) else 50.0
            p        = PROFILES[entry_pk]
            sl_mult  = get_adaptive_sl_mult(entry_pk, atr_rank)
            sl_dist  = atr * sl_mult
            tp_dist  = p.tp_pips / 10000
            key      = f"{entry_d}_{entry_e}_{entry_loc}_{bar['datetime']:%m%d%H%M}"
            open_trades[key] = {
                "direction":        entry_d,
                "local":            entry_loc,
                "profile_key":      entry_pk,
                "session":          session,
                "entry_price":      entry_px,
                "entry_time":       nxt["datetime"],
                "atr_entry":        atr,
                "sl_price":         entry_px - sl_dist if entry_d=="Bull" else entry_px + sl_dist,
                "tp_price":         entry_px + tp_dist if entry_d=="Bull" else entry_px - tp_dist,
                "trail_dist":       p.trail_dist / 10000,
                "trail_trigger":    p.trail_trigger / 10000,
                "trail_active":     False,
                "trail_stop":       0.0,
                "best_price":       entry_px,
                "bars_held":        0,
                "eff_at_entry":     float(b.get("efficiency", 0)) if pd.notna(b.get("efficiency", 0)) else 0.0,
                "atr_rank_at_entry":atr_rank,
                "dir_score_at_entry": float(b.get("dir_score", 0)) if pd.notna(b.get("dir_score", 0)) else 0.0,
                "dir_conf_at_entry":  float(b.get("direction_conf", 0)) if pd.notna(b.get("direction_conf", 0)) else 0.0,
                "hour_at_entry":    hour,
            }

    return pd.DataFrame(trades)


def analyse_pullback_rc(tr: pd.DataFrame):
    pr = tr[tr["local"].isin(LOCALS_OF_INTEREST)].copy()
    if pr.empty:
        print("No Pullback/RC trades found.")
        return

    print(f"\n{'='*64}")
    print(f"  PULLBACK + RC ANALYSIS  (n={len(pr)}  total={pr['pnl_pips'].sum():+.1f}p  avg={pr['pnl_pips'].mean():+.2f}p)")
    print(f"{'='*64}")

    def split(col, bins, labels):
        pr[col+"_bin"] = pd.cut(pr[col], bins=bins, labels=labels)
        g = pr.groupby(col+"_bin", observed=True)["pnl_pips"].agg(["count","sum","mean"])
        print(f"\n  By {col}:")
        for lbl, row in g.iterrows():
            bar = "#" * int(abs(row["mean"]) * 4)
            sign = "+" if row["mean"] >= 0 else "-"
            print(f"    {str(lbl):<14} n={int(row['count']):>4}  "
                  f"sum={row['sum']:>+8.1f}p  avg={row['mean']:>+6.2f}p  {sign}{bar}")

    split("efficiency",
          bins=[-999, 20, 35, 50, 65, 999],
          labels=["<20","20-35","35-50","50-65",">65"])

    split("atr_rank",
          bins=[-1, 30, 50, 70, 85, 101],
          labels=["<30","30-50","50-70","70-85",">85"])

    split("dir_score",
          bins=[-999, -50, -20, 0, 20, 50, 999],
          labels=["<-50","-50:-20","-20:0","0:20","20:50",">50"])

    split("dir_conf",
          bins=[-1, 30, 50, 70, 90, 101],
          labels=["<30","30-50","50-70","70-90",">90"])

    pr["hour_bin"] = pd.cut(pr["hour"],
                            bins=[-1,7,12,16,21,24],
                            labels=["Other","London","Overlap","NY","Late"])
    g = pr.groupby("hour_bin", observed=True)["pnl_pips"].agg(["count","sum","mean"])
    print(f"\n  By session (entry hour):")
    for lbl, row in g.iterrows():
        bar = "#" * int(abs(row["mean"]) * 4)
        sign = "+" if row["mean"] >= 0 else "-"
        print(f"    {str(lbl):<14} n={int(row['count']):>4}  "
              f"sum={row['sum']:>+8.1f}p  avg={row['mean']:>+6.2f}p  {sign}{bar}")

    print(f"\n  By local:")
    g = pr.groupby("local")["pnl_pips"].agg(["count","sum","mean"])
    for lbl, row in g.iterrows():
        print(f"    {str(lbl):<22} n={int(row['count']):>4}  sum={row['sum']:>+8.1f}p  avg={row['mean']:>+6.2f}p")


def summary_line(label, tr, baseline_total):
    wins  = tr[tr["pnl_pips"] > 0]
    total = tr["pnl_pips"].sum()
    dd    = (tr["pnl_pips"].cumsum() - tr["pnl_pips"].cumsum().cummax()).min()
    diff  = total - baseline_total
    print(f"  {label:<42} {len(tr):>5} {total:>+8.1f}p  {tr['pnl_pips'].mean():>+6.2f}p  "
          f"{len(wins)/len(tr)*100:>5.1f}%  {dd:>7.1f}p  {diff:>+8.1f}p")


def main():
    print("Loading and classifying bars...")
    df = load_10min_bars()
    cl = classify_bars(df)
    print(f"  {len(cl)} bars  "
          f"({cl['datetime'].iloc[0]:%Y-%m-%d} -> {cl['datetime'].iloc[-1]:%Y-%m-%d})")

    # Phase 1: baseline + Pullback/RC analysis
    print("\nRunning baseline...")
    tr_base = run(cl)
    analyse_pullback_rc(tr_base)

    baseline_total = tr_base["pnl_pips"].sum()

    # Phase 2: test candidate filters
    print(f"\n{'='*64}")
    print("  FILTER CANDIDATES — block Pullback/RC when condition met")
    print(f"{'='*64}")
    print(f"  {'Filter':<42} {'N':>5} {'Total':>9}  {'Avg':>7}  {'Win%':>6}  {'DD':>8}  {'vs base':>9}")

    candidates = [
        ("Baseline (no filter)",
         None),
        ("Efficiency < 20",
         lambda b, l: float(b.get("efficiency", 99)) < 20),
        ("Efficiency < 30",
         lambda b, l: float(b.get("efficiency", 99)) < 30),
        ("Efficiency < 40",
         lambda b, l: float(b.get("efficiency", 99)) < 40),
        ("ATR rank < 40",
         lambda b, l: float(b.get("atr_rank", 99)) < 40),
        ("ATR rank < 50",
         lambda b, l: float(b.get("atr_rank", 99)) < 50),
        ("Dir score abs < 20",
         lambda b, l: abs(float(b.get("dir_score", 99))) < 20),
        ("Dir score abs < 30",
         lambda b, l: abs(float(b.get("dir_score", 99))) < 30),
        ("Dir conf < 40",
         lambda b, l: float(b.get("direction_conf", 99)) < 40),
        ("Dir conf < 50",
         lambda b, l: float(b.get("direction_conf", 99)) < 50),
        ("Session = NY (16-21h)",
         lambda b, l: 16 <= int(b["datetime"].hour) < 21),
        ("Session = Overlap (12-16h)",
         lambda b, l: 12 <= int(b["datetime"].hour) < 16),
        ("Session NY or Overlap",
         lambda b, l: int(b["datetime"].hour) >= 12),
        ("Eff<30 OR dir_conf<40",
         lambda b, l: float(b.get("efficiency",99)) < 30 or float(b.get("direction_conf",99)) < 40),
        ("Eff<40 AND atr_rank<50",
         lambda b, l: float(b.get("efficiency",99)) < 40 and float(b.get("atr_rank",99)) < 50),
        ("RC only: dir_conf<60",
         lambda b, l: l == "ReversalCandidate" and float(b.get("direction_conf",99)) < 60),
        ("Pullback only: eff<30",
         lambda b, l: l == "Pullback" and float(b.get("efficiency",99)) < 30),
    ]

    for label, fn in candidates:
        tr = run(cl, block_fn=fn)
        summary_line(label, tr, baseline_total)


if __name__ == "__main__":
    main()
