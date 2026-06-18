#!/usr/bin/env python3
"""
backtest_maxbars.py — Does max_bars help or hurt?

Tests: no limit, current limits, higher limits, per-session limits.
Uses the same fixed simulation as backtest_bounce.py (correct TrailSL P&L).
"""
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import live_engine as le
from backtest_v6 import load_10min_bars, SPREAD_PIPS
from backtest_bounce import run_sim, stats, flat_adaptive

HALF_SPREAD = (SPREAD_PIPS / 2) / 10000


def patch_max_bars(limit: int):
    """Set all profiles to the same max_bars limit. Returns originals."""
    orig = dict(le.PROFILES)
    le.PROFILES = {k: replace(v, max_bars=limit) for k, v in orig.items()}
    return orig


def patch_max_bars_per_profile(limits: dict):
    """limits = {profile_key: max_bars}. Returns originals."""
    orig = dict(le.PROFILES)
    le.PROFILES = {k: replace(v, max_bars=limits.get(k, v.max_bars))
                   for k, v in orig.items()}
    return orig


def restore(orig):
    le.PROFILES = orig


def main():
    print("Loading and classifying bars...")
    df = load_10min_bars()
    cl = le.classify_bars(df)
    print(f"  {len(cl)} bars.\n")

    afn = flat_adaptive(2.5, 0.0, 0.0)   # baseline adaptive fn (2.5x SL, no bounce)

    print("=" * 80)
    print("SECTION 1 — FLAT MAX_BARS LIMIT (all profiles same limit)")
    print("  Current: T1=40b T2=35b T3=20b T4=15b")
    print("=" * 80)

    flat_limits = [
        ("no limit (999)",          999),
        ("200 bars (~33h)",         200),
        ("100 bars (~17h)",         100),
        ("72 bars  (~12h)",          72),
        ("48 bars  (~8h)",           48),
        ("current T2=35 (baseline)", None),   # use real profile values
        ("30 bars  (~5h)",           30),
        ("20 bars  (~3h20m)",        20),
        ("15 bars  (~2h30m)",        15),
        ("10 bars  (~1h40m)",        10),
    ]

    for label, limit in flat_limits:
        if limit is None:
            t = run_sim(cl, adaptive_fn=afn)
            stats(t, label)
        else:
            orig = patch_max_bars(limit)
            try:
                t = run_sim(cl, adaptive_fn=afn)
            finally:
                restore(orig)
            stats(t, label)

    # Breakdown for no-limit vs current
    print()
    for label, limit in [("no limit (999)", 999), ("current (T2=35)", None)]:
        if limit is None:
            t = run_sim(cl, adaptive_fn=afn)
        else:
            orig = patch_max_bars(limit)
            try:
                t = run_sim(cl, adaptive_fn=afn)
            finally:
                restore(orig)
        df_t = pd.DataFrame(t)
        mb_n = (df_t["reason"] == "MaxBars").sum()
        mb_avg = df_t[df_t["reason"] == "MaxBars"]["pnl_pips"].mean() if mb_n else 0
        mb_win = (df_t[df_t["reason"] == "MaxBars"]["pnl_pips"] > 0).mean() * 100 if mb_n else 0
        print(f"  {label}: MaxBars exits={mb_n}  avg={mb_avg:+.2f}p  win={mb_win:.0f}%")
        g = df_t.groupby("reason")["pnl_pips"].agg(["count","sum","mean"])
        for r, row in g.iterrows():
            print(f"    {r:<18} n={int(row['count']):>5}  sum={row['sum']:>+9.1f}p  avg={row['mean']:>+6.2f}p")
        print()

    print("=" * 80)
    print("SECTION 2 — PER-SESSION ANALYSIS OF MAX_BARS EXITS (baseline)")
    print("=" * 80)
    t_base = run_sim(cl, adaptive_fn=afn)
    df_b = pd.DataFrame(t_base)
    mb = df_b[df_b["reason"] == "MaxBars"]
    print(f"\n  All MaxBars exits: n={len(mb)}  avg={mb['pnl_pips'].mean():+.2f}p  "
          f"win={(mb['pnl_pips']>0).mean()*100:.0f}%\n")
    if not mb.empty:
        print("  By session:")
        g = mb.groupby("session")["pnl_pips"].agg(["count","sum","mean",
                                                     lambda x: (x>0).mean()*100])
        g.columns = ["count","sum","mean","win%"]
        for sess, row in g.iterrows():
            print(f"    {sess:<10} n={int(row['count']):>4}  sum={row['sum']:>+8.1f}p  "
                  f"avg={row['mean']:>+6.2f}p  win={row['win%']:.0f}%")
        print()
        print("  By bars_held at exit:")
        bins = [0, 20, 35, 50, 75, 100, 999]
        labels = ["<=20", "21-35", "36-50", "51-75", "76-100", ">100"]
        mb2 = mb.copy()
        mb2["bin"] = pd.cut(mb2["bars_held"], bins=bins, labels=labels)
        g2 = mb2.groupby("bin", observed=True)["pnl_pips"].agg(["count","mean",
                                                                  lambda x: (x>0).mean()*100])
        g2.columns = ["count","mean","win%"]
        for b, row in g2.iterrows():
            print(f"    bars {b:<8} n={int(row['count']):>4}  avg={row['mean']:>+6.2f}p  win={row['win%']:.0f}%")

    print("\n" + "=" * 80)
    print("SECTION 3 — SESSION-AWARE LIMITS")
    print("  Idea: looser limit in Overlap/Other, tighter in London/NY")
    print("=" * 80)
    # We can't easily pass session-aware max_bars into check_exit without patching.
    # Approximate by testing combinations and showing session breakdown.
    session_configs = [
        ("no limit",                          999),
        ("72b (~12h) — very loose",            72),
        ("48b (~8h)  — loose",                 48),
        ("35b (current T2 baseline)",         None),
        ("24b (~4h)",                          24),
    ]
    for label, limit in session_configs:
        if limit is None:
            t = run_sim(cl, adaptive_fn=afn)
        else:
            orig = patch_max_bars(limit)
            try:
                t = run_sim(cl, adaptive_fn=afn)
            finally:
                restore(orig)
        df_t = pd.DataFrame(t)
        tot = df_t["pnl_pips"].sum()
        avg = df_t["pnl_pips"].mean()
        mb_n = (df_t["reason"] == "MaxBars").sum()
        print(f"\n  {label}  total={tot:+.0f}p  avg={avg:+.3f}p  MaxBars_exits={mb_n}")
        g = df_t.groupby("session")["pnl_pips"].agg(["count","sum","mean",
                                                       lambda x: (x>0).mean()*100])
        g.columns = ["n","sum","avg","win%"]
        for sess, row in g.iterrows():
            print(f"    {sess:<10} n={int(row['n']):>4}  sum={row['sum']:>+8.1f}p  "
                  f"avg={row['avg']:>+6.2f}p  win={row['win%']:.0f}%")

    print("\n" + "=" * 80)
    print("SECTION 4 — BEST COMBINED: optimal max_bars + adaptive SL + bounce")
    print("=" * 80)
    from backtest_bounce import rank_adaptive
    best_afn   = rank_adaptive([(90, 4.0, 3.0, 2.0), (75, 3.5, 2.0, 1.0), (0, 2.5, 0.0, 0.0)])
    best_bounce = dict(max_wait_bars=2, safety_pips=6, min_bounce_pips=1.0, trail_pips=2.0)

    combos = [
        ("adaptive+bounce  current max_bars", None),
        ("adaptive+bounce  no limit (999)",   999),
        ("adaptive+bounce  48b limit",         48),
        ("adaptive+bounce  72b limit",         72),
    ]
    for label, limit in combos:
        orig = patch_max_bars(limit) if limit is not None else None
        try:
            t = run_sim(cl, adaptive_fn=best_afn, bounce_cfg=best_bounce)
        finally:
            if orig is not None:
                restore(orig)
        stats(t, label)


if __name__ == "__main__":
    main()
