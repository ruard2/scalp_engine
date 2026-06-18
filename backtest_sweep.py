#!/usr/bin/env python3
"""
backtest_sweep.py — Parameter sweep on top of backtest_v6 logic.

Tests, against the same classified 16-month dataset:
  1. MAX_OPEN 1/2/3 with an entry cooldown (min bars between entries)
  2. Trail trigger/dist and SL-mult variations
  3. EnvRange exit on/off

Uses the REAL live_engine check_exit/entry_allowed; profile params are
patched on the live_engine PROFILES objects per config and restored after.
"""
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import live_engine as le
from backtest_v6 import load_10min_bars, SPREAD_PIPS

HALF_SPREAD = (SPREAD_PIPS / 2) / 10000


def run(cl, max_open=1, cooldown_bars=0, ignore_env_range=False,
        trail_mult=1.0, sl_mult=None):
    """Replay with given settings. trail_mult scales trigger+dist; sl_mult overrides sl_atr_mult."""
    trades = []
    open_trades = {}
    last_entry_i = -10**9

    for i in range(200, len(cl) - 1):
        bar  = cl.iloc[i]
        nxt  = cl.iloc[i + 1]
        hour = bar["datetime"].hour
        session = le.get_session_label(hour)

        for key in list(open_trades):
            t = open_trades[key]
            t["bars_held"] += 1
            reason = le.check_exit(t, bar)
            if reason == "EnvRange" and ignore_env_range:
                reason = None
            if reason:
                if reason == "TP":
                    px = t["tp_price"]
                elif reason == "SL":
                    px = t["sl_price"]
                elif reason == "TrailSL":
                    px = t["trail_stop"] if t["trail_active"] and t["trail_stop"] > 0 else t["sl_price"]
                else:
                    px = float(bar["close"])
                if t["direction"] == "Bull":
                    px -= HALF_SPREAD
                    pnl = (px - t["entry_price"]) * 10000
                else:
                    px += HALF_SPREAD
                    pnl = (t["entry_price"] - px) * 10000
                trades.append({"pnl_pips": round(pnl, 2), "reason": reason,
                               "bars": t["bars_held"], "session": t["session"]})
                del open_trades[key]

        d, e, loc = str(bar["direction"]), str(bar["environment"]), str(bar["local"])
        pk = le.get_profile_key(d, e, loc)
        if (len(open_trades) < max_open
                and (i - last_entry_i) >= cooldown_bars
                and le.entry_allowed(d, e, session, hour)
                and pk is not None):
            entry_px = float(nxt["open"]) + (HALF_SPREAD if d == "Bull" else -HALF_SPREAD)
            atr = float(bar["atr"]) if pd.notna(bar["atr"]) and bar["atr"] > 0 else 0.0006
            p = le.PROFILES[pk]
            eff_sl = (sl_mult if sl_mult is not None else p.sl_atr_mult)
            sl_dist, tp_dist = atr * eff_sl, p.tp_pips / 10000
            key = f"{d}_{e}_{loc}_{i}"
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
            last_entry_i = i

    return pd.DataFrame(trades)


def patch_profiles(trail_trigger_add=0.0, trail_dist_add=0.0):
    """Return original PROFILES and install patched copies."""
    orig = dict(le.PROFILES)
    le.PROFILES = {k: replace(v,
                              trail_trigger=v.trail_trigger + trail_trigger_add,
                              trail_dist=v.trail_dist + trail_dist_add)
                   for k, v in orig.items()}
    return orig


def stats(tr: pd.DataFrame) -> str:
    if tr.empty:
        return "no trades"
    cum = tr["pnl_pips"].cumsum()
    dd = (cum - cum.cummax()).min()
    wr = (tr["pnl_pips"] > 0).mean() * 100
    return (f"n={len(tr):>5}  sum={tr['pnl_pips'].sum():>+8.1f}p  "
            f"avg={tr['pnl_pips'].mean():>+6.2f}p  win={wr:4.1f}%  DD={dd:>8.1f}p")


def main():
    df = load_10min_bars()
    print(f"Classifying {len(df)} bars once...")
    cl = le.classify_bars(df)

    configs = [
        # (label, kwargs, profile_patch)
        ("BASELINE  max1",                    dict(), None),
        ("max2 cooldown2",                    dict(max_open=2, cooldown_bars=2), None),
        ("max2 cooldown3",                    dict(max_open=2, cooldown_bars=3), None),
        ("max3 cooldown3",                    dict(max_open=3, cooldown_bars=3), None),
        ("max3 cooldown6",                    dict(max_open=3, cooldown_bars=6), None),
        ("EnvRange exit UIT",                 dict(ignore_env_range=True), None),
        ("trail wijder (+2 trig, +1 dist)",   dict(), dict(trail_trigger_add=2.0, trail_dist_add=1.0)),
        ("trail wijder (+3 trig, +2 dist)",   dict(), dict(trail_trigger_add=3.0, trail_dist_add=2.0)),
        ("SL ruimer (2.0x ATR)",              dict(sl_mult=2.0), None),
        ("SL strakker (1.2x ATR)",            dict(sl_mult=1.2), None),
        ("COMBI: max2 cd3 + trail+2/+1",      dict(max_open=2, cooldown_bars=3), dict(trail_trigger_add=2.0, trail_dist_add=1.0)),
        ("COMBI: max2 cd3 + trail+2/+1 + EnvRange UIT",
                                              dict(max_open=2, cooldown_bars=3, ignore_env_range=True),
                                              dict(trail_trigger_add=2.0, trail_dist_add=1.0)),
    ]

    print(f"\n{'CONFIG':<46} RESULT")
    print("-" * 110)
    results = []
    for label, kwargs, patch in configs:
        orig = patch_profiles(**patch) if patch else None
        try:
            tr = run(cl, **kwargs)
        finally:
            if orig is not None:
                le.PROFILES = orig
        print(f"{label:<46} {stats(tr)}")
        results.append((label, tr))

    # Exit-reason detail for the EnvRange comparison
    print("\nExit-reason detail BASELINE vs EnvRange-UIT:")
    for label in ("BASELINE  max1", "EnvRange exit UIT"):
        tr = dict(results)[label]
        g = tr.groupby("reason")["pnl_pips"].agg(["count", "sum", "mean"])
        print(f"  {label}:")
        for r, row in g.iterrows():
            print(f"    {r:<14} n={int(row['count']):>4}  sum={row['sum']:>+8.1f}p  avg={row['mean']:>+6.2f}p")


if __name__ == "__main__":
    main()
