#!/usr/bin/env python3
"""
backtest_bounce.py v2 — Adaptive regime backtest for v6_engine.

Sweeps three adaptive dimensions together:
  1. SL multiplier gated on ATR rank at entry
  2. Trail trigger/dist widened at high ATR rank
  3. Bounce exit: after SL, wait N bars for recovery before closing

All comparisons share one classified dataset (classify once, sweep fast).
"""
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import live_engine as le
from backtest_v6 import load_10min_bars, SPREAD_PIPS

HALF_SPREAD = (SPREAD_PIPS / 2) / 10000


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive parameter helpers
# ─────────────────────────────────────────────────────────────────────────────
def flat_adaptive(sl=2.5, t_add=0.0, d_add=0.0):
    """Returns adaptive_fn with fixed params (no rank gating)."""
    def fn(atr_rank, profile_key):
        return sl, t_add, d_add
    return fn


def rank_adaptive(thresholds):
    """
    thresholds: list of (min_rank, sl_mult, trail_trigger_add, trail_dist_add)
    sorted highest first. First match wins.
    Example: [(90, 4.0, 3.0, 2.0), (75, 3.5, 2.0, 1.0), (0, 2.5, 0.0, 0.0)]
    """
    thr = sorted(thresholds, key=lambda x: -x[0])
    def fn(atr_rank, profile_key):
        for min_r, sl, tadd, dadd in thr:
            if atr_rank >= min_r:
                return sl, tadd, dadd
        return 2.5, 0.0, 0.0
    return fn


# ─────────────────────────────────────────────────────────────────────────────
# Core simulation — fixed P&L calculation (no min/max slippage on TP)
# ─────────────────────────────────────────────────────────────────────────────
def run_sim(cl, adaptive_fn=None, bounce_cfg=None):
    """
    Simulate all trades with adaptive SL/trail and optional bounce logic.

    adaptive_fn(atr_rank, profile_key) -> (sl_mult, trail_trigger_add, trail_dist_add)
    bounce_cfg: None (no bounce) or dict(max_wait_bars, safety_pips, min_bounce_pips, trail_pips)

    Returns list of trade dicts including full detail for analysis.
    """
    if adaptive_fn is None:
        adaptive_fn = flat_adaptive()

    trades = []
    open_trades = {}
    bounce_pending = {}   # key -> bounce state (for deferred SL close)

    def _open_entry(i, bar, nxt, d, e, loc, pk, session):
        atr      = float(bar["atr"]) if pd.notna(bar["atr"]) and bar["atr"] > 0 else 0.0006
        atr_rank = float(bar["atr_rank"]) if pd.notna(bar["atr_rank"]) else 50.0
        p        = le.PROFILES[pk]
        sl_m, tadd, dadd = adaptive_fn(atr_rank, pk)
        sl_dist  = atr * sl_m
        tp_dist  = p.tp_pips / 10000
        entry_px = float(nxt["open"]) + (HALF_SPREAD if d == "Bull" else -HALF_SPREAD)
        trail_trigger = (p.trail_trigger + tadd) / 10000
        trail_dist    = (p.trail_dist    + dadd) / 10000
        key = f"{d}_{e}_{loc}_{i}"
        open_trades[key] = {
            "combo": f"{d}_{e}_{loc}", "direction": d, "profile_key": pk,
            "session": session, "entry_price": entry_px,
            "atr": atr, "atr_rank": atr_rank, "sl_mult": sl_m,
            "sl_price":  entry_px - sl_dist if d == "Bull" else entry_px + sl_dist,
            "tp_price":  entry_px + tp_dist if d == "Bull" else entry_px - tp_dist,
            "trail_dist": trail_dist, "trail_trigger": trail_trigger,
            "trail_active": False, "trail_stop": 0.0,
            "best_price": entry_px, "bars_held": 0,
        }

    def _record_trade(trade, reason, bar, pnl, exit_px):
        trades.append({
            "bar_i": trade.get("bar_i_open", 0),
            "direction": trade["direction"],
            "combo": trade["combo"],
            "session": trade["session"],
            "atr_rank": trade["atr_rank"],
            "sl_mult": trade["sl_mult"],
            "entry_price": trade["entry_price"],
            "exit_price": round(exit_px, 5),
            "pnl_pips": round(pnl, 2),
            "reason": reason,
            "bars_held": trade["bars_held"],
        })

    for i in range(200, len(cl) - 1):
        bar  = cl.iloc[i]
        nxt  = cl.iloc[i + 1]
        hour = bar["datetime"].hour
        session = le.get_session_label(hour)

        # ── Check bounce-pending positions first ──────────────────────────
        if bounce_cfg:
            to_resolve = []
            for bkey, bs in list(bounce_pending.items()):
                bs["bars_in_bounce"] += 1
                d         = bs["direction"]
                entry_px  = bs["entry_price"]
                sl_trig   = bs["sl_trigger"]
                safety_fl = bs["safety_floor"]
                btrail    = bs["bounce_trail"]
                b_min     = bs["min_bounce_d"]
                b_trail_d = bs["trail_d"]
                best_rec  = bs["best_recovery"]
                max_wait  = bs["max_wait_bars"]

                high = float(bar["high"])
                low  = float(bar["low"])
                close = float(bar["close"])

                closed = False
                if d == "Bull":
                    # Safety floor breach
                    if low - HALF_SPREAD <= safety_fl:
                        px  = safety_fl - HALF_SPREAD
                        pnl = (px - entry_px) * 10000
                        _record_trade(bs, "BounceSafety", bar, pnl, px)
                        closed = True
                    # Profit crossover
                    elif high - HALF_SPREAD >= entry_px:
                        pnl = 0.0
                        _record_trade(bs, "BounceProfit", bar, pnl, entry_px)
                        closed = True
                    else:
                        # Recovery tracking
                        if high > best_rec + b_min:
                            bs["best_recovery"] = high
                            new_trail = high - b_trail_d - HALF_SPREAD
                            bs["bounce_trail"] = max(new_trail, safety_fl) if btrail is None else max(btrail, max(new_trail, safety_fl))
                        # Trail breach
                        if bs["bounce_trail"] is not None and low - HALF_SPREAD <= bs["bounce_trail"]:
                            px  = bs["bounce_trail"]
                            pnl = (px - entry_px) * 10000
                            _record_trade(bs, "BounceTrail", bar, pnl, px)
                            closed = True
                else:  # Bear
                    if high + HALF_SPREAD >= safety_fl:
                        px  = safety_fl + HALF_SPREAD
                        pnl = (entry_px - px) * 10000
                        _record_trade(bs, "BounceSafety", bar, pnl, px)
                        closed = True
                    elif low + HALF_SPREAD <= entry_px:
                        pnl = 0.0
                        _record_trade(bs, "BounceProfit", bar, pnl, entry_px)
                        closed = True
                    else:
                        if low < best_rec - b_min:
                            bs["best_recovery"] = low
                            new_trail = low + b_trail_d + HALF_SPREAD
                            bs["bounce_trail"] = min(new_trail, safety_fl) if btrail is None else min(btrail, min(new_trail, safety_fl))
                        if bs["bounce_trail"] is not None and high + HALF_SPREAD >= bs["bounce_trail"]:
                            px  = bs["bounce_trail"]
                            pnl = (entry_px - px) * 10000
                            _record_trade(bs, "BounceTrail", bar, pnl, px)
                            closed = True

                if not closed and bs["bars_in_bounce"] >= max_wait:
                    if d == "Bull":
                        px  = float(close) - HALF_SPREAD
                        pnl = (px - entry_px) * 10000
                    else:
                        px  = float(close) + HALF_SPREAD
                        pnl = (entry_px - px) * 10000
                    _record_trade(bs, "BounceTimeout", bar, pnl, px)
                    closed = True

                if closed:
                    to_resolve.append(bkey)

            for bkey in to_resolve:
                del bounce_pending[bkey]

        # ── Check open trades for exit ────────────────────────────────────
        for key in list(open_trades):
            open_trades[key]["bars_held"] += 1

        to_close = []
        for key, t in open_trades.items():
            reason = le.check_exit(t, bar)
            if reason:
                to_close.append((key, reason))

        for key, reason in to_close:
            t  = open_trades.pop(key)
            d  = t["direction"]
            ep = t["entry_price"]

            if reason == "TP":
                px = t["tp_price"]
                exit_px = px - HALF_SPREAD if d == "Bull" else px + HALF_SPREAD
                pnl = (exit_px - ep) * 10000 if d == "Bull" else (ep - exit_px) * 10000

            elif reason in ("SL", "TrailSL"):
                if reason == "TrailSL" and t["trail_active"] and t["trail_stop"] > 0:
                    # TrailSL: exit at the trail stop price (no extra slippage —
                    # the trail was already set conservatively and bar just touched it)
                    sl_ref = t["trail_stop"]
                    if d == "Bull":
                        exit_px = sl_ref - HALF_SPREAD
                        pnl     = (exit_px - ep) * 10000
                    else:
                        exit_px = sl_ref + HALF_SPREAD
                        pnl     = (ep - exit_px) * 10000
                else:
                    # Hard SL: may gap through — use worst of sl_price vs bar extreme
                    sl_ref = t["sl_price"]
                    if d == "Bull":
                        exit_px = min(sl_ref, float(bar["low"])) - HALF_SPREAD
                        pnl     = (exit_px - ep) * 10000
                    else:
                        exit_px = max(sl_ref, float(bar["high"])) + HALF_SPREAD
                        pnl     = (ep - exit_px) * 10000

                # Bounce logic: intercept SL exits and defer close
                if bounce_cfg and reason == "SL":
                    safety_d = bounce_cfg["safety_pips"] / 10000
                    safety_fl = (exit_px - safety_d) if d == "Bull" else (exit_px + safety_d)
                    sl_trig   = exit_px
                    bounce_pending[key] = {
                        "direction": d, "entry_price": ep,
                        "sl_trigger": sl_trig, "safety_floor": safety_fl,
                        "bounce_trail": None, "best_recovery": sl_trig,
                        "min_bounce_d": bounce_cfg["min_bounce_pips"] / 10000,
                        "trail_d":      bounce_cfg["trail_pips"] / 10000,
                        "max_wait_bars": bounce_cfg["max_wait_bars"],
                        "bars_in_bounce": 0,
                        "combo": t["combo"], "session": t["session"],
                        "atr_rank": t["atr_rank"], "sl_mult": t["sl_mult"],
                        "bars_held": t["bars_held"],
                        "bar_i_open": t.get("bar_i_open", 0),
                    }
                    continue  # don't record trade yet

            else:
                # DirectionFlip / MaxBars / etc.
                close_p  = float(bar["close"])
                exit_px  = close_p - HALF_SPREAD if d == "Bull" else close_p + HALF_SPREAD
                pnl      = (exit_px - ep) * 10000 if d == "Bull" else (ep - exit_px) * 10000

            t["bar_i_open"] = t.get("bar_i_open", i)
            _record_trade(t, reason, bar, pnl, exit_px)

        # ── Entry ─────────────────────────────────────────────────────────
        d, e, loc = str(bar["direction"]), str(bar["environment"]), str(bar["local"])
        pk = le.get_profile_key(d, e, loc)
        if (len(open_trades) < 1
                and le.entry_allowed(d, e, session, hour)
                and pk is not None):
            _open_entry(i, bar, nxt, d, e, loc, pk, session)

    return trades


# ─────────────────────────────────────────────────────────────────────────────
# Stats helpers
# ─────────────────────────────────────────────────────────────────────────────
def stats(trades, label=""):
    df = pd.DataFrame(trades)
    if df.empty:
        print(f"  {label}: no trades"); return
    n   = len(df)
    tot = df["pnl_pips"].sum()
    avg = df["pnl_pips"].mean()
    cum = df["pnl_pips"].cumsum()
    dd  = (cum - cum.cummax()).min()
    wr  = (df["pnl_pips"] > 0).mean() * 100
    print(f"  {label:<58} n={n:>5}  sum={tot:>+8.1f}p  avg={avg:>+6.2f}p  win={wr:4.1f}%  DD={dd:>8.1f}p")


def reason_breakdown(trades, label=""):
    df = pd.DataFrame(trades)
    if df.empty: return
    print(f"\n  Exit breakdown for: {label}")
    g = df.groupby("reason")["pnl_pips"].agg(["count","sum","mean"])
    for r, row in g.iterrows():
        print(f"    {r:<18} n={int(row['count']):>5}  sum={row['sum']:>+9.1f}p  avg={row['mean']:>+6.2f}p")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("Loading and classifying bars...")
    df = load_10min_bars()
    cl = le.classify_bars(df)
    print(f"  {len(cl)} bars. Running sweeps...\n")

    # ── BASELINE ─────────────────────────────────────────────────────────────
    baseline = run_sim(cl)
    print("=" * 80)
    print("BASELINE (flat 2.5x SL, current PROFILES, no bounce)")
    print("=" * 80)
    stats(baseline, "Current config")
    reason_breakdown(baseline, "Current config")

    # ── SECTION 1: ADAPTIVE SL + TRAIL (no bounce) ───────────────────────────
    print("\n" + "=" * 80)
    print("SECTION 1 -- ADAPTIVE SL + TRAIL BY ATR RANK")
    print("=" * 80)
    print("  (format: rank thresholds | SL mult | trail +trigger/+dist)\n")

    sl_trail_configs = [
        ("flat 2.5x  trail+0/+0  (baseline)",
         flat_adaptive(2.5, 0.0, 0.0)),
        ("flat 3.0x  trail+0/+0",
         flat_adaptive(3.0, 0.0, 0.0)),
        ("flat 3.5x  trail+0/+0",
         flat_adaptive(3.5, 0.0, 0.0)),
        ("flat 2.5x  trail+2/+1",
         flat_adaptive(2.5, 2.0, 1.0)),
        # Rank-gated SL only
        ("rank>75: 3.5x  else 2.5x   trail flat",
         rank_adaptive([(75, 3.5, 0.0, 0.0), (0, 2.5, 0.0, 0.0)])),
        ("rank>75: 4.0x  else 2.5x   trail flat",
         rank_adaptive([(75, 4.0, 0.0, 0.0), (0, 2.5, 0.0, 0.0)])),
        ("rank>90: 4.0x  else 2.5x   trail flat",
         rank_adaptive([(90, 4.0, 0.0, 0.0), (0, 2.5, 0.0, 0.0)])),
        # Rank-gated SL + trail
        ("rank>75: 3.5x +2/+1  else 2.5x +0/+0",
         rank_adaptive([(75, 3.5, 2.0, 1.0), (0, 2.5, 0.0, 0.0)])),
        ("rank>75: 4.0x +3/+2  else 2.5x +0/+0",
         rank_adaptive([(75, 4.0, 3.0, 2.0), (0, 2.5, 0.0, 0.0)])),
        # Three-tier
        ("rank>90: 4.0x +3/+2  rank>75: 3.5x +2/+1  else 2.5x",
         rank_adaptive([(90, 4.0, 3.0, 2.0), (75, 3.5, 2.0, 1.0), (0, 2.5, 0.0, 0.0)])),
        ("rank>90: 4.5x +4/+2  rank>75: 3.5x +2/+1  else 2.5x",
         rank_adaptive([(90, 4.5, 4.0, 2.0), (75, 3.5, 2.0, 1.0), (0, 2.5, 0.0, 0.0)])),
        # Tighter on low-vol
        ("rank>90: 4.0x +3/+2  rank>75: 3.5x +2/+1  rank<50: 2.0x -1/0",
         rank_adaptive([(90, 4.0, 3.0, 2.0), (75, 3.5, 2.0, 1.0), (50, 2.5, 0.0, 0.0), (0, 2.0, -1.0, 0.0)])),
    ]

    for label, afn in sl_trail_configs:
        t = run_sim(cl, adaptive_fn=afn)
        stats(t, label)

    # ── SECTION 2: BOUNCE ONLY (baseline SL) ─────────────────────────────────
    print("\n" + "=" * 80)
    print("SECTION 2 -- BOUNCE EXIT (baseline 2.5x SL)")
    print("=" * 80)
    print("  (shows how much bounce recovers vs immediate SL close)\n")

    bounce_configs = [
        ("wait=2b  safety=6p  min=1p  trail=2p",
         dict(max_wait_bars=2, safety_pips=6, min_bounce_pips=1.0, trail_pips=2.0)),
        ("wait=3b  safety=6p  min=1p  trail=2p  [best from v1]",
         dict(max_wait_bars=3, safety_pips=6, min_bounce_pips=1.0, trail_pips=2.0)),
        ("wait=3b  safety=8p  min=1.5p trail=2.5p",
         dict(max_wait_bars=3, safety_pips=8, min_bounce_pips=1.5, trail_pips=2.5)),
        ("wait=3b  safety=10p min=2p  trail=3p",
         dict(max_wait_bars=3, safety_pips=10, min_bounce_pips=2.0, trail_pips=3.0)),
        ("wait=4b  safety=8p  min=1.5p trail=2.5p",
         dict(max_wait_bars=4, safety_pips=8, min_bounce_pips=1.5, trail_pips=2.5)),
    ]

    for label, bcfg in bounce_configs:
        t = run_sim(cl, bounce_cfg=bcfg)
        stats(t, label)

    # ── SECTION 3: BEST ADAPTIVE SL + BOUNCE COMBINED ────────────────────────
    print("\n" + "=" * 80)
    print("SECTION 3 -- COMBINED: ADAPTIVE SL/TRAIL + BOUNCE")
    print("=" * 80 + "\n")

    best_bounce = dict(max_wait_bars=3, safety_pips=6, min_bounce_pips=1.0, trail_pips=2.0)

    combined_configs = [
        ("BASELINE no bounce no adaptive",
         flat_adaptive(2.5, 0.0, 0.0), None),
        ("bounce only (3b/6p/1p/2p)",
         flat_adaptive(2.5, 0.0, 0.0), best_bounce),
        ("3-tier SL+trail only",
         rank_adaptive([(90, 4.0, 3.0, 2.0), (75, 3.5, 2.0, 1.0), (0, 2.5, 0.0, 0.0)]), None),
        ("3-tier SL+trail + bounce",
         rank_adaptive([(90, 4.0, 3.0, 2.0), (75, 3.5, 2.0, 1.0), (0, 2.5, 0.0, 0.0)]), best_bounce),
        ("4.5x-tier + bounce",
         rank_adaptive([(90, 4.5, 4.0, 2.0), (75, 3.5, 2.0, 1.0), (0, 2.5, 0.0, 0.0)]), best_bounce),
        # Tighter on low-vol + bounce
        ("full adaptive (tight low-vol) + bounce",
         rank_adaptive([(90, 4.0, 3.0, 2.0), (75, 3.5, 2.0, 1.0), (50, 2.5, 0.0, 0.0), (0, 2.0, -1.0, 0.0)]), best_bounce),
    ]

    for label, afn, bcfg in combined_configs:
        t = run_sim(cl, adaptive_fn=afn, bounce_cfg=bcfg)
        stats(t, label)

    # Best combined — reason breakdown
    print()
    best_t = run_sim(cl,
                     adaptive_fn=rank_adaptive([(90, 4.0, 3.0, 2.0), (75, 3.5, 2.0, 1.0), (0, 2.5, 0.0, 0.0)]),
                     bounce_cfg=best_bounce)
    reason_breakdown(best_t, "3-tier SL+trail + bounce")

    # ── ATR rank breakdown for best combined ─────────────────────────────────
    print("\n  ATR rank breakdown (best combined config):")
    df_best = pd.DataFrame(best_t)
    for rmin, rmax, lbl in [(0,50,"rank  0-50%"), (50,75,"rank 50-75%"),
                             (75,90,"rank 75-90%"), (90,101,"rank 90%+")]:
        sub = df_best[(df_best["atr_rank"] >= rmin) & (df_best["atr_rank"] < rmax)]
        if sub.empty: continue
        n = len(sub); tot = sub["pnl_pips"].sum(); avg = sub["pnl_pips"].mean()
        wr = (sub["pnl_pips"] > 0).mean() * 100
        print(f"    {lbl}  n={n:>5}  sum={tot:>+8.1f}p  avg={avg:>+6.2f}p  win={wr:4.1f}%")

    # ── SL exit count comparison ──────────────────────────────────────────────
    print("\n  SL exit count by config (adaptive widens SL -> fewer hard SL hits):")
    for label, afn, bcfg in combined_configs:
        t = run_sim(cl, adaptive_fn=afn, bounce_cfg=bcfg)
        df_t = pd.DataFrame(t)
        sl_n   = (df_t["reason"] == "SL").sum()
        bsl_n  = (df_t["reason"].isin(["BounceTrail","BounceTimeout","BounceProfit","BounceSafety"])).sum()
        print(f"    {label:<50} SL={sl_n:>4}  BounceExit={bsl_n:>4}")


if __name__ == "__main__":
    main()
