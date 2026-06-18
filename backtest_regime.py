#!/usr/bin/env python3
"""
backtest_regime.py — Does regime strength determine the optimal entry delay?

Hypothesis: strong-regime signals (Bull_Trend_RC on high-vol day) are already
multi-bar confirmed and don't need a 1-bar wait. Weak regimes benefit most from
the filter. A regime-gated delay policy should beat flat delay=1.

Regime score (0-4):
  Environment:  Trend=2, Expansion=1, other=0
  Local:        RC=+1, else=0
  ATR rank:     >=75% => +1, else 0
  => score 0-4: higher = stronger conviction

Tests:
  Section 1 — breakdown of flat delay=0 and delay=1 results BY regime score
              (are strong regimes already good without delay?)
  Section 2 — regime-gated delay policies vs flat delay=0 and delay=1
  Section 3 — regime-gated TP multiplier (wider TP on strong regimes)
  Section 4 — best combined regime policy
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import live_engine as le
from backtest_v6 import load_10min_bars, SPREAD_PIPS
from backtest_bounce import flat_adaptive, rank_adaptive
from backtest_entry import run_delayed, stats

HALF_SPREAD = (SPREAD_PIPS / 2) / 10000


# ─────────────────────────────────────────────────────────────────────────────
# Regime score
# ─────────────────────────────────────────────────────────────────────────────
def regime_score(direction, environment, local_lbl, atr_rank):
    score = 0
    if environment == "Trend":
        score += 2
    elif environment == "Expansion":
        score += 1
    if local_lbl == "RC":
        score += 1
    if atr_rank >= 75.0:
        score += 1
    return score


# ─────────────────────────────────────────────────────────────────────────────
# Modified run_delayed that supports per-signal dynamic delay and dynamic TP
# ─────────────────────────────────────────────────────────────────────────────
def run_regime_gated(cl, delay_fn, tp_mult_fn=None, adaptive_fn=None, bounce_cfg=None):
    """
    delay_fn(direction, environment, local_lbl, atr_rank) -> int (0 or 1)
    tp_mult_fn(direction, environment, local_lbl, atr_rank) -> float multiplier on tp_pips
                                                               (1.0 = no change)
    """
    if adaptive_fn is None:
        adaptive_fn = flat_adaptive(2.5)
    if tp_mult_fn is None:
        tp_mult_fn = lambda d, e, loc, r: 1.0

    trades = []
    open_trades = {}
    bounce_pending = {}
    pending_entries = {}

    cl_len = len(cl)

    def _record(t, reason, pnl):
        trades.append({
            "direction": t["direction"], "combo": t["combo"],
            "session": t["session"], "atr_rank": t["atr_rank"],
            "pnl_pips": round(pnl, 2), "reason": reason,
            "bars_held": t["bars_held"], "sl_mult": t["sl_mult"],
            "regime_score": t.get("regime_score", 0),
        })

    for i in range(200, cl_len - 1):
        bar  = cl.iloc[i]
        hour = bar["datetime"].hour
        session = le.get_session_label(hour)

        # ── Activate pending entries ──────────────────────────────────────
        if i in pending_entries:
            nxt = cl.iloc[i + 1] if i + 1 < cl_len else None
            if nxt is not None:
                pending = pending_entries.pop(i)
                entry_px = float(nxt["open"]) + (
                    HALF_SPREAD if pending["direction"] == "Bull" else -HALF_SPREAD)
                atr  = pending["atr"]
                sl_m = pending["sl_mult"]
                pk   = pending["profile_key"]
                p    = le.PROFILES[pk]
                tp_m = pending["tp_mult"]
                sl_dist = atr * sl_m
                tp_dist = (p.tp_pips * tp_m) / 10000
                d = pending["direction"]
                open_trades[pending["key"]] = {
                    "combo": pending["combo"], "direction": d,
                    "profile_key": pk, "session": pending["session"],
                    "entry_price": entry_px, "atr": atr,
                    "atr_rank": pending["atr_rank"], "sl_mult": sl_m,
                    "sl_price":  entry_px - sl_dist if d=="Bull" else entry_px + sl_dist,
                    "tp_price":  entry_px + tp_dist if d=="Bull" else entry_px - tp_dist,
                    "trail_dist": p.trail_dist / 10000,
                    "trail_trigger": p.trail_trigger / 10000,
                    "trail_active": False, "trail_stop": 0.0,
                    "best_price": entry_px, "bars_held": 0,
                    "regime_score": pending["regime_score"],
                }

        # ── Bounce pending ────────────────────────────────────────────────
        bounce_close = []
        for bkey, bs in list(bounce_pending.items()):
            bs["bars_in_bounce"] += 1
            d = bs["direction"]; ep = bs["entry_price"]
            sf = bs["safety_floor"]; btr = bs.get("bounce_trail")
            brec = bs.get("best_recovery", bs["sl_trigger"])
            b_min = 1.0/10000; b_td = 2.0/10000
            high = float(bar["high"]); low = float(bar["low"])
            cr = None
            if d == "Bull":
                if low <= sf: cr = "BounceSafety"
                elif high >= ep: cr = "BounceProfit"
                else:
                    if high > brec + b_min:
                        bs["best_recovery"] = high
                        nt = high - b_td
                        bs["bounce_trail"] = max(nt,sf) if btr is None else max(btr,max(nt,sf))
                    if bs.get("bounce_trail") and low <= bs["bounce_trail"]: cr = "BounceTrail"
            else:
                if high >= sf: cr = "BounceSafety"
                elif low <= ep: cr = "BounceProfit"
                else:
                    if low < brec - b_min:
                        bs["best_recovery"] = low
                        nt = low + b_td
                        bs["bounce_trail"] = min(nt,sf) if btr is None else min(btr,min(nt,sf))
                    if bs.get("bounce_trail") and high >= bs["bounce_trail"]: cr = "BounceTrail"
            if cr is None and bs["bars_in_bounce"] >= 2: cr = "BounceTimeout"
            if cr:
                cp = float(bar["close"])
                pnl = (cp - HALF_SPREAD - ep)*10000 if d=="Bull" else (ep - cp - HALF_SPREAD)*10000
                _record(bs, cr, pnl)
                bounce_close.append(bkey)
        for bk in bounce_close:
            del bounce_pending[bk]

        # ── Open trades: increment bars_held, check exit ──────────────────
        for key in list(open_trades):
            open_trades[key]["bars_held"] += 1

        to_close = []
        for key, t in open_trades.items():
            reason = le.check_exit(t, bar)
            if reason:
                to_close.append((key, reason))

        for key, reason in to_close:
            t  = open_trades.pop(key)
            d  = t["direction"]; ep = t["entry_price"]
            if reason == "TP":
                px = t["tp_price"]
                exit_px = px - HALF_SPREAD if d=="Bull" else px + HALF_SPREAD
                pnl = (exit_px - ep)*10000 if d=="Bull" else (ep - exit_px)*10000
            elif reason == "TrailSL" and t["trail_active"] and t["trail_stop"] > 0:
                sl_ref = t["trail_stop"]
                exit_px = sl_ref - HALF_SPREAD if d=="Bull" else sl_ref + HALF_SPREAD
                pnl = (exit_px - ep)*10000 if d=="Bull" else (ep - exit_px)*10000
            elif reason == "SL":
                sl_ref = t["sl_price"]
                if d == "Bull":
                    exit_px = min(sl_ref, float(bar["low"])) - HALF_SPREAD
                else:
                    exit_px = max(sl_ref, float(bar["high"])) + HALF_SPREAD
                pnl = (exit_px - ep)*10000 if d=="Bull" else (ep - exit_px)*10000
                if bounce_cfg:
                    sf = exit_px - 6.0/10000 if d=="Bull" else exit_px + 6.0/10000
                    bounce_pending[key] = {
                        "direction": d, "entry_price": ep, "sl_trigger": exit_px,
                        "safety_floor": sf, "bounce_trail": None,
                        "best_recovery": exit_px, "bars_in_bounce": 0,
                        "bars_held": t["bars_held"], "atr_rank": t["atr_rank"],
                        "combo": t["combo"], "session": t["session"],
                        "sl_mult": t["sl_mult"], "regime_score": t.get("regime_score",0),
                    }
                    continue
            else:
                cp = float(bar["close"])
                exit_px = cp - HALF_SPREAD if d=="Bull" else cp + HALF_SPREAD
                pnl = (exit_px - ep)*10000 if d=="Bull" else (ep - exit_px)*10000
            _record(t, reason, pnl)

        # ── Entry signal check ────────────────────────────────────────────
        n_active = len(open_trades) + len(bounce_pending) + len(pending_entries)
        if n_active >= 1:
            continue

        d0, e0, loc0 = str(bar["direction"]), str(bar["environment"]), str(bar["local"])
        pk0 = le.get_profile_key(d0, e0, loc0)
        if not le.entry_allowed(d0, e0, session, hour) or pk0 is None:
            continue

        atr      = float(bar["atr"])      if pd.notna(bar["atr"])      and bar["atr"]      > 0 else 0.0006
        atr_rank = float(bar["atr_rank"]) if pd.notna(bar["atr_rank"]) else 50.0
        sl_m, _, _ = adaptive_fn(atr_rank, pk0)
        tp_m     = tp_mult_fn(d0, e0, loc0, atr_rank)
        rscore   = regime_score(d0, e0, loc0, atr_rank)
        delay    = delay_fn(d0, e0, loc0, atr_rank)

        if delay == 0:
            if i + 1 >= cl_len:
                continue
            nxt = cl.iloc[i + 1]
            entry_px = float(nxt["open"]) + (HALF_SPREAD if d0=="Bull" else -HALF_SPREAD)
            p        = le.PROFILES[pk0]
            sl_dist  = atr * sl_m
            tp_dist  = (p.tp_pips * tp_m) / 10000
            key      = f"{d0}_{e0}_{loc0}_{i}"
            open_trades[key] = {
                "combo": f"{d0}_{e0}_{loc0}", "direction": d0, "profile_key": pk0,
                "session": session, "entry_price": entry_px,
                "atr": atr, "atr_rank": atr_rank, "sl_mult": sl_m,
                "sl_price":  entry_px - sl_dist if d0=="Bull" else entry_px + sl_dist,
                "tp_price":  entry_px + tp_dist if d0=="Bull" else entry_px - tp_dist,
                "trail_dist": p.trail_dist / 10000, "trail_trigger": p.trail_trigger / 10000,
                "trail_active": False, "trail_stop": 0.0,
                "best_price": entry_px, "bars_held": 0, "regime_score": rscore,
            }
        else:
            conf_idx = i + delay
            if conf_idx >= cl_len - 1:
                continue
            conf_bar = cl.iloc[conf_idx]
            ch = conf_bar["datetime"].hour
            cs = le.get_session_label(ch)
            cd = str(conf_bar["direction"])
            ce = str(conf_bar["environment"])
            if cd != d0 or ce in ("Range","Compression") or not le.entry_allowed(cd, ce, cs, ch):
                continue
            if conf_idx + 1 >= cl_len:
                continue
            key = f"{d0}_{e0}_{loc0}_{i}"
            pending_entries[conf_idx] = {
                "key": key, "combo": f"{d0}_{e0}_{loc0}",
                "direction": d0, "profile_key": pk0,
                "session": session, "atr": atr, "atr_rank": atr_rank,
                "sl_mult": sl_m, "tp_mult": tp_m, "regime_score": rscore,
            }

    return trades


def regime_stats(trades, label=""):
    df = pd.DataFrame(trades)
    if df.empty:
        print(f"  {label:<58} NO TRADES"); return
    n   = len(df)
    tot = df["pnl_pips"].sum()
    avg = df["pnl_pips"].mean()
    cum = df["pnl_pips"].cumsum()
    dd  = (cum - cum.cummax()).min()
    wr  = (df["pnl_pips"] > 0).mean() * 100
    print(f"  {label:<58} n={n:>5}  sum={tot:>+8.1f}p  avg={avg:>+6.2f}p  "
          f"win={wr:4.1f}%  DD={dd:>8.1f}p")


def main():
    afn_best = rank_adaptive([(90,4.0,3.0,2.0),(75,3.5,2.0,1.0),(0,2.5,0.0,0.0)])
    bounce   = dict(max_wait_bars=2, safety_pips=6, min_bounce_pips=1.0, trail_pips=2.0)

    print("Loading and classifying 10-min bars...")
    df10 = load_10min_bars()
    cl   = le.classify_bars(df10)
    print(f"  {len(cl)} bars.\n")

    # ── SECTION 1: Baseline flat results for comparison ──────────────────────
    print("=" * 80)
    print("SECTION 1 — REFERENCE: flat delay=0 and delay=1 (adaptive+bounce)")
    print("=" * 80 + "\n")
    t0 = run_delayed(cl, delay_bars=0, adaptive_fn=afn_best, bounce_cfg=bounce)
    t1 = run_delayed(cl, delay_bars=1, adaptive_fn=afn_best, bounce_cfg=bounce)
    stats(t0, "flat delay=0 (immediate) + adaptive + bounce")
    stats(t1, "flat delay=1 (10min)    + adaptive + bounce")

    # ── SECTION 2: Per-regime breakdown of delay=0 and delay=1 ──────────────
    print("\n" + "=" * 80)
    print("SECTION 2 — REGIME SCORE BREAKDOWN  (which regimes need the delay?)")
    print("  Score: Trend=2, Expansion=1, RC=+1, ATR>=75%=+1  (max 4)")
    print("=" * 80 + "\n")

    for delay, label in [(0, "delay=0"), (1, "delay=1")]:
        t = run_delayed(cl, delay_bars=delay, adaptive_fn=afn_best, bounce_cfg=bounce)
        df = pd.DataFrame(t)
        if df.empty:
            continue
        df["regime_score"] = df.apply(
            lambda r: regime_score(r["direction"],
                                   r["combo"].split("_")[1] if "_" in r["combo"] else "",
                                   r["combo"].split("_")[2] if r["combo"].count("_") >= 2 else "",
                                   r["atr_rank"]), axis=1)
        print(f"  {label}  by regime score:")
        g = df.groupby("regime_score")["pnl_pips"].agg(
            ["count","sum","mean", lambda x: (x>0).mean()*100])
        g.columns = ["n","sum","avg","win%"]
        for sc, row in g.iterrows():
            print(f"    score={sc}  n={int(row['n']):>5}  sum={row['sum']:>+8.1f}p  "
                  f"avg={row['avg']:>+6.2f}p  win={row['win%']:.0f}%")
        print()

    # ── SECTION 3: Regime-gated delay policies ───────────────────────────────
    print("=" * 80)
    print("SECTION 3 — REGIME-GATED DELAY POLICIES (adaptive+bounce)")
    print("  Policy: skip 1-bar wait when regime is already strong")
    print("=" * 80 + "\n")

    policies = [
        ("flat delay=0 (reference)",
         lambda d,e,loc,r: 0),
        ("flat delay=1 (reference)",
         lambda d,e,loc,r: 1),
        ("gated: score>=4 -> delay=0, else delay=1",
         lambda d,e,loc,r: 0 if regime_score(d,e,loc,r) >= 4 else 1),
        ("gated: score>=3 -> delay=0, else delay=1",
         lambda d,e,loc,r: 0 if regime_score(d,e,loc,r) >= 3 else 1),
        ("gated: score>=2 -> delay=0, else delay=1",
         lambda d,e,loc,r: 0 if regime_score(d,e,loc,r) >= 2 else 1),
        ("gated: score<=1 -> skip  (only trade strong regimes with delay=1)",
         lambda d,e,loc,r: (1 if regime_score(d,e,loc,r) >= 2 else 999)),
        ("gated: score=4 -> delay=0, score<=2 -> delay=1, score<=1 -> skip",
         lambda d,e,loc,r: (0 if regime_score(d,e,loc,r) == 4
                            else 999 if regime_score(d,e,loc,r) <= 1
                            else 1)),
    ]

    for label, delay_fn in policies:
        # Wrap delay_fn to skip (return 999 = never activates) for "skip" case
        def make_dfn(fn):
            def dfn(d,e,loc,r):
                v = fn(d,e,loc,r)
                return v
            return dfn
        t = run_regime_gated(cl, delay_fn=make_dfn(delay_fn),
                             adaptive_fn=afn_best, bounce_cfg=bounce)
        regime_stats(t, label)

    # ── SECTION 4: Regime-gated TP multiplier ───────────────────────────────
    print("\n" + "=" * 80)
    print("SECTION 4 — REGIME-GATED TP MULTIPLIER (wider TP on strong regimes)")
    print("  Baseline TP is per-profile. Multiplier widens/narrows it.")
    print("=" * 80 + "\n")

    tp_policies = [
        ("flat TP x1.0 + delay=1 (reference)",
         lambda d,e,loc,r: 1,
         lambda d,e,loc,r: 1.0),
        ("strong (score>=3) TP x1.5 + delay=0, rest TP x1.0 + delay=1",
         lambda d,e,loc,r: 0 if regime_score(d,e,loc,r) >= 3 else 1,
         lambda d,e,loc,r: 1.5 if regime_score(d,e,loc,r) >= 3 else 1.0),
        ("strong (score>=3) TP x2.0 + delay=0, rest TP x1.0 + delay=1",
         lambda d,e,loc,r: 0 if regime_score(d,e,loc,r) >= 3 else 1,
         lambda d,e,loc,r: 2.0 if regime_score(d,e,loc,r) >= 3 else 1.0),
        ("strong (score>=3) TP x1.5 + delay=1, rest TP x1.0 + delay=1",
         lambda d,e,loc,r: 1,
         lambda d,e,loc,r: 1.5 if regime_score(d,e,loc,r) >= 3 else 1.0),
        ("all TP x1.5 + delay=1",
         lambda d,e,loc,r: 1,
         lambda d,e,loc,r: 1.5),
        ("score=4 TP x2.0 + delay=0, score=3 TP x1.5 + delay=1, else TP x1.0 + delay=1",
         lambda d,e,loc,r: 0 if regime_score(d,e,loc,r) == 4 else 1,
         lambda d,e,loc,r: (2.0 if regime_score(d,e,loc,r) == 4
                            else 1.5 if regime_score(d,e,loc,r) == 3
                            else 1.0)),
    ]

    for label, dfn, tpfn in tp_policies:
        t = run_regime_gated(cl, delay_fn=dfn, tp_mult_fn=tpfn,
                             adaptive_fn=afn_best, bounce_cfg=bounce)
        regime_stats(t, label)

    # ── SECTION 5: Regime breakdown of best policy ───────────────────────────
    print("\n" + "=" * 80)
    print("SECTION 5 — BEST POLICY: breakdown by regime score and combo")
    print("=" * 80 + "\n")

    # Pick score>=3 -> delay=0+TP*1.5, else delay=1+TP*1.0 as candidate
    best_dfn  = lambda d,e,loc,r: 0 if regime_score(d,e,loc,r) >= 3 else 1
    best_tpfn = lambda d,e,loc,r: 1.5 if regime_score(d,e,loc,r) >= 3 else 1.0
    t_best = run_regime_gated(cl, delay_fn=best_dfn, tp_mult_fn=best_tpfn,
                              adaptive_fn=afn_best, bounce_cfg=bounce)
    df_best = pd.DataFrame(t_best)

    if not df_best.empty:
        print("  By regime score:")
        g = df_best.groupby("regime_score")["pnl_pips"].agg(
            ["count","sum","mean", lambda x: (x>0).mean()*100])
        g.columns = ["n","sum","avg","win%"]
        for sc, row in g.iterrows():
            d_label = "delay=0+TP*1.5" if sc >= 3 else "delay=1+TP*1.0"
            print(f"    score={sc} [{d_label}]  n={int(row['n']):>5}  sum={row['sum']:>+8.1f}p  "
                  f"avg={row['avg']:>+6.2f}p  win={row['win%']:.0f}%")

        print("\n  By combo (top 15):")
        g2 = df_best.groupby("combo")["pnl_pips"].agg(
            ["count","sum","mean", lambda x: (x>0).mean()*100])
        g2.columns = ["n","sum","avg","win%"]
        g2 = g2.sort_values("sum", ascending=False).head(15)
        for combo, row in g2.iterrows():
            print(f"    {combo:<35} n={int(row['n']):>5}  sum={row['sum']:>+8.1f}p  "
                  f"avg={row['avg']:>+6.2f}p  win={row['win%']:.0f}%")

        print("\n  By exit reason:")
        g3 = df_best.groupby("reason")["pnl_pips"].agg(["count","sum","mean"])
        for r, row in g3.iterrows():
            print(f"    {r:<18} n={int(row['count']):>5}  sum={row['sum']:>+9.1f}p  avg={row['mean']:>+6.2f}p")


if __name__ == "__main__":
    main()
