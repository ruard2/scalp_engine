#!/usr/bin/env python3
"""
backtest_entry.py v2 — Entry timing analysis. Corrected simulation.

Bug fixed vs v1: positions were added to open_trades immediately with a FUTURE
entry price, causing check_exit to run against phantom prices during the wait.
Fix: use a pending_entries dict; position only becomes active on the actual
entry bar.

Tests on BOTH 10-min bars (current engine) AND 5-min bars.

Delay semantics (10-min bars):
  delay=0: signal on bar i → enter at bar i+1 open   (0 min wait)
  delay=1: confirm on bar i+1 closes → enter at bar i+2 open (10 min wait)
  delay=2: confirm on bar i+2 closes → enter at bar i+3 open (20 min wait)

Delay semantics (5-min bars):
  delay=1: confirm 5 min wait → enter at bar i+2 open
  delay=2: confirm 10 min wait → enter at bar i+3 open
  delay=4: confirm 20 min wait (same as 10-min delay=2) → enter at bar i+5 open
"""
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import live_engine as le
from backtest_v6 import SPREAD_PIPS
from backtest_bounce import flat_adaptive, rank_adaptive

HALF_SPREAD = (SPREAD_PIPS / 2) / 10000


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────
def load_5min_bars():
    from backtest_v6 import load_10min_bars as _load
    # Load the raw 5-min CSV directly (same file, but don't resample)
    import re
    from pathlib import Path as P
    csv = P(r"C:\Users\Ruard\testingversion2\5_min_data\fetched_data_eurusd_401697501.csv")
    df = pd.read_csv(csv)
    df["ts"] = df["BarDate"].str.extract(r"(\d+)")[0].astype(float) / 1000
    df["datetime"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close"})
    return df[["datetime","open","high","low","close"]].copy()


def load_10min_bars():
    from backtest_v6 import load_10min_bars as _load
    return _load()


# ─────────────────────────────────────────────────────────────────────────────
# Classify for 5-min bars (resample to 10-min for classifier, then expand)
# ─────────────────────────────────────────────────────────────────────────────
def classify_5min(df5):
    """
    Classify on 10-min bars, then broadcast labels back to 5-min rows.
    Each 5-min bar gets the label of the most recent completed 10-min bar.
    """
    df5 = df5.copy()
    df5["datetime"] = pd.to_datetime(df5["datetime"], utc=True)

    # Resample to 10-min
    df10 = (df5.set_index("datetime")
              .resample("10min")
              .agg(open=("open","first"), high=("high","max"),
                   low=("low","min"),   close=("close","last"))
              .dropna().reset_index())
    df10["datetime"] = pd.to_datetime(df10["datetime"], utc=True)

    cl10 = le.classify_bars(df10)
    cl10["datetime"] = pd.to_datetime(cl10["datetime"], utc=True)

    # Merge labels onto 5-min bars with merge_asof (backward — use CLOSED bar)
    df5_s = df5.sort_values("datetime").copy()
    cl10_s = cl10[["datetime","direction","environment","local","atr","atr_rank"]].copy()
    cl10_s = cl10_s.sort_values("datetime")
    merged = pd.merge_asof(df5_s, cl10_s, on="datetime", direction="backward")
    return merged.dropna(subset=["direction"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# CORRECT delayed-entry simulation
# Pending entries live in pending_entries until their actual entry bar arrives.
# ─────────────────────────────────────────────────────────────────────────────
def run_delayed(cl, delay_bars=0, require_same_dir=True,
                adaptive_fn=None, bounce_cfg=None, bar_minutes=10):
    """
    delay_bars: how many bars after signal to wait before entering.
    require_same_dir: if True, skip entry if direction changed during wait.
    bar_minutes: 5 or 10 — used only for labelling, not logic.
    """
    if adaptive_fn is None:
        adaptive_fn = flat_adaptive(2.5)

    trades = []
    open_trades = {}      # active positions
    bounce_pending = {}   # SL-intercepted, waiting for bounce
    pending_entries = {}  # {entry_bar_idx: trade_dict} — not active yet

    cl_len = len(cl)

    def _record(t, reason, pnl):
        trades.append({
            "direction": t["direction"], "combo": t["combo"],
            "session": t["session"], "atr_rank": t["atr_rank"],
            "pnl_pips": round(pnl, 2), "reason": reason,
            "bars_held": t["bars_held"], "sl_mult": t["sl_mult"],
        })

    for i in range(200, cl_len - 1):
        bar  = cl.iloc[i]
        hour = bar["datetime"].hour
        session = le.get_session_label(hour)

        # ── Activate pending entries whose entry bar has arrived ──────────
        if i in pending_entries:
            nxt = cl.iloc[i + 1] if i + 1 < cl_len else None
            if nxt is not None:
                pending = pending_entries.pop(i)
                entry_px = float(nxt["open"]) + (
                    HALF_SPREAD if pending["direction"] == "Bull" else -HALF_SPREAD)
                atr   = pending["atr"]
                sl_m  = pending["sl_mult"]
                pk    = pending["profile_key"]
                p     = le.PROFILES[pk]
                sl_dist = atr * sl_m
                tp_dist = p.tp_pips / 10000
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
                        "combo": t["combo"], "session": t["session"], "sl_mult": t["sl_mult"],
                    }
                    continue
            else:
                cp = float(bar["close"])
                exit_px = cp - HALF_SPREAD if d=="Bull" else cp + HALF_SPREAD
                pnl = (exit_px - ep)*10000 if d=="Bull" else (ep - exit_px)*10000
            _record(t, reason, pnl)

        # ── Entry signal check ────────────────────────────────────────────
        # Block if already in a position OR already have a pending entry
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

        if delay_bars == 0:
            # Immediate: enter at next bar open — add directly, no pending needed.
            if i + 1 >= cl_len:
                continue
            nxt = cl.iloc[i + 1]
            entry_px = float(nxt["open"]) + (HALF_SPREAD if d0=="Bull" else -HALF_SPREAD)
            p        = le.PROFILES[pk0]
            sl_dist  = atr * sl_m
            tp_dist  = p.tp_pips / 10000
            key      = f"{d0}_{e0}_{loc0}_{i}"
            open_trades[key] = {
                "combo": f"{d0}_{e0}_{loc0}", "direction": d0, "profile_key": pk0,
                "session": session, "entry_price": entry_px,
                "atr": atr, "atr_rank": atr_rank, "sl_mult": sl_m,
                "sl_price":  entry_px - sl_dist if d0=="Bull" else entry_px + sl_dist,
                "tp_price":  entry_px + tp_dist if d0=="Bull" else entry_px - tp_dist,
                "trail_dist": p.trail_dist / 10000, "trail_trigger": p.trail_trigger / 10000,
                "trail_active": False, "trail_stop": 0.0,
                "best_price": entry_px, "bars_held": 0,
            }
        else:
            # Check confirmation bar (its close, so it's bar i+delay_bars)
            conf_idx = i + delay_bars
            if conf_idx >= cl_len - 1:
                continue
            conf_bar = cl.iloc[conf_idx]
            ch = conf_bar["datetime"].hour
            cs = le.get_session_label(ch)
            cd = str(conf_bar["direction"])
            ce = str(conf_bar["environment"])
            if require_same_dir:
                if cd != d0:
                    continue   # direction flipped — skip
                if ce in ("Range", "Compression"):
                    continue   # environment degraded — skip
                if not le.entry_allowed(cd, ce, cs, ch):
                    continue
            # Register as pending — activates on iteration conf_idx,
            # enters at cl.iloc[conf_idx+1]["open"]
            if conf_idx + 1 >= cl_len:
                continue
            key = f"{d0}_{e0}_{loc0}_{i}"
            pending_entries[conf_idx] = {
                "key": key, "combo": f"{d0}_{e0}_{loc0}",
                "direction": d0, "profile_key": pk0,
                "session": session, "atr": atr, "atr_rank": atr_rank, "sl_mult": sl_m,
            }

    return trades


# ─────────────────────────────────────────────────────────────────────────────
# Stats helper
# ─────────────────────────────────────────────────────────────────────────────
def stats(trades, label=""):
    df = pd.DataFrame(trades)
    if df.empty:
        print(f"  {label:<55} NO TRADES"); return
    n   = len(df)
    tot = df["pnl_pips"].sum()
    avg = df["pnl_pips"].mean()
    cum = df["pnl_pips"].cumsum()
    dd  = (cum - cum.cummax()).min()
    wr  = (df["pnl_pips"] > 0).mean() * 100
    print(f"  {label:<55} n={n:>5}  sum={tot:>+8.1f}p  avg={avg:>+6.2f}p  "
          f"win={wr:4.1f}%  DD={dd:>8.1f}p")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    afn_base = flat_adaptive(2.5)
    afn_best = rank_adaptive([(90,4.0,3.0,2.0),(75,3.5,2.0,1.0),(0,2.5,0.0,0.0)])
    bounce   = dict(max_wait_bars=2, safety_pips=6, min_bounce_pips=1.0, trail_pips=2.0)

    # ── 10-MIN BARS ──────────────────────────────────────────────────────────
    print("Loading 10-min bars and classifying...")
    from backtest_v6 import load_10min_bars
    df10 = load_10min_bars()
    cl10 = le.classify_bars(df10)
    print(f"  {len(cl10)} bars (10-min)\n")

    print("=" * 80)
    print("10-MIN BARS — BASELINE ADAPTIVE PARAMS (2.5x SL, no bounce)")
    print("  delay=1: enter 10min later  delay=2: enter 20min later")
    print("=" * 80 + "\n")
    for delay, req, label in [
        (0, True,  "immediate (baseline)"),
        (1, True,  "delay 1 bar (10min), same dir required"),
        (1, False, "delay 1 bar (10min), any signal"),
        (2, True,  "delay 2 bars (20min), same dir required"),
        (2, False, "delay 2 bars (20min), any signal"),
        (3, True,  "delay 3 bars (30min), same dir required"),
    ]:
        t = run_delayed(cl10, delay_bars=delay, require_same_dir=req, adaptive_fn=afn_base)
        stats(t, label)

    print()
    print("=" * 80)
    print("10-MIN BARS — BEST COMBINED (adaptive SL + bounce)")
    print("=" * 80 + "\n")
    for delay, req, label in [
        (0, True,  "immediate + adaptive + bounce"),
        (1, True,  "delay 1 bar (10min) + adaptive + bounce"),
        (2, True,  "delay 2 bars (20min) + adaptive + bounce"),
        (3, True,  "delay 3 bars (30min) + adaptive + bounce"),
    ]:
        t = run_delayed(cl10, delay_bars=delay, require_same_dir=req,
                        adaptive_fn=afn_best, bounce_cfg=bounce)
        stats(t, label)

    # Exit breakdown for delay=1 best
    print()
    t_d1 = run_delayed(cl10, delay_bars=1, require_same_dir=True,
                        adaptive_fn=afn_best, bounce_cfg=bounce)
    df_d1 = pd.DataFrame(t_d1)
    if not df_d1.empty:
        print("  Reason breakdown (10-min, delay=1bar, adaptive+bounce):")
        g = df_d1.groupby("reason")["pnl_pips"].agg(["count","sum","mean"])
        for r, row in g.iterrows():
            print(f"    {r:<18} n={int(row['count']):>5}  sum={row['sum']:>+9.1f}p  avg={row['mean']:>+6.2f}p")

    # ── 5-MIN BARS ───────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("5-MIN BARS — same classifier (10-min labels broadcast to 5-min bars)")
    print("  delay=1: enter 5min later  delay=2: enter 10min later")
    print("  delay=4: enter 20min later (= 10-min delay=2)")
    print("=" * 80 + "\n")

    print("Loading 5-min bars and classifying...")
    df5 = load_5min_bars()
    cl5 = classify_5min(df5)
    print(f"  {len(cl5)} bars (5-min)\n")

    for delay, req, label in [
        (0, True,  "immediate (baseline)"),
        (1, True,  "delay 1 bar (5min), same dir"),
        (2, True,  "delay 2 bars (10min), same dir"),
        (4, True,  "delay 4 bars (20min), same dir"),
        (6, True,  "delay 6 bars (30min), same dir"),
        (2, False, "delay 2 bars (10min), any signal"),
        (4, False, "delay 4 bars (20min), any signal"),
    ]:
        t = run_delayed(cl5, delay_bars=delay, require_same_dir=req, adaptive_fn=afn_base,
                        bar_minutes=5)
        stats(t, label)

    print()
    print("  5-min + adaptive + bounce:")
    for delay, req, label in [
        (0, True,  "immediate + adaptive + bounce"),
        (1, True,  "delay 1 bar (5min) + adaptive + bounce"),
        (2, True,  "delay 2 bars (10min) + adaptive + bounce"),
        (4, True,  "delay 4 bars (20min) + adaptive + bounce"),
    ]:
        t = run_delayed(cl5, delay_bars=delay, require_same_dir=req,
                        adaptive_fn=afn_best, bounce_cfg=bounce, bar_minutes=5)
        stats(t, label)

    # ── SANITY CHECK: do both sims give same baseline? ────────────────────────
    print("\n" + "=" * 80)
    print("SANITY CHECK — immediate entry: 10-min sim vs 5-min sim should be close")
    print("=" * 80)
    t10 = run_delayed(cl10, delay_bars=0, adaptive_fn=afn_base)
    t5  = run_delayed(cl5,  delay_bars=0, adaptive_fn=afn_base, bar_minutes=5)
    stats(t10, "10-min immediate")
    stats(t5,  "5-min immediate (should be similar)")


if __name__ == "__main__":
    main()
