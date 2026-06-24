#!/usr/bin/env python3
"""
backtest_exhaustion.py — Test TP-ratio decay as a trend exhaustion signal.

Idea: if the recent N closed trades in the same direction are mostly TrailSL
(small wins) rather than TP hits, momentum is fading. Block Pullback/RC in
that regime.

Tracks per-direction: last N exit reasons. When TP fraction < threshold,
Pullback/RC entries in that direction are blocked.

Sweeps:
  - window N = 3, 4, 5
  - TP-rate threshold = 0.0 (0%), 0.33 (1-in-3), 0.50 (half)

Baseline included for comparison. Adaptive SL + 1-bar delay as per live engine.

Also prints what would have happened on June 23-24 for each best config.
"""
import sys
from collections import deque
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from live_engine import (
    PROFILES, get_profile_key, entry_allowed, check_exit,
    get_session_label, classify_bars, get_adaptive_sl_mult,
)

DATA_FILE   = Path(r"C:\Users\Ruard\testingversion2\5_min_data\fetched_data_eurusd_401697501.csv")
HALF_SPREAD = 0.5 / 10000
LOCALS_RESTRICTED = {"Pullback", "ReversalCandidate"}

# Also block all locals (not just Pullback/RC) when exhausted — tested separately
BLOCK_ALL_LOCALS = False


def load_10min_bars() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    ts = df["BarDate"].str.extract(r"(\d+)")[0].astype("int64")
    df["datetime"] = pd.to_datetime(ts, unit="ms", utc=True)
    df = df.rename(columns={"Open":"open","High":"high","Low":"low",
                             "Close":"close","Volume":"volume"})
    df = df[["datetime","open","high","low","close","volume"]]
    df = df.set_index("datetime").sort_index()
    r = df.resample("10min").agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"), close=("close","last"), volume=("volume","sum")
    ).dropna(subset=["open"]).reset_index()
    r["date_utc"] = r["datetime"].dt.date.astype(str)
    return r


def run(cl: pd.DataFrame, window: int, tp_threshold: float,
        block_locals=LOCALS_RESTRICTED) -> pd.DataFrame:
    """
    window        : look at last N same-direction closes
    tp_threshold  : if TP fraction < this → exhaustion gate active
                    0.0 = never (baseline), >0 = active filter
    block_locals  : set of local labels to block when exhausted
    """
    trades = []
    open_trades: dict = {}
    pending_signal = None

    # Per-direction sliding window of recent exit reasons
    recent = {"Bull": deque(maxlen=window), "Bear": deque(maxlen=window)}

    for i in range(200, len(cl) - 1):
        bar = cl.iloc[i]
        nxt = cl.iloc[i + 1]
        hour    = bar["datetime"].hour
        session = get_session_label(hour)

        # ── exits ───────────────────────────────────────────────────────────
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
            px  = px - HALF_SPREAD if d == "Bull" else px + HALF_SPREAD
            pnl = (px - t["entry_price"]) * 10000 if d == "Bull" else (t["entry_price"] - px) * 10000

            # Update sliding window with this exit reason
            recent[d].append(reason)

            trades.append({
                "entry_time":        t["entry_time"],
                "exit_time":         bar["datetime"],
                "direction":         d,
                "local":             t["local"],
                "session":           t["session"],
                "pnl_pips":          round(pnl, 2),
                "bars":              t["bars_held"],
                "reason":            reason,
                "tp_rate_at_entry":  t["tp_rate_at_entry"],
                "window_size":       t["window_size"],
                "exhausted_at_entry":t["exhausted_at_entry"],
            })
            del open_trades[key]

        # ── 1-bar delay ──────────────────────────────────────────────────────
        d   = str(bar["direction"])
        e   = str(bar["environment"])
        loc = str(bar["local"])
        pk  = get_profile_key(d, e, loc)

        do_entry = False
        entry_d = entry_e = entry_loc = entry_pk = entry_bar = None

        if len(open_trades) >= 1:
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
                # Exhaustion gate
                if confirmed and tp_threshold > 0:
                    r_win = recent[ps["d"]]
                    if len(r_win) >= window:
                        tp_rate = sum(1 for r in r_win if r == "TP") / len(r_win)
                        if tp_rate < tp_threshold and ps["loc"] in block_locals:
                            confirmed = False

                if confirmed:
                    do_entry = True
                    entry_d, entry_e, entry_loc, entry_pk, entry_bar = (
                        ps["d"], ps["e"], ps["loc"], ps["pk"], ps["bar"])
                pending_signal = None

            if not do_entry and entry_allowed(d, e, session, hour) and pk is not None:
                pending_signal = {"d":d,"e":e,"loc":loc,"pk":pk,"bar":bar}

        if do_entry:
            b        = entry_bar
            entry_px = float(nxt["open"]) + (HALF_SPREAD if entry_d=="Bull" else -HALF_SPREAD)
            atr      = float(b["atr"])      if pd.notna(b["atr"])      and b["atr"]>0 else 0.0006
            atr_rank = float(b["atr_rank"]) if pd.notna(b["atr_rank"]) else 50.0
            p        = PROFILES[entry_pk]
            sl_mult  = get_adaptive_sl_mult(entry_pk, atr_rank)
            sl_dist  = atr * sl_mult
            tp_dist  = p.tp_pips / 10000

            r_win    = recent[entry_d]
            tp_rate  = sum(1 for r in r_win if r == "TP") / len(r_win) if r_win else 1.0
            exhausted = (len(r_win) >= window and tp_rate < tp_threshold and tp_threshold > 0)

            key = f"{entry_d}_{entry_loc}_{bar['datetime']:%m%d%H%M}"
            open_trades[key] = {
                "direction":          entry_d,
                "local":              entry_loc,
                "profile_key":        entry_pk,
                "session":            session,
                "entry_price":        entry_px,
                "entry_time":         nxt["datetime"],
                "sl_price":           entry_px - sl_dist if entry_d=="Bull" else entry_px + sl_dist,
                "tp_price":           entry_px + tp_dist if entry_d=="Bull" else entry_px - tp_dist,
                "trail_dist":         p.trail_dist / 10000,
                "trail_trigger":      p.trail_trigger / 10000,
                "trail_active":       False,
                "trail_stop":         0.0,
                "best_price":         entry_px,
                "bars_held":          0,
                "tp_rate_at_entry":   round(tp_rate, 2),
                "window_size":        len(r_win),
                "exhausted_at_entry": exhausted,
            }

    return pd.DataFrame(trades)


def stats(tr: pd.DataFrame):
    wins  = tr[tr["pnl_pips"] > 0]
    total = tr["pnl_pips"].sum()
    dd    = (tr["pnl_pips"].cumsum() - tr["pnl_pips"].cumsum().cummax()).min()
    return len(tr), total, tr["pnl_pips"].mean(), len(wins)/len(tr)*100, dd


def main():
    print("Loading and classifying bars...")
    df = load_10min_bars()
    cl = classify_bars(df)
    print(f"  {len(cl)} bars  "
          f"({cl['datetime'].iloc[0]:%Y-%m-%d} -> {cl['datetime'].iloc[-1]:%Y-%m-%d})")

    # Baseline
    tr_base = run(cl, window=3, tp_threshold=0.0)
    n0, t0, a0, w0, d0 = stats(tr_base)
    base_total = t0

    print(f"\n{'='*76}")
    print(f"  TP-RATE EXHAUSTION GATE SWEEP — block Pullback/RC when TP rate < threshold")
    print(f"{'='*76}")
    print(f"  {'Config':<36} {'N':>5} {'Total':>9} {'Avg':>7} {'Win%':>6} {'DD':>8} {'vs base':>9}")
    print(f"  {'-'*74}")

    best_delta = -9999
    best_cfg   = None

    print(f"  {'Baseline (no gate)':<36} {n0:>5} {t0:>+9.1f}p {a0:>+6.2f}p {w0:>5.1f}% {d0:>7.1f}p {'—':>9}")

    for window in [3, 4, 5]:
        for threshold in [0.33, 0.50, 0.67]:
            tr = run(cl, window=window, tp_threshold=threshold)
            n, t, a, w, d = stats(tr)
            delta = t - base_total
            label = f"W={window}  TP<{threshold:.0%}"
            marker = " <-- BEST" if delta > best_delta else ""
            if delta > best_delta:
                best_delta = delta
                best_cfg   = (window, threshold)
                best_tr    = tr
            print(f"  {label:<36} {n:>5} {t:>+9.1f}p {a:>+6.2f}p {w:>5.1f}% {d:>7.1f}p {delta:>+9.1f}p{marker}")
        print()

    # Also test blocking ALL locals (not just Pullback/RC) with best window
    print(f"  --- Block ALL locals when exhausted ---")
    for window in [3, 4, 5]:
        for threshold in [0.33, 0.50]:
            tr = run(cl, window=window, tp_threshold=threshold,
                     block_locals={"Pullback","ReversalCandidate","None","Impulse"})
            n, t, a, w, d = stats(tr)
            delta = t - base_total
            label = f"W={window}  TP<{threshold:.0%}  ALL locals"
            print(f"  {label:<36} {n:>5} {t:>+9.1f}p {a:>+6.2f}p {w:>5.1f}% {d:>7.1f}p {delta:>+9.1f}p")
        print()

    # Deep dive on best config
    if best_cfg:
        w_best, thr_best = best_cfg
        print(f"\n{'='*76}")
        print(f"  BEST CONFIG: window={w_best}  TP<{thr_best:.0%}")
        print(f"{'='*76}")

        tr = best_tr
        print(f"\n  By local:")
        g = tr.groupby("local")["pnl_pips"].agg(["count","sum","mean"])
        for loc, row in g.iterrows():
            print(f"    {loc:<22} n={int(row['count']):>4}  sum={row['sum']:>+8.1f}p  avg={row['mean']:>+6.2f}p")

        blocked = tr_base[~tr_base["entry_time"].isin(tr["entry_time"])]
        print(f"\n  Blocked trades (n={len(blocked)}):")
        if not blocked.empty:
            g2 = blocked.groupby("local")["pnl_pips"].agg(["count","sum","mean"])
            for loc, row in g2.iterrows():
                print(f"    {loc:<22} n={int(row['count']):>4}  sum={row['sum']:>+8.1f}p  avg={row['mean']:>+6.2f}p")

        # June 23-24 breakdown
        print(f"\n  June 23-24 with best config:")
        # Re-run keeping all trade detail for that period
        tr_detail = run(cl, window=w_best, tp_threshold=thr_best)
        recent_tr = tr_detail[tr_detail["entry_time"] >= pd.Timestamp("2026-06-23", tz="UTC")]
        recent_base = tr_base[tr_base["entry_time"] >= pd.Timestamp("2026-06-23", tz="UTC")]

        blocked_recent = recent_base[~recent_base["entry_time"].isin(recent_tr["entry_time"])]

        print(f"\n  {'Entry time':<22} {'Local':<22} {'tp_rate':>8} {'exhausted':>10} {'PnL':>8}  {'status'}")
        print(f"  {'-'*80}")
        for _, t in recent_base.iterrows():
            was_blocked = t["entry_time"] not in recent_tr["entry_time"].values
            tp_r = t["tp_rate_at_entry"]
            exh  = t["exhausted_at_entry"]
            flag = "<<< BLOCKED" if was_blocked else ""
            print(f"  {str(t['entry_time'])[:19]:<22} {t['local']:<22} "
                  f"{tp_r:>8.0%} {str(exh):>10} {t['pnl_pips']:>+8.1f}p  {t['reason']}  {flag}")

        base_r  = recent_base["pnl_pips"].sum()
        filt_r  = recent_tr["pnl_pips"].sum()
        print(f"\n  Baseline June 23-24 : {base_r:+.1f}p")
        print(f"  With exhaustion gate: {filt_r:+.1f}p")
        print(f"  Difference          : {filt_r - base_r:+.1f}p")


if __name__ == "__main__":
    main()
