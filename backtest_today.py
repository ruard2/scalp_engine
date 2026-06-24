#!/usr/bin/env python3
"""
Quick check: what did dir_score look like on June 23-24 for all trades,
and would the |dir_score| < 20 filter have changed anything?
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from live_engine import (
    PROFILES, get_profile_key, entry_allowed, check_exit,
    get_session_label, classify_bars, get_adaptive_sl_mult,
)

DATA_FILE = Path(r"C:\Users\Ruard\testingversion2\5_min_data\fetched_data_eurusd_401697501.csv")
HALF_SPREAD = 0.5 / 10000

def load_10min_bars():
    df = pd.read_csv(DATA_FILE)
    ts = df["BarDate"].str.extract(r"(\d+)")[0].astype("int64")
    df["datetime"] = pd.to_datetime(ts, unit="ms", utc=True)
    df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
    df = df[["datetime","open","high","low","close","volume"]]
    df = df.set_index("datetime").sort_index()
    r = df.resample("10min").agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"), close=("close","last"), volume=("volume","sum")
    ).dropna(subset=["open"]).reset_index()
    r["date_utc"] = r["datetime"].dt.date.astype(str)
    return r

def run_with_dirscore(cl, date_filter=None):
    trades = []
    open_trades = {}
    pending_signal = None

    for i in range(200, len(cl) - 1):
        bar = cl.iloc[i]
        nxt = cl.iloc[i + 1]
        hour    = bar["datetime"].hour
        session = get_session_label(hour)

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
            px = px - HALF_SPREAD if d == "Bull" else px + HALF_SPREAD
            pnl = (px - t["entry_price"]) * 10000 if d == "Bull" else (t["entry_price"] - px) * 10000
            trades.append({**t, "exit_time": bar["datetime"], "pnl_pips": round(pnl,2),
                           "reason": reason, "exit_px": round(px,5)})
            del open_trades[key]

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
                confirmed = (d == ps["d"] and e not in ("Range","Compression")
                             and entry_allowed(d, e, session, hour) and pk is not None)
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
            dir_score= float(b["dir_score"]) if pd.notna(b.get("dir_score")) else 0.0
            p        = PROFILES[entry_pk]
            sl_mult  = get_adaptive_sl_mult(entry_pk, atr_rank)
            sl_dist  = atr * sl_mult
            key = f"{entry_d}_{entry_loc}_{bar['datetime']:%m%d%H%M}"
            open_trades[key] = {
                "direction": entry_d, "local": entry_loc, "profile_key": entry_pk,
                "session": session, "entry_price": entry_px,
                "entry_time": nxt["datetime"],
                "sl_price": entry_px - sl_dist if entry_d=="Bull" else entry_px + sl_dist,
                "tp_price": entry_px + p.tp_pips/10000 if entry_d=="Bull" else entry_px - p.tp_pips/10000,
                "trail_dist": p.trail_dist/10000, "trail_trigger": p.trail_trigger/10000,
                "trail_active": False, "trail_stop": 0.0, "best_price": entry_px, "bars_held": 0,
                "dir_score": dir_score, "atr_rank": atr_rank,
                "blocked": abs(dir_score) < 20 and entry_loc in ("Pullback","ReversalCandidate"),
            }

    return pd.DataFrame(trades)

def main():
    print("Loading and classifying...")
    df = load_10min_bars()
    cl = classify_bars(df)

    # Filter to June 22-24 only for display (enough context)
    recent = cl[cl["datetime"] >= "2026-06-22"].reset_index(drop=True)
    # But need 200 bars of history, so run full then filter output
    tr = run_with_dirscore(cl)
    tr = tr[tr["entry_time"] >= pd.Timestamp("2026-06-23", tz="UTC")].copy()

    print(f"\n{'='*80}")
    print("  JUNE 23-24 TRADES — with dir_score and filter impact")
    print(f"{'='*80}")
    print(f"  {'Entry time':<22} {'Dir':<5} {'Local':<22} {'dir_score':>10} {'|<20?':>6} "
          f"{'blocked?':>8} {'PnL':>8} {'Reason'}")
    print(f"  {'-'*78}")

    total_base   = 0.0
    total_filter = 0.0

    for _, t in tr.iterrows():
        ds       = t["dir_score"]
        loc      = t["local"]
        blocked  = t["blocked"]
        pnl      = t["pnl_pips"]
        weak     = abs(ds) < 20
        flag     = "<<< BLOCK" if blocked else ""
        total_base += pnl
        if not blocked:
            total_filter += pnl

        print(f"  {str(t['entry_time'])[:19]:<22} {t['direction']:<5} {loc:<22} "
              f"{ds:>10.1f} {'YES' if weak else 'no':>6} "
              f"{'BLOCKED' if blocked else '':>8} {pnl:>+8.1f}p  {t['reason']}  {flag}")

    print(f"  {'-'*78}")
    print(f"  {'Baseline total':<56} {total_base:>+8.1f}p")
    print(f"  {'With |dir_score|<20 filter':<56} {total_filter:>+8.1f}p")
    print(f"  {'Difference':<56} {total_filter - total_base:>+8.1f}p")

if __name__ == "__main__":
    main()
