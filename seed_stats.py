#!/usr/bin/env python3
"""
seed_stats.py — One-time seed of live_stats.json from a CityIndex account history export.

Usage:
    python seed_stats.py "AccountHistory6_25_2026 9_43_17 AM.xls"

The XLS is actually an HTML table export from the broker.
After running this, live_stats.json is fully seeded and the live engine
continues appending to it from that point forward — no dependency on the file.
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

STATS_FILE = Path(__file__).parent / "live_stats.json"


def parse_broker_export(path: str) -> pd.DataFrame:
    tables = pd.read_html(path, encoding="utf-8")
    df = tables[0]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["date_utc"] = df["Date"].dt.date.astype(str)
    df["pnl_eur"]  = pd.to_numeric(df["Realised P&L"], errors="coerce").fillna(0.0)
    df["details"]  = df["Details"].astype(str).str.upper()
    return df


def build_buckets(df: pd.DataFrame) -> dict:
    """Aggregate rows into day/week/month/year buckets."""
    buckets: dict = {"daily": {}, "weekly": {}, "monthly": {}, "yearly": {}}

    for date_str, group in df.groupby("date_utc"):
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        iso_cal   = dt.isocalendar()
        week_key  = f"{iso_cal[0]}-W{iso_cal[1]:02d}"
        month_key = dt.strftime("%Y-%m")
        year_key  = dt.strftime("%Y")

        trade_rows  = group[~group["details"].str.contains("COMMISSION|FINANCING")]
        cost_rows   = group[ group["details"].str.contains("COMMISSION|FINANCING")]

        trades         = len(trade_rows)
        gross_eur      = round(trade_rows["pnl_eur"].sum(), 4)
        commission_eur = round(cost_rows["pnl_eur"].sum(), 4)
        net_eur        = round(gross_eur + commission_eur, 4)

        entry = {
            "trades":         trades,
            "gross_pips":     0.0,    # not available from broker EUR export
            "gross_eur":      gross_eur,
            "commission_eur": commission_eur,
            "net_eur":        net_eur,
        }

        for section, key in [
            ("daily", date_str), ("weekly", week_key),
            ("monthly", month_key), ("yearly", year_key),
        ]:
            b = buckets[section].setdefault(key, {
                "trades": 0, "gross_pips": 0.0, "gross_eur": 0.0,
                "commission_eur": 0.0, "net_eur": 0.0,
            })
            b["trades"]         += entry["trades"]
            b["gross_eur"]       = round(b["gross_eur"]       + entry["gross_eur"],      4)
            b["commission_eur"]  = round(b["commission_eur"]  + entry["commission_eur"], 4)
            b["net_eur"]         = round(b["net_eur"]         + entry["net_eur"],        4)

    return buckets


def print_summary(buckets: dict):
    print("\nDaily breakdown:")
    print(f"  {'Date':<12} {'Trades':>7} {'Gross €':>9} {'Comm €':>9} {'Net €':>9}")
    print(f"  {'-'*50}")
    for day, e in sorted(buckets["daily"].items()):
        print(f"  {day:<12} {e['trades']:>7} {e['gross_eur']:>+9.4f} "
              f"{e['commission_eur']:>+9.4f} {e['net_eur']:>+9.4f}")

    print("\nWeekly:")
    for wk, e in sorted(buckets["weekly"].items()):
        print(f"  {wk:<12} trades={e['trades']}  gross={e['gross_eur']:+.4f}€  "
              f"comm={e['commission_eur']:+.4f}€  net={e['net_eur']:+.4f}€")

    print("\nMonthly:")
    for mo, e in sorted(buckets["monthly"].items()):
        print(f"  {mo:<12} trades={e['trades']}  gross={e['gross_eur']:+.4f}€  "
              f"comm={e['commission_eur']:+.4f}€  net={e['net_eur']:+.4f}€")

    print("\nYearly:")
    for yr, e in sorted(buckets["yearly"].items()):
        print(f"  {yr:<12} trades={e['trades']}  gross={e['gross_eur']:+.4f}€  "
              f"comm={e['commission_eur']:+.4f}€  net={e['net_eur']:+.4f}€")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python seed_stats.py <broker_export.xls>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Reading: {path}")
    df = parse_broker_export(path)
    print(f"  {len(df)} rows across {df['date_utc'].nunique()} day(s)")

    buckets = build_buckets(df)
    print_summary(buckets)

    # Merge into existing stats (if engine already wrote some live trades)
    if STATS_FILE.exists():
        with open(STATS_FILE, encoding="utf-8") as f:
            stats = json.load(f)
        print(f"\nMerging with existing {STATS_FILE.name}...")
    else:
        stats = {}

    for section in ("daily", "weekly", "monthly", "yearly"):
        target = stats.setdefault(section, {})
        for key, entry in buckets[section].items():
            if key in target:
                # Add historical numbers to any existing live-engine entries
                t = target[key]
                t["trades"]         += entry["trades"]
                t["gross_eur"]       = round(t.get("gross_eur", 0.0)       + entry["gross_eur"],      4)
                t["commission_eur"]  = round(t.get("commission_eur", 0.0)  + entry["commission_eur"], 4)
                t["net_eur"]         = round(t.get("net_eur", 0.0)         + entry["net_eur"],        4)
            else:
                target[key] = entry

    stats["last_updated"] = datetime.now(timezone.utc).isoformat()
    stats["seeded_from"]  = Path(path).name

    temp = STATS_FILE.with_suffix(".tmp")
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, STATS_FILE)

    print(f"\nWritten to: {STATS_FILE}")


if __name__ == "__main__":
    main()
