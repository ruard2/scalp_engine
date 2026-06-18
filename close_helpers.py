import csv
import os
import pandas as pd
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import re
from datetime import datetime, timezone

def parse_dotnet_datetime(dotnet_date: str) -> datetime:
    """
    Convert .NET-style '/Date(ms)/' to a Python datetime (UTC, tz-aware).
    """
    match = re.match(r"\/Date\((\d+)\)\/", dotnet_date)
    if match:
        timestamp_ms = int(match.group(1))
        # return an *aware* UTC datetime instead of naive
        return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
    raise ValueError(f"Invalid .NET datetime format: {dotnet_date}")





def calculate_effective_hours(start: datetime, end: datetime) -> float:
    """
    Calculate “effective” open hours between two UTC datetimes,
    excluding the weekend block (Fri 22:00 CET → Mon 02:00 CET).
    Returns hours as a float.
    """
    effective_seconds = 0
    current = start
    while current < end:
        next_hour = current + pd.Timedelta(hours=1)
        # Exclude Fri 22:00+ CET, all Saturday/Sunday, Mon before 02:00 CET
        cet = current.astimezone(ZoneInfo("Europe/Amsterdam"))
        weekday = cet.weekday()
        hour = cet.hour
        if not (
            (weekday == 4 and hour >= 22) or  # Fri 22:00+
            (weekday == 5) or                # All Saturday
            (weekday == 6) or                # All Sunday
            (weekday == 0 and hour < 2)      # Mon before 02:00
        ):
            effective_seconds += (next_hour - current).total_seconds()
        current = next_hour
    return effective_seconds / 3600.0


def detect_strong_adverse_signal(
    position: pd.Series,
    indicators_file: str,
    market_id,
    multiplier: float = 6,
    window: int = 3
) -> bool:
    """
    If the last `window` bars from the CSV show large moves in the opposite direction
    to our position, return True to indicate we should close immediately.
    Now mode-aware: determines trading mode from position.
    """
    try:
        # Determine trading mode from position
        trading_mode = None
        if hasattr(position, 'get'):
            # Check if position dict/Series has trading_mode
            trading_mode = position.get('trading_mode', None)
            
            # Fallback: check pattern for timeframe indicators
            if not trading_mode:
                pattern = str(position.get('pattern_name', '')).lower()
                if any(tf in pattern for tf in ['5min', '15min', '30min', '1h', '60min']):
                    trading_mode = "scalp"
                else:
                    trading_mode = "swing"
        
        # Read indicators with mode awareness
        row_dict = get_indicator_values(market_id, trading_mode=trading_mode)
        if not row_dict:
            print(f"[ERROR] MarketId {market_id} not found in indicators.")
            return False

        avg_change = row_dict.get('AverageChange')
        market_data_file = row_dict.get('csv_file')

        # Guard against NaN or non-string paths:
        if not isinstance(market_data_file, str) or market_data_file.strip() == "":
            print(f"[ERROR] Invalid market_data_file for MarketId {market_id}: {market_data_file}")
            return False

        if not os.path.exists(market_data_file):
            print(f"[ERROR] Market data file {market_data_file} not found.")
            return False

        market_data = pd.read_csv(market_data_file)
        market_data['Pct_Change'] = market_data['Close'].pct_change().abs()
        market_data['Direction'] = (
            market_data['Close'].diff() > 0
        ).astype(int) - (market_data['Close'].diff() < 0).astype(int)

        threshold = multiplier * avg_change
        if len(market_data) < window:
            # Not enough bars to decide
            return False

        last_rows = market_data.tail(window)
        window_changes = last_rows['Pct_Change']
        window_directions = last_rows['Direction']

        adverse_dir = -1 if position['Direction'].lower() == 'buy' else 1

        if (window_changes > threshold).all() and (window_directions.eq(adverse_dir).all()):
            return True

        return False

    except Exception as e:
        print(f"[ERROR] Exception during strong signal detection: {e}")
        return False




# ─────────────────────────────────────────────────────────────────────────────
# 2026-05-19: Module-level indicator cache for get_indicator_values.
# Previously called 28+ times per MC cycle (4× per position × 7 positions)
# with a full sequential CSV scan each time. indicators_swing.csv is written
# at most once every 5 hours by DynamicIndicators. Cache by file mtime:
# re-read only when the file actually changes.
# Structure: { file_path: {'mtime': float, 'data': {market_id_str: dict}} }
# ─────────────────────────────────────────────────────────────────────────────
_indicator_file_cache: dict = {}


def _load_indicators_cached(file_path: str) -> dict:
    """Return {market_id_str: row_dict} for file_path, reloading on mtime change."""
    try:
        mtime = os.path.getmtime(file_path)
    except OSError:
        return {}
    cached = _indicator_file_cache.get(file_path)
    if cached is not None and cached['mtime'] == mtime:
        return cached['data']
    data = {}
    try:
        with open(file_path, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                mid = str(row.get('MarketId', '')).strip()
                if mid:
                    data[mid] = dict(row)
        _indicator_file_cache[file_path] = {'mtime': mtime, 'data': data}
    except Exception:
        pass
    return data


def get_indicator_values(market_id, trading_mode=None, indicators_file='indicators.csv'):
    """
    Reads indicators CSV for the given MarketId.
    Now mode-aware: can read from indicators_scalp.csv or indicators_swing.csv

    2026-05-19: Cached by file mtime — re-reads only when file changes.
    Previously did a full sequential CSV scan on every call.
    """
    if trading_mode == "scalp":
        files_to_try = ["indicators_scalp.csv"]
    elif trading_mode == "swing":
        files_to_try = ["indicators_swing.csv"]
    elif trading_mode is None:
        files_to_try = ["indicators_swing.csv", "indicators_scalp.csv", indicators_file]
    else:
        files_to_try = [indicators_file]

    for file_path in files_to_try:
        if not os.path.exists(file_path):
            continue
        data = _load_indicators_cached(file_path)
        row = data.get(str(market_id))
        if row is None:
            continue
        try:
            result = {
                'MarketId':      row.get('MarketId'),
                'Window':        int(row['Window']) if row.get('Window') else 3,
                'MiddleBand':    float(row['MiddleBand']) if row.get('MiddleBand') else 0.0,
                'UpperBand':     float(row['UpperBand']) if row.get('UpperBand') else 0.0,
                'LowerBand':     float(row['LowerBand']) if row.get('LowerBand') else 0.0,
                'SpreadBuffer':  float(row['SpreadBuffer']) if row.get('SpreadBuffer') else 0.5,
                'LossThreshold': float(row['LossThreshold']) if row.get('LossThreshold') else 0.002,
                'csv_file':      row.get('csv_file'),
            }
            optional_fields = [
                'ATR14', 'ATR50', 'AverageChange', 'MarketName',
                'MC_ARM_PCT', 'MC_ARM_CUSHION_SPREADS', 'MC_TS_RATCHET_SPREADS',
                'MC_DEEP_BREACH_MULT', 'MC_DEBOUNCE_SECS', 'trading_mode', 'ATR14_pct', 'ATR50_pct',
            ]
            numeric = {'ATR14', 'ATR50', 'AverageChange', 'MC_ARM_PCT',
                       'MC_ARM_CUSHION_SPREADS', 'MC_TS_RATCHET_SPREADS',
                       'MC_DEEP_BREACH_MULT', 'MC_DEBOUNCE_SECS', 'ATR14_pct', 'ATR50_pct'}
            for field in optional_fields:
                val = row.get(field)
                if val not in (None, '', 'nan'):
                    try:
                        result[field] = float(val) if field in numeric else val
                    except (ValueError, TypeError):
                        result[field] = val
            return result
        except Exception as e:
            print(f"[ERROR] Error reading values from {file_path}: {e}")
            continue

    print(f"[WARN] MarketId {market_id} not found in any indicators file (tried: {files_to_try}).")
    return None


def get_market_name(market_id, csv_file='market_ids.csv') -> str:
    """
    Retrieve the market's name from market_ids.csv. Returns
    the 'name' column matching 'market_id', or an error string.
    """
    try:
        with open(csv_file, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['market_id'] == str(market_id):
                    return row['name']
        return "Market ID not found."
    except FileNotFoundError:
        return "CSV file not found."
    except KeyError:
        return "Invalid CSV format."