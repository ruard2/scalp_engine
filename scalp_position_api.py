"""
Position API Handler
Handles closing positions via CityIndex API.
"""

import requests
import json
import time
import csv
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from close_helpers import get_market_name
from lightstreamer_receiver import LightstreamerReceiver
import scalp_config as cfg


class PositionCloser:
    """
    Handles the API calls to close positions.
    """
    
    def __init__(self, session_token: str, trading_account_id: str, username: str):
        self.session_token = session_token
        self.trading_account_id = trading_account_id
        self.username = username
        self.ls_receiver = None
    
    def refresh_token(self):
        """Reload session token from file."""
        try:
            with open('session_token.json', 'r') as f:
                data = json.load(f)
            self.session_token = data.get('session_token', self.session_token)
        except Exception as e:
            print(f"[WARN] Token refresh failed: {e}")
    
    def last_check(self, order_id: int, direction: str, audit_id: str,
                   ask: float, bid: float, market_id: int) -> tuple:
        """
        Final price check before closing.
        Returns: (order_id, new_bid, new_ask, new_audit_id, market_id) or (None, ...) if aborted
        """
        current_price = (bid + ask) / 2
        threshold = current_price * 0.0005  # 0.05% improvement threshold
        
        time.sleep(cfg.LAST_CHECK_WAIT_SECS)
        
        try:
            receiver = LightstreamerReceiver()
            updated = receiver.fetch_market_data_one(market_id)
            receiver.disconnect()
            
            if updated:
                new_bid = updated['Bid']
                new_ask = updated['Offer']
                new_audit = updated['AuditId']
            else:
                return order_id, bid, ask, audit_id, market_id
        except Exception as e:
            print(f"[WARN] Last check failed: {e}")
            return order_id, bid, ask, audit_id, market_id
        
        # Check if price improved significantly
        if direction == 'buy':
            if new_ask > ask + threshold:
                print(f"[INFO] Price improved, aborting close for {order_id}")
                return None, new_bid, new_ask, new_audit, market_id
        else:
            if new_bid < bid - threshold:
                print(f"[INFO] Price improved, aborting close for {order_id}")
                return None, new_bid, new_ask, new_audit, market_id
        
        return order_id, new_bid, new_ask, new_audit, market_id
    
    def close_position(self, order_id: int, direction: str, price: float,
                       audit_id: str, quantity: float, bid: float, ask: float,
                       entry_price: float, market_id: int, reason: str) -> bool:
        """
        Close a position via the API.
        Returns True if successful.
        """
        # Final price check
        result = self.last_check(order_id, direction, audit_id, ask, bid, market_id)
        checked_order_id, new_bid, new_ask, new_audit_id, _ = result
        
        if not checked_order_id:
            return False
        
        print(f"[CLOSE] Closing {order_id} | {reason}")
        
        opposite = 'sell' if direction == 'buy' else 'buy'
        market_name = get_market_name(market_id)
        
        endpoint = "https://ciapi.cityindex.com/TradingAPI/order/newtradeorder"
        headers = {
            'Session': self.session_token,
            'UserName': self.username,
            'Content-Type': 'application/json'
        }
        
        payload = {
            "IfDone": [],
            "Direction": opposite,
            "BidPrice": new_bid,
            "OfferPrice": new_ask,
            "AuditId": new_audit_id,
            "AutoRollover": False,
            "MarketId": int(market_id),
            "Close": [int(order_id)],
            "Currency": None,
            "Quantity": quantity,
            "QuoteId": None,
            "PositionMethodId": 1,
            "TradingAccountId": self.trading_account_id,
            "MarketName": market_name,
            "Status": None,
            "isTrade": True
        }
        
        try:
            response = requests.post(endpoint, json=payload, headers=headers)
            if response.status_code == 200:
                close_price = new_bid if direction == 'buy' else new_ask
                print(f"[CLOSE] ✓ Closed {order_id} @ {close_price:.5f} | {reason}")
                
                # Log close
                self._log_close(order_id, market_id, direction, entry_price, close_price, reason)
                
                # Refresh token
                self.refresh_token()
                
                return True
            else:
                print(f"[ERROR] Close failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Close request failed: {e}")
            return False
    
    def _log_close(self, order_id: int, market_id: int, direction: str,
                   entry_price: float, close_price: float, reason: str):
        """Log the close to CSV files."""
        now = datetime.now(ZoneInfo("Europe/Amsterdam"))
        timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
        
        # Calculate P&L
        if direction == 'buy':
            pnl = close_price - entry_price
        else:
            pnl = entry_price - close_price
        pnl_pct = (pnl / entry_price) * 100 if entry_price else 0
        outcome = 'profit' if pnl > 0 else 'loss'
        
        # Log to close_reasons.csv
        if cfg.LOG_CLOSES:
            try:
                file_exists = os.path.isfile('close_reasons.csv')
                with open('close_reasons.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(['Timestamp', 'Opening Price', 'Direction', 'Closing Price', 'Close Reason'])
                    writer.writerow([timestamp, entry_price, direction, close_price, reason])
            except Exception as e:
                print(f"[WARN] Failed to log close reason: {e}")
        
        # Log to trade_summary.csv
        if cfg.LOG_SUMMARY:
            try:
                file_exists = os.path.isfile('trade_summary.csv')
                with open('trade_summary.csv', 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=[
                        'ClosedAt', 'OrderId', 'MarketId', 'Direction',
                        'EntryPrice', 'ClosePrice', 'PnL', 'PnLPct', 'Outcome', 'Reason'
                    ])
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow({
                        'ClosedAt': timestamp,
                        'OrderId': order_id,
                        'MarketId': market_id,
                        'Direction': direction,
                        'EntryPrice': f"{entry_price:.5f}",
                        'ClosePrice': f"{close_price:.5f}",
                        'PnL': f"{pnl:.5f}",
                        'PnLPct': f"{pnl_pct:.3f}",
                        'Outcome': outcome,
                        'Reason': reason
                    })
            except Exception as e:
                print(f"[WARN] Failed to log trade summary: {e}")
        
        # Update position_history.csv
        self._update_position_history(order_id, reason, close_price)

        # Post-exit watch: fetch next 5 OHLC bars and log for future optimisation
        self._log_post_exit_watch(order_id, market_id, direction, entry_price,
                                  close_price, reason, pnl_pct)

    def _log_post_exit_watch(self, order_id: int, market_id: int, direction: str,
                              entry_price: float, close_price: float,
                              reason: str, pnl_pct: float):
        """
        Fetch the next 5 OHLC bars after close and log them to post_exit_watch.csv.
        This answers: did price continue in our direction after we exited,
        or did we exit at the right time?
        Columns: OrderId, MarketId, Direction, CloseReason, PnLPct,
                 Bar1..5 (High, Low, Close), BestAfter, WorstAfter
        """
        try:
            import requests as req

            # Fetch 5 bars (5-min) after close via bar history API
            endpoint = (f"https://ciapi.cityindex.com/TradingAPI/market/"
                        f"{market_id}/barhistory"
                        f"?interval=MINUTE&span=5&PriceBars=6&priceType=MID")
            headers = {
                'Session': self.session_token,
                'UserName': self.username,
            }
            resp = req.get(endpoint, headers=headers, timeout=10)
            if resp.status_code != 200:
                return

            bars_raw = resp.json().get('PriceBars', [])
            # Skip the current bar (bar 0 = still-open bar), take next 5
            bars = bars_raw[:5] if len(bars_raw) >= 5 else bars_raw

            if not bars:
                return

            highs  = [b.get('High', 0) for b in bars]
            lows   = [b.get('Low',  0) for b in bars]
            closes = [b.get('Close', 0) for b in bars]

            if direction == 'buy':
                best_after  = (max(highs)  - close_price) / entry_price * 100
                worst_after = (min(lows)   - close_price) / entry_price * 100
            else:
                best_after  = (close_price - min(lows))   / entry_price * 100
                worst_after = (close_price - max(highs))  / entry_price * 100

            now = datetime.now(ZoneInfo("Europe/Amsterdam")).strftime('%Y-%m-%d %H:%M:%S')
            row = {
                'ClosedAt':   now,
                'OrderId':    order_id,
                'MarketId':   market_id,
                'Direction':  direction,
                'EntryPrice': round(entry_price,  5),
                'ClosePrice': round(close_price,  5),
                'PnLPct':     round(pnl_pct,      4),
                'CloseReason': reason,
                'BestAfter5b':  round(best_after,  4),
                'WorstAfter5b': round(worst_after, 4),
            }
            # Add per-bar data
            for i, (h, l, c) in enumerate(zip(highs, lows, closes), 1):
                row[f'Bar{i}_High']  = round(h, 5)
                row[f'Bar{i}_Low']   = round(l, 5)
                row[f'Bar{i}_Close'] = round(c, 5)

            watch_file = 'backtest_folder/post_exit_watch.csv'
            os.makedirs('backtest_folder', exist_ok=True)
            file_exists = os.path.isfile(watch_file)
            with open(watch_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)

        except Exception as e:
            print(f"[WARN] Post-exit watch failed: {e}")
    
    def _update_position_history(self, order_id: int, reason: str, close_price: float):
        """Update the position_history.csv with close info."""
        file_name = 'position_history.csv'
        if not os.path.isfile(file_name):
            return
        
        try:
            import pandas as pd
            df = pd.read_csv(file_name, on_bad_lines='skip', dtype=str)
            
            if 'OrderID' not in df.columns:
                return
            
            df['OrderID'] = df['OrderID'].astype(str)
            mask = df['OrderID'] == str(order_id)
            
            if mask.any():
                now = datetime.now(ZoneInfo("Europe/Amsterdam")).strftime('%Y-%m-%d %H:%M:%S')
                df.loc[mask, 'Status'] = 'Closed'
                df.loc[mask, 'CloseDate'] = now
                df.loc[mask, 'CloseReason'] = reason
                df.loc[mask, 'ClosingPrice'] = f"{close_price:.5f}"
                df.to_csv(file_name, index=False)
        except Exception as e:
            print(f"[WARN] Failed to update position history: {e}")
