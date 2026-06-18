import requests
from config import username, tradingAccountID
from datetime import datetime
import session_manager
import csv
import os

class TradeExecutor:
    def __init__(self):
        self.session = self._get_session()
        # 2026-05-19: Cache min opening amounts — never change intraday
        self._min_amount_cache: dict = {}

    def _get_session(self):
        try:
            return session_manager.get_session_token()
        except AttributeError:
            return session_manager.SessionManager().get_session_token()

    def fetch_min_opening_amount(self, market_id):
        if market_id in self._min_amount_cache:
            return self._min_amount_cache[market_id]
        url = "https://ciapi.cityindex.com/TradingAPI/market/information"
        headers = {
            'Session': self.session,
            'UserName': username,
            'Content-Type': 'application/json'
        }
        payload = {"MarketIds": [market_id]}
        resp = requests.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        for item in resp.json().get("MarketInformation", []):
            if item["MarketId"] == market_id:
                web_min = float(str(item.get("WebMinSize", "0.1")).replace(",", "."))
                inc = float(str(item.get("IncrementSize", "1")).replace(",", "."))
                result = max(web_min, inc)
                self._min_amount_cache[market_id] = result
                return result
        raise ValueError(f"Market ID {market_id} not found")


    def prefetch_min_amounts(self, market_ids: list):
        """Warm the min-amount cache for all markets in one API call. 2026-05-19"""
        if not market_ids:
            return
        try:
            url = "https://ciapi.cityindex.com/TradingAPI/market/information"
            headers = {'Session': self.session, 'UserName': username, 'Content-Type': 'application/json'}
            resp = requests.post(url, headers=headers, json={"MarketIds": list(market_ids)})
            resp.raise_for_status()
            for item in resp.json().get("MarketInformation", []):
                mid = item["MarketId"]
                web_min = float(str(item.get("WebMinSize", "0.1")).replace(",", "."))
                inc = float(str(item.get("IncrementSize", "1")).replace(",", "."))
                self._min_amount_cache[mid] = max(web_min, inc)
            print(f"[EXEC] Pre-warmed min amounts for {len(self._min_amount_cache)} markets")
        except Exception as e:
            print(f"[EXEC] prefetch_min_amounts failed (non-critical): {e}")

    def place_order(self, direction, bid, offer, audit_id, market_id, market_name, quantity_multiplier=1.0):
        """
        Place a new market order. As soon as CIAPI returns success, we force-refresh
        the CIAPI token so streaming can reconnect on its next fetch.
        
        Args:
            quantity_multiplier: Multiplies the base position size (1.0 = normal, 1.5-2.5 = macro boost)
        """
        min_amount = self.fetch_min_opening_amount(market_id)
        quantity = min_amount * quantity_multiplier
        # 2026-05-19: clamp to broker minimum — EQG sets multiplier < 1.0 (e.g. 0.50x)
        # producing quantity below WebMinSize → CityIndex StatusReason 8 rejection.
        quantity = max(quantity, min_amount)
        endpoint = "https://ciapi.cityindex.com/TradingAPI/order/newtradeorder"
        headers = {
            'Session': self.session,
            'UserName': username,
            'Content-Type': 'application/json'
        }

        trade_details = {
            "IfDone": [],
            "Direction": direction,
            "BidPrice": bid,
            "AuditId": audit_id,
            "AutoRollover": False,
            "MarketId": market_id,
            "OfferPrice": offer,
            "OrderId": 0,
            "Currency": market_name.split("/")[-1],
            "Quantity": quantity,
            "TradingAccountId": tradingAccountID,
            "MarketName": market_name,
            "isTrade": True,
            "PositionMethodId": 2
        }

        resp = requests.post(endpoint, json=trade_details, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        order_id = data.get("OrderId", 0)
        # immediately after your requests.post(…) and before logging:
        if direction.lower() == "buy":
            opening_price = offer   # you pay the offer (ask)
        else:
            opening_price = bid     # you receive the bid


        if order_id <= 0:
            raise Exception(f"Invalid OrderId: {order_id} — Response: {data}")

        print(f"[EXECUTED] {direction.upper()} {market_name} at {bid}/{offer} — OrderID {order_id}")



        return order_id, opening_price

    def log_trade_to_csv(self,
            date_time, market_id, order_id, opening_price,
            direction, currency_pair,
            signal, pattern, rsi, macd_hist, volatility, source_file,
            # NEW REQUIRED PARAMETERS for managing_close integration
            trading_mode='swing',
            rule_type='',
            # OPTIONAL PARAMETERS for future enhancements
            signal_source='',
            timeframe='',
            signal_score=0.0,
            regime='normal',
            quantity_multiplier=1.0,  # NEW: Track macro boost multiplier
            **extra_fields
        ):
        """
        Write a new "Open" entry into position_history.csv.
        
        CRITICAL SCHEMA UPDATE (v2.0):
        Now includes TradingMode and RuleType columns which are REQUIRED
        for managing_close.py to correctly determine scalp vs swing parameters.
        
        Parameters:
        -----------
        date_time : str
            Opening timestamp in format "YYYY-MM-DD HH:MM:SS"
        market_id : int
            Market ID from CityIndex
        order_id : int
            Order ID returned by broker
        opening_price : float
            Actual execution price
        direction : str
            'buy' or 'sell'
        currency_pair : str
            e.g., 'EUR/USD'
        signal : str
            'Buy' or 'Sell'
        pattern : str
            Pattern name that triggered the signal
        rsi : str/float
            RSI value at signal generation
        macd_hist : str/float
            MACD histogram value at signal generation
        volatility : str/float
            Volatility measure at signal generation
        source_file : str
            Source identifier (e.g., 'signal_logic', 'rule_engine_swing')
        trading_mode : str, default='swing'
            CRITICAL: 'scalp' or 'swing' - determines which parameters managing_close uses
        rule_type : str, default=''
            Additional classification: 'scalp', 'swing', 'news', 'reversal'
        signal_source : str, optional
            Source system: 'signal_logic', 'rule_engine', 'forex_factory', 'reversal_rules'
        timeframe : str, optional
            Timeframe of signal: '5min', '15min', '1h', 'D1', etc.
        signal_score : float, optional
            Quality score of the signal (0-100)
        regime : str, optional
            Volatility regime: 'quiet', 'normal', 'volatile'
        **extra_fields : dict
            Additional fields for future expansion (ignored for now)
        
        File Schema (20 columns):
        ------------------------
        DateTime, MarketID, OrderID, OpeningPrice, Direction, CurrencyPair,
        Signal, Pattern, RSI, MACD_Hist, Volatility, SourceFile, 
        TradingMode, RuleType, SignalSource, Timeframe, SignalScore, Regime,
        QuantityMultiplier,
        Status, CloseDate, CloseReason, ClosingPrice
        
        Notes:
        ------
        - TradingMode MUST be 'scalp' or 'swing' - managing_close depends on this
        - CloseDate, CloseReason, ClosingPrice are empty until position closes
        - Status starts as "Open", becomes "Closed" when managing_close updates it
        """
        file = "position_history.csv"
        
        # ENHANCED SCHEMA with managing_close integration fields
        fieldnames = [
            # Original 16 columns (maintained for backward compatibility)
            "DateTime","MarketID","OrderID","OpeningPrice",
            "Direction","CurrencyPair",
            "Signal","Pattern","RSI","MACD_Hist","Volatility","SourceFile",
            # NEW: Critical fields for managing_close
            "TradingMode",      # scalp | swing - REQUIRED for correct TS/SL parameters
            "RuleType",         # Additional classification
            "SignalSource",     # Which system generated the signal
            "Timeframe",        # Signal timeframe for analysis
            "SignalScore",      # Quality score for performance tracking
            "Regime",           # Volatility regime context
            "QuantityMultiplier", # NEW: Macro position boost multiplier (1.0 to 2.5)
            # Status tracking (unchanged)
            "Status",
            "CloseDate","CloseReason","ClosingPrice"
        ]
        
        # Build entry with all fields
        new_entry = {
            # Original fields
            "DateTime":     date_time,
            "MarketID":     market_id,
            "OrderID":      order_id,
            "OpeningPrice": opening_price,
            "Direction":    direction,
            "CurrencyPair": currency_pair,
            "Signal":       signal,
            "Pattern":      pattern,
            "RSI":          rsi,
            "MACD_Hist":    macd_hist,
            "Volatility":   volatility,
            "SourceFile":   source_file,
            
            # NEW: Critical managing_close integration fields
            "TradingMode":  trading_mode,      # REQUIRED: managing_close uses this!
            "RuleType":     rule_type,
            "SignalSource": signal_source,
            "Timeframe":    timeframe,
            "SignalScore":  signal_score,
            "Regime":       regime,
            "QuantityMultiplier": quantity_multiplier,  # NEW: Macro boost tracking
            
            # Status fields (empty until close)
            "Status":       "Open",
            "CloseDate":    "",
            "CloseReason":  "",
            "ClosingPrice": ""
        }

        # Write to CSV
        file_exists = os.path.isfile(file)
        with open(file, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(new_entry)
        
        # Debug logging to confirm what was written
        print(f"[CSV] Logged to position_history.csv: OrderID={order_id}, "
              f"TradingMode={trading_mode}, Pattern={pattern}, Source={signal_source}")