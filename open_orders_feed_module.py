import requests
import pandas as pd
import logging
from session_manager import session_manager  # Assuming this handles token retrieval
from config import username, tradingAccountID
import json
from datetime import datetime
import pytz
import v3_config as cfg

class PositionManagementModule:
    @staticmethod
    def get_open_positions():

        # Define Amsterdam timezone
        amsterdam_tz = pytz.timezone("Europe/Amsterdam")

        # Get current time in Amsterdam
        now = datetime.now(amsterdam_tz)


        """ Queries the API for the list of open positions for the specified trading account. """

        token = session_manager.get_session_token()
        url = f"https://ciapi.cityindex.com/TradingAPI/order/activeorders"

        headers = {
            'Session': token,
            'UserName': username,
            'Content-Type': 'application/json'
        }

        body = {
            "TradingAccountId": tradingAccountID
        }

        response = requests.post(url, headers=headers, json=body)

        if response.status_code == 200:
            response_text = response.text
            data = json.loads(response_text)

            #print("\n━━━━━━━━ RAW ACTIVE ORDERS RESPONSE ━━━━━━━━")
            #print(json.dumps(data, indent=2))

            if 'ActiveOrders' not in data:
                print("[WARNING] 'ActiveOrders' not found in response.")
                return []

            open_positions = []

            for order in data['ActiveOrders']:
                trade_order = order.get('TradeOrder', {})
                open_positions.append({
                    'OrderId': trade_order.get('OrderId'),
                    'Direction': trade_order.get('Direction'),
                    'Quantity': trade_order.get('Quantity'),
                    'Price': trade_order.get('Price'),
                    'MarketId': trade_order.get('MarketId'),
                    'LastChangedDateTimeUTC': trade_order.get('LastChangedDateTimeUTC')
                })

            if cfg.DEBUG_MODE:
                print(f"[POS] {len(open_positions)} open position(s) found")
            #for pos in open_positions:
                #print(f"→ OrderId: {pos['OrderId']} | MarketId: {pos['MarketId']} | Direction: {pos['Direction']} | Price: {pos['Price']}")

            return open_positions

        elif response.status_code == 401:
            logging.warning("Session is not valid, refreshing token...")
            token = session_manager.refresh_session_token()
            headers['Session'] = token
            response = requests.post(url, headers=headers, json=body)

            if response.status_code == 200:
                response_text = response.text
                data = json.loads(response_text)

                print("\n[INFO] Refreshed session. Raw response:")
                print(json.dumps(data, indent=2))

                open_positions = []
                for order in data.get('ActiveOrders', []):
                    trade_order = order.get('TradeOrder', {})
                    open_positions.append({
                        'OrderId': trade_order.get('OrderId'),
                        'Direction': trade_order.get('Direction'),
                        'Quantity': trade_order.get('Quantity'),
                        'Price': trade_order.get('Price'),
                        'MarketId': trade_order.get('MarketId'),
                        'LastChangedDateTimeUTC': trade_order.get('LastChangedDateTimeUTC')
                    })

                print(f"\n[INFO] Total positions after refresh: {len(open_positions)}")
                return open_positions
            else:
                logging.error(f"[ERROR] Failed to fetch open positions after token refresh: {response.status_code}, {response.text}")
                return None

        else:
            logging.error(f"[ERROR] Failed to fetch open positions: {response.status_code}, {response.text}")
            return None
