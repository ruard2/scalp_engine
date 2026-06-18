import time
from datetime import datetime
from lightstreamer.client import LightstreamerClient, Subscription, SubscriptionListener
from session_manager import session_manager
from config import username
import asyncio

# Logging suppressed — 2026-05-07
import logging
import websocket
logging.basicConfig(level=logging.WARNING)
logging.getLogger("lightstreamer").setLevel(logging.WARNING)
logging.getLogger("websocket").setLevel(logging.WARNING)

class LightstreamerReceiver:
    def __init__(self, initial_market_ids=None):
        """
        Initialize a new LightstreamerReceiver:
          1) Read the CIAPI session_token from session_token.json
          2) Connect (WebSocket‐only) and subscribe to initial_market_ids (if given)
        """
        self._session_token    = None
        self._ls_client        = None
        self._current_ids      = set()
        self._subscription_obj = None
        self.latest_data       = {}
        self._last_subscribe_time = None

        try:
            import json
            with open("session_token.json", "r") as f:
                data = json.load(f)
            self._session_token = data.get("session_token")
            pass  # debug suppressed 2026-05-07
        except Exception as e:
            print(f"[ERROR {_time_now()}] Failed to read session_token.json: {e}")

        self._rebuild_ls_client()

        if initial_market_ids:
            desired = {str(int(float(mid))) for mid in initial_market_ids}  
            self._subscribe_all_ids(desired)


    def _rebuild_ls_client(self):
        """
        Tear down any old LightstreamerClient and build a brand‐new one (WebSocket‐only).
        After connect succeeds, re‐subscribe to whatever is in self._current_ids.
        """
        if self._ls_client:
            try:
                self._ls_client.disconnect()
            except:
                pass
            self._ls_client = None
            self._subscription_obj = None

        while True:
            try:
                # Use full WSS URL and adapter set
                client = LightstreamerClient("https://push.cityindex.com", "STREAMINGALL")
                client.connectionDetails.setUser(username)
                client.connectionDetails.setPassword(self._session_token)
                #client.connectionDetails.setForcedTransport("websocket")
                client.connect()
                time.sleep(2)
                pass  # debug suppressed 2026-05-07
                self._ls_client = client
                break
            except Exception as e:
                print(f"[ERROR {_time_now()}] WebSocket connect failed: {e}. Retrying in 2 s…")
                time.sleep(2)

        if self._current_ids:
            pass  # debug suppressed 2026-05-07
            self._subscribe_all_ids(self._current_ids)

    def _subscribe_all_ids(self, id_set):
        if not id_set or self._ls_client is None:
            return
        if self._subscription_obj:
            try:
                self._ls_client.unsubscribe(self._subscription_obj)
            except:
                pass
            self._subscription_obj = None

        items = [f"ID.{m}" for m in sorted(id_set)]
        pass  # debug suppressed 2026-05-07

        sub = Subscription(mode="MERGE", items=items, fields=["AuditId", "Bid", "Offer"])
        sub.setDataAdapter("PRICES")
        sub.setRequestedSnapshot("yes")

        class MultiTickListener(SubscriptionListener):
            def __init__(inner):
                inner.seen = set()
            def onItemUpdate(inner, update):
                mid = update.getItemName().replace("ID.", "")
                try:
                    bid = float(update.getValue("Bid"))
                    offer = float(update.getValue("Offer"))
                except:
                    return
                self.latest_data[mid] = {
                    "AuditId": update.getValue("AuditId"),
                    "Bid": bid,
                    "Offer": offer,
                    "timestamp": datetime.utcnow(),
                }

        sub.addListener(MultiTickListener())

        try:
            self._ls_client.subscribe(sub)
            pass  # debug suppressed 2026-05-07
            self._subscription_obj = sub
            self._current_ids = set(id_set)
            self._last_subscribe_time = datetime.utcnow()
        except Exception as e:
            print(f"[ERROR {_time_now()}] Error during subscribe: {e}")
            self._subscription_obj = None

    def fetch_market_data_all(self, market_id_list, timeout_secs=10):
        desired_set = {str(mid) for mid in market_id_list}
        if desired_set != self._current_ids:
            self._subscribe_all_ids(desired_set)
            self._last_subscribe_time = datetime.utcnow()
            pass  # debug suppressed 2026-05-07
        snapshot = {}
        start_time = time.time()
        while time.time() - start_time < timeout_secs:
            for m in desired_set:
                if m in self.latest_data and m not in snapshot:
                    snapshot[m] = self.latest_data[m]
            if set(snapshot.keys()) == desired_set:
                pass  # tick confirmation — suppressed 2026-05-07
                return snapshot
            time.sleep(0.05)
        missing = sorted(desired_set - set(snapshot.keys()))
        if missing:
            print(f"[WARN {_time_now()}] → Timed out after {timeout_secs}s; missing: {missing}")
        return snapshot

    def fetch_market_data_one(self, market_id, timeout_secs=5):
        """
        Subscribe temporarily to a single market_id and return one tick.
        Used for last_check.
        """
        desired_set = {str(market_id)}
        if desired_set != self._current_ids:
            self._subscribe_all_ids(desired_set)
            self._last_subscribe_time = datetime.utcnow()
            pass  # debug suppressed 2026-05-07

        start_time = time.time()
        while time.time() - start_time < timeout_secs:
            if str(market_id) in self.latest_data:
                return self.latest_data[str(market_id)]
            time.sleep(0.05)

        print(f"[WARN {_time_now()}] → No tick received for {market_id} within {timeout_secs}s")
        return None



    def disconnect(self):
        """Unsubscribe and close the Lightstreamer client; safe to call multiple times."""
        try:
            if self._ls_client and self._subscription_obj:
                try:
                    self._ls_client.unsubscribe(self._subscription_obj)
                except Exception:
                    pass
        finally:
            self._subscription_obj = None

        try:
            if self._ls_client:
                try:
                    self._ls_client.disconnect()
                except Exception:
                    pass
        finally:
            self._ls_client = None
            try:
                self._current_ids.clear()
            except Exception:
                pass


# Utility for Timestamps
def _time_now():
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
