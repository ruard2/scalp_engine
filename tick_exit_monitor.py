"""
tick_exit_monitor.py — Real-time exit management via Lightstreamer ticks.

Runs as a daemon thread alongside the 10-min bar loop.
On every bid/offer tick, checks all open positions for:
  - TP:      close immediately
  - TrailSL: update trail high-water mark, close when breached
  - SL:      intercept -> move to bounce_pending (broker position stays open)

Bounce recovery also monitored tick-by-tick:
  - BounceProfit:  price returned to entry -> close
  - BounceSafety:  price fell BOUNCE_SAFETY_P further -> close
  - BounceTrail:   trail activated on recovery, then breached -> close

Bounce timeout (BOUNCE_MAX_BARS bars without recovery) is handled by the main
OHLC loop because it requires bar-counting, not ticks.

DirectionFlip exits stay in the main loop (need classifier labels).

Thread safety: all state reads/writes protected by the Lock passed in at init.
The main bar loop acquires the same lock for the ~2s it touches state, then
releases before the 9.5-min sleep. Monitor runs freely during the sleep.
"""

import threading
import time
import logging
from datetime import datetime, timezone
from typing import Optional, Tuple, Callable

log = logging.getLogger("v6_engine")

POLL_SECS = 0.5   # tick poll interval


class TickExitMonitor:
    def __init__(
        self,
        state:            dict,
        lock:             threading.Lock,  # reserved, not used (GIL + list() snapshots)
        ls_receiver,                       # LightstreamerReceiver instance
        market_id:        str,
        close_order_fn:   Callable,
        log_trade_fn:     Callable,
        quantity:         float,
        paper:            bool  = False,
        bounce_safety_p:  float = 6.0,
        bounce_min_pips:  float = 1.0,
        bounce_trail_pips:float = 2.0,
        min_trail_lock_p: float = 1.5,
    ):
        self._state     = state
        self._lock      = lock
        self._ls        = ls_receiver
        self._market_id = str(market_id)
        self._close_fn  = close_order_fn
        self._log_trade = log_trade_fn
        self._qty       = quantity
        self._paper     = paper

        # Bounce parameters (in price units)
        self._safety_d      = bounce_safety_p  / 10000
        self._bounce_min_d  = bounce_min_pips  / 10000
        self._bounce_trail_d= bounce_trail_pips/ 10000
        self._min_trail_d   = min_trail_lock_p / 10000

        self._reconnect_fn = None   # set by live_engine after init
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_tick_ts: Optional[datetime] = None
        self._last_tick_bid: Optional[float] = None  # detect actual price change

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────
    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="TickExitMonitor")
        self._thread.start()
        log.info("[TickExit] Monitor started (poll interval %.1fs).", POLL_SECS)

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        log.info("[TickExit] Monitor stopped.")

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _get_tick(self) -> Optional[Tuple[float, float, str]]:
        """Return (bid, offer, audit_id) or None if no tick available."""
        if self._ls is None:
            return None
        try:
            data = self._ls.latest_data.get(self._market_id)
            if data is None:
                return None
            return float(data["Bid"]), float(data["Offer"]), str(data.get("AuditId", ""))
        except Exception:
            return None

    def _do_close(self, order_id, direction, entry_price,
                  bid, offer, audit_id, reason) -> Tuple[bool, float, float]:
        """Place close order. Returns (success, close_price, pnl_pips)."""
        api_dir = "buy" if direction == "Bull" else "sell"
        success = self._close_fn(
            order_id=str(order_id),
            entry_direction=api_dir,
            bid=bid, offer=offer, audit_id=audit_id,
            quantity=self._qty,
            entry_price=float(entry_price),
            paper=self._paper,
        )
        close_px = bid if direction == "Bull" else offer
        pnl = ((close_px - entry_price) * 10000 if direction == "Bull"
               else (entry_price - close_px) * 10000)
        status = "OK" if success else "FAIL"
        log.info(f"[TickExit] CLOSE {status}  reason={reason}  dir={direction}  "
                 f"pnl={pnl:+.1f}p  price={close_px:.5f}")
        return success, close_px, pnl

    # ─────────────────────────────────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────────────────────────────────
    STALE_WARN_SECS    = 60    # warn if no new tick for this long
    STALE_RECONNECT_SECS = 90  # force reconnect if still stale

    def _run(self):
        while not self._stop_event.is_set():
            try:
                tick = self._get_tick()
                now  = datetime.now(timezone.utc)

                if tick is not None:
                    bid, offer, audit_id = tick
                    # Detect genuinely new tick (price changed)
                    if bid != self._last_tick_bid:
                        self._last_tick_bid = bid
                        self._last_tick_ts  = now

                    # Staleness watchdog
                    if self._last_tick_ts is not None:
                        stale_secs = (now - self._last_tick_ts).total_seconds()
                        if stale_secs >= self.STALE_RECONNECT_SECS:
                            log.warning("[TickExit] Tick stale for %.0fs — forcing LS reconnect...",
                                        stale_secs)
                            self._do_reconnect()
                        elif stale_secs >= self.STALE_WARN_SECS:
                            log.warning("[TickExit] Tick stale for %.0fs (reconnect at %ds)",
                                        stale_secs, self.STALE_RECONNECT_SECS)

                    now_iso = now.isoformat()
                    # list() snapshots in _check_* prevent RuntimeError on
                    # concurrent dict modification; pop guards handle races.
                    self._check_open_trades(bid, offer, audit_id, now_iso)
                    self._check_bounce_pending(bid, offer, audit_id, now_iso)

                else:
                    # No tick at all — LS not connected yet or lost
                    if self._last_tick_ts is None:
                        stale_secs = 0.0
                    else:
                        stale_secs = (now - self._last_tick_ts).total_seconds()
                    if stale_secs >= self.STALE_RECONNECT_SECS or self._last_tick_ts is None:
                        log.warning("[TickExit] No tick received for %.0fs — forcing LS reconnect...",
                                    stale_secs)
                        self._do_reconnect()

            except Exception as e:
                log.error("[TickExit] Exception in monitor loop: %s", e, exc_info=True)
            time.sleep(POLL_SECS)

    def _do_reconnect(self):
        """Call the reconnect callback (ls_connect from live_engine) and retry until live."""
        if self._reconnect_fn is None:
            return
        attempt = 0
        while not self._stop_event.is_set():
            attempt += 1
            try:
                log.info("[TickExit] LS reconnect attempt %d...", attempt)
                ok = self._reconnect_fn()
                if ok:
                    self._last_tick_ts  = None   # reset — wait for fresh tick
                    self._last_tick_bid = None
                    log.info("[TickExit] LS reconnected OK on attempt %d.", attempt)
                    return
                else:
                    log.warning("[TickExit] LS reconnect attempt %d failed — retry in 15s", attempt)
            except Exception as e:
                log.error("[TickExit] LS reconnect attempt %d exception: %s", attempt, e)
            time.sleep(15)

    # ─────────────────────────────────────────────────────────────────────────
    # Open-trade checker
    # ─────────────────────────────────────────────────────────────────────────
    def _check_open_trades(self, bid, offer, audit_id, now_iso):
        state = self._state
        if "open_trades" not in state:
            return

        to_close  = []   # (key, reason)
        to_bounce = []   # key  — SL hit, don't broker-close; move to bounce

        for key, t in list(state["open_trades"].items()):
            d  = t["direction"]
            ep = t["entry_price"]
            sl = t["sl_price"]
            tp = t["tp_price"]

            # Current executable price (Bull exits at bid, Bear at offer)
            cur = bid if d == "Bull" else offer

            # ── Trail stop: update high-water mark ────────────────────
            if d == "Bull":
                if cur > t.get("best_price", ep):
                    t["best_price"] = cur
                favor = cur - ep
                if not t["trail_active"] and favor >= t["trail_trigger"]:
                    t["trail_active"] = True
                    raw = cur - t["trail_dist"]
                    t["trail_stop"] = max(raw, ep + self._min_trail_d)
                    log.info(f"[TickExit] Trail ARMED  key={key}  "
                             f"stop={t['trail_stop']:.5f}  cur={cur:.5f}")
                elif t["trail_active"]:
                    candidate = max(cur - t["trail_dist"], ep + self._min_trail_d)
                    if candidate > t["trail_stop"]:
                        t["trail_stop"] = candidate
            else:
                if cur < t.get("best_price", ep):
                    t["best_price"] = cur
                favor = ep - cur
                if not t["trail_active"] and favor >= t["trail_trigger"]:
                    t["trail_active"] = True
                    raw = cur + t["trail_dist"]
                    t["trail_stop"] = min(raw, ep - self._min_trail_d)
                    log.info(f"[TickExit] Trail ARMED  key={key}  "
                             f"stop={t['trail_stop']:.5f}  cur={cur:.5f}")
                elif t["trail_active"]:
                    candidate = min(cur + t["trail_dist"], ep - self._min_trail_d)
                    if candidate < t["trail_stop"]:
                        t["trail_stop"] = candidate

            # ── TP ────────────────────────────────────────────────────
            if d == "Bull" and cur >= tp:
                to_close.append((key, "TP")); continue
            if d == "Bear" and cur <= tp:
                to_close.append((key, "TP")); continue

            # ── TrailSL ───────────────────────────────────────────────
            if t["trail_active"] and t.get("trail_stop", 0) > 0:
                if d == "Bull" and cur <= t["trail_stop"]:
                    to_close.append((key, "TrailSL")); continue
                if d == "Bear" and cur >= t["trail_stop"]:
                    to_close.append((key, "TrailSL")); continue

            # ── SL → bounce ───────────────────────────────────────────
            if d == "Bull" and cur <= sl:
                to_bounce.append(key); continue
            if d == "Bear" and cur >= sl:
                to_bounce.append(key); continue

        # Execute TP / TrailSL closes
        for key, reason in to_close:
            t = state["open_trades"].pop(key, None)
            if t is None:
                continue  # already removed (OHLC loop race)
            d = t["direction"]; ep = t["entry_price"]
            success, close_px, pnl = self._do_close(
                t.get("order_id"), d, ep, bid, offer, audit_id, reason)
            self._log_trade({
                "type": "CLOSE", "timestamp": now_iso,
                "combo_key": key, "direction": d,
                "reason": reason, "pnl_pips": round(pnl, 2),
                "close_price": round(close_px, 5),
                "bars_held": t.get("bars_held", 0),
                "source": "tick",
            })

        # SL hits: move to bounce_pending without calling close_order
        for key in to_bounce:
            t = state["open_trades"].pop(key, None)
            if t is None:
                continue  # already removed
            d = t["direction"]; ep = t["entry_price"]
            sl_trigger = t["sl_price"]
            safety_fl  = (sl_trigger - self._safety_d if d == "Bull"
                          else sl_trigger + self._safety_d)
            log.info(f"[TickExit] SL HIT -> BOUNCE_START  key={key}  "
                     f"dir={d}  sl={sl_trigger:.5f}  safety={safety_fl:.5f}  "
                     f"cur={bid if d=='Bull' else offer:.5f}")
            self._log_trade({
                "type": "BOUNCE_START", "timestamp": now_iso,
                "combo_key": key, "direction": d,
                "sl_trigger": round(sl_trigger, 5),
                "bars_held": t.get("bars_held", 0),
                "source": "tick",
            })
            state["bounce_pending"][key] = {
                "order_id":     t.get("order_id"),
                "direction":    d,
                "entry_price":  ep,
                "environment":  t.get("environment", ""),
                "local":        t.get("local", ""),
                "profile_key":  t.get("profile_key", ""),
                "sl_trigger":   sl_trigger,
                "safety_floor": safety_fl,
                "bounce_trail": None,
                "best_recovery": sl_trigger,
                "bars_in_bounce": 0,
                "bars_held":    t.get("bars_held", 0),
                "atr_rank":     t.get("atr_rank", 50.0),
            }

    # ─────────────────────────────────────────────────────────────────────────
    # Bounce-pending checker
    # ─────────────────────────────────────────────────────────────────────────
    def _check_bounce_pending(self, bid, offer, audit_id, now_iso):
        state = self._state
        if "bounce_pending" not in state:
            return

        to_close = []  # (key, reason)

        for bkey, bs in list(state["bounce_pending"].items()):
            d  = bs["direction"]
            ep = bs["entry_price"]
            sf = bs["safety_floor"]
            btr  = bs.get("bounce_trail")
            brec = bs.get("best_recovery", bs["sl_trigger"])

            cur = bid if d == "Bull" else offer

            reason = None
            if d == "Bull":
                if cur <= sf:
                    reason = "BounceSafety"
                elif cur >= ep:
                    reason = "BounceProfit"
                else:
                    # Update recovery tracking
                    if cur > brec + self._bounce_min_d:
                        bs["best_recovery"] = cur
                        nt = cur - self._bounce_trail_d
                        bs["bounce_trail"] = (max(nt, sf) if btr is None
                                              else max(btr, max(nt, sf)))
                    if bs.get("bounce_trail") and cur <= bs["bounce_trail"]:
                        reason = "BounceTrail"
            else:
                if cur >= sf:
                    reason = "BounceSafety"
                elif cur <= ep:
                    reason = "BounceProfit"
                else:
                    if cur < brec - self._bounce_min_d:
                        bs["best_recovery"] = cur
                        nt = cur + self._bounce_trail_d
                        bs["bounce_trail"] = (min(nt, sf) if btr is None
                                              else min(btr, min(nt, sf)))
                    if bs.get("bounce_trail") and cur >= bs["bounce_trail"]:
                        reason = "BounceTrail"

            if reason:
                to_close.append((bkey, reason))

        for bkey, reason in to_close:
            bs = state["bounce_pending"].pop(bkey, None)
            if bs is None:
                continue  # already removed
            d = bs["direction"]; ep = bs["entry_price"]
            success, close_px, pnl = self._do_close(
                bs.get("order_id"), d, ep, bid, offer, audit_id, reason)
            self._log_trade({
                "type": "CLOSE", "timestamp": now_iso,
                "combo_key": bkey, "direction": d,
                "reason": reason, "pnl_pips": round(pnl, 2),
                "close_price": round(close_px, 5),
                "bars_held": bs.get("bars_held", 0),
                "source": "tick",
            })
