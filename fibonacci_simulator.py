import os
import re
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm


# =========================
# CONFIG
# =================
# ========
CSV_FILE = "20250217.csv"
BASE_DATE = None

RESAMPLE_TIMEFRAME = "5min"

# Pivot Detection
PIVOT_PERIOD = 1

# Minimum structure requirement
MIN_CANDLES_LOW_TO_HIGH = 2

# Force first anchor candle seq (set None to disable)
MANUAL_FIRST_ANCHOR_SEQ = None # e.g., 100 to force anchor at seq=100

# ✅ DEBUG: Force extreme to a specific pivot index (H1/H2/H3... or L1/L2/...)
DEBUG_EXTREME_PIVOT_H_INDEX = None  # None = off; 1 means lock at first pivot high/low

# ✅ ANCHOR SHIFT FEATURE (your earlier case)
SHIFT_ANCHOR_ON_LOWER_LOW_BEFORE_HIGH = True
SHIFT_ANCHOR_REQUIRES_PIVOT = False  # False = shift even if not a pivot (recommended)

# QUALITY FILTERS
MIN_PRICE_MOVEMENT_PERCENT = 0.10
MIN_PRICE_MOVEMENT_POINTS = 10
MIN_ATR_MULTIPLIER = 0.8
ATR_PERIOD = 14

# Fibonacci Levels
FIB_LEVELS = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.5, 1.6]

# Golden Zone
GOLDEN_ZONE_LOW = 0.382
GOLDEN_ZONE_HIGH = 0.618

# IMPORTANT:
# If False, golden zone touch is wick/body overlap,
# BUT we now ADD a rule: "If wick crosses 0.382, it's INVALID (not GZ)"
REQUIRE_BODY_IN_GOLDEN_ZONE = False

# Trend
TREND = "UPTREND"  # "UPTREND" or "DOWNTREND"

# Plotting toggles
DO_PLOT = True
SHOW_INVALIDATED_FIBS = True
SHOW_QUALITY_REJECTED = False
PLOT_ONLY_VALID_FIBS = False

# Labels
LABEL_LEVELS = {0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.5, 1.6}
X_LABEL_PADDING = 35

DEFAULT_SYMBOL_NAME = "NIFTY 50"


# =========================
# FIBONACCI STATE
# =========================
class FibState(Enum):
    WAITING_FOR_MIN_CANDLES = 1
    WAITING_FOR_OPPOSITE = 2
    ACTIVE = 3
    VALIDATED = 4
    STOPLOSS_HIT = 5
    TARGET_HIT = 6
    INVALIDATED = 7


class FibonacciLeg:
    """
    One Fibonacci leg:
      - UPTREND: anchor = LOW pivot, extreme = HIGH pivot, fib[0]=anchor, fib[1]=extreme
      - DOWNTREND: anchor = HIGH pivot, extreme = LOW pivot, fib[0]=anchor, fib[1]=extreme
    """

    def __init__(
        self,
        leg_id: int,
        trend: str,
        anchor_seq: int,
        anchor_price: float,
        min_gap: int,
        atr_value: Optional[float] = None,
    ):
        self.leg_id = int(leg_id)
        self.trend = trend
        self.anchor_seq = int(anchor_seq)
        self.anchor_price = float(anchor_price)
        self.min_gap = int(min_gap)
        self.atr_value = None if (atr_value is None or pd.isna(atr_value)) else float(atr_value)

        self.opposite_seq: Optional[int] = None
        self.opposite_price: Optional[float] = None

        self.current_extreme_seq: Optional[int] = None
        self.current_extreme_price: Optional[float] = None

        self.state = FibState.WAITING_FOR_MIN_CANDLES
        self.min_gap_satisfied_seq = self.anchor_seq + self.min_gap

        self.validation_info = {
            "entered_golden_zone": False,
            "golden_zone_entry_seq": None,
            "is_valid_fib": False,
        }

        # ✅ ENTRY tracking (after golden zone only)
        self.entry_seq: Optional[int] = None
        self.entry_price: Optional[float] = None

        self.stoploss_seq: Optional[int] = None
        self.stoploss_price: Optional[float] = None
        self.target_seq: Optional[int] = None
        self.target_price: Optional[float] = None

        self.invalidation_seq: Optional[int] = None
        self.invalidation_reason: Optional[str] = None

        self.quality_rejected = False
        self.quality_rejection_reason: Optional[str] = None

        # ✅ Track pivot extremes (H1/H2/... or L1/L2/...)
        self.extreme_pivot_count = 0
        self.locked_extreme = False
        self.locked_extreme_label: Optional[str] = None  # keep as Hn/Ln

        # ✅ Track last range change to skip golden zone evaluation on same candle
        self.last_range_change_seq: Optional[int] = None

        # Debug log
        self.updates: List[Dict] = []
        self.updates.append(
            {
                "seq": self.anchor_seq,
                "price": self.anchor_price,
                "event": "ANCHOR_SET",
                "detail": f"Anchor set ({self.trend}) at {self.anchor_price:.2f}. Waiting {self.min_gap} candles.",
            }
        )

    # -------------------------
    # Helpers
    # -------------------------
    def _price_move(self) -> float:
        if self.current_extreme_price is None:
            return 0.0
        if self.trend == "UPTREND":
            return float(self.current_extreme_price - self.anchor_price)
        return float(self.anchor_price - self.current_extreme_price)

    def check_quality_filters(self) -> bool:
        if self.current_extreme_price is None:
            return True

        move = self._price_move()
        if self.anchor_price == 0:
            return True

        percent_move = (abs(move) / abs(self.anchor_price)) * 100.0
        if percent_move < MIN_PRICE_MOVEMENT_PERCENT:
            self.quality_rejected = True
            self.quality_rejection_reason = (
                f"Price move too small: {percent_move:.2f}% (min {MIN_PRICE_MOVEMENT_PERCENT}%)"
            )
            return False

        if abs(move) < MIN_PRICE_MOVEMENT_POINTS:
            self.quality_rejected = True
            self.quality_rejection_reason = (
                f"Absolute move too small: {abs(move):.2f} points (min {MIN_PRICE_MOVEMENT_POINTS})"
            )
            return False

        if self.atr_value and self.atr_value > 0:
            atr_mult = abs(move) / self.atr_value
            if atr_mult < MIN_ATR_MULTIPLIER:
                self.quality_rejected = True
                self.quality_rejection_reason = (
                    f"Move too small vs ATR: {atr_mult:.2f}x (min {MIN_ATR_MULTIPLIER}x)"
                )
                return False

        return True

    def get_fib_levels(self) -> Dict[float, float]:
        if self.current_extreme_price is None:
            return {}

        if self.trend == "UPTREND":
            rng = float(self.current_extreme_price - self.anchor_price)
            return {lv: float(self.anchor_price + rng * lv) for lv in FIB_LEVELS}
        else:
            rng = float(self.anchor_price - self.current_extreme_price)
            return {lv: float(self.anchor_price - rng * lv) for lv in FIB_LEVELS}

    def _stoploss_touch_anyway(self, o: float, h: float, l: float, c: float, lvl_0: float, lvl_382: float) -> bool:
        """
        SL zone = between fib(0) and fib(0.382).
        Trigger if wick OR body OR close enters zone OR crosses 0.
        Works for BOTH UPTREND and DOWNTREND because we min/max the band.
        """
        zone_low = min(lvl_0, lvl_382)
        zone_high = max(lvl_0, lvl_382)

        # Wick touches zone
        wick_touches = (l <= zone_high) and (h >= zone_low)

        # Body touches zone
        body_lo = min(o, c)
        body_hi = max(o, c)
        body_touches = (body_lo <= zone_high) and (body_hi >= zone_low)

        # Close inside zone
        close_in = (zone_low <= c <= zone_high)

        # Cross/break 0 (directional)
        crossed_0 = (l < lvl_0) if self.trend == "UPTREND" else (h > lvl_0)

        return wick_touches or body_touches or close_in or crossed_0

    def _golden_zone_touched_no_382_cross(self, o: float, h: float, l: float, c: float, lvl_382: float, lvl_618: float) -> bool:
        """
        Golden zone touch BUT with your new rule:

        ✅ If wick crosses 0.382 -> DO NOT treat as Golden Zone. It becomes INVALID.

        Meaning:
        - UPTREND: if low < lvl_382 => wick crossed below 0.382 (INVALID)
        - DOWNTREND: if high > lvl_382 => wick crossed above 0.382 (INVALID)

        So GZ touch can happen only if candle overlaps zone AND does NOT cross the 0.382 boundary.
        """
        gz_low = min(lvl_382, lvl_618)
        gz_high = max(lvl_382, lvl_618)

        # "cross 0.382" check (INVALID case)
        crossed_382 = (l < lvl_382) if self.trend == "UPTREND" else (h > lvl_382)
        if crossed_382:
            return False

        if REQUIRE_BODY_IN_GOLDEN_ZONE:
            body_hi = max(o, c)
            body_lo = min(o, c)
            overlap = (body_lo <= gz_high) and (body_hi >= gz_low)
            return overlap

        # Wick overlap (but we've already ensured wick did not cross 0.382)
        overlap = (l <= gz_high) and (h >= gz_low)
        return overlap

    # -------------------------
    # Main candle update
    # -------------------------
    def update_candle(
        self,
        seq: int,
        open_price: float,
        high: float,
        low: float,
        close: float,
        is_pivot_high: bool = False,
        is_pivot_low: bool = False,
    ) -> bool:
        if self.state in (FibState.INVALIDATED, FibState.STOPLOSS_HIT, FibState.TARGET_HIT):
            return False

        # ============================================================
        # ✅ ANCHOR SHIFT (only before opposite pivot exists)
        # ============================================================
        if SHIFT_ANCHOR_ON_LOWER_LOW_BEFORE_HIGH and self.opposite_seq is None:
            if self.trend == "UPTREND":
                should_shift = (low < self.anchor_price) and (is_pivot_low or not SHIFT_ANCHOR_REQUIRES_PIVOT)
                if should_shift:
                    old_seq, old_price = self.anchor_seq, self.anchor_price

                    self.anchor_seq = int(seq)
                    self.anchor_price = float(low)

                    self.state = FibState.WAITING_FOR_MIN_CANDLES
                    self.min_gap_satisfied_seq = self.anchor_seq + self.min_gap

                    self.opposite_seq = None
                    self.opposite_price = None
                    self.current_extreme_seq = None
                    self.current_extreme_price = None
                    self.extreme_pivot_count = 0
                    self.locked_extreme = False
                    self.locked_extreme_label = None

                    self.validation_info = {"entered_golden_zone": False, "golden_zone_entry_seq": None, "is_valid_fib": False}
                    self.entry_seq = None
                    self.entry_price = None
                    self.stoploss_seq = None
                    self.stoploss_price = None
                    self.target_seq = None
                    self.target_price = None
                    self.invalidation_seq = None
                    self.invalidation_reason = None
                    self.quality_rejected = False
                    self.quality_rejection_reason = None

                    self.last_range_change_seq = int(seq)

                    self.updates.append(
                        {
                            "seq": seq,
                            "price": float(low),
                            "event": "ANCHOR_SHIFTED",
                            "detail": f"✅ Anchor shifted: old LOW {old_price:.2f} (seq {old_seq}) -> new LOW {low:.2f} (seq {seq}). Restarting min-gap.",
                        }
                    )
                    return True
            else:
                should_shift = (high > self.anchor_price) and (is_pivot_high or not SHIFT_ANCHOR_REQUIRES_PIVOT)
                if should_shift:
                    old_seq, old_price = self.anchor_seq, self.anchor_price

                    self.anchor_seq = int(seq)
                    self.anchor_price = float(high)

                    self.state = FibState.WAITING_FOR_MIN_CANDLES
                    self.min_gap_satisfied_seq = self.anchor_seq + self.min_gap

                    self.opposite_seq = None
                    self.opposite_price = None
                    self.current_extreme_seq = None
                    self.current_extreme_price = None
                    self.extreme_pivot_count = 0
                    self.locked_extreme = False
                    self.locked_extreme_label = None

                    self.validation_info = {"entered_golden_zone": False, "golden_zone_entry_seq": None, "is_valid_fib": False}
                    self.entry_seq = None
                    self.entry_price = None
                    self.stoploss_seq = None
                    self.stoploss_price = None
                    self.target_seq = None
                    self.target_price = None
                    self.invalidation_seq = None
                    self.invalidation_reason = None
                    self.quality_rejected = False
                    self.quality_rejection_reason = None

                    self.last_range_change_seq = int(seq)

                    self.updates.append(
                        {
                            "seq": seq,
                            "price": float(high),
                            "event": "ANCHOR_SHIFTED",
                            "detail": f"✅ Anchor shifted: old HIGH {old_price:.2f} (seq {old_seq}) -> new HIGH {high:.2f} (seq {seq}). Restarting min-gap.",
                        }
                    )
                    return True

        # 1) Wait minimum gap
        if self.state == FibState.WAITING_FOR_MIN_CANDLES:
            if seq >= self.min_gap_satisfied_seq:
                self.state = FibState.WAITING_FOR_OPPOSITE
                self.updates.append(
                    {
                        "seq": seq,
                        "price": float((high + low) / 2),
                        "event": "MIN_GAP_SATISFIED",
                        "detail": f"Minimum {self.min_gap} candles passed; now looking for opposite pivot.",
                    }
                )
            return True

        # 2) Activate only on confirmed opposite pivot AFTER gap
        if self.state == FibState.WAITING_FOR_OPPOSITE:
            if self.trend == "UPTREND":
                if is_pivot_high and high > self.anchor_price:
                    self.opposite_seq = int(seq)
                    self.opposite_price = float(high)
                    self.current_extreme_seq = int(seq)
                    self.current_extreme_price = float(high)

                    self.extreme_pivot_count = 1
                    self.locked_extreme_label = "H1"
                    if DEBUG_EXTREME_PIVOT_H_INDEX == 1:
                        self.locked_extreme = True

                    if not self.check_quality_filters():
                        self.state = FibState.INVALIDATED
                        self.invalidation_seq = int(seq)
                        self.invalidation_reason = f"QUALITY FILTER: {self.quality_rejection_reason}"
                        self.updates.append({"seq": seq, "price": float(high), "event": "QUALITY_REJECTED",
                                             "detail": self.quality_rejection_reason})
                        return False

                    self.state = FibState.ACTIVE
                    self.updates.append({"seq": seq, "price": float(high), "event": "ACTIVATED",
                                         "detail": f"Activated on pivot HIGH {high:.2f} (H1)"})
                    return True
            else:
                if is_pivot_low and low < self.anchor_price:
                    self.opposite_seq = int(seq)
                    self.opposite_price = float(low)
                    self.current_extreme_seq = int(seq)
                    self.current_extreme_price = float(low)

                    self.extreme_pivot_count = 1
                    self.locked_extreme_label = "L1"
                    if DEBUG_EXTREME_PIVOT_H_INDEX == 1:
                        self.locked_extreme = True

                    if not self.check_quality_filters():
                        self.state = FibState.INVALIDATED
                        self.invalidation_seq = int(seq)
                        self.invalidation_reason = f"QUALITY FILTER: {self.quality_rejection_reason}"
                        self.updates.append({"seq": seq, "price": float(low), "event": "QUALITY_REJECTED",
                                             "detail": self.quality_rejection_reason})
                        return False

                    self.state = FibState.ACTIVE
                    self.updates.append({"seq": seq, "price": float(low), "event": "ACTIVATED",
                                         "detail": f"Activated on pivot LOW {low:.2f} (L1)"})
                    return True

            return True

        # 3) ACTIVE / VALIDATED
        if self.state in (FibState.ACTIVE, FibState.VALIDATED):

            allow_extreme_updates = (self.entry_seq is None)

            if allow_extreme_updates and (not self.locked_extreme):
                if self.trend == "UPTREND":
                    if self.current_extreme_price is not None and is_pivot_high and high > float(self.current_extreme_price):
                        self.extreme_pivot_count += 1
                        self.current_extreme_seq = int(seq)
                        self.current_extreme_price = float(high)
                        self.locked_extreme_label = f"H{self.extreme_pivot_count}"
                        self.last_range_change_seq = int(seq)

                        self.updates.append({"seq": seq, "price": float(high), "event": "EXTREME_UPDATE",
                                             "detail": f"New pivot HIGH {high:.2f} ({self.locked_extreme_label})"})

                        if self.validation_info.get("entered_golden_zone") and self.entry_seq is None:
                            old_gz = self.validation_info.get("golden_zone_entry_seq")
                            self.validation_info = {"entered_golden_zone": False, "golden_zone_entry_seq": None, "is_valid_fib": False}
                            self.state = FibState.ACTIVE
                            self.updates.append({"seq": seq, "price": float(high), "event": "GZ_RESET_AFTER_EXTREME_UPDATE",
                                                 "detail": f"♻️ Extreme updated before ENTRY, so old GZ={old_gz} reset. Waiting for NEW golden zone."})

                        if DEBUG_EXTREME_PIVOT_H_INDEX is not None and int(DEBUG_EXTREME_PIVOT_H_INDEX) > 0:
                            if self.extreme_pivot_count == int(DEBUG_EXTREME_PIVOT_H_INDEX):
                                self.locked_extreme = True
                                self.updates.append({"seq": seq, "price": float(high), "event": "EXTREME_LOCKED",
                                                     "detail": f"✅ DEBUG: Locked extreme at {self.locked_extreme_label} ({high:.2f})"})
                else:
                    if self.current_extreme_price is not None and is_pivot_low and low < float(self.current_extreme_price):
                        self.extreme_pivot_count += 1
                        self.current_extreme_seq = int(seq)
                        self.current_extreme_price = float(low)
                        self.locked_extreme_label = f"L{self.extreme_pivot_count}"
                        self.last_range_change_seq = int(seq)

                        self.updates.append({"seq": seq, "price": float(low), "event": "EXTREME_UPDATE",
                                             "detail": f"New pivot LOW {low:.2f} ({self.locked_extreme_label})"})

                        if self.validation_info.get("entered_golden_zone") and self.entry_seq is None:
                            old_gz = self.validation_info.get("golden_zone_entry_seq")
                            self.validation_info = {"entered_golden_zone": False, "golden_zone_entry_seq": None, "is_valid_fib": False}
                            self.state = FibState.ACTIVE
                            self.updates.append({"seq": seq, "price": float(low), "event": "GZ_RESET_AFTER_EXTREME_UPDATE",
                                                 "detail": f"♻️ Extreme updated before ENTRY, so old GZ={old_gz} reset. Waiting for NEW golden zone."})

                        if DEBUG_EXTREME_PIVOT_H_INDEX is not None and int(DEBUG_EXTREME_PIVOT_H_INDEX) > 0:
                            if self.extreme_pivot_count == int(DEBUG_EXTREME_PIVOT_H_INDEX):
                                self.locked_extreme = True
                                self.updates.append({"seq": seq, "price": float(low), "event": "EXTREME_LOCKED",
                                                     "detail": f"✅ DEBUG: Locked extreme at {self.locked_extreme_label} ({low:.2f})"})

            fib = self.get_fib_levels()
            if not fib or 0.382 not in fib or 0.618 not in fib:
                return True

            if self.last_range_change_seq is not None and seq == self.last_range_change_seq:
                return True

            lvl_0 = float(fib[0])
            lvl_382 = float(fib[0.382])
            lvl_618 = float(fib[0.618])

            lvl_15 = float(fib[1.5]) if 1.5 in fib else None
            lvl_16 = float(fib[1.6]) if 1.6 in fib else None

            # ------------------------------------------------------------
            # INVALIDATION (ONLY BEFORE golden zone touch)
            # + YOUR NEW RULE:
            #   If wick crosses fib(0.382) -> INVALID (not golden zone)
            # ------------------------------------------------------------
            if not self.validation_info["entered_golden_zone"]:

                # NEW: wick-cross-0.382 invalidation
                crossed_382 = (low < lvl_382) if self.trend == "UPTREND" else (high > lvl_382)
                if crossed_382:
                    self.state = FibState.INVALIDATED
                    self.invalidation_seq = int(seq)
                    self.invalidation_reason = "Wick crossed fib(0.382) before Golden Zone (treat as INVALID)"
                    self.updates.append({"seq": seq, "price": float(close), "event": "INVALIDATED",
                                         "detail": self.invalidation_reason})
                    return False

                # existing invalidation rules
                if self.trend == "UPTREND":
                    if low < self.anchor_price:
                        self.state = FibState.INVALIDATED
                        self.invalidation_seq = int(seq)
                        self.invalidation_reason = "Low broke anchor (0) before golden zone"
                        self.updates.append({"seq": seq, "price": float(low), "event": "INVALIDATED",
                                             "detail": self.invalidation_reason})
                        return False
                    if close < lvl_382:
                        self.state = FibState.INVALIDATED
                        self.invalidation_seq = int(seq)
                        self.invalidation_reason = f"Close {close:.2f} < fib(0.382) {lvl_382:.2f} before golden zone"
                        self.updates.append({"seq": seq, "price": float(close), "event": "INVALIDATED",
                                             "detail": self.invalidation_reason})
                        return False
                else:
                    if high > self.anchor_price:
                        self.state = FibState.INVALIDATED
                        self.invalidation_seq = int(seq)
                        self.invalidation_reason = "High broke anchor (0) before golden zone"
                        self.updates.append({"seq": seq, "price": float(high), "event": "INVALIDATED",
                                             "detail": self.invalidation_reason})
                        return False
                    if close > lvl_382:
                        self.state = FibState.INVALIDATED
                        self.invalidation_seq = int(seq)
                        self.invalidation_reason = f"Close {close:.2f} > fib(0.382) {lvl_382:.2f} before golden zone"
                        self.updates.append({"seq": seq, "price": float(close), "event": "INVALIDATED",
                                             "detail": self.invalidation_reason})
                        return False

            # ------------------------------------------------------------
            # GOLDEN ZONE TOUCH => VALID
            # BUT: must NOT cross 0.382 wick, otherwise it would have invalidated above.
            # ------------------------------------------------------------
            if not self.validation_info["entered_golden_zone"]:
                if self._golden_zone_touched_no_382_cross(open_price, high, low, close, lvl_382, lvl_618):
                    self.validation_info["entered_golden_zone"] = True
                    self.validation_info["golden_zone_entry_seq"] = int(seq)
                    self.validation_info["is_valid_fib"] = True
                    self.state = FibState.VALIDATED
                    self.updates.append({"seq": seq, "price": float((low + high) / 2),
                                         "event": "GOLDEN_ZONE_TOUCHED", "detail": "✓ VALID (touched 0.382–0.618 without crossing 0.382 wick)"})

            # AFTER golden zone rules
            if self.validation_info["entered_golden_zone"]:

                lvl_786 = float(fib[0.786]) if 0.786 in fib else None
                lvl_1 = float(fib[1.0]) if 1.0 in fib else None
                gz_seq = self.validation_info.get("golden_zone_entry_seq")

                # A) BEFORE ENTRY => INVALIDATED (SL-zone touch in ANY way)
                if self.entry_seq is None:
                    if self._stoploss_touch_anyway(open_price, high, low, close, lvl_0, lvl_382):
                        self.state = FibState.INVALIDATED
                        self.invalidation_seq = int(seq)
                        self.invalidation_reason = "Invalid BEFORE ENTRY: candle touched 0–0.382 zone (wick/body/close) or crossed 0 after GZ"
                        self.updates.append({"seq": seq, "price": float(close), "event": "INVALIDATED",
                                             "detail": self.invalidation_reason})
                        return False

                # B) ✅ ENTRY (after GZ): candle CLOSE must be inside 0.786–1.0
                if self.entry_seq is None and (lvl_786 is not None and lvl_1 is not None):
                    entry_low = min(lvl_786, lvl_1)
                    entry_high = max(lvl_786, lvl_1)
                    close_in_786_1 = (entry_low <= close <= entry_high)

                    same_as_anchor = (seq == self.anchor_seq)
                    same_as_opposite = (self.opposite_seq is not None and seq == self.opposite_seq)
                    same_as_extreme = (self.current_extreme_seq is not None and seq == self.current_extreme_seq)
                    same_as_gz = (gz_seq is not None and seq == int(gz_seq))

                    if close_in_786_1 and not (same_as_anchor or same_as_opposite or same_as_extreme or same_as_gz):
                        self.entry_seq = int(seq)
                        self.entry_price = float(close)
                        self.locked_extreme = True
                        self.updates.append({"seq": seq, "price": self.entry_price, "event": "ENTRY",
                                             "detail": "✓ ENTRY (CLOSE inside 0.786–1.0). Extreme locked after entry."})

                # C) AFTER ENTRY => STOPLOSS / TARGET
                if self.entry_seq is not None:

                    # STOPLOSS (touch 0–0.382 zone in ANY way OR cross 0)
                    if self.stoploss_seq is None:
                        if self._stoploss_touch_anyway(open_price, high, low, close, lvl_0, lvl_382):
                            self.stoploss_seq = int(seq)
                            self.stoploss_price = float(close)
                            self.state = FibState.STOPLOSS_HIT
                            self.updates.append({"seq": seq, "price": self.stoploss_price, "event": "STOPLOSS_HIT",
                                                 "detail": "✗ STOPLOSS (after ENTRY: wick/body/close touched 0–0.382 or crossed 0)"})
                            return False

                    # TARGET
                    if self.target_seq is None and (lvl_15 is not None and lvl_16 is not None):
                        target_low = min(lvl_15, lvl_16)
                        target_high = max(lvl_15, lvl_16)

                        if (low <= target_high) and (high >= target_low):
                            self.target_seq = int(seq)
                            self.target_price = float((low + high) / 2)
                            self.state = FibState.TARGET_HIT
                            self.updates.append({"seq": seq, "price": self.target_price, "event": "TARGET_HIT",
                                                 "detail": "✓ TARGET (after ENTRY: 1.5–1.6 zone)"})
                            return False

            return True

        return True


# =========================
# PDF EXPORT
# =========================
def export_fib_summary_pdf(filename: str, symbol: str, timeframe: str, sym_df: pd.DataFrame, legs: List[FibonacciLeg]):
    if "datetime" in sym_df.columns and not sym_df["datetime"].isna().all():
        dmin = pd.to_datetime(sym_df["datetime"].min()).date()
        dmax = pd.to_datetime(sym_df["datetime"].max()).date()
        intraday_date_str = f"{dmin}" if dmin == dmax else f"{dmin} to {dmax}"
    else:
        intraday_date_str = "N/A"

    total_applied = len(legs)
    valid_legs = [x for x in legs if x.validation_info.get("is_valid_fib")]
    invalid_legs = [x for x in legs if (x.state == FibState.INVALIDATED and not x.quality_rejected)]

    valid_target_hit = sum(1 for x in legs if x.validation_info.get("is_valid_fib") and x.state == FibState.TARGET_HIT)
    valid_stoploss_hit = sum(1 for x in legs if x.validation_info.get("is_valid_fib") and x.state == FibState.STOPLOSS_HIT)
    invalid_count = len(invalid_legs)

    valid_no_sl_tgt = sum(
        1 for x in legs
        if x.validation_info.get("is_valid_fib")
        and x.state in (FibState.VALIDATED, FibState.ACTIVE)
        and x.stoploss_seq is None and x.target_seq is None
    )

    seq_to_dt = {}
    if "seq" in sym_df.columns and "datetime" in sym_df.columns:
        tmp = sym_df[["seq", "datetime"]].dropna()
        for _, r in tmp.iterrows():
            seq_to_dt[int(r["seq"])] = pd.to_datetime(r["datetime"])

    # valid_entry_3pm_or_after = 0
    # for leg in valid_legs:
    #     gz_seq = leg.validation_info.get("golden_zone_entry_seq")
    #     if gz_seq is None:
    #         continue
    #     dt = seq_to_dt.get(int(gz_seq))
    #     if dt is None:
    #         continue
    #         # Count ENTRY at 15:00 or after 15:00
    #     if (dt.hour > 15) or (dt.hour == 15 and dt.minute >= 0):
    #         valid_entry_3pm_or_after += 1
    valid_entry_3pm_or_after = 0
    for leg in legs:
        if leg.entry_seq is None:
            continue

        dt = seq_to_dt.get(int(leg.entry_seq))
        if dt is None:
            continue

        # ENTRY at or after 15:00
        if dt.hour > 15 or (dt.hour == 15 and dt.minute >= 0):
            valid_entry_3pm_or_after += 1


    c = canvas.Canvas(filename, pagesize=A4)
    w, h = A4
    y = h - 2 * cm

    c.setFont("Helvetica-Bold", 16)
    c.drawString(2 * cm, y, f"Fibonacci Retracement Summary Report {TREND}")


    y -= 1.2 * cm
    c.setFont("Helvetica", 12)
    c.drawString(2 * cm, y, f"Symbol: {symbol}")
    y -= 0.7 * cm
    c.drawString(2 * cm, y, f"Timeframe: {timeframe}")
    y -= 0.7 * cm
    c.drawString(2 * cm, y, f"Intraday Date: {intraday_date_str}")

    y -= 1.0 * cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Counts:")

    y -= 0.8 * cm
    c.setFont("Helvetica", 12)
    c.drawString(2 * cm, y, f"Total Fibonacci Applied: {total_applied}")
    y -= 0.6 * cm
    c.drawString(2 * cm, y, f"Valid Fibonacci (Golden Zone touched): {len(valid_legs)}")
    y -= 0.6 * cm
    c.drawString(2 * cm, y, f"Valid + Target Hit: {valid_target_hit}")
    y -= 0.6 * cm
    c.drawString(2 * cm, y, f"Valid + Stoploss Hit: {valid_stoploss_hit}")
    y -= 0.6 * cm
    c.drawString(2 * cm, y, f"Valid but NO SL & NO TGT yet: {valid_no_sl_tgt}")
    y -= 0.6 * cm
    c.drawString(2 * cm, y, f"Invalid Fibonacci: {invalid_count}")
    y -= 0.6 * cm
    # c.drawString(2 * cm, y, f"Valid + Entry at 3:00 PM (GZ entry at 15:00): {valid_entry_3pm}")
    c.drawString(2 * cm, y, f"Valid + Entry at/after 3:00 PM (ENTRY >= 15:00): {valid_entry_3pm_or_after}")


    y -= 1.2 * cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Leg Details (latest first):")

    y -= 0.8 * cm
    c.setFont("Helvetica", 9)

    legs_sorted = sorted(legs, key=lambda x: x.anchor_seq, reverse=True)
    for leg in legs_sorted[:90]:
        ext = f"{leg.current_extreme_price:.2f}" if leg.current_extreme_price is not None else "-"
        ext_seq = f"{leg.current_extreme_seq}" if leg.current_extreme_seq is not None else "-"

        anchor_dt = seq_to_dt.get(leg.anchor_seq)
        anchor_time = anchor_dt.strftime("%H:%M") if anchor_dt is not None else "-"

        ext_dt = seq_to_dt.get(int(ext_seq)) if ext_seq != "-" else None
        ext_time = ext_dt.strftime("%H:%M") if ext_dt is not None else "-"

        gz_seq = leg.validation_info.get("golden_zone_entry_seq")
        gz_dt = seq_to_dt.get(int(gz_seq)) if gz_seq is not None else None
        gz_time = gz_dt.strftime("%H:%M") if gz_dt is not None else "-"

        ent_seq = leg.entry_seq if leg.entry_seq is not None else None
        ent_dt = seq_to_dt.get(int(ent_seq)) if ent_seq is not None else None
        ent_time = ent_dt.strftime("%H:%M") if ent_dt is not None else "-"

        sls_seq = leg.stoploss_seq if leg.stoploss_seq is not None else None
        sls_dt = seq_to_dt.get(int(sls_seq)) if sls_seq is not None else None
        sls_time = sls_dt.strftime("%H:%M") if sls_dt is not None else "-"

        tgs_seq = leg.target_seq if leg.target_seq is not None else None
        tgs_dt = seq_to_dt.get(int(tgs_seq)) if tgs_seq is not None else None
        tgs_time = tgs_dt.strftime("%H:%M") if tgs_dt is not None else "-"

        lock = f" LOCK={leg.locked_extreme_label}" if leg.locked_extreme_label else ""

        line = (
            f"Fib-{leg.leg_id} | anchor={anchor_time}@{leg.anchor_price:.2f} | "
            f"ext={ext_time}@{ext} | state={leg.state.name} | "
            f"GZ={gz_time} | ENTRY={ent_time} | SL={sls_time} | TGT={tgs_time}{lock}"
        )
        c.drawString(2 * cm, y, line)
        y -= 0.45 * cm

        if y < 2 * cm:
            c.showPage()
            y = h - 2 * cm
            c.setFont("Helvetica", 9)

    c.save()


# =========================
# ATR
# =========================
def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


# =========================
# PIVOT DETECTION
# =========================
def detect_pivot_low(df: pd.DataFrame, idx: int, period: int) -> bool:
    if idx < period or idx >= len(df) - period:
        return False
    low = float(df.iloc[idx]["low"])
    left = df.iloc[idx - period: idx]["low"].values
    right = df.iloc[idx + 1: idx + period + 1]["low"].values
    return (low < float(np.min(left))) and (low < float(np.min(right)))


def detect_pivot_high(df: pd.DataFrame, idx: int, period: int) -> bool:
    if idx < period or idx >= len(df) - period:
        return False
    high = float(df.iloc[idx]["high"])
    left = df.iloc[idx - period: idx]["high"].values
    right = df.iloc[idx + 1: idx + period + 1]["high"].values
    return (high > float(np.max(left))) and (high > float(np.max(right)))


# =========================
# DATETIME BUILDING
# =========================
def extract_yyyymmdd_from_filename(csv_file: str) -> Optional[str]:
    name = os.path.splitext(os.path.basename(csv_file))[0]
    m = re.search(r"(20\d{6})", name)
    return m.group(1) if m else None


def build_datetime(df: pd.DataFrame, base_date: Optional[str], csv_file: Optional[str] = None) -> pd.DataFrame:
    df.columns = [c.strip().lower() for c in df.columns]

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        return df

    if "date" in df.columns and "time" in df.columns:
        d = df["date"].astype(str).str.strip()
        t = df["time"].astype(str).str.strip()
        t = t.apply(lambda x: x + ":00" if (":" in x and len(x) == 5) else x)

        dt = pd.to_datetime(d + " " + t, errors="coerce")
        if dt.isna().all():
            d2 = d.str.replace(r"\D", "", regex=True)
            t2 = t.str.replace(r"\D", "", regex=True).str.zfill(6)
            dt = pd.to_datetime(d2 + t2, format="%Y%m%d%H%M%S", errors="coerce")

        df["datetime"] = dt
        return df

    if "time" in df.columns:
        inferred = None
        if base_date is None and csv_file is not None:
            inferred = extract_yyyymmdd_from_filename(csv_file)

        base = base_date or inferred
        if base is None:
            raise ValueError("CSV has only 'time' column but no BASE_DATE/filename date")

        t = df["time"].astype(str).str.strip()
        t = t.apply(lambda x: x + ":00" if (":" in x and len(x) == 5) else x)

        dt = pd.to_datetime(base + " " + t, format="%Y%m%d %H:%M:%S", errors="coerce")
        if dt.isna().all():
            t2 = df["time"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
            dt = pd.to_datetime(base + t2, format="%Y%m%d%H%M%S", errors="coerce")

        df["datetime"] = dt
        return df

    raise ValueError("Need 'datetime' OR ('date'+'time') OR ('time' + BASE_DATE).")


def load_and_resample(csv_file: str, timeframe: str, base_date: Optional[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    df = build_datetime(df, base_date, csv_file=csv_file)

    required = {"open", "high", "low", "close", "datetime"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    if "symbol" not in df.columns:
        df["symbol"] = DEFAULT_SYMBOL_NAME

    df = df.dropna(subset=["datetime"]).copy()
    df = df.sort_values(["symbol", "datetime"]).reset_index(drop=True)

    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"

    out = []
    for sym, g in df.groupby("symbol"):
        g = g.set_index("datetime")
        g_res = g.resample(timeframe).agg(agg).dropna(subset=["open", "high", "low", "close"])
        if len(g_res) > 0:
            g_res["symbol"] = sym
            out.append(g_res.reset_index())

    if not out:
        raise ValueError("No objects to concatenate (resample produced empty groups). Check datetime parsing/timeframe.")

    df_res = pd.concat(out, ignore_index=True)
    df_res = df_res.sort_values(["symbol", "datetime"]).reset_index(drop=True)
    df_res["seq"] = df_res.groupby("symbol").cumcount()
    return df_res


# =========================
# LIVE SIMULATOR
# =========================
def simulate_live_fibonacci_professional(
    df: pd.DataFrame,
    symbol: str,
    trend: str,
    pivot_period: int,
    min_gap: int,
) -> List[FibonacciLeg]:
    sym_df = df[df["symbol"] == symbol].copy().reset_index(drop=True)

    sym_df["atr"] = calculate_atr(sym_df, ATR_PERIOD)

    sym_df["is_pivot_low"] = False
    sym_df["is_pivot_high"] = False
    for i in range(pivot_period, len(sym_df) - pivot_period):
        sym_df.at[i, "is_pivot_low"] = detect_pivot_low(sym_df, i, pivot_period)
        sym_df.at[i, "is_pivot_high"] = detect_pivot_high(sym_df, i, pivot_period)

    legs: List[FibonacciLeg] = []
    rejected: List[FibonacciLeg] = []
    current_leg: Optional[FibonacciLeg] = None
    leg_id = 1

    start_idx = pivot_period

    if MANUAL_FIRST_ANCHOR_SEQ is not None:
        matches = sym_df.index[sym_df["seq"] == MANUAL_FIRST_ANCHOR_SEQ].tolist()
        if not matches:
            raise ValueError(f"MANUAL_FIRST_ANCHOR_SEQ={MANUAL_FIRST_ANCHOR_SEQ} not found in seq column.")
        start_idx = matches[0]

        r0 = sym_df.iloc[start_idx]
        seq0 = int(r0["seq"])
        atr0 = r0["atr"] if "atr" in r0 else None
        anchor_price = float(r0["low"]) if trend == "UPTREND" else float(r0["high"])

        current_leg = FibonacciLeg(leg_id, trend, seq0, anchor_price, min_gap, atr0)
        current_leg.updates.append({"seq": seq0, "price": anchor_price, "event": "MANUAL_ANCHOR_SET",
                                    "detail": f"Manual anchor set at seq={seq0} price={anchor_price:.2f}"})
        leg_id += 1

    for idx in range(start_idx, len(sym_df) - pivot_period):
        row = sym_df.iloc[idx]
        seq = int(row["seq"])
        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])
        atr = row["atr"] if "atr" in row else None

        is_pivot_low = bool(row["is_pivot_low"])
        is_pivot_high = bool(row["is_pivot_high"])

        if current_leg is None:
            if trend == "UPTREND" and is_pivot_low:
                current_leg = FibonacciLeg(leg_id, trend, seq, l, min_gap, atr)
                leg_id += 1
            elif trend != "UPTREND" and is_pivot_high:
                current_leg = FibonacciLeg(leg_id, trend, seq, h, min_gap, atr)
                leg_id += 1

        if current_leg is not None:
            alive = current_leg.update_candle(seq, o, h, l, c, is_pivot_high=is_pivot_high, is_pivot_low=is_pivot_low)

            if not alive:
                if current_leg.quality_rejected:
                    rejected.append(current_leg)
                else:
                    legs.append(current_leg)
                current_leg = None

    if current_leg is not None:
        if current_leg.quality_rejected:
            rejected.append(current_leg)
        else:
            legs.append(current_leg)

    if SHOW_QUALITY_REJECTED:
        legs = legs + rejected

    return legs


# =========================
# PLOT
# =========================
def plot_professional_fibonacci(df: pd.DataFrame, symbol: str, legs: List[FibonacciLeg], min_gap: int):
    sym_df = df[df["symbol"] == symbol].copy().reset_index(drop=True)

    display_legs: List[FibonacciLeg] = []
    for leg in legs:
        if leg.quality_rejected and not SHOW_QUALITY_REJECTED:
            continue
        if not SHOW_INVALIDATED_FIBS and leg.state == FibState.INVALIDATED and not leg.quality_rejected:
            continue
        if PLOT_ONLY_VALID_FIBS and not leg.validation_info.get("is_valid_fib"):
            continue
        display_legs.append(leg)

    display_legs = sorted(display_legs, key=lambda x: x.anchor_seq)
    if not display_legs:
        print(f"⚠️ No legs to plot for {symbol} with current display toggles.")
        return

    # seq -> datetime mapping
    seq_to_dt = {}
    if "seq" in sym_df.columns and "datetime" in sym_df.columns:
        for _, r in sym_df.iterrows():
            seq_to_dt[int(r["seq"])] = pd.to_datetime(r["datetime"])

    # ✅ seq -> candle row (fast lookup)
    sym_df_seq_indexed = sym_df.set_index("seq", drop=False)

    def seq_to_datetime(seq_val):
        return seq_to_dt.get(int(seq_val), pd.NaT)

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=sym_df["datetime"],
            open=sym_df["open"],
            high=sym_df["high"],
            low=sym_df["low"],
            close=sym_df["close"],
            increasing_line_color="#00ff5f",
            decreasing_line_color="#ff4d4d",
            increasing_fillcolor="#00ff5f",
            decreasing_fillcolor="#ff4d4d",
            name=symbol,
        )
    )

    palette = ["cyan", "magenta", "yellow", "orange", "lime", "pink", "white", "deepskyblue", "gold"]

    def leg_color(leg: FibonacciLeg) -> str:
        return palette[(leg.leg_id - 1) % len(palette)]

    y_min = float(sym_df["low"].min())
    y_max = float(sym_df["high"].max())

    for leg in display_legs:
        color = leg_color(leg)

        opacity = 0.55
        dash = "dot"
        if leg.state == FibState.TARGET_HIT:
            opacity = 0.90
            dash = "solid"
        elif leg.state == FibState.STOPLOSS_HIT:
            opacity = 0.55
            dash = "dot"
        elif leg.state == FibState.VALIDATED:
            opacity = 0.80
            dash = "dash"
        elif leg.state == FibState.INVALIDATED:
            opacity = 0.35
            dash = "dot"

        fig.add_shape(
            type="rect",
            x0=seq_to_datetime(leg.anchor_seq),
            x1=seq_to_datetime(leg.min_gap_satisfied_seq),
            y0=y_min,
            y1=y_max,
            fillcolor="rgba(255, 0, 0, 0.05)",
            line=dict(width=0),
            layer="below",
        )

        fig.add_trace(
            go.Scatter(
                x=[seq_to_datetime(leg.anchor_seq)],
                y=[leg.anchor_price],
                mode="markers+text",
                marker=dict(symbol="star", size=18, color=color, line=dict(color="black", width=2)),
                text=[f"Fib-{leg.leg_id}"],
                textposition="bottom center",
                textfont=dict(size=11, color=color, family="Arial Black"),
                showlegend=False,
            )
        )

        fib = leg.get_fib_levels()
        if not fib:
            continue

        end_seq = int(sym_df["seq"].max())
        if leg.state == FibState.TARGET_HIT and leg.target_seq is not None:
            end_seq = int(leg.target_seq)
        elif leg.state == FibState.STOPLOSS_HIT and leg.stoploss_seq is not None:
            end_seq = int(leg.stoploss_seq)
        elif leg.state == FibState.INVALIDATED and leg.invalidation_seq is not None:
            end_seq = int(leg.invalidation_seq)

        i = display_legs.index(leg)
        if i < len(display_legs) - 1:
            next_anchor = int(display_legs[i + 1].anchor_seq)
            if next_anchor > leg.anchor_seq:
                end_seq = min(end_seq, next_anchor - 1)

        anchor_dt = seq_to_datetime(leg.anchor_seq)
        end_dt = seq_to_datetime(end_seq)

        for lv, price in fib.items():
            width = 1
            if lv in (0.382, 0.5, 0.618):
                width = 2
            elif lv in (0, 1.0):
                width = 2.5
            elif lv in (1.5, 1.6):
                width = 2

            fig.add_shape(
                type="line",
                x0=anchor_dt,
                x1=end_dt,
                y0=float(price),
                y1=float(price),
                line=dict(color=color, width=width, dash=dash),
                opacity=opacity,
            )

            if lv in LABEL_LEVELS:
                fig.add_annotation(
                    x=end_dt,
                    y=float(price),
                    text=f"Fib-{leg.leg_id}|{lv:.3f}",
                    showarrow=False,
                    xanchor="left",
                    font=dict(size=12, color=color),
                    bgcolor="rgba(0,0,0,0.0)",
                    opacity=0.95,
                )

        # Golden zone shading
        if 0.382 in fib and 0.618 in fib:
            gz_low = min(float(fib[0.382]), float(fib[0.618]))
            gz_high = max(float(fib[0.382]), float(fib[0.618]))
            fig.add_shape(
                type="rect",
                x0=anchor_dt,
                x1=end_dt,
                y0=gz_low,
                y1=gz_high,
                fillcolor="rgba(0, 255, 255, 0.08)",
                line=dict(width=0),
                layer="below",
            )

        # Stoploss zone shading
        if 0 in fib and 0.382 in fib:
            sl_low = min(float(fib[0]), float(fib[0.382]))
            sl_high = max(float(fib[0]), float(fib[0.382]))
            fig.add_shape(
                type="rect",
                x0=anchor_dt,
                x1=end_dt,
                y0=sl_low,
                y1=sl_high,
                fillcolor="rgba(255, 0, 0, 0.08)",
                line=dict(width=0),
                layer="below",
            )

        # Target zone shading
        if 1.5 in fib and 1.6 in fib:
            t_low = min(float(fib[1.5]), float(fib[1.6]))
            t_high = max(float(fib[1.5]), float(fib[1.6]))
            fig.add_shape(
                type="rect",
                x0=anchor_dt,
                x1=end_dt,
                y0=t_low,
                y1=t_high,
                fillcolor="rgba(0, 255, 0, 0.08)",
                line=dict(width=0),
                layer="below",
            )

        # Extreme marker
        if leg.current_extreme_seq is not None and leg.current_extreme_price is not None:
            text = (leg.locked_extreme_label + " (locked)") if (leg.locked_extreme and leg.locked_extreme_label) else ""
            fig.add_trace(
                go.Scatter(
                    x=[seq_to_datetime(leg.current_extreme_seq)],
                    y=[leg.current_extreme_price],
                    mode="markers+text" if text else "markers",
                    marker=dict(symbol="diamond", size=14, color=color, line=dict(color="white", width=1)),
                    text=[text] if text else None,
                    textposition="top center",
                    textfont=dict(size=11, color=color),
                    showlegend=False,
                )
            )

        # Anchor shift markers
        shift_updates = [u for u in leg.updates if u.get("event") == "ANCHOR_SHIFTED"]
        for u in shift_updates:
            fig.add_trace(
                go.Scatter(
                    x=[seq_to_datetime(int(u["seq"]))],
                    y=[float(u["price"])],
                    mode="markers+text",
                    marker=dict(symbol="triangle-down", size=14, color="yellow", line=dict(color="black", width=1)),
                    text=["SHIFT"],
                    textposition="top center",
                    textfont=dict(size=10, color="yellow"),
                    showlegend=False,
                )
            )

        # GOLDEN ZONE marker
        gz_seq = leg.validation_info.get("golden_zone_entry_seq")
        if gz_seq is not None and int(gz_seq) in sym_df_seq_indexed.index:
            candle = sym_df_seq_indexed.loc[int(gz_seq)]
            candle_low = float(candle["low"])
            candle_high = float(candle["high"])

            if leg.trend == "UPTREND":
                gz_y = candle_low
                textpos = "bottom right"
            else:
                gz_y = candle_high
                textpos = "top right"

            fig.add_trace(
                go.Scatter(
                    x=[seq_to_datetime(int(gz_seq))],
                    y=[gz_y],
                    mode="markers+text",
                    marker=dict(symbol="circle", size=18, color="lime", line=dict(color="white", width=2)),
                    text=["GZ"],
                    textposition=textpos,
                    textfont=dict(size=12, color="white"),
                    showlegend=False,
                )
            )

        # ENTRY marker
        if leg.entry_seq is not None:
            fig.add_trace(
                go.Scatter(
                    x=[seq_to_datetime(leg.entry_seq)],
                    y=[leg.entry_price],
                    mode="markers+text",
                    marker=dict(symbol="triangle-up", size=16, color="deepskyblue", line=dict(color="white", width=1)),
                    text=["ENTRY"],
                    textfont=dict(size=11, color="white"),
                    showlegend=False,
                )
            )

        # STOPLOSS marker
        if leg.state == FibState.STOPLOSS_HIT and leg.stoploss_seq is not None:
            fig.add_trace(
                go.Scatter(
                    x=[seq_to_datetime(leg.stoploss_seq)],
                    y=[leg.stoploss_price],
                    mode="markers+text",
                    marker=dict(symbol="x", size=20, color="red", line=dict(color="white", width=2)),
                    text=["SL"],
                    textfont=dict(size=12, color="white"),
                    showlegend=False,
                )
            )

        # TARGET marker
        if leg.state == FibState.TARGET_HIT and leg.target_seq is not None:
            fig.add_trace(
                go.Scatter(
                    x=[seq_to_datetime(leg.target_seq)],
                    y=[leg.target_price],
                    mode="markers+text",
                    marker=dict(symbol="circle", size=20, color="lime", line=dict(color="white", width=2)),
                    text=["TGT"],
                    textfont=dict(size=12, color="white"),
                    showlegend=False,
                )
            )

    valid = sum(1 for x in display_legs if x.validation_info.get("is_valid_fib"))
    invalidated = sum(1 for x in display_legs if x.state == FibState.INVALIDATED and not x.quality_rejected)
    sl_hits = sum(1 for x in display_legs if x.state == FibState.STOPLOSS_HIT)
    tgt_hits = sum(1 for x in display_legs if x.state == FibState.TARGET_HIT)
    active = sum(1 for x in display_legs if x.state in (FibState.ACTIVE, FibState.VALIDATED))

    fig.update_layout(
        title=dict(
            text=f"{symbol} | Legs:{len(display_legs)} | Valid:{valid} | SL:{sl_hits} | TGT:{tgt_hits} | Invalid:{invalidated} | Active:{active} | MinGap={min_gap}",
            font=dict(color="white", size=16),
        ),
        xaxis=dict(
            title="Time",
            showgrid=True,
            gridcolor="rgba(128,128,128,0.15)",
            color="white",
            type="date",
        ),
        yaxis=dict(title="Price", showgrid=True, gridcolor="rgba(128,128,128,0.15)", color="white"),
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        xaxis_rangeslider_visible=False,
        height=900,
        width=1750,
        margin=dict(l=60, r=320, t=80, b=40),
        hovermode="x unified",
    )

    fig.show()


# =========================
# MAIN
# =========================
def main():
    try:
        print("Loading & resampling...")
        df = load_and_resample(CSV_FILE, RESAMPLE_TIMEFRAME, BASE_DATE)
        symbols = df["symbol"].unique()
        print(f"Loaded candles: {len(df)} | Symbols: {list(symbols)}")

        for sym in symbols:
            legs = simulate_live_fibonacci_professional(
                df=df,
                symbol=sym,
                trend=TREND,
                pivot_period=PIVOT_PERIOD,
                min_gap=MIN_CANDLES_LOW_TO_HIGH,
            )

            if DO_PLOT:
                plot_professional_fibonacci(df, sym, legs, MIN_CANDLES_LOW_TO_HIGH)

            sym_df = df[df["symbol"] == sym].copy().reset_index(drop=True)
            csv_base = os.path.splitext(os.path.basename(CSV_FILE))[0]
            tf_safe = RESAMPLE_TIMEFRAME.replace(" ", "").replace("/", "_")

            safe_sym = re.sub(r'[\\/:*?"<>|]+', "_", str(sym))
            pdf_name = f"{csv_base}_{tf_safe}_{TREND}_{safe_sym}_fib_summary.pdf"

            export_fib_summary_pdf(pdf_name, sym, RESAMPLE_TIMEFRAME, sym_df, legs)
            print(f"📄 PDF saved: {pdf_name}")

        print("\n✅ Done.")

    except FileNotFoundError:
        print(f"❌ CSV file not found: {CSV_FILE}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()