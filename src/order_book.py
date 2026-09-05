import math
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class OrderBookLevel:
    level: int
    bid_price: float
    bid_volume: int
    bid_orders: int
    ask_price: float
    ask_volume: int
    ask_orders: int


@dataclass
class OrderBookSnapshot:
    symbol: str
    levels: List[OrderBookLevel] = field(default_factory=list)

    @property
    def best_bid(self) -> Optional[float]:
        return self.levels[0].bid_price if self.levels else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.levels[0].ask_price if self.levels else None

    @property
    def total_bid_volume(self) -> int:
        return sum(l.bid_volume for l in self.levels)

    @property
    def total_ask_volume(self) -> int:
        return sum(l.ask_volume for l in self.levels)


@dataclass
class OrderBookReading:
    score: float  # -1 (heavy sell pressure) .. +1 (heavy buy pressure)
    weight: float
    bid_ask_ratio: float
    spread_pct: Optional[float]
    likely_queue: Optional[
        str
    ]  # "buy", "sell", or None - approximate, see module docstring
    reason_en: str
    reason_fa: str
    available: bool = True


class OrderBookAnalyzer:

    RATIO_SCALE = 3.0
    QUEUE_VOLUME_DOMINANCE = 50.0
    QUEUE_MAX_OPPOSING_ORDERS = 2  

    def parse(self, symbol: str, raw: dict) -> Optional[OrderBookSnapshot]:
        best_limits = raw.get("bestLimits") if raw else None
        if not best_limits:
            return None

        levels = []
        for row in best_limits:
            try:
                levels.append(
                    OrderBookLevel(
                        level=row.get("number", 0),
                        bid_price=float(row["pMeDem"]),
                        bid_volume=int(row["qTitMeDem"]),
                        bid_orders=int(row["zOrdMeDem"]),
                        ask_price=float(row["pMeOf"]),
                        ask_volume=int(row["qTitMeOf"]),
                        ask_orders=int(row["zOrdMeOf"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue

        if not levels:
            return None
        levels.sort(key=lambda l: l.level)
        return OrderBookSnapshot(symbol=symbol, levels=levels)

    def analyze(
        self, snapshot: Optional[OrderBookSnapshot], weight: float = 14.0
    ) -> OrderBookReading:
        if snapshot is None or not snapshot.levels:
            return OrderBookReading(
                score=0.0,
                weight=weight,
                bid_ask_ratio=1.0,
                spread_pct=None,
                likely_queue=None,
                available=False,
                reason_en="Order book unavailable (market may be closed, or the fetch failed).",
                reason_fa="اطلاعات تابلو در دسترس نیست (احتمالاً بازار بسته است یا دریافت داده ناموفق بود).",
            )

        total_bid = snapshot.total_bid_volume
        total_ask = snapshot.total_ask_volume
        eps = 1  # avoid division by zero without materially affecting real ratios
        ratio = (total_bid + eps) / (total_ask + eps)

        log_ratio = math.log2(ratio)
        score = max(-1.0, min(1.0, log_ratio / self.RATIO_SCALE))

        spread_pct = None
        if snapshot.best_bid and snapshot.best_ask:
            spread_pct = (snapshot.best_ask - snapshot.best_bid) / snapshot.best_bid

        likely_queue = None
        top = snapshot.levels[0]
        if (
            total_bid >= self.QUEUE_VOLUME_DOMINANCE * max(total_ask, 1)
            and top.ask_orders <= self.QUEUE_MAX_OPPOSING_ORDERS
        ):
            likely_queue = "buy"
        elif (
            total_ask >= self.QUEUE_VOLUME_DOMINANCE * max(total_bid, 1)
            and top.bid_orders <= self.QUEUE_MAX_OPPOSING_ORDERS
        ):
            likely_queue = "sell"

        if likely_queue == "buy":
            en = (
                f"Order book heavily bid-dominated ({ratio:.1f}x ask volume) with almost no sellers - "
                f"looks like a possible buy queue (صف خرید), though this isn't confirmed against the official price band."
            )
            fa = (
                f"تابلو به‌شدت سمت خرید سنگین است (نسبت {ratio:.1f} برابر حجم فروش) با فروشنده تقریباً صفر — "
                f"احتمال صف خرید وجود دارد، هرچند این تشخیص با دامنه رسمی نوسان قیمت تأیید نشده است."
            )
        elif likely_queue == "sell":
            en = (
                f"Order book heavily ask-dominated ({1/ratio:.1f}x bid volume) with almost no buyers - "
                f"looks like a possible sell queue (صف فروش), though this isn't confirmed against the official price band."
            )
            fa = (
                f"تابلو به‌شدت سمت فروش سنگین است (نسبت {1/ratio:.1f} برابر حجم خرید) با خریدار تقریباً صفر — "
                f"احتمال صف فروش وجود دارد، هرچند این تشخیص با دامنه رسمی نوسان قیمت تأیید نشده است."
            )
        elif ratio >= 2:
            en = f"Bid volume outweighs ask volume {ratio:.1f}x - more buying interest visible than selling."
            fa = f"حجم خرید {ratio:.1f} برابر حجم فروش است — تقاضای بیشتری نسبت به عرضه در تابلو دیده می‌شود."
        elif ratio <= 0.5:
            en = f"Ask volume outweighs bid volume {1/ratio:.1f}x - more selling interest visible than buying."
            fa = f"حجم فروش {1/ratio:.1f} برابر حجم خرید است — عرضه بیشتری نسبت به تقاضا در تابلو دیده می‌شود."
        else:
            en = "Order book is roughly balanced between bid and ask volume."
            fa = "تابلو از نظر حجم خرید و فروش تقریباً متعادل است."

        return OrderBookReading(
            score=score,
            weight=weight,
            bid_ask_ratio=round(ratio, 2),
            spread_pct=round(spread_pct, 4) if spread_pct is not None else None,
            likely_queue=likely_queue,
            reason_en=en,
            reason_fa=fa,
            available=True,
        )
