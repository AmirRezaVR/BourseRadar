import math
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class OrderBookLevel:
    level: int
    bid_price: float
    bid_volume: int
    ask_price: float
    ask_volume: int


@dataclass
class OrderBookSnapshot:
    symbol: str
    levels: List[OrderBookLevel] = field(default_factory=list)

    @property
    def total_bid_volume(self) -> int:
        return sum(l.bid_volume for l in self.levels)

    @property
    def total_ask_volume(self) -> int:
        return sum(l.ask_volume for l in self.levels)


@dataclass
class OrderBookReading:
    """Just enough for the signal engine to use this as a factor."""
    score: float
    weight: float
    available: bool = True


class OrderBookAnalyzer:
    # Every doubling of the bid:ask volume ratio moves the score by 1/RATIO_SCALE.
    RATIO_SCALE = 3.0

    def parse(self, symbol: str, raw: dict) -> Optional[OrderBookSnapshot]:
        best_limits = raw.get("bestLimits") if raw else None
        if not best_limits:
            return None

        levels = []
        for row in best_limits:
            try:
                levels.append(OrderBookLevel(
                    level=row.get("number", 0),
                    bid_price=float(row["pMeDem"]), bid_volume=int(row["qTitMeDem"]),
                    ask_price=float(row["pMeOf"]), ask_volume=int(row["qTitMeOf"]),
                ))
            except (KeyError, TypeError, ValueError):
                continue

        if not levels:
            return None
        levels.sort(key=lambda l: l.level)
        return OrderBookSnapshot(symbol=symbol, levels=levels)

    def analyze(self, snapshot: Optional[OrderBookSnapshot], weight: float = 14.0) -> OrderBookReading:
        if snapshot is None or not snapshot.levels:
            return OrderBookReading(score=0.0, weight=weight, available=False)

        total_bid = snapshot.total_bid_volume
        total_ask = snapshot.total_ask_volume
        eps = 1  # avoid division by zero without materially affecting real ratios
        ratio = (total_bid + eps) / (total_ask + eps)

        score = max(-1.0, min(1.0, math.log2(ratio) / self.RATIO_SCALE))
        return OrderBookReading(score=score, weight=weight, available=True)
