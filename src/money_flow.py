import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class MoneyFlowReading:

    score: float
    weight: float
    available: bool = True


class MoneyFlowAnalyzer:
    # Every doubling of the per-capita buy:sell ratio moves that half of the
    # score by 1/PER_CAPITA_SCALE.
    PER_CAPITA_SCALE = 2.0

    def analyze(self, raw: Optional[dict], weight: float = 14.0) -> MoneyFlowReading:
        ct = raw.get("clientType") if raw else None
        if not ct:
            return MoneyFlowReading(score=0.0, weight=weight, available=False)

        try:
            buy_val = float(ct["buy_I_Value"])
            sell_val = float(ct["sell_I_Value"])
            buy_count = float(ct.get("buy_I_Count", 0))
            sell_count = float(ct.get("sell_I_Count", 0))
        except (KeyError, TypeError, ValueError):
            return MoneyFlowReading(score=0.0, weight=weight, available=False)

        total = buy_val + sell_val
        if total <= 0:
            return MoneyFlowReading(score=0.0, weight=weight, available=False)

        # Net individual (real) money flow, naturally in [-1, 1]
        net_ratio = (buy_val - sell_val) / total

        # Per-capita buy vs. sell size (سرانه) - catches cases net value alone
        # can miss, e.g. many small sellers vs a few large buyers
        pc_score = 0.0
        if buy_count > 0 and sell_count > 0:
            per_capita_ratio = (buy_val / buy_count) / (sell_val / sell_count)
            pc_score = max(
                -1.0, min(1.0, math.log2(per_capita_ratio) / self.PER_CAPITA_SCALE)
            )

        score = max(-1.0, min(1.0, 0.5 * net_ratio + 0.5 * pc_score))
        return MoneyFlowReading(score=score, weight=weight, available=True)
