"""
Real vs. institutional (حقیقی/حقوقی) money flow analysis.

TSETMC's ClientType/GetClientTypeHistory endpoint splits each day's buy/sell
activity into two categories. Confirmed field names (from a real captured
response), prefixed I (individual/real - حقیقی) and N (institutional/legal
- حقوقی): buy_I_Value, buy_N_Value, sell_I_Value, sell_N_Value, plus
matching _Volume and _Count fields.

The I/N meaning is inferred by elimination (there are only two categories,
and Iranian market convention always splits حقیقی/حقوقی) rather than
confirmed from official documentation, since none exists for this endpoint.

Two signals are combined here, both specific to how TSE retail activity is
commonly read:
  1. Net individual money flow as a fraction of individual trade value -
     are real/retail investors net buying or net selling today.
  2. Per-capita buy vs. sell size (سرانه خرید/فروش حقیقی) - is the average
     individual buyer spending more or less than the average individual
     seller. This catches cases net value alone can miss (e.g. a lot of
     small sellers vs a few large buyers looks very different from raw
     net value than from per-capita size).
"""

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class MoneyFlowReading:
    score: float          # -1 (real money leaving) .. +1 (real money entering)
    weight: float
    net_ratio: Optional[float]
    per_capita_ratio: Optional[float]
    reason_en: str
    reason_fa: str
    available: bool = True


class MoneyFlowAnalyzer:
    PER_CAPITA_SCALE = 2.0  # a 4x per-capita ratio maxes out that sub-signal

    def analyze(self, raw: Optional[dict], weight: float = 14.0) -> MoneyFlowReading:
        ct = raw.get("clientType") if raw else None
        if not ct:
            return MoneyFlowReading(
                score=0.0, weight=weight, net_ratio=None, per_capita_ratio=None, available=False,
                reason_en="Money-flow data unavailable (market may be closed, or the fetch failed).",
                reason_fa="اطلاعات ورود و خروج پول در دسترس نیست (احتمالاً بازار بسته است یا دریافت داده ناموفق بود).",
            )

        try:
            buy_val = float(ct["buy_I_Value"])
            sell_val = float(ct["sell_I_Value"])
            buy_count = float(ct.get("buy_I_Count", 0))
            sell_count = float(ct.get("sell_I_Count", 0))
        except (KeyError, TypeError, ValueError):
            return MoneyFlowReading(
                score=0.0, weight=weight, net_ratio=None, per_capita_ratio=None, available=False,
                reason_en="Money-flow data malformed.", reason_fa="داده ورود و خروج پول ناقص است.",
            )

        total = buy_val + sell_val
        if total <= 0:
            return MoneyFlowReading(
                score=0.0, weight=weight, net_ratio=None, per_capita_ratio=None, available=False,
                reason_en="No individual trading value recorded for this day.",
                reason_fa="ارزش معاملات حقیقی برای این روز ثبت نشده است.",
            )

        # Sub-signal 1: net flow, naturally in [-1, 1] by construction
        net_ratio = (buy_val - sell_val) / total

        # Sub-signal 2: per-capita buy vs sell size, symmetric via log2 (same
        # pattern as the order-book imbalance score)
        per_capita_ratio = None
        pc_score = 0.0
        if buy_count > 0 and sell_count > 0:
            pc_buy = buy_val / buy_count
            pc_sell = sell_val / sell_count
            per_capita_ratio = pc_buy / pc_sell
            pc_score = max(-1.0, min(1.0, math.log2(per_capita_ratio) / self.PER_CAPITA_SCALE))

        score = max(-1.0, min(1.0, 0.5 * net_ratio + 0.5 * pc_score))

        net_pct = net_ratio * 100
        if net_ratio > 0.05 and (per_capita_ratio is None or per_capita_ratio >= 1):
            en = (f"Real/individual investors were net BUYERS today ({net_pct:+.1f}% of their trade value)"
                  + (f", with average buy size {per_capita_ratio:.2f}x average sell size." if per_capita_ratio else "."))
            fa = (f"سرمایه‌گذاران حقیقی امروز خالص خریدار بودند ({net_pct:+.1f}٪ از ارزش معاملات آن‌ها)"
                  + (f"، با سرانه خرید {per_capita_ratio:.2f} برابر سرانه فروش." if per_capita_ratio else "."))
        elif net_ratio < -0.05 and (per_capita_ratio is None or per_capita_ratio <= 1):
            en = (f"Real/individual investors were net SELLERS today ({net_pct:+.1f}% of their trade value)"
                  + (f", with average sell size {1/per_capita_ratio:.2f}x average buy size." if per_capita_ratio else "."))
            fa = (f"سرمایه‌گذاران حقیقی امروز خالص فروشنده بودند ({net_pct:+.1f}٪ از ارزش معاملات آن‌ها)"
                  + (f"، با سرانه فروش {1/per_capita_ratio:.2f} برابر سرانه خرید." if per_capita_ratio else "."))
        elif per_capita_ratio and abs(per_capita_ratio - 1) > 0.15 and abs(net_ratio) <= 0.05:
            direction_en = "larger" if per_capita_ratio > 1 else "smaller"
            direction_fa = "بزرگ‌تر" if per_capita_ratio > 1 else "کوچک‌تر"
            en = f"Net individual flow is roughly balanced, but average buy size is {direction_en} than average sell size ({per_capita_ratio:.2f}x)."
            fa = f"جریان خالص حقیقی تقریباً متعادل است، اما سرانه خرید {direction_fa} از سرانه فروش است ({per_capita_ratio:.2f} برابر)."
        else:
            en = f"Real/individual money flow is roughly balanced today ({net_pct:+.1f}%)."
            fa = f"جریان پول حقیقی امروز تقریباً متعادل است ({net_pct:+.1f}٪)."

        return MoneyFlowReading(
            score=score, weight=weight, net_ratio=round(net_ratio, 4),
            per_capita_ratio=round(per_capita_ratio, 3) if per_capita_ratio else None,
            reason_en=en, reason_fa=fa, available=True,
        )
