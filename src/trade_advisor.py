import pandas as pd
from dataclasses import dataclass, field
from typing import List


@dataclass
class TradeSignal:
    verdict: str  # "BUY", "WATCH", or "AVOID"
    score: int  # 0-100 composite technical score
    current_price: float
    entry_low: float
    entry_high: float
    stop_loss: float
    target: float
    risk_reward_ratio: float
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class TradeAdvisor:
    """
    Rule-based technical signal generator for SHORT-TERM SWING TRADES
    with a MODERATE risk/reward profile (target risk:reward ~= 1:2).

    This is NOT financial advice. It is a deterministic, rule-based
    summary of technical indicators already computed on the price history.
    It has no knowledge of fundamentals, news, market-wide conditions,
    or your personal financial situation.
    """

    ATR_STOP_MULTIPLIER = 1.5  # moderate risk: 1.5x ATR below entry
    RISK_REWARD_RATIO = 2.0  # moderate: aim for 2x the risked amount
    MAX_STOP_DISTANCE_PCT = 0.08  # never suggest a stop wider than 8% for a swing trade
    OVERBOUGHT_RSI = 75
    OVERSOLD_RSI = 30

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        if df.empty or len(df) < 50:
            raise ValueError(
                "Not enough price history to generate a reliable signal "
                "(need at least 50 trading days for EMA50/RSI/MACD to stabilize)."
            )

        latest = df.iloc[-1]
        reasons: List[str] = []
        warnings: List[str] = []
        score = 0

        close = latest["close_price"]
        ema20 = latest["EMA_20"]
        ema50 = latest["EMA_50"]
        rsi = latest["RSI_14"]
        macd = latest["MACD"]
        macd_signal = latest["MACD_Signal"]
        atr = latest["ATR_14"]
        recent_low = latest["RECENT_LOW_20"]
        recent_high = latest["RECENT_HIGH_20"]

        # --- 1. Trend: EMA20 vs EMA50 (max 30 pts) ---
        if pd.notna(ema20) and pd.notna(ema50) and ema20 > ema50:
            score += 30
            reasons.append(
                "Short-term trend is above the medium-term trend (EMA20 > EMA50) — uptrend intact."
            )
        else:
            reasons.append(
                "EMA20 is not above EMA50 — short-term trend is weak or bearish."
            )

        # --- 2. Price position vs EMA20 (max 15 pts) ---
        if pd.notna(ema20) and close > ema20:
            score += 15
            reasons.append("Price is trading above its 20-day EMA.")
        else:
            reasons.append(
                "Price is below its 20-day EMA — short-term momentum is soft."
            )

        # --- 3. RSI zone (max 25 pts) ---
        if pd.notna(rsi):
            if 45 <= rsi <= 65:
                score += 25
                reasons.append(
                    f"RSI ({rsi:.1f}) is in a healthy zone for a swing entry — not extended."
                )
            elif 35 <= rsi < 45 or 65 < rsi <= 70:
                score += 10
                reasons.append(
                    f"RSI ({rsi:.1f}) is borderline — acceptable but not ideal."
                )
            elif rsi > 70:
                reasons.append(
                    f"RSI ({rsi:.1f}) is elevated — stock may be short-term overbought."
                )
            else:
                reasons.append(
                    f"RSI ({rsi:.1f}) is weak — momentum has not turned up yet."
                )

        # --- 4. MACD momentum (max 30 pts) ---
        if pd.notna(macd) and pd.notna(macd_signal):
            if macd > macd_signal:
                score += 30
                reasons.append("MACD is above its signal line — bullish momentum.")
            else:
                reasons.append(
                    "MACD is below its signal line — momentum has not confirmed a turn yet."
                )

        score = min(100, max(0, score))

        # --- Verdict ---
        if pd.notna(rsi) and rsi >= self.OVERBOUGHT_RSI:
            verdict = "AVOID"
            warnings.append(
                f"RSI ({rsi:.1f}) is in overbought territory (>= {self.OVERBOUGHT_RSI}). "
                "Entering now risks buying into a short-term top — wait for a pullback."
            )
        elif score >= 70:
            verdict = "BUY"
        elif score >= 45:
            verdict = "WATCH"
        else:
            verdict = "AVOID"

        # --- Entry zone ---
        if pd.notna(ema20):
            extension_pct = (close - ema20) / ema20 if ema20 else 0
            if extension_pct > 0.03:
                # Price is extended above EMA20 — suggest waiting for a pullback
                entry_low = round(ema20)
                entry_high = round(ema20 * 1.02)
                warnings.append(
                    f"Price is {extension_pct*100:.1f}% above its EMA20 — considered extended for a "
                    "short-term entry. Consider waiting for a pullback toward the entry zone below "
                    "rather than chasing the current price."
                )
            else:
                entry_low = round(min(close, ema20))
                entry_high = round(close)
        else:
            entry_low = entry_high = round(close)

        entry_reference = (entry_low + entry_high) / 2

        # --- Stop-loss (volatility-based, capped) ---
        if pd.notna(atr) and atr > 0:
            atr_stop = entry_reference - (self.ATR_STOP_MULTIPLIER * atr)
        else:
            atr_stop = entry_reference * (1 - self.MAX_STOP_DISTANCE_PCT)

        max_stop_distance = entry_reference * self.MAX_STOP_DISTANCE_PCT
        min_allowed_stop = entry_reference - max_stop_distance
        stop_loss = round(max(atr_stop, min_allowed_stop))

        if pd.notna(recent_low) and recent_low > stop_loss:
            warnings.append(
                f"Suggested stop-loss ({stop_loss:,.0f}) sits below the recent 20-day low "
                f"({recent_low:,.0f}). Verify this against your own risk tolerance."
            )

        # --- Target (risk:reward based, checked against resistance) ---
        risk_per_share = entry_reference - stop_loss
        target = round(entry_reference + (risk_per_share * self.RISK_REWARD_RATIO))

        if pd.notna(recent_high) and target > recent_high:
            warnings.append(
                f"Calculated target ({target:,.0f}) is above recent resistance "
                f"({recent_high:,.0f}). Price may stall or reverse near that level before reaching target."
            )

        risk_reward = round(target - entry_reference) / max(risk_per_share, 1)

        return TradeSignal(
            verdict=verdict,
            score=score,
            current_price=round(close),
            entry_low=entry_low,
            entry_high=entry_high,
            stop_loss=stop_loss,
            target=target,
            risk_reward_ratio=round(risk_reward, 2),
            reasons=reasons,
            warnings=warnings,
        )
