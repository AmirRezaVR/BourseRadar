import pandas as pd
from dataclasses import dataclass
from typing import List, Optional

from src.order_book import OrderBookAnalyzer
from src.money_flow import MoneyFlowAnalyzer


@dataclass
class SignalWeights:

    rsi: float = 15
    macd: float = 19
    ma_trend: float = 19
    volume: float = 10
    support_resistance: float = 10
    order_book: float = 13
    money_flow: float = 14

    def total(self) -> float:
        return (
            self.rsi
            + self.macd
            + self.ma_trend
            + self.volume
            + self.support_resistance
            + self.order_book
            + self.money_flow
        )

    def __post_init__(self):
        if abs(self.total() - 100) > 0.01:
            raise ValueError(f"SignalWeights must sum to 100, got {self.total()}")


@dataclass
class SignalThresholds:
    strong: float = 0.60  # |composite| >= this -> Strong Buy / Strong Sell
    moderate: float = 0.25  # |composite| >= this -> Buy / Sell


@dataclass
class VolatilityConfig:
    expansion_ratio: float = (
        1.5  # ATR% > 1.5x its own 60-day average -> volatility expansion
    )
    expansion_multiplier: float = (
        0.7  # confidence is scaled by this much when expanding
    )


VERDICT_LABELS = {
    "STRONG_BUY": {"en": "Strong Buy", "fa": "خرید قوی"},
    "BUY": {"en": "Buy", "fa": "خرید"},
    "NEUTRAL": {"en": "Neutral / Hold", "fa": "خنثی / نگهداری"},
    "SELL": {"en": "Sell", "fa": "فروش / اجتناب از خرید"},
    "STRONG_SELL": {"en": "Strong Sell", "fa": "فروش قوی / اجتناب جدی از خرید"},
}
VERDICT_ICONS = {
    "STRONG_BUY": "🟢",
    "BUY": "🟢",
    "NEUTRAL": "🟡",
    "SELL": "🔴",
    "STRONG_SELL": "🔴",
}
MIN_HISTORY_DAYS = 60


@dataclass
class FactorResult:
    """Just enough to compute the composite score"""

    name: str
    score: float
    weight: float
    available: bool = True

    @property
    def weighted_contribution(self) -> float:
        return self.score * self.weight


@dataclass
class Recommendation:
    """Trimmed to exactly what main.py's short-mode output reads. Nothing here is unused."""

    symbol: str
    verdict: str
    verdict_en: str
    verdict_fa: str
    confidence: float

    current_price: float
    entry_low: float
    entry_high: float
    stop_loss: float
    target: float
    risk_reward_ratio: float


class SignalEngine:

    ATR_STOP_MULTIPLIER = 1.5
    MAX_STOP_DISTANCE_PCT = 0.08
    RISK_REWARD_RATIO = 2.0
    ENTRY_EXTENSION_THRESHOLD = 0.03

    def __init__(
        self,
        weights: Optional[SignalWeights] = None,
        thresholds: Optional[SignalThresholds] = None,
        volatility_cfg: Optional[VolatilityConfig] = None,
    ):
        self.weights = weights or SignalWeights()
        self.thresholds = thresholds or SignalThresholds()
        self.volatility_cfg = volatility_cfg or VolatilityConfig()
        self.order_book_analyzer = OrderBookAnalyzer()
        self.money_flow_analyzer = MoneyFlowAnalyzer()

        self.FACTOR_METHODS = [
            ("rsi", self._score_rsi),
            ("macd", self._score_macd),
            ("ma_trend", self._score_ma_trend),
            ("volume", self._score_volume),
            ("support_resistance", self._score_support_resistance),
        ]

    def _score_rsi(self, df: pd.DataFrame, latest: pd.Series) -> FactorResult:
        rsi = latest.get("RSI_14")
        weight = self.weights.rsi
        if pd.isna(rsi):
            return FactorResult("rsi", 0.0, weight)

        if rsi > 80:
            score = -0.3
        elif rsi > 70:
            score = 0.3
        elif rsi >= 55:
            score = 0.6
        elif rsi > 45:
            score = 0.0
        elif rsi >= 30:
            score = -0.6
        elif rsi >= 20:
            score = -0.3
        else:
            score = 0.3

        return FactorResult("rsi", score, weight)

    def _score_macd(self, df: pd.DataFrame, latest: pd.Series) -> FactorResult:
        weight = self.weights.macd
        macd, signal = latest.get("MACD"), latest.get("MACD_Signal")
        if pd.isna(macd) or pd.isna(signal) or len(df) < 4:
            return FactorResult("macd", 0.0, weight)

        hist_recent = df["MACD_Hist"].tail(3)
        hist_up = hist_recent.diff().mean() > 0 if len(hist_recent) >= 2 else False
        bullish_cross = macd > signal

        if bullish_cross and hist_up:
            score = 1.0
        elif bullish_cross and not hist_up:
            score = 0.5
        elif not bullish_cross and hist_up:
            score = -0.5
        else:
            score = -1.0

        return FactorResult("macd", score, weight)

    def _score_ma_trend(self, df: pd.DataFrame, latest: pd.Series) -> FactorResult:
        weight = self.weights.ma_trend
        close, ema20, ema50 = (
            latest.get("close_price"),
            latest.get("EMA_20"),
            latest.get("EMA_50"),
        )
        if pd.isna(ema20) or pd.isna(ema50):
            return FactorResult("ma_trend", 0.0, weight)

        short_term = 0.5 if close > ema20 else -0.5
        medium_term = 0.5 if ema20 > ema50 else -0.5
        score = short_term + medium_term

        return FactorResult("ma_trend", score, weight)

    def _score_volume(self, df: pd.DataFrame, latest: pd.Series) -> FactorResult:
        weight = self.weights.volume
        vol = latest.get("volume")
        avg_vol = df["volume"].tail(20).mean() if len(df) >= 20 else None

        if avg_vol is None or pd.isna(avg_vol) or avg_vol == 0:
            return FactorResult("volume", 0.0, weight)

        ratio = vol / avg_vol
        price_change = (
            latest.get("close_price") - df["close_price"].iloc[-2]
            if len(df) >= 2
            else 0
        )

        if ratio >= 1.5 and price_change > 0:
            score = 1.0
        elif ratio >= 1.5 and price_change < 0:
            score = -1.0
        elif ratio < 0.8:
            score = 0.0
        else:
            score = 0.3 if price_change > 0 else (-0.3 if price_change < 0 else 0.0)

        return FactorResult("volume", score, weight)

    def _score_support_resistance(
        self, df: pd.DataFrame, latest: pd.Series
    ) -> FactorResult:
        weight = self.weights.support_resistance
        close = latest.get("close_price")
        recent_low = latest.get("RECENT_LOW_20")
        recent_high = latest.get("RECENT_HIGH_20")

        if pd.isna(recent_low) or pd.isna(recent_high) or recent_high == recent_low:
            return FactorResult("support_resistance", 0.0, weight)

        position = (close - recent_low) / (recent_high - recent_low)

        if position <= 0.15:
            score = 0.5
        elif position >= 0.85:
            score = -0.5
        elif position <= 0.35:
            score = 0.2
        elif position >= 0.65:
            score = -0.2
        else:
            score = 0.0

        return FactorResult("support_resistance", score, weight)

    def _volatility_multiplier(self, df: pd.DataFrame, latest: pd.Series) -> float:
        atr = latest.get("ATR_14")
        close = latest.get("close_price")
        if pd.isna(atr) or not close or len(df) < 60:
            return 1.0

        atr_pct_series = df["ATR_14"] / df["close_price"]
        atr_pct = atr / close
        atr_pct_avg = atr_pct_series.tail(60).mean()

        if pd.isna(atr_pct_avg) or atr_pct_avg == 0:
            return 1.0

        ratio = atr_pct / atr_pct_avg
        if ratio >= self.volatility_cfg.expansion_ratio:
            return self.volatility_cfg.expansion_multiplier

        return 1.0

    def _classify(self, composite: float) -> str:
        if composite >= self.thresholds.strong:
            return "STRONG_BUY"
        if composite >= self.thresholds.moderate:
            return "BUY"
        if composite <= -self.thresholds.strong:
            return "STRONG_SELL"
        if composite <= -self.thresholds.moderate:
            return "SELL"
        return "NEUTRAL"

    def _compute_entry_stop_target(self, latest: pd.Series) -> dict:
        close, ema20 = latest.get("close_price"), latest.get("EMA_20")
        atr = latest.get("ATR_14")

        if pd.notna(ema20) and ema20:
            extension = (close - ema20) / ema20
            if extension > self.ENTRY_EXTENSION_THRESHOLD:
                entry_low, entry_high = round(ema20), round(ema20 * 1.02)
            else:
                entry_low, entry_high = round(min(close, ema20)), round(close)
        else:
            entry_low = entry_high = round(close)

        entry_ref = (entry_low + entry_high) / 2
        if pd.notna(atr) and atr > 0:
            atr_stop = entry_ref - (self.ATR_STOP_MULTIPLIER * atr)
        else:
            atr_stop = entry_ref * (1 - self.MAX_STOP_DISTANCE_PCT)
        min_allowed_stop = entry_ref * (1 - self.MAX_STOP_DISTANCE_PCT)
        stop_loss = round(max(atr_stop, min_allowed_stop))

        risk_per_share = max(entry_ref - stop_loss, 1)
        target = round(entry_ref + risk_per_share * self.RISK_REWARD_RATIO)
        rr = round((target - entry_ref) / risk_per_share, 2)

        return dict(
            entry_low=entry_low,
            entry_high=entry_high,
            stop_loss=stop_loss,
            target=target,
            risk_reward_ratio=rr,
        )

    def generate(
        self,
        df: pd.DataFrame,
        symbol: Optional[str] = None,
        order_book_raw: Optional[dict] = None,
        money_flow_raw: Optional[dict] = None,
    ) -> Recommendation:
        if df.empty or len(df) < MIN_HISTORY_DAYS:
            raise ValueError(
                f"Not enough price history (need {MIN_HISTORY_DAYS}+ trading days) / "
                f"داده تاریخی کافی نیست (حداقل {MIN_HISTORY_DAYS} روز لازم است)"
            )

        latest = df.iloc[-1]
        name = symbol or "This stock"

        factors = [method(df, latest) for _, method in self.FACTOR_METHODS]

        order_book_snapshot = None
        if order_book_raw:
            order_book_snapshot = self.order_book_analyzer.parse(name, order_book_raw)
        ob_reading = self.order_book_analyzer.analyze(
            order_book_snapshot, weight=self.weights.order_book
        )
        factors.append(
            FactorResult(
                "order_book",
                ob_reading.score,
                ob_reading.weight,
                available=ob_reading.available,
            )
        )

        mf_reading = self.money_flow_analyzer.analyze(
            money_flow_raw, weight=self.weights.money_flow
        )
        factors.append(
            FactorResult(
                "money_flow",
                mf_reading.score,
                mf_reading.weight,
                available=mf_reading.available,
            )
        )

        available_factors = [f for f in factors if f.available]
        available_weight = sum(f.weight for f in available_factors)
        raw_composite = (
            sum(f.weighted_contribution for f in available_factors) / available_weight
            if available_weight > 0
            else 0.0
        )
        raw_composite = max(-1.0, min(1.0, raw_composite))

        vol_multiplier = self._volatility_multiplier(df, latest)
        composite = max(-1.0, min(1.0, raw_composite * vol_multiplier))
        confidence = round(abs(composite) * 100, 1)
        verdict = self._classify(composite)

        levels = self._compute_entry_stop_target(latest)
        label = VERDICT_LABELS[verdict]

        return Recommendation(
            symbol=name,
            verdict=verdict,
            verdict_en=label["en"],
            verdict_fa=label["fa"],
            confidence=confidence,
            current_price=round(latest["close_price"]),
            entry_low=levels["entry_low"],
            entry_high=levels["entry_high"],
            stop_loss=levels["stop_loss"],
            target=levels["target"],
            risk_reward_ratio=levels["risk_reward_ratio"],
        )
