from dataclasses import dataclass
from datetime import date, datetime, time as datetime_time
import math
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd


@dataclass
class SignalWeights:

    rsi: float = 16
    macd: float = 16
    ma_trend: float = 28
    volume: float = 20
    support_resistance: float = 20

    def total(self) -> float:
        return (
            self.rsi + self.macd + self.ma_trend + self.volume + self.support_resistance
        )

    def __post_init__(self):
        if abs(self.total() - 100) > 0.01:
            raise ValueError(f"SignalWeights must sum to 100, got {self.total()}")


@dataclass
class SignalThresholds:
    strong: float = 0.60  # |composite| >= this -> Strong Buy / Strong Sell
    moderate: float = 0.25  # |composite| >= this -> Buy / Sell


VERDICT_LABELS = {
    "STRONG_BUY": {"en": "Strong Buy", "fa": "خرید قوی"},
    "BUY": {"en": "Buy", "fa": "خرید"},
    "NEUTRAL": {"en": "Neutral / Hold", "fa": "خنثی / نگهداری"},
    "SELL": {"en": "Sell", "fa": "فروش / اجتناب از خرید"},
    "STRONG_SELL": {"en": "Strong Sell", "fa": "فروش قوی / اجتناب جدی از خرید"},
}
MIN_HISTORY_DAYS = 60
MIN_RISK_REWARD = 1.0
TEHRAN_TZ = ZoneInfo("Asia/Tehran")
MARKET_CLOSE_TIME = datetime_time(12, 30)


@dataclass
class FactorResult:
    name: str
    score: float
    weight: float

    @property
    def weighted_contribution(self) -> float:
        return self.score * self.weight


@dataclass
class Recommendation:
    symbol: str
    verdict: str
    verdict_en: str
    verdict_fa: str
    signal_strength: float

    current_price: float
    entry_low: Optional[float]
    entry_high: Optional[float]
    stop_loss: Optional[float]
    target: Optional[float]
    risk_reward_ratio: Optional[float]
    setup_valid: bool


class SignalEngine:

    ATR_STOP_MULTIPLIER = 1.5
    MAX_STOP_DISTANCE_PCT = 0.08
    ENTRY_EXTENSION_THRESHOLD = 0.03

    def __init__(
        self,
        weights: Optional[SignalWeights] = None,
        thresholds: Optional[SignalThresholds] = None,
    ):
        self.weights = weights or SignalWeights()
        self.thresholds = thresholds or SignalThresholds()

        self.FACTOR_METHODS = [
            ("rsi", self._score_rsi),
            ("macd", self._score_macd),
            ("ma_trend", self._score_ma_trend),
            ("volume", self._score_volume),
            ("support_resistance", self._score_support_resistance),
        ]

    def _score_rsi(self, df: pd.DataFrame, latest: pd.Series) -> FactorResult:
        """Use contrarian scores at extreme RSI values."""
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

        if latest.get("RSI_BULLISH_DIVERGENCE", False):
            score += 0.4
        elif latest.get("RSI_BEARISH_DIVERGENCE", False):
            score -= 0.4

        score = max(-1.0, min(1.0, score))
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

        if latest.get("MACD_BULLISH_DIVERGENCE", False):
            score += 0.4
        elif latest.get("MACD_BEARISH_DIVERGENCE", False):
            score -= 0.4

        score = max(-1.0, min(1.0, score))
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
        trend_score = short_term + medium_term

        close_prices = pd.to_numeric(df["close_price"], errors="coerce")
        momentum_horizons = ((5, 0.25), (20, 0.40), (60, 0.35))
        momentum_scores = []
        atr = latest.get("ATR_14")
        for days, horizon_weight in momentum_horizons:
            if len(close_prices) <= days:
                continue
            prior_close = close_prices.iloc[-days - 1]
            if (
                pd.isna(prior_close)
                or prior_close <= 0
                or pd.isna(close)
                or pd.isna(atr)
                or atr <= 0
            ):
                continue
            period_return = close / prior_close - 1
            expected_move = (atr / close) * math.sqrt(days)
            normalized_return = period_return / expected_move
            momentum_score = normalized_return / (1 + abs(normalized_return))
            momentum_scores.append((momentum_score, horizon_weight))

        if momentum_scores:
            total_horizon_weight = sum(
                horizon_weight for _, horizon_weight in momentum_scores
            )
            momentum_score = (
                sum(score * horizon_weight for score, horizon_weight in momentum_scores)
                / total_horizon_weight
            )
            score = 0.55 * trend_score + 0.45 * momentum_score
        else:
            score = trend_score

        return FactorResult("ma_trend", score, weight)

    def _score_volume(self, df: pd.DataFrame, latest: pd.Series) -> FactorResult:
        weight = self.weights.volume
        vol = latest.get("volume")
        avg_vol = df["volume"].iloc[-21:-1].mean() if len(df) >= 21 else None

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
        """Prioritize breakout continuation over range mean reversion."""
        weight = self.weights.support_resistance
        close = latest.get("close_price")

        if latest.get("BREAKOUT_UP_20", False):
            return FactorResult("support_resistance", 0.8, weight)
        if latest.get("BREAKOUT_DOWN_20", False):
            return FactorResult("support_resistance", -0.8, weight)
        if latest.get("FAKE_BREAKOUT_UP_20", False):
            return FactorResult("support_resistance", -0.6, weight)
        if latest.get("FAKE_BREAKOUT_DOWN_20", False):
            return FactorResult("support_resistance", 0.6, weight)

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

    @staticmethod
    def _freshness_multiplier(
        latest: pd.Series, as_of_date: Optional[date] = None
    ) -> float:
        date_value = latest.get("date")
        try:
            date_text = str(date_value)
            if len(date_text) == 8 and date_text.isdigit():
                market_date = datetime.strptime(date_text, "%Y%m%d").date()
            else:
                market_date = datetime.strptime(date_text[:10], "%Y-%m-%d").date()
        except (TypeError, ValueError):
            return 0.85

        reference_date = as_of_date or datetime.now(TEHRAN_TZ).date()
        age_days = (reference_date - market_date).days
        if age_days <= 3:
            return 1.0
        if age_days <= 14:
            return 1.0 - 0.35 * (age_days - 3) / 11
        return 0.65

    @staticmethod
    def _is_partial_session(
        latest: pd.Series, as_of_datetime: Optional[datetime] = None
    ) -> bool:
        date_value = latest.get("date")
        time_value = latest.get("last_update_time")
        try:
            date_text = str(date_value)
            if len(date_text) == 8 and date_text.isdigit():
                market_date = datetime.strptime(date_text, "%Y%m%d").date()
            else:
                market_date = datetime.strptime(date_text[:10], "%Y-%m-%d").date()
            market_time = datetime.strptime(str(time_value)[:5], "%H:%M").time()
        except (TypeError, ValueError):
            return False

        reference_datetime = as_of_datetime or datetime.now(TEHRAN_TZ).replace(
            tzinfo=None
        )
        return (
            market_date == reference_datetime.date()
            and reference_datetime.time() < MARKET_CLOSE_TIME
            and market_time < MARKET_CLOSE_TIME
            and market_time <= reference_datetime.time()
        )

    def _compute_entry_stop_target(
        self, latest: pd.Series, is_short: bool = False
    ) -> dict:
        close, ema20 = latest.get("close_price"), latest.get("EMA_20")
        atr = latest.get("ATR_14")

        if pd.notna(ema20) and ema20:
            extension = (close - ema20) / ema20
            if not is_short and extension > self.ENTRY_EXTENSION_THRESHOLD:
                entry_low, entry_high = round(ema20), round(ema20 * 1.02)
            elif is_short and extension < -self.ENTRY_EXTENSION_THRESHOLD:
                entry_low, entry_high = round(ema20 * 0.98), round(ema20)
            elif is_short:
                entry_low, entry_high = round(close), round(max(close, ema20))
            else:
                entry_low, entry_high = round(min(close, ema20)), round(close)
        else:
            entry_low = entry_high = round(close)

        entry_ref = (entry_low + entry_high) / 2
        if pd.notna(atr) and atr > 0:
            stop_distance = self.ATR_STOP_MULTIPLIER * atr
        else:
            stop_distance = entry_ref * self.MAX_STOP_DISTANCE_PCT
        stop_distance = min(stop_distance, entry_ref * self.MAX_STOP_DISTANCE_PCT)

        recent_level = latest.get("RECENT_HIGH_20" if is_short else "RECENT_LOW_20")
        if is_short:
            atr_buffer = 0.5 * atr if pd.notna(atr) and atr > 0 else 0
            stop_candidate = (
                max(recent_level + atr_buffer, entry_ref + atr_buffer)
                if pd.notna(recent_level)
                else entry_ref + stop_distance
            )
            stop_loss = round(
                min(stop_candidate, entry_ref * (1 + self.MAX_STOP_DISTANCE_PCT))
            )
            risk_per_share = max(stop_loss - entry_ref, 1)
            support = latest.get("RECENT_LOW_20")
            if pd.notna(support) and support < entry_ref:
                target = support
            elif pd.notna(atr) and atr > 0:
                target = entry_ref - 2 * atr
            else:
                target = entry_ref * 0.96
            target = round(max(1, target))
            rr = round((entry_ref - target) / risk_per_share, 2)
        else:
            atr_buffer = 0.5 * atr if pd.notna(atr) and atr > 0 else 0
            stop_candidate = (
                min(recent_level - atr_buffer, entry_ref - atr_buffer)
                if pd.notna(recent_level)
                else entry_ref - stop_distance
            )
            stop_loss = round(
                max(stop_candidate, entry_ref * (1 - self.MAX_STOP_DISTANCE_PCT))
            )
            risk_per_share = max(entry_ref - stop_loss, 1)
            resistance = latest.get("RECENT_HIGH_20")
            if pd.notna(resistance) and resistance > entry_ref:
                target = resistance
            elif pd.notna(atr) and atr > 0:
                target = entry_ref + 2 * atr
            else:
                target = entry_ref * 1.04
            target = round(target)
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
        as_of_date: Optional[date] = None,
        as_of_datetime: Optional[datetime] = None,
    ) -> Recommendation:
        if df.empty or len(df) < MIN_HISTORY_DAYS:
            raise ValueError(
                f"Not enough price history (need {MIN_HISTORY_DAYS}+ trading days) / "
                f"داده تاریخی کافی نیست (حداقل {MIN_HISTORY_DAYS} روز لازم است)"
            )

        market_latest = df.iloc[-1]
        partial_session = self._is_partial_session(market_latest, as_of_datetime)
        scoring_df = df.iloc[:-1] if partial_session else df
        if len(scoring_df) < MIN_HISTORY_DAYS:
            raise ValueError(
                f"Not enough complete price history (need {MIN_HISTORY_DAYS}+ "
                "trading days)"
            )
        latest = scoring_df.iloc[-1]
        name = symbol or "This stock"

        factors = [method(scoring_df, latest) for _, method in self.FACTOR_METHODS]

        raw_composite = (
            sum(f.weighted_contribution for f in factors) / self.weights.total()
        )
        raw_composite = max(-1.0, min(1.0, raw_composite))

        composite = raw_composite
        signal_strength = round(
            abs(composite)
            * 100
            * self._freshness_multiplier(
                latest,
                as_of_date
                or (as_of_datetime.date() if as_of_datetime is not None else None),
            ),
            1,
        )
        verdict = self._classify(composite)

        current_price = market_latest.get("last_price")
        if pd.isna(current_price):
            current_price = market_latest["close_price"]
        level_inputs = latest.copy()
        level_inputs["close_price"] = current_price
        levels = self._compute_entry_stop_target(
            level_inputs, is_short=verdict in {"SELL", "STRONG_SELL"}
        )
        is_short = verdict in {"SELL", "STRONG_SELL"}
        entry_ref = (levels["entry_low"] + levels["entry_high"]) / 2
        direction_ok = (
            levels["target"] < entry_ref if is_short else levels["target"] > entry_ref
        )
        setup_valid = (
            verdict != "NEUTRAL"
            and direction_ok
            and levels["risk_reward_ratio"] >= MIN_RISK_REWARD
        )
        if not setup_valid:
            levels = {key: None for key in levels}
        label = VERDICT_LABELS[verdict]

        return Recommendation(
            symbol=name,
            verdict=verdict,
            verdict_en=label["en"],
            verdict_fa=label["fa"],
            signal_strength=signal_strength,
            current_price=round(current_price),
            entry_low=levels["entry_low"],
            entry_high=levels["entry_high"],
            stop_loss=levels["stop_loss"],
            target=levels["target"],
            risk_reward_ratio=levels["risk_reward_ratio"],
            setup_valid=setup_valid,
        )
