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
    # Added factors. Weights are not yet validated against history (see backtest step).
    trend_strength: float = 8
    bollinger: float = 5
    money_flow: float = 8
    vwap: float = 4
    weekly_trend: float = 5

    def total(self) -> float:
        return (
            self.rsi
            + self.macd
            + self.ma_trend
            + self.volume
            + self.support_resistance
            + self.trend_strength
            + self.bollinger
            + self.money_flow
            + self.vwap
            + self.weekly_trend
        )

    def __post_init__(self):
        if self.total() <= 0:
            raise ValueError(f"SignalWeights must have a positive total, got {self.total()}")


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
            ("trend_strength", self._score_trend_strength),
            ("bollinger", self._score_bollinger),
            ("money_flow", self._score_money_flow),
            ("vwap", self._score_vwap),
            ("weekly_trend", self._score_weekly_trend),
        ]

    def _score_rsi(self, df: pd.DataFrame, latest: pd.Series) -> FactorResult:
        """Momentum score: above 50 is bullish, below 50 bearish, full score at 70 or 30.

        RSI follows momentum here. It does not call a reversal at overbought or
        oversold levels, and it does not add divergence.
        """
        weight = self.weights.rsi
        rsi = self._num(latest.get("RSI_14"))
        if rsi is None:
            return FactorResult("rsi", 0.0, weight)
        return FactorResult("rsi", self._clip((rsi - 50) / 20), weight)

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
        if pd.isna(close) or pd.isna(ema20) or pd.isna(ema50):
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
        # Median, not mean: one past spike should not raise the baseline for weeks.
        avg_vol = df["volume"].iloc[-21:-1].median() if len(df) >= 21 else None

        if avg_vol is None or pd.isna(avg_vol) or avg_vol == 0:
            return FactorResult("volume", 0.0, weight)
        if vol is None or pd.isna(vol):
            return FactorResult("volume", 0.0, weight)

        ratio = vol / avg_vol
        price_change = (
            latest.get("close_price") - df["close_price"].iloc[-2]
            if len(df) >= 2
            else 0
        )
        if pd.isna(price_change):
            price_change = 0

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

    @staticmethod
    def _num(value) -> Optional[float]:
        """Return a finite float, or None for missing or non-finite values."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    @staticmethod
    def _clip(value: float) -> float:
        return max(-1.0, min(1.0, value))

    def _score_trend_strength(self, df: pd.DataFrame, latest: pd.Series) -> FactorResult:
        """ADX gates the direction from +DI/-DI: no directional view in a range (ADX < 20)."""
        weight = self.weights.trend_strength
        adx = self._num(latest.get("ADX_14"))
        plus_di = self._num(latest.get("PLUS_DI_14"))
        minus_di = self._num(latest.get("MINUS_DI_14"))
        if adx is None or plus_di is None or minus_di is None or adx < 20:
            return FactorResult("trend_strength", 0.0, weight)

        direction = 1.0 if plus_di > minus_di else (-1.0 if minus_di > plus_di else 0.0)
        strength = min(1.0, (adx - 20) / 20)
        return FactorResult("trend_strength", direction * strength, weight)

    def _score_bollinger(self, df: pd.DataFrame, latest: pd.Series) -> FactorResult:
        """%B on the upper or lower half; neutral while bands are squeezed (no direction yet)."""
        weight = self.weights.bollinger
        pct_b = self._num(latest.get("BB_PCT_B_20"))
        if pct_b is None or bool(latest.get("BB_SQUEEZE_20", False)):
            return FactorResult("bollinger", 0.0, weight)
        return FactorResult("bollinger", self._clip(2 * (pct_b - 0.5)), weight)

    def _score_money_flow(self, df: pd.DataFrame, latest: pd.Series) -> FactorResult:
        """Average of CMF, MFI, OBV flow and trade-size direction (available parts only)."""
        weight = self.weights.money_flow
        components = []

        cmf = self._num(latest.get("CMF_20"))
        if cmf is not None:
            components.append(self._clip(cmf / 0.25))

        mfi = self._num(latest.get("MFI_14"))
        if mfi is not None:
            components.append(self._clip((mfi - 50) / 30))

        obv_flow = self._num(latest.get("OBV_FLOW_20"))
        if obv_flow is not None:
            components.append(self._clip(obv_flow))

        ratio = self._num(latest.get("TRADE_SIZE_RATIO_20"))
        if ratio is not None and len(df) >= 2:
            change = self._num(latest.get("close_price") - df["close_price"].iloc[-2])
            if change is not None and change != 0:
                sign = 1.0 if change > 0 else -1.0
                components.append(sign * min(1.0, max(0.0, (ratio - 1) / 0.5)))

        if not components:
            return FactorResult("money_flow", 0.0, weight)
        return FactorResult("money_flow", sum(components) / len(components), weight)

    def _score_vwap(self, df: pd.DataFrame, latest: pd.Series) -> FactorResult:
        """Price relative to the 20-day VWAP, scaled by ATR."""
        weight = self.weights.vwap
        close = self._num(latest.get("close_price"))
        vwap = self._num(latest.get("VWAP_20"))
        atr = self._num(latest.get("ATR_14"))
        if close is None or vwap is None or atr is None or atr <= 0:
            return FactorResult("vwap", 0.0, weight)
        return FactorResult("vwap", self._clip((close - vwap) / (2 * atr)), weight)

    def _score_weekly_trend(self, df: pd.DataFrame, latest: pd.Series) -> FactorResult:
        """Last completed week: above (+1) or below (-1) its weekly EMA."""
        weight = self.weights.weekly_trend
        trend = self._num(latest.get("WEEKLY_TREND_10"))
        return FactorResult("weekly_trend", self._clip(trend) if trend else 0.0, weight)

    def composite_from_scores(self, scores: dict, excluded=frozenset(), scales=None) -> float:
        """Weighted composite of factor scores. The one formula used by generate() and the backtest.

        sum(score * weight * scale) / sum(weight * scale), clipped to [-1, 1].
        scales optionally multiplies individual weights (for example the regime scaling).
        """
        scales = scales or {}
        weighted, total = 0.0, 0.0
        for name, score in scores.items():
            if name in excluded:
                continue
            weight = getattr(self.weights, name) * scales.get(name, 1.0)
            weighted += score * weight
            total += weight
        if total <= 0:
            return 0.0
        return self._clip(weighted / total)

    TREND_FOLLOWING = ("ma_trend", "macd", "weekly_trend")

    def _regime_scales(self, latest: pd.Series) -> dict:
        """Halve trend-following weights when ADX shows a range (below 20)."""
        adx = self._num(latest.get("ADX_14"))
        if adx is not None and adx < 20:
            return {name: 0.5 for name in self.TREND_FOLLOWING}
        return {}

    @staticmethod
    def _agreement(factors, effective: dict, composite: float) -> float:
        """Share of the active factor weight that points the same way as the composite.

        This is what the README calls agreement: magnitude alone can be high even when
        many active factors disagree. Returns 0 when there is no direction or no
        active factor.
        """
        if composite == 0:
            return 0.0
        direction = 1.0 if composite > 0 else -1.0
        active = [f for f in factors if abs(f.score) > 1e-9]
        total = sum(effective[f.name] for f in active)
        if total <= 0:
            return 0.0
        aligned = sum(
            effective[f.name] for f in active if f.score * direction > 0
        )
        return aligned / total

    @staticmethod
    def _traded_ratio(df: pd.DataFrame) -> float:
        """Share of the last 20 sessions with any trade. Thin trading lowers strength."""
        if "volume" not in df.columns:
            return 1.0
        recent = pd.to_numeric(df["volume"].tail(20), errors="coerce")
        if recent.empty or recent.isna().all():
            return 1.0
        return float((recent.fillna(0) > 0).mean())

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
        latest: pd.Series,
        as_of_datetime: Optional[datetime] = None,
        session_close: datetime_time = MARKET_CLOSE_TIME,
    ) -> bool:
        date_value = latest.get("date")
        time_value = latest.get("last_update_time")
        try:
            date_text = str(date_value)
            if len(date_text) == 8 and date_text.isdigit():
                market_date = datetime.strptime(date_text, "%Y%m%d").date()
            else:
                market_date = datetime.strptime(date_text[:10], "%Y-%m-%d").date()
        except (TypeError, ValueError):
            return False

        try:
            market_time = datetime.strptime(str(time_value)[:5], "%H:%M").time()
        except (TypeError, ValueError):
            market_time = None  # update time missing on the row

        reference_datetime = as_of_datetime or datetime.now(TEHRAN_TZ).replace(
            tzinfo=None
        )
        if (
            market_date != reference_datetime.date()
            or reference_datetime.time() >= session_close
        ):
            return False
        if market_time is None:
            # Today's row before the close with no timestamp is still an open session.
            return True
        return market_time < session_close and market_time <= reference_datetime.time()

    @staticmethod
    def _session_close_time(history: pd.DataFrame) -> datetime_time:
        """Close time of this symbol's session, from its recent update times.

        Uses the latest update time among the last 10 completed sessions, and falls
        back to the standard close when no usable time is available.
        """
        if history.empty or "last_update_time" not in history.columns:
            return MARKET_CLOSE_TIME
        times = []
        for value in history["last_update_time"].tail(10):
            text = str(value)
            for fmt, width in (("%H:%M:%S", 8), ("%H:%M", 5)):
                try:
                    times.append(datetime.strptime(text[:width], fmt).time())
                    break
                except ValueError:
                    continue
        return max(times) if times else MARKET_CLOSE_TIME

    def _compute_entry_stop_target(self, latest: pd.Series) -> dict:
        """Long-side entry zone, stop loss, target and risk/reward for one day."""
        close, ema20 = latest.get("close_price"), latest.get("EMA_20")
        atr = latest.get("ATR_14")

        if pd.notna(ema20) and ema20:
            extension = (close - ema20) / ema20
            if extension > self.ENTRY_EXTENSION_THRESHOLD:
                entry_low, entry_high = round(ema20), round(ema20 * 1.02)
            else:
                # The zone spans from the lower to the higher of close and EMA20.
                entry_low = round(min(close, ema20))
                entry_high = round(max(close, ema20))
        else:
            entry_low = entry_high = round(close)

        entry_ref = (entry_low + entry_high) / 2
        if pd.notna(atr) and atr > 0:
            stop_distance = self.ATR_STOP_MULTIPLIER * atr
        else:
            stop_distance = entry_ref * self.MAX_STOP_DISTANCE_PCT
        stop_distance = min(stop_distance, entry_ref * self.MAX_STOP_DISTANCE_PCT)

        recent_low = latest.get("RECENT_LOW_20")
        atr_buffer = 0.5 * atr if pd.notna(atr) and atr > 0 else 0
        stop_candidate = (
            min(recent_low - atr_buffer, entry_ref - atr_buffer)
            if pd.notna(recent_low)
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
        partial_session = self._is_partial_session(
            market_latest,
            as_of_datetime,
            self._session_close_time(df.iloc[:-1]),
        )
        scoring_df = df.iloc[:-1] if partial_session else df
        if len(scoring_df) < MIN_HISTORY_DAYS:
            raise ValueError(
                f"Not enough complete price history (need {MIN_HISTORY_DAYS}+ "
                "trading days)"
            )
        latest = scoring_df.iloc[-1]
        name = symbol or "This stock"

        factors = [method(scoring_df, latest) for _, method in self.FACTOR_METHODS]

        # Regime: in a range (ADX below 20) trend-following factors are less reliable,
        # so their weight is halved for that day.
        scales = self._regime_scales(latest)
        effective = {f.name: f.weight * scales.get(f.name, 1.0) for f in factors}
        raw_composite = self.composite_from_scores(
            {f.name: f.score for f in factors}, scales=scales
        )

        composite = raw_composite
        agreement = self._agreement(factors, effective, composite)
        liquidity = self._traded_ratio(scoring_df)
        signal_strength = round(
            abs(composite)
            * 100
            * agreement
            * liquidity
            * self._freshness_multiplier(
                latest,
                as_of_date
                or (as_of_datetime.date() if as_of_datetime is not None else None),
            ),
            1,
        )
        verdict = self._classify(composite)

        current_price = market_latest.get("last_price")
        if pd.isna(current_price) or current_price <= 0:
            current_price = market_latest["close_price"]
        level_inputs = latest.copy()
        level_inputs["close_price"] = current_price
        # Long-side levels: entry zone, stop below and target above.
        # For a sell verdict the entry is hidden below and the stop and target are
        # used as exit levels by a holder.
        levels = self._compute_entry_stop_target(level_inputs)
        entry_ref = (levels["entry_low"] + levels["entry_high"]) / 2
        direction_ok = levels["target"] > entry_ref
        setup_valid = (
            verdict in {"BUY", "STRONG_BUY"}
            and direction_ok
            and levels["risk_reward_ratio"] >= MIN_RISK_REWARD
        )
        if verdict in {"SELL", "STRONG_SELL"}:
            # A sell verdict is not a buy setup: no entry zone and no risk/reward.
            # Stop loss and target stay as exit levels for someone who holds the stock.
            levels = {
                **levels,
                "entry_low": None,
                "entry_high": None,
                "risk_reward_ratio": None,
            }
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
