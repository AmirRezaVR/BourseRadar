"""
Multi-factor signal scoring engine (v2).

Five independent, weighted DIRECTIONAL factors combine into a composite
score in [-1, 1]:
    RSI, MACD, Moving-Average trend alignment, Volume confirmation,
    Support/Resistance position.

Volatility is deliberately NOT a sixth directional factor. It doesn't tell
you whether to lean bullish or bearish - it tells you how much to trust
whatever direction the other factors point to. So it's applied as a
confidence MULTIPLIER on the composite score instead: high volatility
shrinks confidence toward neutral without ever flipping direction. This is
a design correction from an earlier version, where volatility was folded
into the weighted sum as a bipolar score and it turned out to add a small
persistent negative bias on undirected data - technically correct in
isolation, but not the right shape for what volatility actually represents.

Every factor's scoring buckets are DELIBERATELY SYMMETRIC around neutral
(score(mirror_input) == -score(input)), verified with a dedicated bias
test (see test suite) rather than assumed. This matters: an earlier
version of this engine had ~3x more bullish-scoring states than bearish
ones in one factor, which silently biased every recommendation toward
"Buy" regardless of the actual data.
"""

import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from src.order_book import OrderBookAnalyzer, OrderBookSnapshot


@dataclass
class SignalWeights:
    """Weights for the six directional factors. Must sum to 100."""
    rsi: float = 18
    macd: float = 22
    ma_trend: float = 22
    volume: float = 12
    support_resistance: float = 12
    order_book: float = 14

    def total(self) -> float:
        return self.rsi + self.macd + self.ma_trend + self.volume + self.support_resistance + self.order_book

    def __post_init__(self):
        if abs(self.total() - 100) > 0.01:
            raise ValueError(f"SignalWeights must sum to 100, got {self.total()}")


@dataclass
class SignalThresholds:
    strong: float = 0.60    # |composite| >= this -> Strong Buy / Strong Sell
    moderate: float = 0.25  # |composite| >= this -> Buy / Sell


@dataclass
class VolatilityConfig:
    expansion_ratio: float = 1.5     # ATR% > 1.5x its own 60-day average -> volatility expansion
    expansion_multiplier: float = 0.7  # confidence is scaled by this much when expanding
    # Contraction (calm markets) deliberately does NOT boost confidence above 1.0 -
    # a quiet market isn't evidence the signal is MORE trustworthy, just that it's quiet.


VERDICT_LABELS = {
    "STRONG_BUY": {"en": "Strong Buy", "fa": "خرید قوی"},
    "BUY": {"en": "Buy", "fa": "خرید"},
    "NEUTRAL": {"en": "Neutral / Hold", "fa": "خنثی / نگهداری"},
    "SELL": {"en": "Sell", "fa": "فروش / اجتناب از خرید"},
    "STRONG_SELL": {"en": "Strong Sell", "fa": "فروش قوی / اجتناب جدی از خرید"},
}
VERDICT_ICONS = {"STRONG_BUY": "🟢", "BUY": "🟢", "NEUTRAL": "🟡", "SELL": "🔴", "STRONG_SELL": "🔴"}
MIN_HISTORY_DAYS = 60


@dataclass
class FactorResult:
    name: str
    label_en: str
    label_fa: str
    score: float   # -1 (bearish) .. +1 (bullish)
    weight: float
    reason_en: str
    reason_fa: str
    available: bool = True  # False = excluded from composite (not just zeroed)

    @property
    def weighted_contribution(self) -> float:
        return self.score * self.weight


@dataclass
class Recommendation:
    symbol: str
    generated_at: str
    verdict: str
    verdict_en: str
    verdict_fa: str
    confidence: float          # 0-100, AFTER the volatility multiplier is applied
    composite_score: float     # -1..1, final (post-multiplier) value used for classification
    raw_composite_score: float  # -1..1, BEFORE the volatility multiplier - shown for transparency

    current_price: float
    entry_low: float
    entry_high: float
    stop_loss: float
    target: float
    risk_reward_ratio: float

    factors: List[FactorResult] = field(default_factory=list)
    volatility_multiplier: float = 1.0
    volatility_reason_en: str = ""
    volatility_reason_fa: str = ""

    confirmations_en: List[str] = field(default_factory=list)
    confirmations_fa: List[str] = field(default_factory=list)
    risks_en: List[str] = field(default_factory=list)
    risks_fa: List[str] = field(default_factory=list)

    summary_en: str = ""
    summary_fa: str = ""


class SignalEngine:
    """
    NOT financial advice. A deterministic, rule-based reading of price,
    volume, and momentum only - no fundamentals, no news, no market-wide
    context. TSE retail accounts generally can't short-sell, so Sell/Strong
    Sell should be read as "avoid new entries / consider exiting an
    existing position", not a short-entry instruction.
    """

    ATR_STOP_MULTIPLIER = 1.5
    MAX_STOP_DISTANCE_PCT = 0.08
    RISK_REWARD_RATIO = 2.0
    ENTRY_EXTENSION_THRESHOLD = 0.03

    def __init__(self, weights: Optional[SignalWeights] = None,
                 thresholds: Optional[SignalThresholds] = None,
                 volatility_cfg: Optional[VolatilityConfig] = None):
        self.weights = weights or SignalWeights()
        self.thresholds = thresholds or SignalThresholds()
        self.volatility_cfg = volatility_cfg or VolatilityConfig()
        self.order_book_analyzer = OrderBookAnalyzer()

        # Registry pattern: adding a factor later means adding one method
        # here and one weight field above - nothing else changes.
        self.FACTOR_METHODS = [
            ("rsi", self._score_rsi),
            ("macd", self._score_macd),
            ("ma_trend", self._score_ma_trend),
            ("volume", self._score_volume),
            ("support_resistance", self._score_support_resistance),
        ]

    # ------------------------------------------------------------------
    # Directional factors - every bucket set is symmetric by construction.
    # ------------------------------------------------------------------

    def _score_rsi(self, df: pd.DataFrame, latest: pd.Series) -> FactorResult:
        rsi = latest.get("RSI_14")
        weight = self.weights.rsi
        if pd.isna(rsi):
            return FactorResult("rsi", "RSI", "RSI", 0.0, weight, "RSI unavailable.", "RSI در دسترس نیست.")

        if rsi > 80:
            score, en, fa = -0.3, f"RSI ({rsi:.1f}) extremely overbought - stretched, reversal risk rising.", f"RSI ({rsi:.1f}) اشباع خرید شدید — ریسک بازگشت بالا."
        elif rsi > 70:
            score, en, fa = 0.3, f"RSI ({rsi:.1f}) overbought but still reflects real bullish momentum.", f"RSI ({rsi:.1f}) اشباع خرید اما همچنان نشان‌دهنده مومنتوم صعودی واقعی."
        elif rsi >= 55:
            score, en, fa = 0.6, f"RSI ({rsi:.1f}) shows healthy bullish momentum.", f"RSI ({rsi:.1f}) نشان‌دهنده مومنتوم صعودی سالم."
        elif rsi > 45:
            score, en, fa = 0.0, f"RSI ({rsi:.1f}) is neutral.", f"RSI ({rsi:.1f}) خنثی است."
        elif rsi >= 30:
            score, en, fa = -0.6, f"RSI ({rsi:.1f}) shows weakening, bearish momentum.", f"RSI ({rsi:.1f}) نشان‌دهنده مومنتوم نزولی."
        elif rsi >= 20:
            score, en, fa = -0.3, f"RSI ({rsi:.1f}) oversold but still reflects real bearish momentum.", f"RSI ({rsi:.1f}) اشباع فروش اما همچنان نزولی."
        else:
            score, en, fa = 0.3, f"RSI ({rsi:.1f}) extremely oversold - stretched, bounce risk rising.", f"RSI ({rsi:.1f}) اشباع فروش شدید — ریسک بازگشت رو به بالا."

        return FactorResult("rsi", "RSI (Overbought/Oversold)", "RSI (اشباع خرید/فروش)", score, weight, en, fa)

    def _score_macd(self, df: pd.DataFrame, latest: pd.Series) -> FactorResult:
        weight = self.weights.macd
        macd, signal = latest.get("MACD"), latest.get("MACD_Signal")
        if pd.isna(macd) or pd.isna(signal) or len(df) < 4:
            return FactorResult("macd", "MACD", "مکدی", 0.0, weight, "MACD unavailable.", "مکدی در دسترس نیست.")

        hist_recent = df["MACD_Hist"].tail(3)
        hist_up = hist_recent.diff().mean() > 0 if len(hist_recent) >= 2 else False
        bullish_cross = macd > signal

        if bullish_cross and hist_up:
            score, en, fa = 1.0, "MACD above signal, histogram expanding - strong bullish momentum.", "مکدی بالای خط سیگنال، هیستوگرام در حال رشد — مومنتوم صعودی قوی."
        elif bullish_cross and not hist_up:
            score, en, fa = 0.5, "MACD above signal, but momentum leveling off.", "مکدی بالای خط سیگنال، اما شتاب کاهش می‌یابد."
        elif not bullish_cross and hist_up:
            score, en, fa = -0.5, "MACD below signal but the gap is narrowing - weakening bearish momentum.", "مکدی زیر خط سیگنال اما فاصله کم می‌شود — مومنتوم نزولی تضعیف می‌شود."
        else:
            score, en, fa = -1.0, "MACD below signal, histogram falling - strong bearish momentum.", "مکدی زیر خط سیگنال، هیستوگرام در حال افت — مومنتوم نزولی قوی."

        return FactorResult("macd", "MACD (Trend & Crossover)", "مکدی (روند و تقاطع)", score, weight, en, fa)

    def _score_ma_trend(self, df: pd.DataFrame, latest: pd.Series) -> FactorResult:
        weight = self.weights.ma_trend
        close, ema20, ema50 = latest.get("close_price"), latest.get("EMA_20"), latest.get("EMA_50")
        if pd.isna(ema20) or pd.isna(ema50):
            return FactorResult("ma_trend", "Moving Average Alignment", "همراستایی میانگین‌های متحرک", 0.0, weight,
                                 "Moving averages unavailable.", "میانگین‌های متحرک در دسترس نیستند.")

        # Additive model: two independent +/-0.5 sub-signals. This is symmetric
        # by construction, unlike an earlier ad-hoc 4-case bucket that gave
        # 3 of 4 states a positive score.
        short_term = 0.5 if close > ema20 else -0.5
        medium_term = 0.5 if ema20 > ema50 else -0.5
        score = short_term + medium_term

        if score == 1.0:
            en, fa = "Full bullish alignment: price > EMA20 > EMA50.", "همراستایی کامل صعودی: قیمت > EMA20 > EMA50."
        elif score == -1.0:
            en, fa = "Full bearish alignment: price < EMA20 < EMA50.", "همراستایی کامل نزولی: قیمت < EMA20 < EMA50."
        elif short_term > 0:
            en, fa = "Mixed: price reclaimed EMA20, but EMA50 (medium-term) hasn't confirmed yet.", "ترکیبی: قیمت بالای EMA20، اما EMA50 هنوز تأیید نکرده."
        else:
            en, fa = "Mixed: price below EMA20 despite EMA20 still above EMA50 - pullback in a fading uptrend.", "ترکیبی: قیمت زیر EMA20 با وجود EMA20 بالای EMA50 — اصلاح در روند صعودی رو به تضعیف."

        return FactorResult("ma_trend", "Moving Average Alignment", "همراستایی میانگین‌های متحرک", score, weight, en, fa)

    def _score_volume(self, df: pd.DataFrame, latest: pd.Series) -> FactorResult:
        weight = self.weights.volume
        vol = latest.get("volume")
        avg_vol = df["volume"].tail(20).mean() if len(df) >= 20 else None

        if avg_vol is None or pd.isna(avg_vol) or avg_vol == 0:
            return FactorResult("volume", "Volume Confirmation", "تأیید حجم معاملات", 0.0, weight,
                                 "Volume average unavailable.", "میانگین حجم در دسترس نیست.")

        ratio = vol / avg_vol
        price_change = latest.get("close_price") - df["close_price"].iloc[-2] if len(df) >= 2 else 0

        if ratio >= 1.5 and price_change > 0:
            score, en, fa = 1.0, f"Volume {ratio:.1f}x average on an up day - strong buying confirmation.", f"حجم {ratio:.1f} برابر میانگین در روز مثبت — تأیید قوی تقاضا."
        elif ratio >= 1.5 and price_change < 0:
            score, en, fa = -1.0, f"Volume {ratio:.1f}x average on a down day - strong selling confirmation.", f"حجم {ratio:.1f} برابر میانگین در روز منفی — تأیید قوی عرضه."
        elif ratio < 0.8:
            score, en, fa = 0.0, f"Volume only {ratio:.1f}x average - low participation, no confirmation either way.", f"حجم تنها {ratio:.1f} برابر میانگین — مشارکت پایین، بدون تأیید."
        else:
            direction = "up" if price_change > 0 else ("down" if price_change < 0 else "flat")
            score = 0.3 if price_change > 0 else (-0.3 if price_change < 0 else 0.0)
            dir_fa = "مثبت" if direction == "up" else ("منفی" if direction == "down" else "خنثی")
            en, fa = f"Volume near average ({ratio:.1f}x) on a {direction} day - mild, unconfirmed signal.", f"حجم نزدیک به میانگین ({ratio:.1f} برابر) در روز {dir_fa} — سیگنال ضعیف."

        return FactorResult("volume", "Volume Confirmation", "تأیید حجم معاملات", score, weight, en, fa)

    def _score_support_resistance(self, df: pd.DataFrame, latest: pd.Series) -> FactorResult:
        weight = self.weights.support_resistance
        close = latest.get("close_price")
        recent_low = df["low_price"].tail(20).min() if len(df) >= 20 else None
        recent_high = df["high_price"].tail(20).max() if len(df) >= 20 else None

        if recent_low is None or recent_high is None or recent_high == recent_low:
            return FactorResult("support_resistance", "Support/Resistance", "حمایت/مقاومت", 0.0, weight,
                                 "Support/resistance unavailable.", "حمایت/مقاومت در دسترس نیست.")

        position = (close - recent_low) / (recent_high - recent_low)

        if position <= 0.15:
            score, en, fa = 0.5, "Price near its 20-day support - potential bounce zone.", "قیمت نزدیک حمایت ۲۰ روزه — منطقه احتمالی بازگشت."
        elif position >= 0.85:
            score, en, fa = -0.5, "Price near its 20-day resistance - risk of rejection.", "قیمت نزدیک مقاومت ۲۰ روزه — ریسک برگشت."
        elif position <= 0.35:
            score, en, fa = 0.2, "Price in the lower part of its recent range.", "قیمت در بخش پایینی محدوده اخیر."
        elif position >= 0.65:
            score, en, fa = -0.2, "Price in the upper part of its recent range, approaching resistance.", "قیمت در بخش بالایی محدوده اخیر، نزدیک مقاومت."
        else:
            score, en, fa = 0.0, "Price in the middle of its recent trading range.", "قیمت در میانه محدوده اخیر."

        return FactorResult("support_resistance", "Support/Resistance Position", "موقعیت نسبت به حمایت/مقاومت", score, weight, en, fa)

    # ------------------------------------------------------------------
    # Volatility: confidence modifier, not a directional vote.
    # ------------------------------------------------------------------

    def _volatility_multiplier(self, df: pd.DataFrame, latest: pd.Series) -> tuple:
        atr = latest.get("ATR_14")
        close = latest.get("close_price")
        if pd.isna(atr) or not close or len(df) < 60:
            return 1.0, "Volatility baseline unavailable.", "میانگین نوسان تاریخی در دسترس نیست."

        atr_pct_series = df["ATR_14"] / df["close_price"]
        atr_pct = atr / close
        atr_pct_avg = atr_pct_series.tail(60).mean()

        if pd.isna(atr_pct_avg) or atr_pct_avg == 0:
            return 1.0, "Volatility baseline unavailable.", "میانگین نوسان تاریخی در دسترس نیست."

        ratio = atr_pct / atr_pct_avg
        if ratio >= self.volatility_cfg.expansion_ratio:
            m = self.volatility_cfg.expansion_multiplier
            en = f"Volatility is expanding ({ratio:.1f}x normal) - confidence reduced to reflect higher uncertainty."
            fa = f"نوسانات در حال گسترش است ({ratio:.1f} برابر عادی) — اطمینان برای انعکاس عدم قطعیت بیشتر کاهش یافت."
            return m, en, fa

        return 1.0, "Volatility is within its normal historical range.", "نوسانات در محدوده عادی تاریخی است."

    # ------------------------------------------------------------------

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

        return dict(entry_low=entry_low, entry_high=entry_high, stop_loss=stop_loss, target=target, risk_reward_ratio=rr)

    def generate(self, df: pd.DataFrame, symbol: Optional[str] = None,
                 order_book_raw: Optional[dict] = None) -> Recommendation:
        if df.empty or len(df) < MIN_HISTORY_DAYS:
            raise ValueError(
                f"Not enough price history (need {MIN_HISTORY_DAYS}+ trading days). / "
                f"داده تاریخی کافی نیست (حداقل {MIN_HISTORY_DAYS} روز لازم است)."
            )

        latest = df.iloc[-1]
        name, name_fa = symbol or "This stock", symbol or "این سهم"

        factors = [method(df, latest) for _, method in self.FACTOR_METHODS]

        order_book_snapshot = None
        if order_book_raw:
            order_book_snapshot = self.order_book_analyzer.parse(name, order_book_raw)
        ob_reading = self.order_book_analyzer.analyze(order_book_snapshot, weight=self.weights.order_book)
        factors.append(FactorResult(
            "order_book", "Order Book (Bid/Ask)", "تابلو (عرضه و تقاضا)",
            ob_reading.score, ob_reading.weight, ob_reading.reason_en, ob_reading.reason_fa,
            available=ob_reading.available,
        ))

        # Composite is a weighted average over only the AVAILABLE factors -
        # an unavailable factor (e.g. no order book snapshot) is excluded
        # from both numerator and denominator, not silently zeroed with
        # full weight still counted (which would just dilute confidence
        # for a reason that has nothing to do with the actual data).
        available_factors = [f for f in factors if f.available]
        available_weight = sum(f.weight for f in available_factors)
        raw_composite = (sum(f.weighted_contribution for f in available_factors) / available_weight
                          if available_weight > 0 else 0.0)
        raw_composite = max(-1.0, min(1.0, raw_composite))

        vol_multiplier, vol_reason_en, vol_reason_fa = self._volatility_multiplier(df, latest)
        composite = max(-1.0, min(1.0, raw_composite * vol_multiplier))
        confidence = round(abs(composite) * 100, 1)
        verdict = self._classify(composite)

        levels = self._compute_entry_stop_target(latest)

        direction_sign = 1 if composite > 0 else (-1 if composite < 0 else 0)
        confirmations_en, confirmations_fa, risks_en, risks_fa = [], [], [], []
        for f in factors:
            if f.score == 0:
                continue
            agrees = (f.score > 0 and direction_sign >= 0) or (f.score < 0 and direction_sign <= 0)
            (confirmations_en if agrees else risks_en).append(f.reason_en)
            (confirmations_fa if agrees else risks_fa).append(f.reason_fa)

        if vol_multiplier < 1.0:
            risks_en.append(vol_reason_en)
            risks_fa.append(vol_reason_fa)

        risks_en.append("Rule-based technical signal only - not financial advice. Data may be delayed.")
        risks_fa.append("این یک سیگنال خودکار مبتنی بر تحلیل تکنیکال است، نه توصیه مالی. داده‌ها ممکن است با تأخیر باشند.")

        label = VERDICT_LABELS[verdict]
        summary_en, summary_fa = self._build_summary(name, name_fa, verdict, label, confidence, composite, levels)

        return Recommendation(
            symbol=name, generated_at=datetime.now().isoformat(timespec="seconds"),
            verdict=verdict, verdict_en=label["en"], verdict_fa=label["fa"],
            confidence=confidence, composite_score=round(composite, 3), raw_composite_score=round(raw_composite, 3),
            current_price=round(latest["close_price"]),
            entry_low=levels["entry_low"], entry_high=levels["entry_high"], stop_loss=levels["stop_loss"],
            target=levels["target"], risk_reward_ratio=levels["risk_reward_ratio"],
            factors=factors, volatility_multiplier=vol_multiplier,
            volatility_reason_en=vol_reason_en, volatility_reason_fa=vol_reason_fa,
            confirmations_en=confirmations_en, confirmations_fa=confirmations_fa,
            risks_en=risks_en, risks_fa=risks_fa, summary_en=summary_en, summary_fa=summary_fa,
        )

    def _build_summary(self, name, name_fa, verdict, label, confidence, composite, levels):
        summary_en = (f"{name}: composite score {composite:+.2f} (confidence {confidence:.0f}%), "
                      f"a {label['en']} reading. ")
        summary_fa = (f"{name_fa}: امتیاز ترکیبی {composite:+.2f} (اطمینان {confidence:.0f}٪)، "
                       f"نتیجه «{label['fa']}». ")

        if verdict in ("BUY", "STRONG_BUY"):
            summary_en += (f"Suggested entry {levels['entry_low']:,.0f}-{levels['entry_high']:,.0f} Rial, "
                            f"stop-loss {levels['stop_loss']:,.0f}, target {levels['target']:,.0f} "
                            f"(risk:reward 1:{levels['risk_reward_ratio']}).")
            summary_fa += (f"محدوده ورود پیشنهادی {levels['entry_low']:,.0f} تا {levels['entry_high']:,.0f} ریال، "
                            f"حد ضرر {levels['stop_loss']:,.0f}، هدف {levels['target']:,.0f} ریال "
                            f"(ریسک به بازده ۱ به {levels['risk_reward_ratio']}).")
        elif verdict in ("SELL", "STRONG_SELL"):
            summary_en += ("Conditions don't favor a new long entry. If already holding, the technical "
                            "picture suggests caution. (TSE retail accounts generally can't short-sell.)")
            summary_fa += ("شرایط فعلی از ورود خرید جدید حمایت نمی‌کند. در صورت داشتن این سهم، احتیاط بیشتری لازم است. "
                            "(در بورس ایران معمولاً امکان فروش استقراضی برای خرد وجود ندارد.)")
        else:
            summary_en += "Signals are mixed - no clear edge in either direction right now."
            summary_fa += "سیگنال‌ها ترکیبی هستند — در حال حاضر مزیت واضحی در هیچ جهتی وجود ندارد."

        return summary_en, summary_fa
