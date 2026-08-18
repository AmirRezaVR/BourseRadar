import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TradeSignal:
    verdict: str  # "BUY", "WATCH", or "AVOID"
    verdict_fa: str  # Persian verdict label
    score: int  # 0-100 composite technical score
    current_price: float
    entry_low: float
    entry_high: float
    stop_loss: float
    target: float
    risk_reward_ratio: float
    reasons: List[str] = field(default_factory=list)
    reasons_fa: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    warnings_fa: List[str] = field(default_factory=list)
    summary_en: str = ""
    summary_fa: str = ""


class TradeAdvisor:


    ATR_STOP_MULTIPLIER = 1.5  # moderate risk: 1.5x ATR below entry
    RISK_REWARD_RATIO = 2.0  # moderate: aim for 2x the risked amount
    MAX_STOP_DISTANCE_PCT = 0.08  # never suggest a stop wider than 8% for a swing trade
    OVERBOUGHT_RSI = 75
    OVERSOLD_RSI = 30

    def generate_signal(
        self, df: pd.DataFrame, symbol: Optional[str] = None
    ) -> TradeSignal:
        if df.empty or len(df) < 50:
            raise ValueError(
                "Not enough price history to generate a reliable signal "
                "(need at least 50 trading days for EMA50/RSI/MACD to stabilize). / "
                "داده تاریخی کافی برای تولید سیگنال قابل اعتماد وجود ندارد "
                "(حداقل ۵۰ روز کاری برای پایدار شدن EMA50، RSI و MACD لازم است)."
            )

        name = symbol if symbol else "This stock"
        name_fa = symbol if symbol else "این سهم"

        latest = df.iloc[-1]
        reasons, reasons_fa = [], []
        warnings, warnings_fa = [], []
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
        trend_bullish = pd.notna(ema20) and pd.notna(ema50) and ema20 > ema50
        if trend_bullish:
            score += 30
            reasons.append(
                "Short-term trend is above the medium-term trend (EMA20 > EMA50) — uptrend intact."
            )
            reasons_fa.append(
                "روند کوتاه‌مدت بالاتر از روند میان‌مدت است (EMA20 بالاتر از EMA50) — روند صعودی برقرار است."
            )
        else:
            reasons.append(
                "EMA20 is not above EMA50 — short-term trend is weak or bearish."
            )
            reasons_fa.append(
                "EMA20 بالاتر از EMA50 نیست — روند کوتاه‌مدت ضعیف یا نزولی است."
            )

        # --- 2. Price position vs EMA20 (max 15 pts) ---
        price_above_ema20 = pd.notna(ema20) and close > ema20
        if price_above_ema20:
            score += 15
            reasons.append("Price is trading above its 20-day EMA.")
            reasons_fa.append(
                "قیمت بالاتر از میانگین متحرک نمایی ۲۰ روزه (EMA20) قرار دارد."
            )
        else:
            reasons.append(
                "Price is below its 20-day EMA — short-term momentum is soft."
            )
            reasons_fa.append(
                "قیمت پایین‌تر از EMA20 قرار دارد — مومنتوم کوتاه‌مدت ضعیف است."
            )

        # --- 3. RSI zone (max 25 pts) ---
        rsi_zone = "unknown"
        if pd.notna(rsi):
            if 45 <= rsi <= 65:
                score += 25
                rsi_zone = "healthy"
                reasons.append(
                    f"RSI ({rsi:.1f}) is in a healthy zone for a swing entry — not extended."
                )
                reasons_fa.append(
                    f"RSI ({rsi:.1f}) در محدوده سالم برای ورود نوسانی قرار دارد — قیمت اشباع نشده است."
                )
            elif 35 <= rsi < 45 or 65 < rsi <= 70:
                score += 10
                rsi_zone = "borderline"
                reasons.append(
                    f"RSI ({rsi:.1f}) is borderline — acceptable but not ideal."
                )
                reasons_fa.append(
                    f"RSI ({rsi:.1f}) در محدوده مرزی قرار دارد — قابل قبول اما ایده‌آل نیست."
                )
            elif rsi > 70:
                rsi_zone = "overbought"
                reasons.append(
                    f"RSI ({rsi:.1f}) is elevated — stock may be short-term overbought."
                )
                reasons_fa.append(
                    f"RSI ({rsi:.1f}) بالا است — احتمال اشباع خرید کوتاه‌مدت وجود دارد."
                )
            else:
                rsi_zone = "weak"
                reasons.append(
                    f"RSI ({rsi:.1f}) is weak — momentum has not turned up yet."
                )
                reasons_fa.append(
                    f"RSI ({rsi:.1f}) پایین است — مومنتوم هنوز به سمت بالا برنگشته است."
                )

        # --- 4. MACD momentum (max 30 pts) ---
        macd_bullish = pd.notna(macd) and pd.notna(macd_signal) and macd > macd_signal
        if pd.notna(macd) and pd.notna(macd_signal):
            if macd_bullish:
                score += 30
                reasons.append("MACD is above its signal line — bullish momentum.")
                reasons_fa.append(
                    "مکدی (MACD) بالاتر از خط سیگنال است — مومنتوم صعودی."
                )
            else:
                reasons.append(
                    "MACD is below its signal line — momentum has not confirmed a turn yet."
                )
                reasons_fa.append(
                    "مکدی (MACD) پایین‌تر از خط سیگنال است — مومنتوم هنوز تغییر روند را تأیید نکرده است."
                )

        score = min(100, max(0, score))

        # --- Verdict ---
        if pd.notna(rsi) and rsi >= self.OVERBOUGHT_RSI:
            verdict = "AVOID"
            verdict_fa = "اجتناب از خرید"
            warnings.append(
                f"RSI ({rsi:.1f}) is in overbought territory (>= {self.OVERBOUGHT_RSI}). "
                "Entering now risks buying into a short-term top — wait for a pullback."
            )
            warnings_fa.append(
                f"RSI ({rsi:.1f}) در محدوده اشباع خرید قرار دارد (بالاتر از {self.OVERBOUGHT_RSI}). "
                "ورود در این نقطه ریسک خرید در سقف کوتاه‌مدت را دارد — بهتر است منتظر اصلاح قیمت بمانید."
            )
        elif score >= 70:
            verdict = "BUY"
            verdict_fa = "پیشنهاد خرید"
        elif score >= 45:
            verdict = "WATCH"
            verdict_fa = "زیر نظر (صبر کنید)"
        else:
            verdict = "AVOID"
            verdict_fa = "اجتناب از خرید"

        # --- Entry zone ---
        extended = False
        if pd.notna(ema20):
            extension_pct = (close - ema20) / ema20 if ema20 else 0
            if extension_pct > 0.03:
                extended = True
                entry_low = round(ema20)
                entry_high = round(ema20 * 1.02)
                warnings.append(
                    f"Price is {extension_pct*100:.1f}% above its EMA20 — considered extended for a "
                    "short-term entry. Consider waiting for a pullback toward the entry zone below "
                    "rather than chasing the current price."
                )
                warnings_fa.append(
                    f"قیمت {extension_pct*100:.1f}٪ بالاتر از EMA20 قرار دارد — برای یک ورود کوتاه‌مدت "
                    "کشیده (extended) محسوب می‌شود. بهتر است به‌جای خرید در قیمت فعلی، منتظر اصلاح قیمت "
                    "تا محدوده ورود پیشنهادی بمانید."
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
            warnings_fa.append(
                f"حد ضرر پیشنهادی ({stop_loss:,.0f}) پایین‌تر از کف قیمتی ۲۰ روز اخیر "
                f"({recent_low:,.0f}) قرار دارد. این عدد را با میزان ریسک‌پذیری خودتان تطبیق دهید."
            )

        # --- Target (risk:reward based, checked against resistance) ---
        risk_per_share = entry_reference - stop_loss
        target = round(entry_reference + (risk_per_share * self.RISK_REWARD_RATIO))

        if pd.notna(recent_high) and target > recent_high:
            warnings.append(
                f"Calculated target ({target:,.0f}) is above recent resistance "
                f"({recent_high:,.0f}). Price may stall or reverse near that level before reaching target."
            )
            warnings_fa.append(
                f"هدف قیمتی محاسبه‌شده ({target:,.0f}) بالاتر از مقاومت ۲۰ روز اخیر "
                f"({recent_high:,.0f}) قرار دارد. ممکن است قیمت پیش از رسیدن به هدف، در این محدوده متوقف یا برگردد."
            )

        risk_reward = round(target - entry_reference) / max(risk_per_share, 1)

        # --- Narrative summary (English) ---
        trend_txt_en = "an uptrend" if trend_bullish else "a weak or downward trend"
        rsi_txt_en = {
            "healthy": "healthy, non-overbought momentum",
            "borderline": "borderline momentum — not clearly strong or weak",
            "overbought": "overbought conditions",
            "weak": "weak, unconfirmed momentum",
            "unknown": "unclear momentum (insufficient data)",
        }.get(rsi_zone, "unclear momentum")
        macd_txt_en = (
            "confirming bullish momentum"
            if macd_bullish
            else "not yet confirming a bullish turn"
        )

        summary_en = (
            f"{name} is currently in {trend_txt_en}, "
            f"{'trading above' if price_above_ema20 else 'trading below'} its 20-day average, "
            f"with RSI showing {rsi_txt_en} and MACD {macd_txt_en}. "
        )
        if verdict == "BUY":
            summary_en += (
                f"Overall, conditions support a short-term swing entry. A suggested entry zone is "
                f"{entry_low:,.0f}–{entry_high:,.0f} Rial, with a stop-loss at {stop_loss:,.0f} Rial "
                f"to limit downside, and a target near {target:,.0f} Rial (roughly a 1:{risk_reward:.1f} "
                f"risk-to-reward ratio). This does not guarantee a profitable trade — it only means the "
                f"technical setup currently favors buyers over sellers."
            )
        elif verdict == "WATCH":
            summary_en += (
                f"The setup is mixed — some factors favor buyers, others don't yet confirm. It may be "
                f"worth watching for a clearer signal (e.g. RSI moving into a healthier zone, or MACD "
                f"crossing bullish) before committing capital. If you still want reference levels, a "
                f"possible entry zone would be {entry_low:,.0f}–{entry_high:,.0f} Rial with a stop-loss "
                f"near {stop_loss:,.0f} Rial."
            )
        else:
            summary_en += (
                "Overall, current conditions do not support a new long entry. It's generally better to "
                "wait for the trend or momentum to improve rather than entering against the prevailing signals."
            )

        # --- Narrative summary (Persian) ---
        trend_txt_fa = "روند صعودی" if trend_bullish else "روند ضعیف یا نزولی"
        rsi_txt_fa = {
            "healthy": "مومنتوم سالم و بدون اشباع خرید",
            "borderline": "مومنتوم مرزی — نه به‌وضوح قوی و نه ضعیف",
            "overbought": "شرایط اشباع خرید",
            "weak": "مومنتوم ضعیف و تأییدنشده",
            "unknown": "مومنتوم نامشخص (داده کافی نیست)",
        }.get(rsi_zone, "مومنتوم نامشخص")
        macd_txt_fa = (
            "مومنتوم صعودی را تأیید می‌کند"
            if macd_bullish
            else "هنوز تغییر روند صعودی را تأیید نکرده است"
        )

        summary_fa = (
            f"{name_fa} در حال حاضر در {trend_txt_fa} قرار دارد، "
            f"{'بالاتر از' if price_above_ema20 else 'پایین‌تر از'} میانگین ۲۰ روزه خود معامله می‌شود، "
            f"RSI نشان‌دهنده {rsi_txt_fa} است و MACD {macd_txt_fa}. "
        )
        if verdict == "BUY":
            summary_fa += (
                f"در مجموع، شرایط فعلی از یک ورود نوسانی کوتاه‌مدت حمایت می‌کند. محدوده ورود پیشنهادی "
                f"{entry_low:,.0f} تا {entry_high:,.0f} ریال است، با حد ضرر در {stop_loss:,.0f} ریال "
                f"برای محدود کردن ریسک، و هدف قیمتی نزدیک به {target:,.0f} ریال "
                f"(نسبت ریسک به بازده تقریباً ۱ به {risk_reward:.1f}). این به معنای تضمین سودآوری معامله "
                f"نیست — فقط نشان می‌دهد که شرایط تکنیکال فعلی به نفع خریداران است تا فروشندگان."
            )
        elif verdict == "WATCH":
            summary_fa += (
                f"شرایط ترکیبی و نامشخص است — برخی عوامل به نفع خریداران و برخی هنوز تأییدنشده هستند. "
                f"بهتر است منتظر سیگنال واضح‌تری بمانید (مثلاً ورود RSI به محدوده سالم‌تر، یا کراس صعودی "
                f"MACD) پیش از تخصیص سرمایه. در صورت تمایل به داشتن سطوح مرجع، محدوده ورود احتمالی "
                f"{entry_low:,.0f} تا {entry_high:,.0f} ریال با حد ضرر نزدیک به {stop_loss:,.0f} ریال خواهد بود."
            )
        else:
            summary_fa += (
                "در مجموع، شرایط فعلی از یک ورود خرید جدید حمایت نمی‌کند. معمولاً بهتر است منتظر بهبود "
                "روند یا مومنتوم بمانید تا اینکه بر خلاف سیگنال‌های موجود وارد معامله شوید."
            )

        return TradeSignal(
            verdict=verdict,
            verdict_fa=verdict_fa,
            score=score,
            current_price=round(close),
            entry_low=entry_low,
            entry_high=entry_high,
            stop_loss=stop_loss,
            target=target,
            risk_reward_ratio=round(risk_reward, 2),
            reasons=reasons,
            reasons_fa=reasons_fa,
            warnings=warnings,
            warnings_fa=warnings_fa,
            summary_en=summary_en,
            summary_fa=summary_fa,
        )
