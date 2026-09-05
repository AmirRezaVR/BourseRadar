import pandas as pd
from src.tsetmc_fetcher import TSETMCFetcher
from src.database import DatabaseManager
from src.signal_engine import SignalEngine, VERDICT_ICONS, MIN_HISTORY_DAYS
from src.analysis_service import (
    analyze_symbol_core,
    get_fetch_timestamp,
    iran_time_available,
    parse_symbols,
)

EXIT_COMMANDS = {"exit", "quit", "q", "خروج", "پایان"}

LANG_CHOICES = {
    "1": "en",
    "en": "en",
    "english": "en",
    "2": "fa",
    "fa": "fa",
    "farsi": "fa",
    "persian": "fa",
    "فارسی": "fa",
}
DETAIL_CHOICES = {
    "1": "short",
    "short": "short",
    "2": "detailed",
    "detailed": "detailed",
    "detail": "detailed",
}


def choose_language() -> str:
    print("=" * 50)
    print("🌐 Choose your language / زبان خود را انتخاب کنید")
    print("=" * 50)
    print("[1] English")
    print("[2] فارسی")
    while True:
        choice = input("> ").strip().lower()
        if choice in LANG_CHOICES:
            return LANG_CHOICES[choice]
        print("Please enter 1 or 2. / لطفاً عدد ۱ یا ۲ را وارد کنید.")


def choose_detail_level(lang: str) -> str:
    if lang == "en":
        print("\nHow much detail do you want per recommendation?")
        print("[1] Short - verdict, confidence, entry/stop/target only (default)")
        print("[2] Detailed - full reasoning behind every recommendation")
    else:
        print("\nچقدر جزئیات برای هر توصیه می‌خواهید؟")
        print("[1] خلاصه — فقط نتیجه، اطمینان، محدوده ورود/حد ضرر/هدف (پیش‌فرض)")
        print("[2] کامل — توضیح کامل دلیل هر توصیه")
    while True:
        choice = input("> ").strip().lower()
        if choice in DETAIL_CHOICES:
            return DETAIL_CHOICES[choice]
        if choice == "":
            return "short"
        print(
            "Please enter 1 or 2." if lang == "en" else "لطفاً عدد ۱ یا ۲ را وارد کنید."
        )


def print_short_result(symbol: str, rec, lang: str, fetch_ts: str):
    icon = VERDICT_ICONS.get(rec.verdict, "")
    if lang == "en":
        print(
            f"\n{icon} {symbol} — {rec.verdict_en}  ({rec.confidence:.0f}% confidence)"
        )
        print(
            f"   Price: {rec.current_price:,.0f}  |  Entry: {rec.entry_low:,.0f}–{rec.entry_high:,.0f}  "
            f"|  Stop: {rec.stop_loss:,.0f}  |  Target: {rec.target:,.0f}  (1:{rec.risk_reward_ratio})"
        )
        print(f"   Fetched: {fetch_ts}")
        print("   Not financial advice - a rule-based technical read only.")
    else:
        print(f"\n{icon} {symbol} — {rec.verdict_fa}  (اطمینان {rec.confidence:.0f}٪)")
        print(
            f"   قیمت: {rec.current_price:,.0f}  |  ورود: {rec.entry_low:,.0f} تا {rec.entry_high:,.0f}  "
            f"|  حد ضرر: {rec.stop_loss:,.0f}  |  هدف: {rec.target:,.0f}  (۱ به {rec.risk_reward_ratio})"
        )
        print(f"   زمان دریافت: {fetch_ts}")
        print("   این توصیه مالی نیست — فقط یک خوانش خودکار تکنیکال است.")


def print_technical_report(symbol: str, latest: pd.Series, lang: str):
    ema20, ema50, rsi = latest["EMA_20"], latest["EMA_50"], latest["RSI_14"]
    macd, macd_signal, atr = latest["MACD"], latest["MACD_Signal"], latest["ATR_14"]

    print("\n" + "=" * 50)
    if lang == "en":
        print(f"📊 Technical Report — {symbol}")
        print("=" * 50)
        print(f"📅 Last trade date (TSETMC): {latest['date']}")
        print(f"🔹 Closing price: {latest['close_price']:,.0f} Rial")
        print(f"🔹 EMA 20 / EMA 50: {ema20:,.0f} / {ema50:,.0f}")
        print(f"🔹 RSI (14): {rsi:.2f}" if not pd.isna(rsi) else "🔹 RSI (14): -")
        print(f"🔹 MACD: {macd:.2f} | Signal: {macd_signal:.2f}")
        print(f"🔹 ATR (14): {atr:.2f}" if not pd.isna(atr) else "🔹 ATR (14): -")
    else:
        print(f"📊 گزارش فنی — {symbol}")
        print("=" * 50)
        print(f"📅 تاریخ آخرین معامله (TSETMC): {latest['date']}")
        print(f"🔹 قیمت پایانی: {latest['close_price']:,.0f} ریال")
        print(f"🔹 EMA20 / EMA50: {ema20:,.0f} / {ema50:,.0f}")
        print(f"🔹 RSI (۱۴ روزه): {rsi:.2f}" if not pd.isna(rsi) else "🔹 RSI: -")
        print(f"🔹 مکدی: {macd:.2f} | سیگنال: {macd_signal:.2f}")
        print(f"🔹 ATR (۱۴ روزه): {atr:.2f}" if not pd.isna(atr) else "🔹 ATR: -")
    print("=" * 50)


def print_detailed_result(symbol: str, rec, lang: str, fetch_ts: str):
    icon = VERDICT_ICONS.get(rec.verdict, "")

    if lang == "en":
        print(f"\n📌 {symbol} — Multi-Factor Signal")
        print("-" * 50)
        print(f"Verdict:          {icon} {rec.verdict_en}")
        print(
            f"Confidence:       {rec.confidence:.0f}%   "
            f"(composite {rec.composite_score:+.2f}, before volatility adjustment {rec.raw_composite_score:+.2f})"
        )
        print(f"Current price:    {rec.current_price:,.0f} Rial")
        print(f"Suggested entry:  {rec.entry_low:,.0f} – {rec.entry_high:,.0f} Rial")
        print(f"Stop-loss:        {rec.stop_loss:,.0f} Rial")
        print(f"Target:           {rec.target:,.0f} Rial   (1:{rec.risk_reward_ratio})")
        print(f"Fetched:          {fetch_ts}")

        print("\nWhy:")
        for f in rec.factors:
            avail = "" if f.available else "  [unavailable, excluded]"
            print(
                f"  {f.label_en:32s} {f.score:+.2f}  (weight {f.weight:4.1f})  ->  {f.reason_en}{avail}"
            )
        print(
            f"  {'Volatility (confidence modifier)':32s} x{rec.volatility_multiplier:.2f}       ->  {rec.volatility_reason_en}"
        )

        if rec.confirmations_en:
            print("\n✅ Confirmations:")
            for c in rec.confirmations_en:
                print(f"  • {c}")
        if rec.risks_en:
            print("\n⚠️  Risks:")
            for r in rec.risks_en:
                print(f"  • {r}")

        print(f"\nSummary: {rec.summary_en}")
        print("\n" + "-" * 50)
        print(
            "Not financial advice - a rule-based technical read only, no fundamentals or news considered."
        )
        print("-" * 50)
    else:
        print(f"\n📌 {symbol} — سیگنال چندعاملی")
        print("-" * 50)
        print(f"نتیجه:              {icon} {rec.verdict_fa}")
        print(
            f"اطمینان:            {rec.confidence:.0f}٪   "
            f"(امتیاز ترکیبی {rec.composite_score:+.2f}، پیش از تعدیل نوسان {rec.raw_composite_score:+.2f})"
        )
        print(f"قیمت فعلی:           {rec.current_price:,.0f} ریال")
        print(
            f"محدوده ورود پیشنهادی: {rec.entry_low:,.0f} تا {rec.entry_high:,.0f} ریال"
        )
        print(f"حد ضرر:              {rec.stop_loss:,.0f} ریال")
        print(
            f"هدف قیمتی:           {rec.target:,.0f} ریال   (۱ به {rec.risk_reward_ratio})"
        )
        print(f"زمان دریافت:          {fetch_ts}")

        print("\nدلیل:")
        for f in rec.factors:
            avail = "" if f.available else "  [در دسترس نیست، حذف شد]"
            print(
                f"  {f.label_fa:35s} {f.score:+.2f}  (وزن {f.weight:.0f})  ->  {f.reason_fa}{avail}"
            )
        print(
            f"  {'نوسان (تعدیل‌کننده اطمینان)':35s} x{rec.volatility_multiplier:.2f}     ->  {rec.volatility_reason_fa}"
        )

        if rec.confirmations_fa:
            print("\n✅ تأییدات:")
            for c in rec.confirmations_fa:
                print(f"  • {c}")
        if rec.risks_fa:
            print("\n⚠️  ریسک‌ها:")
            for r in rec.risks_fa:
                print(f"  • {r}")

        print(f"\nجمع‌بندی: {rec.summary_fa}")
        print("\n" + "-" * 50)
        print(
            "این توصیه مالی نیست — فقط یک خوانش خودکار تکنیکال است، بدون در نظر گرفتن بنیاد یا اخبار."
        )
        print("-" * 50)


def analyze_symbol(
    symbol: str,
    fetcher: TSETMCFetcher,
    db: DatabaseManager,
    engine: SignalEngine,
    lang: str,
    detail: str,
):
    """Analyze a single symbol via the shared core. Returns the Recommendation on success, or None."""
    print(f"\n⏳ {symbol}...")

    result = analyze_symbol_core(symbol, fetcher, db, engine)

    if not result.success:
        if result.error == "no_data":
            print("❌ No data retrieved." if lang == "en" else "❌ داده‌ای دریافت نشد.")
        elif result.error == "insufficient_history":
            msg = (
                f"⚠️  Not enough history yet ({result.error_detail} days) for a signal."
                if lang == "en"
                else f"⚠️  داده کافی نیست ({result.error_detail} روز)."
            )
            print(msg)
        else:
            print(f"⚠️  {result.error_detail}")
        return None

    rec = result.recommendation
    if detail == "detailed":
        print_technical_report(symbol, result.df.iloc[-1], lang)
        print_detailed_result(symbol, rec, lang, result.fetch_ts)
    else:
        print_short_result(symbol, rec, lang, result.fetch_ts)

    return rec


def print_batch_summary(results: list, lang: str):
    if len(results) < 2:
        return

    print("\n" + "=" * 50)
    print("📋 Batch Summary" if lang == "en" else "📋 خلاصه گروهی")
    print("=" * 50)

    if lang == "en":
        print(f"{'Symbol':<15}{'Verdict':<16}{'Confidence':<12}{'Entry Zone'}")
        print("-" * 50)
        for symbol, rec in results:
            if rec is None:
                print(f"{symbol:<15}{'N/A':<16}{'-':<12}-")
            else:
                icon = VERDICT_ICONS.get(rec.verdict, "")
                print(
                    f"{symbol:<15}{icon + ' ' + rec.verdict_en:<16}{f'{rec.confidence:.0f}%':<12}"
                    f"{rec.entry_low:,.0f}–{rec.entry_high:,.0f}"
                )
    else:
        print(f"{'نماد':<15}{'نتیجه':<20}{'اطمینان':<12}{'محدوده ورود'}")
        print("-" * 50)
        for symbol, rec in results:
            if rec is None:
                print(f"{symbol:<15}{'بدون داده':<20}{'-':<12}-")
            else:
                icon = VERDICT_ICONS.get(rec.verdict, "")
                print(
                    f"{symbol:<15}{icon + ' ' + rec.verdict_fa:<20}{f'{rec.confidence:.0f}٪':<12}"
                    f"{rec.entry_low:,.0f} تا {rec.entry_high:,.0f}"
                )
    print("=" * 50)


def main():
    lang = choose_language()
    detail = choose_detail_level(lang)

    fetcher = TSETMCFetcher()
    db = DatabaseManager()
    engine = SignalEngine()

    print("\n" + "=" * 50)
    print("📈 BourseRadar")
    print("=" * 50)
    if lang == "en":
        print(
            "Type one symbol, or several separated by spaces (e.g. فملی فولاد خودرو)."
        )
        print(f"Type one of {sorted(EXIT_COMMANDS)} to quit.")
    else:
        print(
            "یک نماد یا چند نماد را با فاصله جدا کرده وارد کنید (مثال: فملی فولاد خودرو)."
        )
        print(f"برای خروج: {sorted(EXIT_COMMANDS)}")

    while True:
        prompt = "\n🔎 Symbol(s): " if lang == "en" else "\n🔎 نماد(ها): "
        raw = input(prompt).strip()

        if not raw:
            continue
        if raw.lower() in EXIT_COMMANDS or raw in EXIT_COMMANDS:
            print("\n👋 Goodbye!" if lang == "en" else "\n👋 خداحافظ!")
            break

        symbols = parse_symbols(raw)
        if not symbols:
            continue

        results = []
        for i, symbol in enumerate(symbols, 1):
            if len(symbols) > 1:
                print(f"\n[{i}/{len(symbols)}]", end="")
            rec = analyze_symbol(symbol, fetcher, db, engine, lang, detail)
            results.append((symbol, rec))

        print_batch_summary(results, lang)


if __name__ == "__main__":
    main()
