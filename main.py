import pandas as pd
from src.tsetmc_fetcher import TSETMCFetcher
from src.database import DatabaseManager
from src.indicators import TechnicalIndicators
from src.signal_engine import SignalEngine, VERDICT_ICONS, MIN_HISTORY_DAYS

EXIT_COMMANDS = {"exit", "quit", "q", "خروج", "پایان"}
SEPARATORS = ["،", ";", "؛", ","]  # accepted alongside the primary space separator

LANG_CHOICES = {
    "1": "en", "en": "en", "english": "en",
    "2": "fa", "fa": "fa", "farsi": "fa", "persian": "fa", "فارسی": "fa",
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


def parse_symbols(raw: str) -> list:
    normalized = raw
    for sep in SEPARATORS:
        normalized = normalized.replace(sep, " ")
    parts = [p.strip() for p in normalized.split() if p.strip()]
    seen, unique_parts = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p)
            unique_parts.append(p)
    return unique_parts


def print_technical_report(symbol: str, latest: pd.Series, lang: str):
    sma20, ema20, ema50 = latest["SMA_20"], latest["EMA_20"], latest["EMA_50"]
    rsi, macd, macd_signal, atr = latest["RSI_14"], latest["MACD"], latest["MACD_Signal"], latest["ATR_14"]

    print("\n" + "=" * 50)
    if lang == "en":
        print(f"📊 Technical Report — {symbol}")
        print("=" * 50)
        print(f"📅 Last trade date: {latest['date']}")
        print(f"🔹 Closing price: {latest['close_price']:,.0f} Rial")
        print(f"🔹 SMA 20: {sma20:,.0f}" if not pd.isna(sma20) else "🔹 SMA 20: -")
        print(f"🔹 EMA 20: {ema20:,.0f}" if not pd.isna(ema20) else "🔹 EMA 20: -")
        print(f"🔹 EMA 50: {ema50:,.0f}" if not pd.isna(ema50) else "🔹 EMA 50: -")
        print(f"🔹 RSI (14): {rsi:.2f}" if not pd.isna(rsi) else "🔹 RSI (14): -")
        print(f"🔹 MACD: {macd:.2f} | Signal: {macd_signal:.2f}")
        print(f"🔹 ATR (14): {atr:.2f}" if not pd.isna(atr) else "🔹 ATR (14): -")
    else:
        print(f"📊 گزارش فنی — {symbol}")
        print("=" * 50)
        print(f"📅 تاریخ آخرین معامله: {latest['date']}")
        print(f"🔹 قیمت پایانی: {latest['close_price']:,.0f} ریال")
        print(f"🔹 میانگین متحرک ساده ۲۰ روزه (SMA20): {sma20:,.0f}" if not pd.isna(sma20) else "🔹 SMA 20: -")
        print(f"🔹 میانگین متحرک نمایی ۲۰ روزه (EMA20): {ema20:,.0f}" if not pd.isna(ema20) else "🔹 EMA 20: -")
        print(f"🔹 میانگین متحرک نمایی ۵۰ روزه (EMA50): {ema50:,.0f}" if not pd.isna(ema50) else "🔹 EMA 50: -")
        print(f"🔹 شاخص قدرت نسبی RSI (۱۴ روزه): {rsi:.2f}" if not pd.isna(rsi) else "🔹 RSI: -")
        print(f"🔹 مکدی (MACD): {macd:.2f} | خط سیگنال: {macd_signal:.2f}")
        print(f"🔹 میانگین محدوده واقعی ATR (۱۴ روزه): {atr:.2f}" if not pd.isna(atr) else "🔹 ATR: -")
    print("=" * 50)


def print_recommendation(symbol: str, rec, lang: str):
    icon = VERDICT_ICONS.get(rec.verdict, "")

    if lang == "en":
        print(f"\n📌 Multi-Factor Signal — {symbol}")
        print("-" * 50)
        print(f"Verdict:          {icon} {rec.verdict_en}")
        print(f"Confidence:       {rec.confidence:.0f}%   (composite {rec.composite_score:+.2f})")
        print(f"Current price:    {rec.current_price:,.0f} Rial")
        print(f"Suggested entry:  {rec.entry_low:,.0f} – {rec.entry_high:,.0f} Rial")
        print(f"Stop-loss:        {rec.stop_loss:,.0f} Rial")
        print(f"Target:           {rec.target:,.0f} Rial   (1:{rec.risk_reward_ratio})")

        print("\nFactor breakdown:")
        for f in rec.factors:
            print(f"  {f.label_en:32s} {f.score:+.2f}  (weight {f.weight:4.1f})  ->  {f.reason_en}")
        print(f"  {'Volatility (confidence modifier)':32s} x{rec.volatility_multiplier:.2f}       ->  {rec.volatility_reason_en}")

        print("\n✅ Positive confirmations:")
        for c in rec.confirmations_en:
            print(f"  • {c}")
        print("\n⚠️  Risk warnings:")
        for r in rec.risks_en:
            print(f"  • {r}")

        print("\nSummary:")
        print(f"  {rec.summary_en}")

        print("\n" + "-" * 50)
        print(
            "This is an automated, rule-based technical signal only — NOT financial advice.\n"
            "It does not account for news, fundamentals, or your personal risk tolerance,\n"
            "and TSETMC data may be delayed or incomplete. Always verify independently."
        )
        print("-" * 50)
    else:
        print(f"\n📌 سیگنال چندعاملی — {symbol}")
        print("-" * 50)
        print(f"نتیجه:              {icon} {rec.verdict_fa}")
        print(f"اطمینان:            {rec.confidence:.0f}٪   (امتیاز ترکیبی {rec.composite_score:+.2f})")
        print(f"قیمت فعلی:           {rec.current_price:,.0f} ریال")
        print(f"محدوده ورود پیشنهادی: {rec.entry_low:,.0f} تا {rec.entry_high:,.0f} ریال")
        print(f"حد ضرر:              {rec.stop_loss:,.0f} ریال")
        print(f"هدف قیمتی:           {rec.target:,.0f} ریال   (۱ به {rec.risk_reward_ratio})")

        print("\nتفکیک عوامل:")
        for f in rec.factors:
            print(f"  {f.label_fa:35s} {f.score:+.2f}  (وزن {f.weight:.0f})  ->  {f.reason_fa}")
        print(f"  {'نوسان (تعدیل‌کننده اطمینان)':35s} x{rec.volatility_multiplier:.2f}     ->  {rec.volatility_reason_fa}")

        print("\n✅ تأییدات مثبت:")
        for c in rec.confirmations_fa:
            print(f"  • {c}")
        print("\n⚠️  هشدارهای ریسک:")
        for r in rec.risks_fa:
            print(f"  • {r}")

        print("\nجمع‌بندی:")
        print(f"  {rec.summary_fa}")

        print("\n" + "-" * 50)
        print(
            "این فقط یک سیگنال خودکار و مبتنی بر تحلیل تکنیکال است و توصیه مالی محسوب نمی‌شود.\n"
            "این تحلیل اخبار، بنیاد شرکت و میزان ریسک‌پذیری شخصی شما را در نظر نمی‌گیرد،\n"
            "و داده‌های TSETMC ممکن است با تأخیر یا ناقص باشند. همیشه پیش از تصمیم‌گیری، به‌صورت مستقل بررسی کنید."
        )
        print("-" * 50)


def analyze_symbol(symbol: str, fetcher: TSETMCFetcher, db: DatabaseManager, engine: SignalEngine, lang: str):
    """Analyze a single symbol. Returns the Recommendation on success, or None on failure."""
    if lang == "en":
        print(f"\n🚀 Analyzing: {symbol}\n")
        print("⏳ Fetching data from TSETMC...")
    else:
        print(f"\n🚀 در حال تحلیل: {symbol}\n")
        print("⏳ در حال دریافت داده از TSETMC...")

    df = fetcher.fetch_daily_history(symbol)

    if df.empty:
        print("❌ No data was retrieved." if lang == "en" else "❌ داده‌ای دریافت نشد.")
        return None

    print(f"✅ Retrieved {len(df)} trading days." if lang == "en" else f"✅ {len(df)} روز کاری دریافت شد.")

    db.save_history(symbol, df)
    print("💾 Data saved to SQLite database." if lang == "en" else "💾 داده‌ها ذخیره شدند.")

    df_analyzed = TechnicalIndicators.apply_all(df)

    if len(df_analyzed) < MIN_HISTORY_DAYS:
        msg = (f"⚠️  Not enough history for a multi-factor signal (need {MIN_HISTORY_DAYS}+ days)."
               if lang == "en" else f"⚠️  داده کافی برای سیگنال چندعاملی نیست (حداقل {MIN_HISTORY_DAYS} روز لازم است).")
        print(msg)
        print_technical_report(symbol, df_analyzed.iloc[-1], lang)
        return None

    print_technical_report(symbol, df_analyzed.iloc[-1], lang)

    try:
        rec = engine.generate(df_analyzed, symbol=symbol)
        print_recommendation(symbol, rec, lang)
        return rec
    except ValueError as e:
        print(f"\n⚠️  Signal unavailable: {e}" if lang == "en" else f"\n⚠️  سیگنال در دسترس نیست: {e}")
        return None


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
                print(f"{symbol:<15}{icon + ' ' + rec.verdict_en:<16}{f'{rec.confidence:.0f}%':<12}"
                      f"{rec.entry_low:,.0f}–{rec.entry_high:,.0f}")
    else:
        print(f"{'نماد':<15}{'نتیجه':<20}{'اطمینان':<12}{'محدوده ورود'}")
        print("-" * 50)
        for symbol, rec in results:
            if rec is None:
                print(f"{symbol:<15}{'بدون داده':<20}{'-':<12}-")
            else:
                icon = VERDICT_ICONS.get(rec.verdict, "")
                print(f"{symbol:<15}{icon + ' ' + rec.verdict_fa:<20}{f'{rec.confidence:.0f}٪':<12}"
                      f"{rec.entry_low:,.0f} تا {rec.entry_high:,.0f}")
    print("=" * 50)


def main():
    lang = choose_language()

    fetcher = TSETMCFetcher()
    db = DatabaseManager()
    engine = SignalEngine()

    print("\n" + "=" * 50)
    print("📈 BourseRadar — Multi-Factor TSE Signal Engine")
    print("=" * 50)
    if lang == "en":
        print("Type one symbol, or several separated by spaces, to analyze them (e.g. فملی فولاد خودرو).")
        print(f"Type one of {sorted(EXIT_COMMANDS)} to quit.")
    else:
        print("برای تحلیل، یک نماد یا چند نماد را با فاصله جدا کرده وارد کنید (مثال: فملی فولاد خودرو).")
        print(f"برای خروج، یکی از این کلمات را وارد کنید: {sorted(EXIT_COMMANDS)}")

    while True:
        prompt = "\n🔎 Enter symbol(s): " if lang == "en" else "\n🔎 نماد(ها) را وارد کنید: "
        raw = input(prompt).strip()

        if not raw:
            print("❌ No symbol entered. Try again, or type 'exit' to quit." if lang == "en"
                  else "❌ نمادی وارد نشد. دوباره تلاش کنید یا 'خروج' را تایپ کنید.")
            continue

        if raw.lower() in EXIT_COMMANDS or raw in EXIT_COMMANDS:
            print("\n👋 Goodbye!" if lang == "en" else "\n👋 خداحافظ!")
            break

        symbols = parse_symbols(raw)
        if not symbols:
            print("❌ Couldn't parse any symbols from that input." if lang == "en" else "❌ نمادی شناسایی نشد.")
            continue

        if len(symbols) > 1:
            print(f"\n📦 Batch mode: {len(symbols)} symbols queued" if lang == "en"
                  else f"\n📦 حالت گروهی: {len(symbols)} نماد در صف")

        results = []
        for i, symbol in enumerate(symbols, 1):
            if len(symbols) > 1:
                print(f"\n{'#' * 50}\n[{i}/{len(symbols)}] {symbol}\n{'#' * 50}")
            rec = analyze_symbol(symbol, fetcher, db, engine, lang)
            results.append((symbol, rec))

        print_batch_summary(results, lang)


if __name__ == "__main__":
    main()
