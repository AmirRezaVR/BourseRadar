import pandas as pd
from src.tsetmc_fetcher import TSETMCFetcher
from src.database import DatabaseManager
from src.indicators import TechnicalIndicators
from src.trade_advisor import TradeAdvisor

EXIT_COMMANDS = {"exit", "quit", "q", "خروج", "پایان"}
SEPARATORS = ["،", ";", "؛", ","]  # accepted alongside the primary space separator

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


def choose_language() -> str:
    """Ask the user once, at startup, which language all output should be in."""
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
    rsi, macd, macd_signal, atr = (
        latest["RSI_14"],
        latest["MACD"],
        latest["MACD_Signal"],
        latest["ATR_14"],
    )

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
        print(
            f"🔹 میانگین متحرک ساده ۲۰ روزه (SMA20): {sma20:,.0f}"
            if not pd.isna(sma20)
            else "🔹 SMA 20: -"
        )
        print(
            f"🔹 میانگین متحرک نمایی ۲۰ روزه (EMA20): {ema20:,.0f}"
            if not pd.isna(ema20)
            else "🔹 EMA 20: -"
        )
        print(
            f"🔹 میانگین متحرک نمایی ۵۰ روزه (EMA50): {ema50:,.0f}"
            if not pd.isna(ema50)
            else "🔹 EMA 50: -"
        )
        print(
            f"🔹 شاخص قدرت نسبی RSI (۱۴ روزه): {rsi:.2f}"
            if not pd.isna(rsi)
            else "🔹 RSI: -"
        )
        print(f"🔹 مکدی (MACD): {macd:.2f} | خط سیگنال: {macd_signal:.2f}")
        print(
            f"🔹 میانگین محدوده واقعی ATR (۱۴ روزه): {atr:.2f}"
            if not pd.isna(atr)
            else "🔹 ATR: -"
        )
    print("=" * 50)


def print_trade_signal(symbol: str, signal, lang: str):
    verdict_icons = {"BUY": "🟢", "WATCH": "🟡", "AVOID": "🔴"}
    icon = verdict_icons.get(signal.verdict, "")

    if lang == "en":
        print(f"\n📌 Swing-Trade Signal — {symbol}")
        print("-" * 50)
        print(f"Verdict:          {icon} {signal.verdict}")
        print(f"Signal score:     {signal.score}/100")
        print(f"Current price:    {signal.current_price:,.0f} Rial")
        print(
            f"Suggested entry:  {signal.entry_low:,.0f} – {signal.entry_high:,.0f} Rial"
        )
        print(f"Stop-loss:        {signal.stop_loss:,.0f} Rial")
        print(f"Target:           {signal.target:,.0f} Rial")
        print(f"Risk : Reward:    1 : {signal.risk_reward_ratio}")

        print("\nWhy:")
        for r in signal.reasons:
            print(f"  • {r}")
        if signal.warnings:
            print("\n⚠️  Notes:")
            for w in signal.warnings:
                print(f"  • {w}")

        print("\nSummary:")
        print(f"  {signal.summary_en}")

        print("\n" + "-" * 50)
        print(
            "This is an automated, rule-based technical signal only — NOT financial advice.\n"
            "It does not account for news, fundamentals, or your personal risk tolerance,\n"
            "and TSETMC data may be delayed or incomplete. Always verify independently."
        )
        print("-" * 50)
    else:
        print(f"\n📌 سیگنال معاملاتی — {symbol}")
        print("-" * 50)
        print(f"نتیجه:              {icon} {signal.verdict_fa}")
        print(f"امتیاز سیگنال:        {signal.score} از ۱۰۰")
        print(f"قیمت فعلی:           {signal.current_price:,.0f} ریال")
        print(
            f"محدوده ورود پیشنهادی: {signal.entry_low:,.0f} تا {signal.entry_high:,.0f} ریال"
        )
        print(f"حد ضرر:              {signal.stop_loss:,.0f} ریال")
        print(f"هدف قیمتی:           {signal.target:,.0f} ریال")
        print(f"نسبت ریسک به بازده:    ۱ به {signal.risk_reward_ratio}")

        print("\nدلایل:")
        for r in signal.reasons_fa:
            print(f"  • {r}")
        if signal.warnings_fa:
            print("\n⚠️  نکات مهم:")
            for w in signal.warnings_fa:
                print(f"  • {w}")

        print("\nجمع‌بندی:")
        print(f"  {signal.summary_fa}")

        print("\n" + "-" * 50)
        print(
            "این فقط یک سیگنال خودکار و مبتنی بر تحلیل تکنیکال است و توصیه مالی محسوب نمی‌شود.\n"
            "این تحلیل اخبار، بنیاد شرکت و میزان ریسک‌پذیری شخصی شما را در نظر نمی‌گیرد،\n"
            "و داده‌های TSETMC ممکن است با تأخیر یا ناقص باشند. همیشه پیش از تصمیم‌گیری، به‌صورت مستقل بررسی کنید."
        )
        print("-" * 50)


def analyze_symbol(symbol: str, fetcher: TSETMCFetcher, db: DatabaseManager, lang: str):
    """Analyze a single symbol. Returns the TradeSignal on success, or None on failure."""
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

    if lang == "en":
        print(f"✅ Retrieved {len(df)} trading days.")
    else:
        print(f"✅ {len(df)} روز کاری دریافت شد.")

    db.save_history(symbol, df)
    print(
        "💾 Data saved to SQLite database."
        if lang == "en"
        else "💾 داده‌ها ذخیره شدند."
    )

    df_analyzed = TechnicalIndicators.apply_all(df)
    latest = df_analyzed.iloc[-1]

    print_technical_report(symbol, latest, lang)

    try:
        advisor = TradeAdvisor()
        signal = advisor.generate_signal(df_analyzed, symbol=symbol)
        print_trade_signal(symbol, signal, lang)
        return signal
    except ValueError as e:
        print(
            f"\n⚠️  Trade signal unavailable: {e}"
            if lang == "en"
            else f"\n⚠️  سیگنال معاملاتی در دسترس نیست: {e}"
        )
        return None


def print_batch_summary(results: list, lang: str):
    if len(results) < 2:
        return

    verdict_icons = {"BUY": "🟢", "WATCH": "🟡", "AVOID": "🔴"}

    print("\n" + "=" * 50)
    print("📋 Batch Summary" if lang == "en" else "📋 خلاصه گروهی")
    print("=" * 50)

    if lang == "en":
        print(f"{'Symbol':<15}{'Verdict':<12}{'Score':<8}{'Entry Zone'}")
        print("-" * 50)
        for symbol, signal in results:
            if signal is None:
                print(f"{symbol:<15}{'N/A':<12}{'-':<8}-")
            else:
                icon = verdict_icons.get(signal.verdict, "")
                print(
                    f"{symbol:<15}{icon + ' ' + signal.verdict:<12}"
                    f"{signal.score:<8}{signal.entry_low:,.0f}–{signal.entry_high:,.0f}"
                )
    else:
        print(f"{'نماد':<15}{'نتیجه':<15}{'امتیاز':<10}{'محدوده ورود'}")
        print("-" * 50)
        for symbol, signal in results:
            if signal is None:
                print(f"{symbol:<15}{'بدون داده':<15}{'-':<10}-")
            else:
                icon = verdict_icons.get(signal.verdict, "")
                print(
                    f"{symbol:<15}{icon + ' ' + signal.verdict_fa:<15}"
                    f"{signal.score:<10}{signal.entry_low:,.0f} تا {signal.entry_high:,.0f}"
                )
    print("=" * 50)


def main():
    lang = choose_language()

    fetcher = TSETMCFetcher()
    db = DatabaseManager()

    print("\n" + "=" * 50)
    print("📈 BourseRadar — TSE Stock Analyzer")
    print("=" * 50)
    if lang == "en":
        print(
            "Type one symbol, or several separated by spaces, to analyze them (e.g. فملی فولاد خودرو)."
        )
        print(f"Type one of {sorted(EXIT_COMMANDS)} to quit.")
    else:
        print(
            "برای تحلیل، یک نماد یا چند نماد را با فاصله جدا کرده وارد کنید (مثال: فملی فولاد خودرو)."
        )
        print(f"برای خروج، یکی از این کلمات را وارد کنید: {sorted(EXIT_COMMANDS)}")

    while True:
        prompt = (
            "\n🔎 Enter symbol(s): " if lang == "en" else "\n🔎 نماد(ها) را وارد کنید: "
        )
        raw = input(prompt).strip()

        if not raw:
            print(
                "❌ No symbol entered. Try again, or type 'exit' to quit."
                if lang == "en"
                else "❌ نمادی وارد نشد. دوباره تلاش کنید یا 'خروج' را تایپ کنید."
            )
            continue

        if raw.lower() in EXIT_COMMANDS or raw in EXIT_COMMANDS:
            print("\n👋 Goodbye!" if lang == "en" else "\n👋 خداحافظ!")
            break

        symbols = parse_symbols(raw)
        if not symbols:
            print(
                "❌ Couldn't parse any symbols from that input."
                if lang == "en"
                else "❌ نمادی شناسایی نشد."
            )
            continue

        if len(symbols) > 1:
            if lang == "en":
                print(f"\n📦 Batch mode: {len(symbols)} symbols queued")
            else:
                print(f"\n📦 حالت گروهی: {len(symbols)} نماد در صف")

        results = []
        for i, symbol in enumerate(symbols, 1):
            if len(symbols) > 1:
                print(f"\n{'#' * 50}")
                print(f"[{i}/{len(symbols)}] {symbol}")
                print(f"{'#' * 50}")
            signal = analyze_symbol(symbol, fetcher, db, lang)
            results.append((symbol, signal))

        print_batch_summary(results, lang)


if __name__ == "__main__":
    main()
