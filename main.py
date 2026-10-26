from src.tsetmc_fetcher import TSETMCFetcher
from src.signal_engine import SignalEngine
from src.analysis_service import analyze_symbol_core, parse_symbols


def fmt_or_dash(value, pattern="{:,.0f}"):
    return "—" if value is None else pattern.format(value)


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


def choose_language() -> str:
    print("=" * 50)
    print("Choose your language / انتخاب زبان")
    print("=" * 50)
    print("[1] English")
    print("[2] فارسی")
    while True:
        choice = input("> ").strip().lower()
        if choice in LANG_CHOICES:
            return LANG_CHOICES[choice]
        print("Please enter 1 or 2. / لطفاً عدد ۱ یا ۲ را وارد کنید")


def print_short_result(symbol: str, rec, lang: str, market_update: str):
    if lang == "en":
        entry_zone = (
            "—"
            if rec.entry_low is None
            else f"{fmt_or_dash(rec.entry_low)}–{fmt_or_dash(rec.entry_high)}"
        )
        ratio = fmt_or_dash(rec.risk_reward_ratio, "{:.1f}")
        ratio_display = f"1:{ratio}" if rec.risk_reward_ratio is not None else ratio
        print(
            f"\n{symbol} — {rec.verdict_en}  "
            f"({rec.signal_strength:.0f}% signal strength)"
        )
        print(
            f"   Price: {rec.current_price:,.0f}  |  Entry: {entry_zone}  "
            f"|  Stop: {fmt_or_dash(rec.stop_loss)}  |  Target: {fmt_or_dash(rec.target)}  "
            f"({ratio_display})"
        )
        # Real last-update time from the exchange, not our own fetch time.
        print(f"   Last market update: {market_update}")
    else:
        entry_zone = (
            "—"
            if rec.entry_low is None
            else f"{fmt_or_dash(rec.entry_low)} تا {fmt_or_dash(rec.entry_high)}"
        )
        ratio = fmt_or_dash(rec.risk_reward_ratio, "{:.1f}")
        ratio_display = f"۱ به {ratio}" if rec.risk_reward_ratio is not None else ratio
        print(
            f"\n{symbol} — {rec.verdict_fa}  "
            f"(قدرت سیگنال {rec.signal_strength:.0f}٪)"
        )
        print(
            f"   قیمت: {rec.current_price:,.0f}  |  ورود: {entry_zone}  "
            f"|  حد ضرر: {fmt_or_dash(rec.stop_loss)}  |  هدف: {fmt_or_dash(rec.target)}  "
            f"({ratio_display})"
        )
        print(f"   آخرین بروزرسانی بازار: {market_update}")


def analyze_symbol(
    symbol: str,
    fetcher: TSETMCFetcher,
    engine: SignalEngine,
    lang: str,
):
    print(f"\n{symbol}...")

    result = analyze_symbol_core(symbol, fetcher, engine)

    if not result.success:
        if result.error == "no_data":
            print("No data retrieved." if lang == "en" else "داده‌ای دریافت نشد")
        elif result.error == "symbol_not_found":
            print(
                "Symbol not found."
                if lang == "en"
                else "نماد با این نام کوتاه پیدا نشد"
            )
        elif result.error == "insufficient_history":
            msg = (
                f"Not enough history yet ({result.error_detail} days) for a signal"
                if lang == "en"
                else f"داده کافی نیست ({result.error_detail} روز)"
            )
            print(msg)
        elif result.error == "data_source_error":
            print(
                "Data source error. Try again later."
                if lang == "en"
                else "خطا در دریافت داده از TSETMC؛ بعداً دوباره تلاش کنید"
            )
        else:
            print(result.error_detail or "")
        return None

    rec = result.recommendation
    market_update = (
        result.market_update_en if lang == "en" else result.market_update_fa
    ) or "—"
    print_short_result(symbol, rec, lang, market_update)
    warnings = result.warnings_en if lang == "en" else result.warnings_fa
    for warning in warnings or []:
        print(
            f"   Data warning: {warning}"
            if lang == "en"
            else f"   هشدار داده: {warning}"
        )

    return rec


def print_batch_summary(results: list, lang: str):
    if len(results) < 2:
        return

    print("\n" + "=" * 50)
    print("Batch Summary" if lang == "en" else "خلاصه گروهی")
    print("=" * 50)

    if lang == "en":
        print(f"{'Symbol':<20}{'Verdict':<18}{'Entry Zone'}")
        print("-" * 50)
        for symbol, rec in results:
            if rec is None:
                print(f"{symbol:<16}{'N/A':<20}-")
            else:
                entry_zone = (
                    "—"
                    if rec.entry_low is None
                    else f"{fmt_or_dash(rec.entry_low)}–{fmt_or_dash(rec.entry_high)}"
                )
                print(f"{symbol:<16}{rec.verdict_en:<20}" f"{entry_zone}")
    else:
        print(f"{'نتیجه':<20}{'محدوده ورود':<25}{'نماد'}")
        print("-" * 50)
        for symbol, rec in results:
            if rec is None:
                print(f"{'بدون داده':<20}{'-':<25}{symbol}")
            else:
                entry_zone = (
                    "—"
                    if rec.entry_low is None
                    else f"{fmt_or_dash(rec.entry_low)} تا {fmt_or_dash(rec.entry_high)}"
                )
                print(f"{rec.verdict_fa:<20}{entry_zone:<25}{symbol}")
    print("=" * 50)


def main():
    lang = choose_language()

    fetcher = TSETMCFetcher()
    engine = SignalEngine()

    print("\n" + "=" * 50)
    print("BourseRadar")
    print("=" * 50)

    commands = " / ".join(sorted(EXIT_COMMANDS))
    if lang == "en":
        print(
            "Type one short ticker, or several separated by spaces "
            "(e.g. فملی فولاد خودرو)"
        )
        print(f"Type one of these to quit: {commands}")
    else:
        print(
            "یک یا چند نماد کوتاه را با فاصله جدا و وارد کنید (مثال: فملی فولاد خودرو)"
        )
        print(f"برای خروج یکی از این کلمات را وارد کنید: {commands}")

    while True:
        prompt = "\n Symbol(s): " if lang == "en" else "\n نماد(ها): "
        raw = input(prompt).strip()

        if not raw:
            continue
        if raw.lower() in EXIT_COMMANDS:
            break

        symbols = parse_symbols(raw)
        if not symbols:
            continue

        results = []
        for i, symbol in enumerate(symbols, 1):
            if len(symbols) > 1:
                print(f"\n[{i}/{len(symbols)}]", end="")
            rec = analyze_symbol(symbol, fetcher, engine, lang)
            results.append((symbol, rec))

        print_batch_summary(results, lang)


if __name__ == "__main__":
    main()
