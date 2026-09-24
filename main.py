from src.tsetmc_fetcher import TSETMCFetcher
from src.signal_engine import SignalEngine, VERDICT_ICONS
from src.analysis_service import analyze_symbol_core, parse_symbols

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
    """The whole point of the program."""
    icon = VERDICT_ICONS.get(rec.verdict, "")
    if lang == "en":
        print(
            f"\n{icon} {symbol} — {rec.verdict_en}  ({rec.confidence:.0f}% confidence)"
        )
        print(
            f"   Price: {rec.current_price:,.0f}  |  Entry: {rec.entry_low:,.0f}–{rec.entry_high:,.0f}  "
            f"|  Stop: {rec.stop_loss:,.0f}  |  Target: {rec.target:,.0f}  (1:{rec.risk_reward_ratio})"
        )
        # Real last-update time from the exchange, not our own fetch time.
        print(f"   Last market update: {market_update}")
    else:
        print(f"\n{icon} {symbol} — {rec.verdict_fa}  (اطمینان {rec.confidence:.0f}٪)")
        print(
            f"   قیمت: {rec.current_price:,.0f}  |  ورود: {rec.entry_low:,.0f} تا {rec.entry_high:,.0f}  "
            f"|  حد ضرر: {rec.stop_loss:,.0f}  |  هدف: {rec.target:,.0f}  (۱ به {rec.risk_reward_ratio})"
        )
        print(f"   آخرین بروزرسانی بازار: {market_update}")


def analyze_symbol(
    symbol: str,
    fetcher: TSETMCFetcher,
    engine: SignalEngine,
    lang: str,
):
    """Runs one symbol through the shared pipeline and prints the result."""
    print(f"\n⏳ {symbol}...")

    result = analyze_symbol_core(symbol, fetcher, engine)

    if not result.success:
        if result.error == "no_data":
            print("No data retrieved." if lang == "en" else "داده‌ای دریافت نشد")
        elif result.error == "insufficient_history":
            msg = (
                f"Not enough history yet ({result.error_detail} days) for a signal"
                if lang == "en"
                else f"داده کافی نیست ({result.error_detail} روز)"
            )
            print(msg)
        else:
            print(f"{result.error_detail}")
        return None

    rec = result.recommendation
    market_update = result.market_update_en if lang == "en" else result.market_update_fa
    print_short_result(symbol, rec, lang, market_update)

    return rec


def print_batch_summary(results: list, lang: str):
    """Recap table at the end of a multi-symbol run."""
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
                icon = VERDICT_ICONS.get(rec.verdict, "")
                print(
                    f"{symbol:<16}{icon + ' ' + rec.verdict_en:<20}"
                    f"{rec.entry_low:,.0f}–{rec.entry_high:,.0f}"
                )
    else:
        print(f"{'نتیجه':<20}{'محدوده ورود':<25}{'نماد'}")
        print("-" * 50)
        for symbol, rec in results:
            if rec is None:
                print(f"{symbol:<15}{'بدون داده':<20}-")
            else:
                icon = VERDICT_ICONS.get(rec.verdict, "")
                print(
                    f"{symbol:<15}{icon + ' ' + rec.verdict_fa:<20}"
                    f"{rec.entry_low:,.0f} تا {rec.entry_high:,.0f}"
                )
    print("=" * 50)


def main():
    lang = choose_language()

    fetcher = TSETMCFetcher()
    engine = SignalEngine()

    print("\n" + "=" * 50)
    print("BourseRadar")
    print("=" * 50)
    if lang == "en":
        print("Type one symbol, or several separated by spaces (e.g. فملی فولاد خودرو)")
        print(f"Use {sorted(EXIT_COMMANDS)} to quit.")
    else:
        print("یک یا چند نماد را با فاصله جدا و وارد کنید (مثال: فملی فولاد خودرو)")
        print(f"استفاده کنید {sorted(EXIT_COMMANDS)} برای خروج از")

    # Main loop: keep asking for symbols until the user types an exit word.
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
            # Only bother numbering these when there's actually a batch -
            # looks noisy on a single symbol.
            if len(symbols) > 1:
                print(f"\n[{i}/{len(symbols)}]", end="")
            rec = analyze_symbol(symbol, fetcher, engine, lang)
            results.append((symbol, rec))

        print_batch_summary(results, lang)


if __name__ == "__main__":
    main()
