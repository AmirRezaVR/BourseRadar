import pandas as pd
from src.tsetmc_fetcher import TSETMCFetcher
from src.database import DatabaseManager
from src.indicators import TechnicalIndicators
from src.trade_advisor import TradeAdvisor


def print_technical_report(symbol: str, latest: pd.Series):
    print("\n" + "=" * 44)
    print(f"📊 Technical report — {symbol}")
    print(f"📅 Last trade date: {latest['date']}")
    print(f"🔹 Closing price: {latest['close_price']:,.0f} Rial")
    print(
        f"🔹 SMA 20: {latest['SMA_20']:,.0f}"
        if not pd.isna(latest["SMA_20"])
        else "🔹 SMA 20: -"
    )
    print(
        f"🔹 EMA 20: {latest['EMA_20']:,.0f}"
        if not pd.isna(latest["EMA_20"])
        else "🔹 EMA 20: -"
    )
    print(
        f"🔹 EMA 50: {latest['EMA_50']:,.0f}"
        if not pd.isna(latest["EMA_50"])
        else "🔹 EMA 50: -"
    )
    print(
        f"🔹 RSI (14): {latest['RSI_14']:.2f}"
        if not pd.isna(latest["RSI_14"])
        else "🔹 RSI (14): -"
    )
    print(f"🔹 MACD: {latest['MACD']:.2f} | Signal: {latest['MACD_Signal']:.2f}")
    print(
        f"🔹 ATR (14): {latest['ATR_14']:.2f}"
        if not pd.isna(latest["ATR_14"])
        else "🔹 ATR (14): -"
    )
    print("=" * 44)


def print_trade_signal(symbol: str, signal):
    verdict_icons = {"BUY": "🟢 BUY", "WATCH": "🟡 WATCH", "AVOID": "🔴 AVOID"}
    print(f"\n📌 Swing-trade signal — {symbol} (short-term, moderate risk/reward)")
    print("-" * 44)
    print(f"Verdict:            {verdict_icons.get(signal.verdict, signal.verdict)}")
    print(f"Signal score:       {signal.score}/100")
    print(f"Current price:      {signal.current_price:,.0f} Rial")
    print(
        f"Suggested entry:    {signal.entry_low:,.0f} – {signal.entry_high:,.0f} Rial"
    )
    print(f"Stop-loss:          {signal.stop_loss:,.0f} Rial")
    print(f"Target:             {signal.target:,.0f} Rial")
    print(f"Risk:Reward:        1 : {signal.risk_reward_ratio}")

    if signal.reasons:
        print("\nWhy:")
        for r in signal.reasons:
            print(f"  • {r}")

    if signal.warnings:
        print("\n⚠️  Notes:")
        for w in signal.warnings:
            print(f"  • {w}")

    print("\n" + "-" * 44)
    print(
        "This is an automated, rule-based technical signal only.\n"
        "It is NOT financial advice, does not account for news, fundamentals,\n"
        "or your personal risk tolerance, and TSETMC data may be delayed or incomplete.\n"
        "Always verify independently before making any trading decision."
    )
    print("-" * 44)


def main():
    symbol = input("Enter stock symbol (e.g. فملی): ").strip()

    if not symbol:
        print("❌ No symbol entered.")
        return

    print(f"\n🚀 Analyzing {symbol}...\n")

    fetcher = TSETMCFetcher()
    print("⏳ Fetching data from TSETMC...")
    df = fetcher.fetch_daily_history(symbol)

    if df.empty:
        print("❌ No data was retrieved.")
        return

    print(f"✅ Retrieved {len(df)} trading days.")

    db = DatabaseManager()
    db.save_history(symbol, df)
    print("💾 Data saved to SQLite database.")

    df_analyzed = TechnicalIndicators.apply_all(df)
    latest = df_analyzed.iloc[-1]

    print_technical_report(symbol, latest)

    try:
        advisor = TradeAdvisor()
        signal = advisor.generate_signal(df_analyzed)
        print_trade_signal(symbol, signal)
    except ValueError as e:
        print(f"\n⚠️  Trade signal unavailable: {e}")


if __name__ == "__main__":
    main()
