import pandas as pd
from src.tsetmc_fetcher import TSETMCFetcher
from src.database import DatabaseManager
from src.indicators import TechnicalIndicators


def main():
    symbol = "فملی"
    print(f"🚀 Starting v0.1 test for symbol: {symbol}\n")

    # 1. Fetch data from TSETMC
    fetcher = TSETMCFetcher()
    print("⏳ Fetching data from TSETMC...")
    df = fetcher.fetch_daily_history(symbol)

    if df.empty:
        print("❌ No data was retrieved.")
        return

    print(f"✅ Retrieved {len(df)} trading days.")

    # 2. Save to SQLite database
    db = DatabaseManager()
    db.save_history(symbol, df)
    print("💾 Data saved to SQLite database.")

    # 3. Calculate indicators
    df_analyzed = TechnicalIndicators.apply_all(df)

    # 4. Show latest status for the symbol
    latest = df_analyzed.iloc[-1]
    print("\n" + "=" * 40)
    print(f"📊 Initial technical report for {symbol}")
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
        f"🔹 RSI (14): {latest['RSI_14']:.2f}"
        if not pd.isna(latest["RSI_14"])
        else "🔹 RSI (14): -"
    )
    print(f"🔹 MACD: {latest['MACD']:.2f} | Signal: {latest['MACD_Signal']:.2f}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    main()
