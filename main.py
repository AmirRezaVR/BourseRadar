from src.tsetmc_fetcher import TSETMCFetcher
from src.database import DatabaseManager
from src.indicators import TechnicalIndicators


def main():
    symbol = "فملی"
    print(f"🚀 شروع تست نسخه 0.1 برای نماد: {symbol}\n")

    # ۱. دریافت داده از TSETMC
    fetcher = TSETMCFetcher()
    print("⏳ در حال دریافت داده از TSETMC...")
    df = fetcher.fetch_daily_history(symbol)

    if df.empty:
        print("❌ متأسفانه داده‌ای دریافت نشد.")
        return

    print(f"✅ تعداد {len(df)} روز کاری دریافت شد.")

    # ۲. ذخیره در دیتابیس SQLite
    db = DatabaseManager()
    db.save_history(symbol, df)
    print("💾 داده‌ها در دیتابیس SQLite ذخیره شدند.")

    # ۳. محاسبه اندیکاتورها
    df_analyzed = TechnicalIndicators.apply_all(df)

    # ۴. نمایش آخرین وضعیت نماد
    latest = df_analyzed.iloc[-1]
    print("\n" + "=" * 40)
    print(f"📊 گزارش فنی اولیه برای نماد {symbol}")
    print(f"📅 تاریخ آخرین معامله: {latest['date']}")
    print(f"🔹 قیمت پایانی: {latest['close_price']:,.0f} ریال")
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
    import pandas as pd

    main()
