import requests
import pandas as pd
from typing import Optional


class TSETMCFetcher:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        # شناسه برخی نمادهای شاخص برای تست (قابل توسعه در فازهای بعدی)
        self.SYMBOL_MAP = {
            "فملی": "35425587644337450",
            "فولاد": "46348084566421241",
            "خودرو": "65883838195688438",
            "وبملت": "35366681030756137",
        }

    def get_symbol_id(self, symbol: str) -> Optional[str]:
        return self.SYMBOL_MAP.get(symbol)

    def fetch_daily_history(self, symbol: str) -> pd.DataFrame:
        """دریافت تاریخچه قیمت روزانه نماد از API رسمی TSETMC"""
        inscode = self.get_symbol_id(symbol)
        if not inscode:
            print(f"❌ نماد {symbol} در نقشه اولیه یافت نشد.")
            return pd.DataFrame()

        url = f"https://cdn.tsetmc.com/api/ClosingPrice/GetClosingPriceDailyList/{inscode}/0"

        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json().get("closingPriceDaily", [])

            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data)

            # استخراج و مرتب‌سازی ستون‌های مورد نیاز
            df_cleaned = pd.DataFrame(
                {
                    "date": df["dEven"].astype(str),
                    "close_price": df["pClosing"],
                    "last_price": df["pDrPri"],
                    "open_price": df["priceFirst"],
                    "high_price": df["priceMax"],
                    "low_price": df["priceMin"],
                    "volume": df["qTitTran"],
                    "value": df["qTitCap"],
                    "count": df["zTran"],
                }
            )

            # مرتب‌سازی بر اساس تاریخ (قدیمی به جدید)
            df_cleaned = df_cleaned.sort_values("date").reset_index(drop=True)
            return df_cleaned

        except Exception as e:
            print(f"❌ خطا در دریافت داده‌های TSETMC برای {symbol}: {e}")
            return pd.DataFrame()
