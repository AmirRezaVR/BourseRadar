import requests
import pandas as pd
from typing import Optional


class TSETMCFetcher:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        # Cache of resolved symbol -> insCode, so we don't re-search on every call
        self._symbol_cache = {}

    def search_symbol(self, keyword: str) -> list:
        """Search TSETMC for instruments matching a keyword. Returns raw list of matches."""
        url = f"https://cdn.tsetmc.com/api/Instrument/GetInstrumentSearch/{keyword}"

        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get("instrumentSearch", [])
        except Exception as e:
            print(f"❌ Error searching for '{keyword}': {e}")
            return []

    def get_symbol_id(self, symbol: str) -> Optional[str]:
        """Resolve a typed symbol name to its insCode, preferring an exact ticker match."""
        if symbol in self._symbol_cache:
            return self._symbol_cache[symbol]

        results = self.search_symbol(symbol)
        if not results:
            return None

        # Prefer an exact match on the short ticker (lVal18AFC)
        exact_matches = [r for r in results if r.get("lVal18AFC") == symbol]

        if exact_matches:
            # If multiple exact matches, prefer main market/OTC stock (flow 1 or 2)
            # over derivatives (flow 3) or others
            exact_matches.sort(key=lambda r: r.get("flow", 99))
            chosen = exact_matches[0]
        else:
            # No exact match — fall back to the first result, but warn the user
            chosen = results[0]
            print(
                f"⚠️  No exact match for '{symbol}'. Using closest result: "
                f"{chosen.get('lVal18AFC')} ({chosen.get('lVal30')})"
            )

        inscode = chosen.get("insCode")
        self._symbol_cache[symbol] = inscode
        return inscode

    def fetch_daily_history(self, symbol: str) -> pd.DataFrame:
        inscode = self.get_symbol_id(symbol)
        if not inscode:
            print(f"❌ Symbol {symbol} not found.")
            return pd.DataFrame()

        url = f"https://cdn.tsetmc.com/api/ClosingPrice/GetClosingPriceDailyList/{inscode}/0"

        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json().get("closingPriceDaily", [])

            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data)

            df_cleaned = pd.DataFrame(
                {
                    "date": df["dEven"].astype(str),
                    "close_price": df["pClosing"],
                    "last_price": df["pDrCotVal"],
                    "open_price": df["priceFirst"],
                    "high_price": df["priceMax"],
                    "low_price": df["priceMin"],
                    "volume": df["qTotTran5J"],
                    "value": df["qTotCap"],
                    "count": df["zTotTran"],
                }
            )

            df_cleaned = df_cleaned.sort_values("date").reset_index(drop=True)
            return df_cleaned

        except Exception as e:
            print(f"❌ Error fetching TSETMC data for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_client_type(self, symbol: str, deven: str) -> dict:
        """Real vs legal buyer/seller money flow for a specific trading day"""
        inscode = self.get_symbol_id(symbol)
        if not inscode:
            print(f"❌ Symbol {symbol} not found.")
            return {}

        url = f"https://cdn.tsetmc.com/api/ClientType/GetClientTypeHistory/{inscode}/{deven}"

        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Error fetching client type data for {symbol}: {e}")
            return {}

    def fetch_order_book(self, symbol: str) -> dict:
        """Buy/sell queue (best limits)"""
        inscode = self.get_symbol_id(symbol)
        if not inscode:
            print(f"❌ Symbol {symbol} not found.")
            return {}

        url = f"https://cdn.tsetmc.com/api/BestLimits/{inscode}"

        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Error fetching order book for {symbol}: {e}")
            return {}
