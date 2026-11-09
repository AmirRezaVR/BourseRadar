import logging
import threading
import time
from urllib.parse import quote

import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)
REQUEST_INTERVAL_SECONDS = 0.25
MAX_RETRIES = 3


class TSETMCAPIError(RuntimeError):
    """Raised when TSETMC cannot return a valid response for a known endpoint."""


class SymbolNotFoundError(LookupError):
    """Raised when no instrument matches the typed ticker exactly."""


def normalize_symbol(text: str) -> str:
    return (
        text.replace("\u064a", "\u06cc")
        .replace("\u0643", "\u06a9")
        .replace("\u200c", "")
        .strip()
    )


def _format_heven(h) -> str:
    try:
        h = str(int(h)).zfill(6)
        return f"{h[0:2]}:{h[2:4]}:{h[4:6]}"
    except (ValueError, TypeError):
        return ""


class TSETMCFetcher:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        self._symbol_cache = {}
        self._request_lock = threading.Lock()
        self._last_request_time = 0.0

    def _wait_for_rate_limit(self) -> None:
        with self._request_lock:
            elapsed = time.monotonic() - self._last_request_time
            if elapsed < REQUEST_INTERVAL_SECONDS:
                time.sleep(REQUEST_INTERVAL_SECONDS - elapsed)
            self._last_request_time = time.monotonic()

    def _get_json(self, url: str, endpoint: str) -> dict:
        last_error = None
        for attempt in range(MAX_RETRIES):
            self._wait_for_rate_limit()
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise TSETMCAPIError(
                        f"{endpoint} returned {type(data).__name__}; expected a JSON object"
                    )
                return data
            except TSETMCAPIError:
                LOGGER.warning("Unexpected TSETMC response schema for %s", endpoint)
                raise
            except (requests.RequestException, ValueError) as error:
                last_error = error
                if attempt + 1 < MAX_RETRIES:
                    time.sleep(0.5 * (2**attempt))

        raise TSETMCAPIError(
            f"TSETMC request failed for {endpoint} after {MAX_RETRIES} attempts: "
            f"{last_error}"
        ) from last_error

    def search_symbol(self, keyword: str) -> list:
        encoded_keyword = quote(keyword, safe="")
        url = (
            "https://cdn.tsetmc.com/api/Instrument/"
            f"GetInstrumentSearch/{encoded_keyword}"
        )
        data = self._get_json(url, "instrument search")
        results = data.get("instrumentSearch")
        if not isinstance(results, list):
            LOGGER.warning("Unexpected TSETMC response schema for instrument search")
            raise TSETMCAPIError(
                "Instrument search response is missing the instrumentSearch list"
            )
        return results

    def get_symbol_id(self, symbol: str) -> str:
        """Resolve a ticker only when an exact normalized match is available."""
        if symbol in self._symbol_cache:
            return self._symbol_cache[symbol]

        results = self.search_symbol(symbol)
        if not all(isinstance(result, dict) for result in results):
            LOGGER.warning("Unexpected TSETMC response schema for symbol search rows")
            raise TSETMCAPIError("Instrument search contains an invalid result row")

        candidates = [
            r
            for r in results
            if normalize_symbol(symbol) == normalize_symbol(str(r.get("lVal18AFC", "")))
        ]

        def _accepted_flow(row: dict) -> int | None:
            flow = row.get("flow")
            if isinstance(flow, bool):
                return None
            if isinstance(flow, int):
                return flow if flow in (1, 2) else None
            if isinstance(flow, float):
                return int(flow) if flow in (1.0, 2.0) else None
            if isinstance(flow, str):
                stripped_flow = flow.strip()
                return int(stripped_flow) if stripped_flow in ("1", "2") else None
            return None

        accepted_matches = [
            (row, flow)
            for row in candidates
            if (flow := _accepted_flow(row)) is not None
        ]
        if not accepted_matches:
            raise SymbolNotFoundError(f"No exact ticker match for '{symbol}'")

        selected_row = next(
            (row for row, flow in accepted_matches if flow == 1),
            accepted_matches[0][0],
        )
        inscode = selected_row.get("insCode")
        if not inscode:
            LOGGER.warning("TSETMC symbol search result is missing insCode")
            raise TSETMCAPIError("Instrument search result is missing insCode")
        self._symbol_cache[symbol] = inscode
        return inscode

    def fetch_daily_history(self, symbol: str) -> pd.DataFrame:
        inscode = self.get_symbol_id(symbol)

        url = f"https://cdn.tsetmc.com/api/ClosingPrice/GetClosingPriceDailyList/{inscode}/0"
        data = self._get_json(url, "daily closing prices")
        rows = data.get("closingPriceDaily")
        if not isinstance(rows, list):
            LOGGER.warning("Unexpected TSETMC response schema for daily closing prices")
            raise TSETMCAPIError(
                "Daily closing-price response is missing the closingPriceDaily list"
            )
        if not rows:
            return pd.DataFrame()
        if not all(isinstance(row, dict) for row in rows):
            LOGGER.warning("Unexpected TSETMC response schema for daily price rows")
            raise TSETMCAPIError("Daily closing-price response contains an invalid row")

        df = pd.DataFrame(rows)
        required_columns = {
            "dEven",
            "pClosing",
            "pDrCotVal",
            "priceFirst",
            "priceMax",
            "priceMin",
            "qTotTran5J",
            "qTotCap",
            "zTotTran",
        }
        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            LOGGER.warning(
                "TSETMC daily price response is missing required fields: %s", missing
            )
            raise TSETMCAPIError(
                f"Daily closing-price response is missing required fields: {missing}"
            )

        numeric_columns = [
            "pClosing",
            "pDrCotVal",
            "priceFirst",
            "priceMax",
            "priceMin",
            "qTotTran5J",
            "qTotCap",
            "zTotTran",
        ]
        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        if df[numeric_columns].isna().all(axis=None):
            LOGGER.warning("TSETMC daily price response contains no numeric data")
            raise TSETMCAPIError(
                "Daily closing-price response contains no usable numeric data"
            )

        df_cleaned = pd.DataFrame(
            {
                "date": df["dEven"].astype(str),
                "last_update_time": (
                    df["hEven"].apply(_format_heven) if "hEven" in df.columns else ""
                ),
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
        # Sort by date and time first, so "keep last" really keeps the latest row.
        df_cleaned = df_cleaned.sort_values(["date", "last_update_time"]).drop_duplicates(
            subset="date", keep="last"
        )

        # A zero or missing low/high makes the true range (and therefore ATR and the
        # stop distance) meaningless, so such rows are removed rather than analysed.
        price_frame = df_cleaned[["close_price", "high_price", "low_price"]]
        unusable = ~(price_frame > 0).all(axis=1)  # NaN > 0 is False
        if unusable.any():
            LOGGER.info(
                "Dropped %d TSETMC rows with missing or non-positive prices",
                int(unusable.sum()),
            )
            df_cleaned = df_cleaned[~unusable]

        return df_cleaned.sort_values("date").reset_index(drop=True)
