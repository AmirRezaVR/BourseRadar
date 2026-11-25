import logging
import math
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime
from urllib.parse import quote
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)
REQUEST_INTERVAL_SECONDS = 0.25
MAX_RETRIES = 3
TEHRAN_TZ = ZoneInfo("Asia/Tehran")


@dataclass(frozen=True)
class LiveQuote:
    price: float
    trade_date: date
    trade_time: str


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


def _format_heven(h) -> str | None:
    try:
        if isinstance(h, (float, np.floating)) and not math.isfinite(float(h)):
            return None
        value = int(h)
        if isinstance(h, float) and not h.is_integer():
            return None
        formatted = str(value).zfill(6)
        if len(formatted) != 6:
            return None
        parsed = datetime.strptime(formatted, "%H%M%S")
        return parsed.strftime("%H:%M:%S")
    except (ValueError, TypeError, OverflowError):
        return None


class TSETMCFetcher:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        self._symbol_cache = {}
        self._live_quote_cache = {}
        self._live_quote_cache_lock = threading.Lock()
        self._live_quote_request_locks = {}
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

    def _clean_history(self, df_cleaned: pd.DataFrame) -> pd.DataFrame:
        parsed_dates = pd.to_datetime(
            df_cleaned["date"], format="%Y%m%d", errors="coerce"
        )
        valid_dates = parsed_dates.notna() & (
            parsed_dates.dt.date <= datetime.now(TEHRAN_TZ).date()
        )
        df_cleaned = df_cleaned.loc[valid_dates].copy()
        df_cleaned["date"] = df_cleaned["date"].astype(str)

        # Missing times remain blank as before; malformed nonempty times cannot win
        # duplicate selection over a valid exchange update.
        df_cleaned = df_cleaned.loc[df_cleaned["last_update_time"].notna()].copy()

        price_columns = [
            "open_price",
            "high_price",
            "low_price",
            "close_price",
            "last_price",
        ]
        finite_prices = np.isfinite(df_cleaned[price_columns]).all(axis=1)
        df_cleaned = df_cleaned.loc[finite_prices].copy()

        # Sort by date and time first, so "keep last" really keeps the latest row.
        df_cleaned = df_cleaned.sort_values(
            ["date", "last_update_time"]
        ).drop_duplicates(subset="date", keep="last")

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

        columns = [
            "date",
            "last_update_time",
            "close_price",
            "last_price",
            "open_price",
            "high_price",
            "low_price",
            "volume",
            "value",
            "count",
        ]
        return df_cleaned[columns].sort_values("date").reset_index(drop=True)

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
                    df["hEven"].apply(
                        lambda value: (
                            ""
                            if pd.isna(value)
                            or (isinstance(value, str) and not value.strip())
                            else _format_heven(value)
                        )
                    )
                    if "hEven" in df.columns
                    else ""
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
        return self._clean_history(df_cleaned)

    def fetch_adjusted_history(self, ins_code: str) -> pd.DataFrame:
        if not isinstance(ins_code, str) or not ins_code:
            return pd.DataFrame()

        url = (
            "https://members.tsetmc.com/tsev2/chart/data/Financial.aspx"
            f"?i={quote(ins_code, safe='')}&t=ph&a=1"
        )
        self._wait_for_rate_limit()
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
        except requests.RequestException as error:
            raise TSETMCAPIError(
                f"TSETMC request failed for adjusted price history: {error}"
            ) from error

        if not isinstance(response.text, str):
            raise TSETMCAPIError("Adjusted price history response is not text")
        rows = []
        for record in response.text.split(";"):
            record = record.strip()
            if not record:
                continue
            fields = record.split(",")
            if len(fields) != 7:
                raise TSETMCAPIError(
                    "Adjusted price history contains an invalid row"
                )
            rows.append(fields)
        if not rows:
            return pd.DataFrame()

        adjusted = pd.DataFrame(
            rows,
            columns=[
                "date",
                "high_price",
                "low_price",
                "open_price",
                "last_price",
                "volume",
                "close_price",
            ],
        )
        for column in [
            "high_price",
            "low_price",
            "open_price",
            "last_price",
            "volume",
            "close_price",
        ]:
            adjusted[column] = pd.to_numeric(adjusted[column], errors="coerce")
        adjusted["last_update_time"] = ""
        adjusted["value"] = np.nan
        adjusted["count"] = np.nan
        return self._clean_history(adjusted)

    def get_live_quote(self, ins_code: str) -> LiveQuote | None:
        if not isinstance(ins_code, str) or not ins_code:
            return None

        with self._live_quote_cache_lock:
            now = time.monotonic()
            cached = self._live_quote_cache.get(ins_code)
            if cached is not None and cached[0] > now:
                return cached[1]
            request_lock = self._live_quote_request_locks.setdefault(
                ins_code, threading.Lock()
            )

        with request_lock:
            with self._live_quote_cache_lock:
                now = time.monotonic()
                cached = self._live_quote_cache.get(ins_code)
                if cached is not None and cached[0] > now:
                    return cached[1]

            quote = None
            url = (
                "https://cdn.tsetmc.com/api/ClosingPrice/"
                f"GetClosingPriceInfo/{ins_code}"
            )
            try:
                response = requests.get(url, headers=self.headers, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    record = (
                        data.get("closingPriceInfo")
                        if isinstance(data, dict)
                        else None
                    )
                    if isinstance(record, dict):
                        raw_price = record.get("pDrCotVal")
                        raw_date = record.get("dEven")
                        raw_time = record.get("hEven")

                        if not isinstance(raw_price, bool):
                            try:
                                price = float(raw_price)
                            except (TypeError, ValueError, OverflowError):
                                price = float("nan")
                        else:
                            price = float("nan")

                        if isinstance(raw_date, int) and not isinstance(raw_date, bool):
                            date_text = str(raw_date)
                        elif isinstance(raw_date, str):
                            date_text = raw_date
                        else:
                            date_text = ""
                        try:
                            trade_date = (
                                datetime.strptime(date_text, "%Y%m%d").date()
                                if len(date_text) == 8 and date_text.isdigit()
                                else None
                            )
                        except ValueError:
                            trade_date = None

                        if isinstance(raw_time, int) and not isinstance(raw_time, bool):
                            time_text = str(raw_time).zfill(6)
                        elif isinstance(raw_time, str) and raw_time.isdigit():
                            time_text = raw_time.zfill(6)
                        else:
                            time_text = ""
                        try:
                            parsed_time = (
                                datetime.strptime(time_text, "%H%M%S")
                                if len(time_text) == 6 and time_text.isdigit()
                                else None
                            )
                        except ValueError:
                            parsed_time = None

                        if (
                            math.isfinite(price)
                            and price > 0
                            and trade_date is not None
                            and trade_date <= datetime.now(TEHRAN_TZ).date()
                            and parsed_time is not None
                            and datetime.strptime("090000", "%H%M%S").time()
                            <= parsed_time.time()
                            <= datetime.strptime("123000", "%H%M%S").time()
                        ):
                            quote = LiveQuote(
                                price=price,
                                trade_date=trade_date,
                                trade_time=parsed_time.strftime("%H:%M:%S"),
                            )
            except (requests.RequestException, ValueError, TypeError, OverflowError):
                quote = None

            with self._live_quote_cache_lock:
                cached = self._live_quote_cache.get(ins_code)
                cached_quote = cached[1] if cached is not None else None
                if cached_quote is not None and (
                    quote is None
                    or (cached_quote.trade_date, cached_quote.trade_time)
                    > (quote.trade_date, quote.trade_time)
                ):
                    quote = cached_quote
                self._live_quote_cache[ins_code] = (time.monotonic() + 5, quote)
            return quote
