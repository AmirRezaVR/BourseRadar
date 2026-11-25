from dataclasses import dataclass
from datetime import date, datetime
import logging
import math
from typing import Optional

import pandas as pd

from src.tsetmc_fetcher import (
    SymbolNotFoundError,
    TSETMCAPIError,
    TSETMCFetcher,
    LiveQuote,
    TEHRAN_TZ,
    normalize_symbol,
)
from src.indicators import TechnicalIndicators
from src.signal_engine import (
    InsufficientHistoryError,
    MIN_HISTORY_DAYS,
    SignalEngine,
)

WEEKDAYS = {
    "en": {
        "Saturday": "Saturday",
        "Sunday": "Sunday",
        "Monday": "Monday",
        "Tuesday": "Tuesday",
        "Wednesday": "Wednesday",
        "Thursday": "Thursday",
        "Friday": "Friday",
    },
    "fa": {
        "Saturday": "شنبه",
        "Sunday": "یکشنبه",
        "Monday": "دوشنبه",
        "Tuesday": "سه‌شنبه",
        "Wednesday": "چهارشنبه",
        "Thursday": "پنجشنبه",
        "Friday": "جمعه",
    },
}
LOGGER = logging.getLogger(__name__)

SEPARATORS = ["،", ";", "؛", ","]


def parse_symbols(raw: str) -> list:
    normalized = raw
    for sep in SEPARATORS:
        normalized = normalized.replace(sep, " ")
    parts = [normalize_symbol(p) for p in normalized.split() if p.strip()]
    seen, unique_parts = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p)
            unique_parts.append(p)
    return unique_parts


def format_market_timestamp(date_str: str, time_str: str, lang: str) -> str:
    """Format TSETMC's exchange update time, not the local fetch time."""
    try:
        dt = datetime.strptime(date_str, "%Y%m%d")
    except (ValueError, TypeError):
        return date_str or ""

    weekday_en = dt.strftime("%A")
    weekday = WEEKDAYS.get(lang, WEEKDAYS["en"]).get(weekday_en, weekday_en)
    date_part = dt.strftime("%Y-%m-%d")

    if time_str:
        return f"{weekday} {date_part} {time_str[:5]}"
    return f"{weekday} {date_part}"


@dataclass
class AnalysisResult:
    symbol: str
    success: bool
    error: Optional[str] = None
    error_detail: Optional[str] = None
    df: Optional[pd.DataFrame] = None
    recommendation: Optional[object] = None
    market_update_en: Optional[str] = None
    market_update_fa: Optional[str] = None
    warnings_en: Optional[list] = None
    warnings_fa: Optional[list] = None


JUMP_THRESHOLD = 0.12


def _adjacent_jump_mask(df: pd.DataFrame) -> pd.Series:
    if "date" not in df.columns:
        return pd.Series(False, index=df.index)

    dates = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    calendar_gaps = dates.diff().dt.days
    skipped_gaps = calendar_gaps.gt(4)
    if skipped_gaps.any():
        LOGGER.debug(
            "Skipped %d price comparisons across history gaps",
            int(skipped_gaps.sum()),
        )
    close = pd.to_numeric(df["close_price"], errors="coerce")
    returns = close.pct_change(fill_method=None)
    return returns.abs().gt(JUMP_THRESHOLD) & calendar_gaps.gt(0) & calendar_gaps.le(4)


def _start_after_last_jump(df: pd.DataFrame) -> pd.DataFrame:
    """Keep history from the latest adjacent-session price discontinuity."""
    jumps = _adjacent_jump_mask(df)
    if not jumps.any():
        return df
    first_index = jumps[jumps].index[-1]
    return df.loc[first_index:].reset_index(drop=True)


def _data_warnings(df: pd.DataFrame) -> tuple:
    warnings_en = []
    warnings_fa = []

    if _adjacent_jump_mask(df).any():
        warnings_en.append(
            "Large price discontinuity detected; corporate-action adjustment "
            "status is unverified."
        )
        warnings_fa.append(
            "پرش بزرگ قیمت دیده شد؛ وضعیت تعدیل بابت رویدادهای شرکتی تأیید نشده است."
        )

    return warnings_en, warnings_fa


def analyze_symbol_core(
    symbol: str, fetcher: TSETMCFetcher, engine: SignalEngine
) -> AnalysisResult:
    try:
        df = pd.DataFrame()
        fetch_adjusted_history = getattr(fetcher, "fetch_adjusted_history", None)
        if callable(fetch_adjusted_history):
            try:
                ins_code = fetcher.get_symbol_id(symbol)
                adjusted_history = fetch_adjusted_history(ins_code)
                if (
                    isinstance(adjusted_history, pd.DataFrame)
                    and not adjusted_history.empty
                ):
                    df = adjusted_history
                    try:
                        raw_history = fetcher.fetch_daily_history(symbol)
                    except TSETMCAPIError as error:
                        LOGGER.warning(
                            "Raw history unavailable for update-time metadata: %s",
                            error,
                        )
                        raw_history = pd.DataFrame()
                    if (
                        not raw_history.empty
                        and "last_update_time" in raw_history.columns
                    ):
                        raw_times = raw_history.drop_duplicates(
                            subset="date", keep="last"
                        ).set_index("date")["last_update_time"]
                        df["last_update_time"] = df["date"].map(raw_times).fillna(
                            df["last_update_time"]
                        )
            except TSETMCAPIError as error:
                LOGGER.warning(
                    "Adjusted history unavailable; using daily history: %s", error
                )
                pass
        if df.empty:
            df = fetcher.fetch_daily_history(symbol)
    except SymbolNotFoundError as e:
        return AnalysisResult(
            symbol=symbol,
            success=False,
            error="symbol_not_found",
            error_detail=str(e),
        )
    except TSETMCAPIError as e:
        return AnalysisResult(
            symbol=symbol,
            success=False,
            error="data_source_error",
            error_detail=str(e),
        )
    if df.empty:
        return AnalysisResult(symbol=symbol, success=False, error="no_data")

    analysis_datetime = datetime.now(TEHRAN_TZ)
    df = df.sort_values(
        ["date", "last_update_time"], kind="mergesort"
    ).reset_index(drop=True)
    history_last = df.iloc[-1]
    latest_date = str(history_last["date"])
    latest_time = history_last.get("last_update_time", "")

    live_quote = None
    get_live_quote = getattr(fetcher, "get_live_quote", None)
    if callable(get_live_quote):
        try:
            ins_code = fetcher.get_symbol_id(symbol)
            candidate = get_live_quote(ins_code)
            if isinstance(candidate, LiveQuote):
                try:
                    candidate_price = float(candidate.price)
                    candidate_time = datetime.strptime(
                        candidate.trade_time, "%H:%M:%S"
                    ).strftime("%H:%M:%S")
                    if (
                        math.isfinite(candidate_price)
                        and candidate_price > 0
                        and isinstance(candidate.trade_date, date)
                        and not isinstance(candidate.trade_date, datetime)
                        and candidate.trade_date == analysis_datetime.date()
                        and candidate_time == candidate.trade_time
                    ):
                        live_quote = candidate
                except (TypeError, ValueError, OverflowError):
                    live_quote = None
        except (SymbolNotFoundError, TSETMCAPIError):
            live_quote = None

    chosen_price = history_last.get("last_price")
    live_is_newer_or_equal = False
    if live_quote is not None:
        try:
            history_date = datetime.strptime(latest_date, "%Y%m%d").date()
            history_time_text = str(latest_time)
            try:
                history_time = datetime.strptime(
                    history_time_text, "%H:%M:%S"
                ).time()
            except ValueError:
                history_time = datetime.min.time()
            live_time = datetime.strptime(live_quote.trade_time, "%H:%M:%S").time()
            live_is_newer_or_equal = (live_quote.trade_date, live_time) >= (
                history_date,
                history_time,
            )
        except (TypeError, ValueError):
            live_is_newer_or_equal = False

    if live_is_newer_or_equal:
        latest_date = live_quote.trade_date.strftime("%Y%m%d")
        latest_time = live_quote.trade_time
        chosen_price = live_quote.price

    # Warnings use the full history, so a jump is still reported after it is cut off.
    warnings_en, warnings_fa = _data_warnings(df)
    df_analyzed = TechnicalIndicators.apply_all(_start_after_last_jump(df))
    if live_is_newer_or_equal:
        df_analyzed.attrs["market_update_date"] = latest_date

    market_update_en = format_market_timestamp(latest_date, latest_time, "en")
    market_update_fa = format_market_timestamp(latest_date, latest_time, "fa")

    if len(df_analyzed) < MIN_HISTORY_DAYS:
        return AnalysisResult(
            symbol=symbol,
            success=False,
            error="insufficient_history",
            error_detail=f"{len(df_analyzed)}/{MIN_HISTORY_DAYS}",
            df=df_analyzed,
            market_update_en=market_update_en,
            market_update_fa=market_update_fa,
            warnings_en=warnings_en,
            warnings_fa=warnings_fa,
        )

    try:
        if live_is_newer_or_equal:
            df_analyzed.loc[df_analyzed.index[-1], "last_price"] = chosen_price
        rec = engine.generate(
            df_analyzed,
            symbol=symbol,
            as_of_datetime=analysis_datetime.replace(tzinfo=None),
        )
    except InsufficientHistoryError as e:
        return AnalysisResult(
            symbol=symbol,
            success=False,
            error="insufficient_history",
            error_detail=f"{e.complete_rows}/{e.min_history_days}",
            df=df_analyzed,
            market_update_en=market_update_en,
            market_update_fa=market_update_fa,
            warnings_en=warnings_en,
            warnings_fa=warnings_fa,
        )
    except ValueError as e:
        return AnalysisResult(
            symbol=symbol,
            success=False,
            error="engine_error",
            error_detail=str(e),
            df=df_analyzed,
            market_update_en=market_update_en,
            market_update_fa=market_update_fa,
            warnings_en=warnings_en,
            warnings_fa=warnings_fa,
        )

    return AnalysisResult(
        symbol=symbol,
        success=True,
        df=df_analyzed,
        recommendation=rec,
        market_update_en=market_update_en,
        market_update_fa=market_update_fa,
        warnings_en=warnings_en,
        warnings_fa=warnings_fa,
    )
