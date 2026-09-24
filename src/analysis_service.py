from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd

from src.tsetmc_fetcher import TSETMCFetcher
from src.indicators import TechnicalIndicators
from src.signal_engine import SignalEngine, MIN_HISTORY_DAYS

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

SEPARATORS = ["،", ";", "؛", ","]


def parse_symbols(raw: str) -> list:
    """Split raw user input into symbols. Space is the primary separator."""
    normalized = raw
    for sep in SEPARATORS:
        normalized = normalized.replace(sep, " ")
    parts = [p.strip() for p in normalized.split() if p.strip()]
    seen, unique_parts = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p)
            unique_parts.append(p)
    return unique_parts


def format_market_timestamp(date_str: str, time_str: str, lang: str) -> str:
    """
    Turns TSETMC's own date (dEven) + time (hEven) into something readable,
    e.g. "Wednesday 2026-08-12 12:29" - this is when the price was actually
    last updated on the exchange.
    """
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


def analyze_symbol_core(
    symbol: str, fetcher: TSETMCFetcher, engine: SignalEngine
) -> AnalysisResult:
    df = fetcher.fetch_daily_history(symbol)
    if df.empty:
        return AnalysisResult(symbol=symbol, success=False, error="no_data")

    df_analyzed = TechnicalIndicators.apply_all(df)

    latest_date = df_analyzed.iloc[-1]["date"]
    latest_time = df_analyzed.iloc[-1].get("last_update_time", "")
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
        )

    order_book_raw = fetcher.fetch_order_book(symbol)
    money_flow_raw = fetcher.fetch_client_type(symbol, deven=latest_date)

    try:
        rec = engine.generate(
            df_analyzed,
            symbol=symbol,
            order_book_raw=order_book_raw,
            money_flow_raw=money_flow_raw,
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
        )

    return AnalysisResult(
        symbol=symbol,
        success=True,
        df=df_analyzed,
        recommendation=rec,
        market_update_en=market_update_en,
        market_update_fa=market_update_fa,
    )
