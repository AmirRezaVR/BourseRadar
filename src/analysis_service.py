"""
Shared analysis pipeline: fetch -> indicators -> order book -> money flow ->
signal engine. Kept separate from main.py so the CLI's presentation logic
(printing) stays cleanly separated from the actual business logic.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd

from src.tsetmc_fetcher import TSETMCFetcher
from src.database import DatabaseManager
from src.indicators import TechnicalIndicators
from src.signal_engine import SignalEngine, MIN_HISTORY_DAYS

try:
    from zoneinfo import ZoneInfo
    IRAN_TZ = ZoneInfo("Asia/Tehran")
except Exception:
    # zoneinfo needs the 'tzdata' package on some systems (notably Windows).
    IRAN_TZ = None


def get_fetch_timestamp() -> str:
    """Exact moment data was pulled, in Iran time when possible."""
    now = datetime.now(IRAN_TZ) if IRAN_TZ else datetime.now()
    formatted = now.strftime("%Y-%m-%d %H:%M")
    return formatted if IRAN_TZ else f"{formatted} (local time, not Iran time)"


def iran_time_available() -> bool:
    return IRAN_TZ is not None


SEPARATORS = ["،", ";", "؛", ","]  # accepted alongside the primary space separator


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


@dataclass
class AnalysisResult:
    symbol: str
    success: bool
    error: Optional[str] = None
    error_detail: Optional[str] = None
    df: Optional[pd.DataFrame] = None
    recommendation: Optional[object] = None
    fetch_ts: Optional[str] = None


def analyze_symbol_core(symbol: str, fetcher: TSETMCFetcher, db: DatabaseManager,
                         engine: SignalEngine) -> AnalysisResult:
    """Runs the full pipeline for one symbol. Never raises - failures come back as AnalysisResult(success=False, ...)."""
    df = fetcher.fetch_daily_history(symbol)
    if df.empty:
        return AnalysisResult(symbol=symbol, success=False, error="no_data")

    fetch_ts = get_fetch_timestamp()
    db.save_history(symbol, df)
    df_analyzed = TechnicalIndicators.apply_all(df)

    if len(df_analyzed) < MIN_HISTORY_DAYS:
        return AnalysisResult(
            symbol=symbol, success=False, error="insufficient_history",
            error_detail=f"{len(df_analyzed)}/{MIN_HISTORY_DAYS}", df=df_analyzed, fetch_ts=fetch_ts,
        )

    order_book_raw = fetcher.fetch_order_book(symbol)
    latest_date = df_analyzed.iloc[-1]["date"]
    money_flow_raw = fetcher.fetch_client_type(symbol, deven=latest_date)

    try:
        rec = engine.generate(df_analyzed, symbol=symbol, order_book_raw=order_book_raw, money_flow_raw=money_flow_raw)
    except ValueError as e:
        return AnalysisResult(symbol=symbol, success=False, error="engine_error", error_detail=str(e),
                               df=df_analyzed, fetch_ts=fetch_ts)

    return AnalysisResult(symbol=symbol, success=True, df=df_analyzed, recommendation=rec, fetch_ts=fetch_ts)
