import sys
from dataclasses import dataclass, field

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.analysis_service import analyze_symbol_core, parse_symbols
from src.signal_engine import SignalEngine
from src.tsetmc_fetcher import TSETMCFetcher
from gui.settings import (
    DEFAULT_FONT,
    FONT_LABELS,
    FONT_SIZES,
    load_settings,
    save_settings,
)
from gui.theme import ERROR_COLOR, PENDING_COLOR, VERDICT_COLORS, build_stylesheet

COLUMNS = [
    "نماد",
    "نتیجه",
    "قدرت سیگنال",
    "قیمت",
    "محدوده ورود",
    "حد ضرر",
    "هدف",
    "ریسک به پاداش",
    "هشدار داده",
    "آخرین بروزرسانی بازار",
]
WARNING_COLUMN = COLUMNS.index("هشدار داده")

ERROR_TEXT = {
    "no_data": "داده‌ای دریافت نشد",
    "symbol_not_found": "نماد با این نام کوتاه پیدا نشد",
    "data_source_error": "خطا در دریافت داده از TSETMC",
    "insufficient_history": "داده تاریخی کافی نیست",
    "engine_error": "تحلیل انجام نشد",
}

_FA_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def to_fa(text: str) -> str:
    return text.translate(_FA_DIGITS).replace(",", "٬")


def fa_money(value) -> str:
    return "—" if value is None else to_fa(f"{value:,.0f}")


@dataclass
class ResultRow:
    cells: list
    state: str = "pending"
    verdict: str = None
    tooltips: dict = field(default_factory=dict)


def _pending_row(symbol: str) -> ResultRow:
    return ResultRow(cells=[symbol, "در انتظار"] + [""] * (len(COLUMNS) - 2))


def _failed_row(symbol: str, text: str, detail: str = "") -> ResultRow:
    tooltips = {1: detail} if detail else {}
    return ResultRow(
        cells=[symbol, text] + [""] * (len(COLUMNS) - 2),
        state="error",
        tooltips=tooltips,
    )


def build_row(symbol: str, result) -> ResultRow:
    if not result.success:
        if result.error == "insufficient_history":
            text = f"{ERROR_TEXT[result.error]} ({to_fa(result.error_detail)} روز)"
            return _failed_row(symbol, text)
        text = ERROR_TEXT.get(result.error, "خطا")
        return _failed_row(symbol, text, result.error_detail or "")

    rec = result.recommendation

    # Neutral results and invalid setups have no price levels.
    if rec.entry_low is None:
        entry_text = "—"
    else:
        entry_text = f"{fa_money(rec.entry_low)} تا {fa_money(rec.entry_high)}"

    ratio = rec.risk_reward_ratio
    ratio_text = "—" if ratio is None else "۱ به " + to_fa(f"{ratio:.1f}")

    warnings = result.warnings_fa or []
    tooltips = {WARNING_COLUMN: "\n".join(warnings)} if warnings else {}

    cells = [
        symbol,
        rec.verdict_fa,
        to_fa(f"{rec.signal_strength:.0f}") + "٪",
        fa_money(rec.current_price),
        entry_text,
        fa_money(rec.stop_loss),
        fa_money(rec.target),
        ratio_text,
        "؛ ".join(warnings),
        result.market_update_fa or "",
    ]
    return ResultRow(
        cells=cells,
        state="ok",
        verdict=rec.verdict,
        tooltips=tooltips,
    )


class AnalysisWorker(QThread):
    row_ready = Signal(int, object)
    progress = Signal(int, int)
    finished_all = Signal(int, int)

    def __init__(self, symbols: list):
        super().__init__()
        self.symbols = symbols

    def run(self):
        total = len(self.symbols)
        try:
            fetcher = TSETMCFetcher()
            engine = SignalEngine()
        except Exception as error:
            for index, symbol in enumerate(self.symbols):
                self.row_ready.emit(
                    index, _failed_row(symbol, "خطا در شروع تحلیل", str(error))
                )
            self.finished_all.emit(0, total)
            return

        succeeded = failed = 0
        for index, symbol in enumerate(self.symbols):
            if self.isInterruptionRequested():
                break
            try:
                row = build_row(symbol, analyze_symbol_core(symbol, fetcher, engine))
            except Exception as error:
                row = _failed_row(symbol, "خطای غیرمنتظره", str(error))

            if row.state == "ok":
                succeeded += 1
            else:
                failed += 1
            self.row_ready.emit(index, row)
            self.progress.emit(index + 1, total)

        self.finished_all.emit(succeeded, failed)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BourseRadar")
        self.worker = None
        self.settings = load_settings()

        self.input = QLineEdit()
        self.input.setPlaceholderText("نماد کوتاه را وارد کنید، مثال: فملی فولاد خودرو")
        self.input.returnPressed.connect(self.start_analysis)

        self.button = QPushButton("تحلیل")
        self.button.clicked.connect(self.start_analysis)

        input_row = QHBoxLayout()
        input_row.setSpacing(10)
        input_row.addWidget(self.input, 1)
        input_row.addWidget(self.button)

        self.font_box = QComboBox()
        for key in FONT_SIZES:
            self.font_box.addItem(FONT_LABELS[key], key)
        current = self.settings.get("font_size", DEFAULT_FONT)
        if current not in FONT_SIZES:
            current = DEFAULT_FONT
        self.font_box.setCurrentIndex(list(FONT_SIZES).index(current))
        self.font_box.currentIndexChanged.connect(self.change_font)

        hint = QLabel(
            "نمادهای کوتاه را با فاصله جدا کنید. نام کامل شرکت پشتیبانی نمی‌شود."
        )
        hint.setObjectName("hint")

        options_row = QHBoxLayout()
        options_row.addWidget(QLabel("اندازه فونت:"))
        options_row.addWidget(self.font_box)
        options_row.addStretch(1)
        options_row.addWidget(hint)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(WARNING_COLUMN, QHeaderView.ResizeMode.Stretch)

        self.status = QLabel("")
        self.status.setObjectName("status")

        disclaimer = QLabel(
            "هشدار: این برنامه فقط برای اطلاع‌رسانی است و توصیه‌ی سرمایه‌گذاری نیست. "
            "مسئولیت تصمیم‌های معاملاتی با شماست."
        )
        disclaimer.setObjectName("disclaimer")
        disclaimer.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 12)
        layout.setSpacing(10)
        layout.addLayout(input_row)
        layout.addLayout(options_row)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.status)
        layout.addWidget(disclaimer)

    def change_font(self, index: int):
        key = self.font_box.itemData(index)
        self.settings["font_size"] = key
        save_settings(self.settings)
        apply_font(key)

    def start_analysis(self):
        if self.worker and self.worker.isRunning():
            return

        symbols = parse_symbols(self.input.text())
        if not symbols:
            return

        self.table.setRowCount(len(symbols))
        for row, symbol in enumerate(symbols):
            self._set_row(row, _pending_row(symbol))

        self.button.setEnabled(False)
        self.status.setText(f"در حال تحلیل ... ۰ از {to_fa(str(len(symbols)))}")

        self.worker = AnalysisWorker(symbols)
        self.worker.row_ready.connect(self._set_row)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_all.connect(self._on_done)
        self.worker.start()

    def _set_row(self, row: int, data: ResultRow):
        for col, text in enumerate(data.cells):
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            tip = data.tooltips.get(col)
            if tip:
                item.setToolTip(tip)

            if data.state == "pending":
                item.setForeground(QBrush(QColor(PENDING_COLOR)))
            elif data.state == "error" and col == 1:
                item.setForeground(QBrush(QColor(ERROR_COLOR)))
            elif data.state == "ok" and col == 1 and data.verdict in VERDICT_COLORS:
                item.setForeground(QBrush(QColor(VERDICT_COLORS[data.verdict])))
                font = item.font()
                font.setBold(True)
                item.setFont(font)

            self.table.setItem(row, col, item)

    def _on_progress(self, done: int, total: int):
        self.status.setText(
            f"در حال تحلیل ... {to_fa(str(done))} از {to_fa(str(total))}"
        )

    def _on_done(self, succeeded: int, failed: int):
        self.button.setEnabled(True)
        self.status.setText(
            f"تحلیل کامل شد: {to_fa(str(succeeded))} موفق، "
            f"{to_fa(str(failed))} ناموفق."
        )

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait()
        event.accept()


def apply_font(key: str) -> None:
    size = FONT_SIZES.get(key, FONT_SIZES[DEFAULT_FONT])
    app = QApplication.instance()
    if isinstance(app, QApplication):
        app.setStyleSheet(build_stylesheet(size))


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
    apply_font(load_settings().get("font_size", DEFAULT_FONT))

    window = MainWindow()
    window.resize(1100, 620)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
