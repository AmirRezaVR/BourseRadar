import sys

from PySide6.QtCore import Qt, QThread, Signal
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

from src.tsetmc_fetcher import TSETMCFetcher
from src.signal_engine import SignalEngine
from src.analysis_service import analyze_symbol_core, parse_symbols
from src.app_settings import (
    FONT_SIZES,
    FONT_LABELS,
    DEFAULT_FONT,
    load_settings,
    save_settings,
)

COLUMNS = [
    "نماد",
    "نتیجه",
    "اطمینان",
    "قیمت",
    "محدوده ورود",
    "حد ضرر",
    "هدف",
    "ریسک به پاداش",
    "آخرین بروزرسانی بازار",
]


def apply_font(key: str) -> None:
    app = QApplication.instance()
    font = app.font()
    font.setPointSize(FONT_SIZES.get(key, FONT_SIZES[DEFAULT_FONT]))
    app.setFont(font)


class AnalysisWorker(QThread):
    """Runs the analysis in a background thread so the window stays responsive."""

    row_ready = Signal(list)
    finished_all = Signal()

    def __init__(self, symbols: list):
        super().__init__()
        self.symbols = symbols

    def run(self):
        fetcher = TSETMCFetcher()
        engine = SignalEngine()
        for symbol in self.symbols:
            try:
                result = analyze_symbol_core(symbol, fetcher, engine)
                row = self._to_row(symbol, result)
            except Exception as e:
                row = [symbol, f"خطا: {e}"] + [""] * (len(COLUMNS) - 2)
            self.row_ready.emit(row)
        self.finished_all.emit()

    @staticmethod
    def _to_row(symbol: str, result) -> list:
        empty = [""] * (len(COLUMNS) - 2)

        if not result.success:
            if result.error == "no_data":
                text = "داده‌ای دریافت نشد"
            elif result.error == "insufficient_history":
                text = f"داده کافی نیست ({result.error_detail} روز)"
            else:
                text = result.error_detail or "خطا"
            return [symbol, text] + empty

        rec = result.recommendation
        return [
            symbol,
            rec.verdict_fa,
            f"{rec.confidence:.0f}٪",
            f"{rec.current_price:,.0f}",
            f"{rec.entry_low:,.0f} تا {rec.entry_high:,.0f}",
            f"{rec.stop_loss:,.0f}",
            f"{rec.target:,.0f}",
            f"۱ به {rec.risk_reward_ratio}",
            result.market_update_fa or "",
        ]


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BourseRadar")
        self.worker = None
        self.settings = load_settings()

        # ----- ورودی نماد -----
        self.input = QLineEdit()
        self.input.setPlaceholderText(
            "نماد(ها) را با فاصله وارد کنید، مثال: فملی فولاد خودرو"
        )
        self.input.returnPressed.connect(self.start_analysis)

        self.button = QPushButton("تحلیل")
        self.button.clicked.connect(self.start_analysis)

        top = QHBoxLayout()
        top.addWidget(self.input, 1)
        top.addWidget(self.button)

        # ----- اندازه فونت -----
        self.font_box = QComboBox()
        for key in FONT_SIZES:
            self.font_box.addItem(FONT_LABELS[key], key)
        current = self.settings.get("font_size", DEFAULT_FONT)
        if current not in FONT_SIZES:
            current = DEFAULT_FONT
        self.font_box.setCurrentIndex(list(FONT_SIZES).index(current))
        self.font_box.currentIndexChanged.connect(self.change_font)

        font_row = QHBoxLayout()
        font_row.addWidget(QLabel("اندازه فونت:"))
        font_row.addWidget(self.font_box)
        font_row.addStretch()

        # ----- جدول نتایج -----
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)

        self.status = QLabel("")

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(font_row)
        layout.addWidget(self.table)
        layout.addWidget(self.status)

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

        self.table.setRowCount(0)
        self.button.setEnabled(False)
        self.status.setText(f"در حال تحلیل {len(symbols)} نماد...")

        self.worker = AnalysisWorker(symbols)
        self.worker.row_ready.connect(self.add_row)
        self.worker.finished_all.connect(self.on_done)
        self.worker.start()

    def add_row(self, values: list):
        row = self.table.rowCount()
        self.table.insertRow(row)
        for col, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, col, item)

    def on_done(self):
        self.button.setEnabled(True)
        self.status.setText("تحلیل کامل شد.")

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.wait()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
    apply_font(load_settings().get("font_size", DEFAULT_FONT))

    window = MainWindow()
    window.resize(1100, 600)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
