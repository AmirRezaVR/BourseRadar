"""
BourseRadar — simple desktop window around the existing analysis code.

Why a window instead of the terminal?
Terminals can't change their own font size, and most of them break Persian
(letters not joined, RTL/LTR text mixed up). Qt renders Persian correctly,
so this window reuses the functions from main.py and just shows the output.

Run from source:   python gui.py
"""
import contextlib
import html
import os
import re
import sys
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QThread, Signal
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import (
    QApplication, QComboBox, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QTextBrowser, QVBoxLayout, QWidget,
)

# Existing project code (unchanged).
from main import analyze_symbol, parse_symbols
from src.database import DatabaseManager
from src.signal_engine import VERDICT_ICONS, SignalEngine
from src.tsetmc_fetcher import TSETMCFetcher

FROZEN = getattr(sys, "frozen", False)

# Text sizes in points: regular / large / extra large
SIZES = {"regular": 12, "large": 15, "xlarge": 19}

FALLBACK_FONTS = ["Segoe UI", "Tahoma", "Noto Sans Arabic", "Noto Naskh Arabic",
                  "DejaVu Sans", "Arial"]

TEXTS = {
    "fa": {
        "language": "زبان",
        "size": "اندازه متن",
        "sizes": {"regular": "معمولی", "large": "بزرگ", "xlarge": "خیلی بزرگ"},
        "placeholder": "نماد یا چند نماد با فاصله، مثال: فملی فولاد خودرو",
        "analyze": "تحلیل",
        "clear": "پاک کردن",
        "busy": "در حال تحلیل…",
        "empty": "نمادی وارد نشد.",
        "batch": "📦 حالت گروهی: {n} نماد در صف",
        "error": "❌ خطا: {e}",
        "summary": "📋 خلاصه گروهی",
        "columns": ("نماد", "نتیجه", "اطمینان", "محدوده ورود"),
        "no_data": "بدون داده",
        "pct": "{v:.0f}٪",
        "range": "{lo:,.0f} تا {hi:,.0f}",
        "hint": "یک نماد یا چند نماد را با فاصله جدا کنید و روی «تحلیل» بزنید.",
    },
    "en": {
        "language": "Language",
        "size": "Text size",
        "sizes": {"regular": "Regular", "large": "Large", "xlarge": "Extra large"},
        "placeholder": "One or more symbols separated by spaces, e.g. فملی فولاد خودرو",
        "analyze": "Analyze",
        "clear": "Clear",
        "busy": "Analyzing…",
        "empty": "No symbol entered.",
        "batch": "📦 Batch mode: {n} symbols queued",
        "error": "❌ Error: {e}",
        "summary": "📋 Batch Summary",
        "columns": ("Symbol", "Verdict", "Confidence", "Entry zone"),
        "no_data": "N/A",
        "pct": "{v:.0f}%",
        "range": "{lo:,.0f}–{hi:,.0f}",
        "hint": "Enter one or more symbols separated by spaces, then press Analyze.",
    },
}

STYLE = """
QWidget { background: #F5F6F8; color: #1F2933; }
QLineEdit, QComboBox {
    background: #FFFFFF; border: 1px solid #C9CED6; border-radius: 6px; padding: 6px 10px;
}
QLineEdit:focus, QComboBox:focus { border-color: #2F5D8C; }
QPushButton {
    background: #2F5D8C; color: #FFFFFF; border: none; border-radius: 6px; padding: 7px 20px;
}
QPushButton:hover { background: #274E78; }
QPushButton:disabled { background: #9DB2C8; }
QPushButton#secondary { background: #E3E7EC; color: #1F2933; }
QPushButton#secondary:hover { background: #D6DBE2; }
QTextBrowser {
    background: #FFFFFF; border: 1px solid #D8DCE2; border-radius: 8px; padding: 10px;
}
"""

DOC_CSS = "p { margin: 3px 0; }"

# A thin divider. (<hr> makes Qt draw a line under every following paragraph.)
RULE_HTML = '<p style="margin:6px 0; font-size:2px; background-color:#D8DCE2;">&nbsp;</p>'

HEADING_RE = re.compile(r"^(📊|📌|📋|🚀|📈|\[\d+/\d+\])")
VERDICT_RE = re.compile(r"^(Verdict:|نتیجه:)")
RULE_RE = re.compile(r"[=\-#]{10,}")


def resource_path(rel: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / rel


def setup_data_dir() -> None:
    """
    In the packaged app, put the database in a folder we can write to
    (next to the program, or in the home folder as a fallback), instead of
    wherever the program happened to be launched from.
    """
    if not FROZEN:
        return
    here = Path(sys.executable).resolve().parent
    candidates = [here / "BourseRadar_data", Path.home() / "BourseRadar_data"]
    if sys.platform == "darwin":  # inside an .app bundle, prefer the home folder
        candidates.reverse()
    for folder in candidates:
        try:
            folder.mkdir(parents=True, exist_ok=True)
            probe = folder / ".write_test"
            probe.write_text("ok")
            probe.unlink()
            os.chdir(folder)
            return
        except OSError:
            continue


def format_line(raw: str, rtl: bool) -> str:
    """Turn one line of the terminal output into clean HTML for the window."""
    line = raw.rstrip()
    if not line.strip():
        return '<p style="margin:0; font-size:6pt;">&nbsp;</p>'
    stripped = line.strip()
    if RULE_RE.fullmatch(stripped):
        return RULE_HTML

    indented = len(line) - len(line.lstrip()) > 0
    text = re.sub(r" {2,}", " ", stripped)          # drop terminal column padding
    text = text.replace("->", "←" if rtl else "→")  # '->' flips badly in RTL text
    text = html.escape(text)

    if HEADING_RE.match(stripped) or VERDICT_RE.match(stripped):
        text = f"<b>{text}</b>"

    side = "right" if rtl else "left"
    margin = f"margin-{side}:18px;" if indented else ""
    direction, align = ("rtl", "right") if rtl else ("ltr", "left")
    if rtl:
        text = "\u200f" + text  # right-to-left mark: keeps RTL even if the line starts with Latin text
    return f'<p dir="{direction}" align="{align}" style="{margin}">{text}</p>'


def summary_html(rows: list, columns: tuple, rtl: bool) -> str:
    """Batch summary as a real table (columns stay aligned in any language)."""
    align = "right" if rtl else "left"
    columns = list(columns)
    rows = [list(r) for r in rows]
    if rtl:  # Qt lays tables out left-to-right, so flip the columns for Persian
        columns.reverse()
        for r in rows:
            r.reverse()
    head = "".join(
        f'<th align="{align}" bgcolor="#EEF1F5">{html.escape(c)}</th>' for c in columns
    )
    body = ""
    for row in rows:
        cells = "".join(f'<td align="{align}">{html.escape(c)}</td>' for c in row)
        body += f"<tr>{cells}</tr>"
    return ('<table width="100%" cellspacing="0" cellpadding="7">'
            f"<tr>{head}</tr>{body}</table>")


class LineWriter:
    """File-like object that hands each finished line to a callback."""

    def __init__(self, emit):
        self._emit = emit
        self._buf = ""

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._emit(line)
        return len(s)

    def flush(self) -> None:
        if self._buf:
            self._emit(self._buf)
            self._buf = ""


class Worker(QThread):
    line = Signal(str)
    summary = Signal(list)

    def __init__(self, symbols: list, lang: str):
        super().__init__()
        self.symbols = symbols
        self.lang = lang

    def summary_rows(self, results: list) -> list:
        t, rows = TEXTS[self.lang], []
        for symbol, rec in results:
            if rec is None:
                rows.append((symbol, t["no_data"], "-", "-"))
                continue
            icon = VERDICT_ICONS.get(rec.verdict, "")
            verdict = rec.verdict_fa if self.lang == "fa" else rec.verdict_en
            rows.append((
                symbol,
                f"{icon} {verdict}".strip(),
                t["pct"].format(v=rec.confidence),
                t["range"].format(lo=rec.entry_low, hi=rec.entry_high),
            ))
        return rows

    def run(self) -> None:
        writer = LineWriter(self.line.emit)
        t = TEXTS[self.lang]
        try:
            with contextlib.redirect_stdout(writer):
                fetcher, db, engine = TSETMCFetcher(), DatabaseManager(), SignalEngine()
                total = len(self.symbols)
                if total > 1:
                    print(t["batch"].format(n=total))
                results = []
                for i, symbol in enumerate(self.symbols, 1):
                    if total > 1:
                        print(f"\n[{i}/{total}] {symbol}")
                    rec = analyze_symbol(symbol, fetcher, db, engine, self.lang)
                    results.append((symbol, rec))
                if total > 1:
                    self.summary.emit(self.summary_rows(results))
        except Exception as e:  # network errors etc. — show them, don't crash
            self.line.emit(t["error"].format(e=e))
        finally:
            writer.flush()


class Window(QWidget):
    def __init__(self, app: QApplication, family: str):
        super().__init__()
        self.app = app
        self.family = family
        self.worker = None
        self.showing_hint = False
        self.settings = QSettings("BourseRadar", "BourseRadar")
        self.lang = self.settings.value("lang", "fa")
        self.size_key = self.settings.value("size", "regular")
        if self.lang not in TEXTS:
            self.lang = "fa"
        if self.size_key not in SIZES:
            self.size_key = "regular"

        self.setWindowTitle("BourseRadar")
        self.resize(820, 700)

        self.lang_label, self.size_label = QLabel(), QLabel()
        self.lang_box, self.size_box = QComboBox(), QComboBox()
        self.lang_box.addItem("فارسی", "fa")
        self.lang_box.addItem("English", "en")
        for key in SIZES:
            self.size_box.addItem("", key)

        self.input = QLineEdit()
        self.run_btn = QPushButton()
        self.clear_btn = QPushButton()
        self.clear_btn.setObjectName("secondary")
        self.out = QTextBrowser()
        self.out.document().setDefaultStyleSheet(DOC_CSS)
        self.out.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        top = QHBoxLayout()
        for w in (self.lang_label, self.lang_box, self.size_label, self.size_box):
            top.addWidget(w)
        top.addStretch(1)

        row = QHBoxLayout()
        row.addWidget(self.input, 1)
        row.addWidget(self.run_btn)
        row.addWidget(self.clear_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(10)
        layout.addLayout(top)
        layout.addLayout(row)
        layout.addWidget(self.out, 1)

        self.lang_box.setCurrentIndex(self.lang_box.findData(self.lang))
        self.size_box.setCurrentIndex(self.size_box.findData(self.size_key))

        self.lang_box.currentIndexChanged.connect(self.on_language)
        self.size_box.currentIndexChanged.connect(self.on_size)
        self.run_btn.clicked.connect(self.start)
        self.input.returnPressed.connect(self.start)
        self.clear_btn.clicked.connect(self.clear)

        self.apply_size()
        self.apply_language()

    # ----- settings ---------------------------------------------------
    def apply_size(self) -> None:
        font = QFont(self.family, SIZES[self.size_key])
        if self.family == "":
            font.setFamilies(FALLBACK_FONTS)
        self.app.setFont(font)
        self.setFont(font)       # app-level alone doesn't reach widgets that already exist
        for child in self.findChildren(QWidget):  # labels, boxes, buttons: set explicitly
            child.setFont(font)

    def apply_language(self) -> None:
        t = TEXTS[self.lang]
        rtl = self.lang == "fa"
        self.app.setLayoutDirection(Qt.RightToLeft if rtl else Qt.LeftToRight)
        self.lang_label.setText(t["language"])
        self.size_label.setText(t["size"])
        for i, key in enumerate(SIZES):
            self.size_box.setItemText(i, t["sizes"][key])
        self.input.setPlaceholderText(t["placeholder"])
        self.run_btn.setText(t["busy"] if self.worker else t["analyze"])
        self.clear_btn.setText(t["clear"])
        if not self.worker and (self.showing_hint or not self.out.toPlainText().strip()):
            self.show_hint()

    def on_language(self) -> None:
        self.lang = self.lang_box.currentData()
        self.settings.setValue("lang", self.lang)
        self.apply_language()

    def on_size(self) -> None:
        self.size_key = self.size_box.currentData()
        self.settings.setValue("size", self.size_key)
        self.apply_size()

    # ----- output -----------------------------------------------------
    def show_hint(self) -> None:
        self.out.clear()
        self.out.append(format_line(TEXTS[self.lang]["hint"], rtl=self.lang == "fa"))
        self.showing_hint = True

    def append_line(self, raw: str) -> None:
        if self.showing_hint:
            self.out.clear()
            self.showing_hint = False
        self.out.append(format_line(raw, rtl=self.lang == "fa"))
        bar = self.out.verticalScrollBar()
        bar.setValue(bar.maximum())

    def append_summary(self, rows: list) -> None:
        t, rtl = TEXTS[self.lang], self.lang == "fa"
        self.append_line("")
        self.append_line(t["summary"])
        self.out.append(summary_html(rows, t["columns"], rtl))
        bar = self.out.verticalScrollBar()
        bar.setValue(bar.maximum())

    def clear(self) -> None:
        self.show_hint()

    # ----- running ----------------------------------------------------
    def start(self) -> None:
        if self.worker:
            return
        symbols = parse_symbols(self.input.text().strip())
        if not symbols:
            self.append_line(TEXTS[self.lang]["empty"])
            return
        if self.showing_hint:
            self.out.clear()
            self.showing_hint = False
        self.run_btn.setEnabled(False)
        self.worker = Worker(symbols, self.lang)
        self.worker.line.connect(self.append_line)
        self.worker.summary.connect(self.append_summary)
        self.worker.finished.connect(self.finished)
        self.apply_language()
        self.worker.start()

    def finished(self) -> None:
        self.worker = None
        self.run_btn.setEnabled(True)
        self.apply_language()


def load_font() -> str:
    """Use the bundled Vazirmatn font (clean Persian) if present; else system fonts."""
    path = resource_path("fonts/Vazirmatn-Regular.ttf")
    if path.exists():
        font_id = QFontDatabase.addApplicationFont(str(path))
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            return families[0]
    return ""


def main() -> None:
    setup_data_dir()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = Window(app, load_font())
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
