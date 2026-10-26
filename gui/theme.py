PENDING_COLOR = "#7d8898"
ERROR_COLOR = "#f87171"

VERDICT_COLORS = {
    "STRONG_BUY": "#22d3a0",
    "BUY": "#4ade80",
    "NEUTRAL": "#fbbf24",
    "SELL": "#fb7185",
    "STRONG_SELL": "#ff4d4f",
}

STYLESHEET_TEMPLATE = """
QWidget {
    background-color: #161a21;
    color: #f5f7fa;
    font-size: __FONT__pt;
}
QLineEdit {
    background-color: #0f1217;
    border: 1px solid #3d4756;
    border-radius: 6px;
    padding: 8px 10px;
    color: #ffffff;
    selection-background-color: #14b8a6;
}
QLineEdit:focus {
    border: 1px solid #2dd4bf;
}
QPushButton {
    background-color: #0d9488;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 8px 22px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #14b8a6;
}
QPushButton:disabled {
    background-color: #3a4450;
    color: #9aa4b2;
}
QComboBox {
    background-color: #0f1217;
    border: 1px solid #3d4756;
    border-radius: 6px;
    padding: 6px 10px;
    color: #ffffff;
    min-width: 120px;
}
QComboBox QAbstractItemView {
    background-color: #0f1217;
    color: #ffffff;
    selection-background-color: #0d9488;
}
QLabel#status {
    color: #aab4c3;
}
QLabel#hint {
    color: #7d8898;
}
QLabel#disclaimer {
    color: #fbbf24;
    padding: 6px 0;
}
QTableWidget {
    background-color: #1c2129;
    alternate-background-color: #232932;
    gridline-color: #333b47;
    border: 1px solid #333b47;
    border-radius: 6px;
    color: #f5f7fa;
    font-size: __FONT__pt;
}
QTableWidget::item {
    padding: 8px 12px;
}
QTableWidget::item:selected {
    background-color: #0f766e;
    color: #ffffff;
}
QHeaderView::section {
    background-color: #2a313c;
    color: #ffffff;
    padding: 10px 12px;
    border: none;
    border-bottom: 2px solid #14b8a6;
    font-weight: bold;
    font-size: __FONT__pt;
}
"""


def build_stylesheet(point_size: int) -> str:
    return STYLESHEET_TEMPLATE.replace("__FONT__", str(point_size))
