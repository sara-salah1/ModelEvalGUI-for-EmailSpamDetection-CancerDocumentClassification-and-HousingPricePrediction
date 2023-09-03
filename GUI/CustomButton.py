from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QPainter, QBrush, QColor, QFont
from PyQt5.QtWidgets import QPushButton


class RoundButton(QPushButton):
    def __init__(self, name):
        super().__init__()

        # self.setFixedSize(100, 40)

        self.normal_color = QColor("#40826d")
        self.hover_color = QColor("#87a96b")

        palette = QPalette()
        palette.setColor(QPalette.ButtonText, Qt.white)
        self.setPalette(palette)
        self.setText(name)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if self.underMouse():
            painter.setBrush(QBrush(self.hover_color))
        else:
            painter.setBrush(QBrush(self.normal_color))

        painter.drawRoundedRect(self.rect(), 15, 15)

        painter.setPen(Qt.white)
        painter.setFont(QFont("Arial", 12, QFont.Bold))
        text_rect = self.rect()
        painter.drawText(text_rect, Qt.AlignCenter, self.text())