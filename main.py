from PyQt5 import QtWidgets
from IORT.widgets.mainwindow import Mainwindow
import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPalette, QColor
os.environ["QT_STYLE_OVERRIDE"] = "Fusion"

def main():
    app = QtWidgets.QApplication([''])
    # 强制 Fusion 样式
    app.setStyle("Fusion")
    # 设置浅色调色板
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(255, 255, 255))
    palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
    palette.setColor(QPalette.Base, QColor(245, 245, 245))
    palette.setColor(QPalette.AlternateBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
    palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
    palette.setColor(QPalette.Text, QColor(0, 0, 0))
    palette.setColor(QPalette.Button, QColor(240, 240, 240))
    palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    app.setPalette(palette)
    mainwindow = Mainwindow()
    mainwindow.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    import tarfile
    main()

    