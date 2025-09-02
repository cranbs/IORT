from PyQt5 import QtWidgets
from IORT.widgets.mainwindow import Mainwindow
import sys

import torch
torch.cuda.is_available()

def main():
    app = QtWidgets.QApplication([''])
    mainwindow = Mainwindow()
    mainwindow.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    import tarfile
    with tarfile.open("IORT.tar.gz", "w:gz") as tar:
        tar.add("IORT", arcname="IORT")
        tar.add("LaMaProject", arcname="LaMaProject")
        tar.add("main.py")
    main()

    