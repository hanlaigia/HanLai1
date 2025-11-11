import sys
from PyQt6.QtWidgets import QApplication
from RetailsManagerEx import App

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = App()
    w.show()
    sys.exit(app.exec())
