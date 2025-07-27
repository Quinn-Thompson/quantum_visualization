from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6 import QtWidgets
from PyQt6 import QtCore

class BlochWindowWidgets():
    def __init__(self):
        self.figure = Figure(figsize=(14, 8))
        self.bloch_visual_widget: FigureCanvas = FigureCanvas(self.figure)


class BlochWindow(QtWidgets.QFrame):
    def __init__(self):
        super().__init__()
        self.main_layout = QtWidgets.QGridLayout()
        self.widgets: BlochWindowWidgets = BlochWindowWidgets()
        self.setObjectName("BlochWindow")
        self.setStyleSheet(
            "QWidget#BlochWindow {" 
            "border: 2px solid red;"
            "}"
        )
        self.setLayout(self.main_layout)
        self.main_layout.addWidget(
            self.widgets.bloch_visual_widget,
            0, 
            0,
            alignment=QtCore.Qt.AlignmentFlag.AlignCenter
        )


        