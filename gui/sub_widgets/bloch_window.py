"""The window for displaying the bloch spheres."""
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6 import QtWidgets
from PyQt6 import QtCore

class BlochWindowWidgets():
    """The widgets for the bloch spheres window.
    """
    def __init__(self) -> None:
        """Initialize each widget within the window.
        """
        self.figure = Figure(figsize=(14, 8))
        self.bloch_visual_widget: FigureCanvas = FigureCanvas(self.figure)


class BlochWindow(QtWidgets.QFrame):
    """The frame for displaying the bloch spheres.
    """
    def __init__(self) -> None:
        """Initialize the bloch spheres window.
        """
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


        