from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from PyQt6 import QtWidgets
from PyQt6 import QtCore

class EntanglementWindowWidgets():
    def __init__(self):
        self.figure, self.axis = plt.subplots(figsize=(14, 8))
        self.entanglement_visual_widget: FigureCanvas = FigureCanvas(self.figure)

class EntanglementWindow(QtWidgets.QFrame):
    def __init__(self):
        super().__init__()
        self.main_layout = QtWidgets.QGridLayout()
        self.widgets: EntanglementWindowWidgets = EntanglementWindowWidgets()
        self.setObjectName("EntanglementWindow")
        self.setStyleSheet(
            "QWidget#EntanglementWindow {" 
            "border: 2px solid red;"
            "}"
        )
        self.setLayout(self.main_layout)
        self.main_layout.addWidget(
            self.widgets.entanglement_visual_widget,
            0, 
            0,
            alignment=QtCore.Qt.AlignmentFlag.AlignCenter
        )

