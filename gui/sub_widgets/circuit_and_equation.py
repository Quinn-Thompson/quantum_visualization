"""The window for displaying the bloch spheres."""
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from PyQt6 import QtWidgets, QtCore, QtGui
from typing import List

class CircuitWindowWidgets():
    """The widgets for the bloch spheres window.
    """
    def __init__(self) -> None:
        """Initialize each widget within the window.
        """
        self.equation = QtWidgets.QLabel()
        self.figure, self.axis = plt.subplots(figsize=(16, 8))
        self.circuit_visual_widget: FigureCanvas = FigureCanvas(self.figure)
        self.circuit_slider_horizontal = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.circuit_slider_vertical = QtWidgets.QSlider(QtCore.Qt.Orientation.Vertical)
        self.circuit_slider_zoom_horizontal = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.circuit_slider_zoom_vertical = QtWidgets.QSlider(QtCore.Qt.Orientation.Vertical)



class CircuitWindow(QtWidgets.QFrame):
    """The frame for displaying the bloch spheres.
    """
    def __init__(self) -> None:
        """Initialize the bloch spheres window.
        """
        super().__init__()
        self.main_layout = QtWidgets.QGridLayout()
        self.widgets: CircuitWindowWidgets = CircuitWindowWidgets()
        self.setObjectName("Window")
        self.setLayout(self.main_layout)
        self.main_layout.addWidget(
            self.widgets.equation,
            0, 
            2,
            alignment=QtCore.Qt.AlignmentFlag.AlignCenter
        )
        self.main_layout.addWidget(
            self.widgets.circuit_visual_widget,
            1, 
            2,
            alignment=QtCore.Qt.AlignmentFlag.AlignCenter
        )
        

        self.main_layout.addWidget(
            self.widgets.circuit_slider_vertical,
            1, 
            1,
            alignment=QtCore.Qt.AlignmentFlag.AlignCenter
        )
        self.main_layout.addWidget(
            self.widgets.circuit_slider_horizontal,
            2, 
            2,
            alignment=QtCore.Qt.AlignmentFlag.AlignCenter
        )
        self.main_layout.addWidget(
            self.widgets.circuit_slider_zoom_vertical,
            1, 
            0,
            alignment=QtCore.Qt.AlignmentFlag.AlignCenter
        )
        self.main_layout.addWidget(
            self.widgets.circuit_slider_zoom_horizontal,
            3, 
            2,
            alignment=QtCore.Qt.AlignmentFlag.AlignCenter
        )
        
        self.widgets.circuit_slider_zoom_horizontal.setRange(20, 100)
        self.widgets.circuit_slider_zoom_vertical.setRange(20, 100)
        self.widgets.circuit_slider_zoom_horizontal.setValue(100)
        self.widgets.circuit_slider_zoom_vertical.setValue(100)
        self.widgets.circuit_slider_vertical.setRange(0, 100)
        self.widgets.circuit_slider_horizontal.setRange(0, 100)
        

    def set_elided_text(self, state_list: List[str]):
        metrics = QtGui.QFontMetrics(self.widgets.equation.font())
        line_length = 0
        label_to_print = ""
        line_number = 0
        for text in state_list:
            text_width = metrics.horizontalAdvance(text)
            if line_length + text_width > self.width():
                label_to_print += "\n"
                line_length = 0
                line_number += 1
                if line_number > 5:
                    break

            label_to_print += text
            line_length += text_width
                

        self.widgets.equation.setText(label_to_print)
        