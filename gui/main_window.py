from PyQt6 import QtWidgets
from PyQt6 import QtCore
from gui.helpers import global_budget_window_style
from gui.sub_widgets.bloch_window import BlochWindow
from gui.sub_widgets.entanglement_window import EntanglementWindow
from gui.sub_widgets.animation_control import AnimationControl
from typing import Dict
from dataclasses import dataclass


def start_application():
    return QtWidgets.QApplication([])  

@dataclass
class SubWidgets():
    bloch_window: BlochWindow
    animation_control: AnimationControl
    entanglement_window: EntanglementWindow
    

class MainWindow(QtWidgets.QMainWindow):
    """
    desc:
        The main widget for the window.
    """
    
    def __init__(self):
        """
        desc:
            The initialization for the main window.
        """
        super().__init__(parent=None)
        self.setWindowFlag(QtCore.Qt.WindowType.FramelessWindowHint)
        self.setWindowTitle("Main Window")

        self.setStyleSheet(global_budget_window_style)
        
        self.sub_window_widgets: SubWidgets = SubWidgets(BlochWindow(), AnimationControl(), EntanglementWindow())

        # add the main widget to the root
        self.central_widget = QtWidgets.QWidget()

        self._init_widgets()

        self.central = self.setCentralWidget(self.central_widget)
        
    def _init_widgets(self):
        self.root_layoutV = QtWidgets.QVBoxLayout()
        self.root_layoutH = QtWidgets.QHBoxLayout()
        
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.sub_window_widgets.bloch_window, "Bloch Window")
        self.tabs.addTab(self.sub_window_widgets.entanglement_window, "Entanglement Window")
        self.timeout_label = QtWidgets.QLabel()
        self.timeout_label.setText("Active")
        self.status_bar = QtWidgets.QToolBar()
        self.status_bar.addWidget(self.timeout_label)

        self.root_layoutV.addLayout(self.root_layoutH)

        self.root_layoutH.addWidget(self.tabs)
        self.root_layoutH.addWidget(self.sub_window_widgets.animation_control)
        self.root_layoutV.addWidget(self.status_bar)
        self.central_widget.setLayout(self.root_layoutV)