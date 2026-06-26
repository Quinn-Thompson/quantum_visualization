"""The main window for the gui."""
from PyQt6 import QtWidgets
from PyQt6 import QtCore
from gui.helpers import global_budget_window_style
from gui.sub_widgets.bloch_window import BlochWindow
from gui.sub_widgets.entanglement_window import EntanglementWindow
from gui.sub_widgets.animation_control import AnimationControl
from gui.sub_widgets.circuit_and_equation import CircuitWindow
from dataclasses import dataclass

def start_application() -> QtWidgets.QApplication:
    """The application object which is used to start the window thread."""
    return QtWidgets.QApplication([])  

@dataclass
class SubWidgets():
    """The different windows that exist within the main one.
    """
    bloch_window: BlochWindow
    circuit_window: CircuitWindow
    animation_control: AnimationControl
    entanglement_window: EntanglementWindow
    

class MainWindow(QtWidgets.QMainWindow):
    """The main widget for the window.
    """
    
    def __init__(self) -> None:
        """The initialization for the main window.
        """
        super().__init__(parent=None)
        self.setWindowFlag(QtCore.Qt.WindowType.FramelessWindowHint)
        self.setWindowTitle("Main Window")

        self.setStyleSheet(global_budget_window_style)
        
        self.sub_window_widgets: SubWidgets = SubWidgets(
            BlochWindow(), CircuitWindow(), AnimationControl(), EntanglementWindow()
        )

        # add the main widget to the root
        self.central_widget = QtWidgets.QWidget()

        self._init_widgets()

        self.central = self.setCentralWidget(self.central_widget)
        
    def _init_widgets(self) -> None:
        """Initialize th separate sub windows and toolbar.
        """
        self.root_layoutV = QtWidgets.QVBoxLayout()
        self.root_layoutH = QtWidgets.QHBoxLayout()
        
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.sub_window_widgets.bloch_window, "Bloch Window")
        self.tabs.addTab(self.sub_window_widgets.entanglement_window, "Entanglement Window")
        self.tabs.addTab(self.sub_window_widgets.circuit_window, "Circuit Window")
        self.timeout_label = QtWidgets.QLabel()
        self.timeout_label.setText("Active")
        self.status_bar = QtWidgets.QToolBar()
        self.status_bar.addWidget(self.timeout_label)

        self.root_layoutV.addLayout(self.root_layoutH)

        self.root_layoutH.addWidget(self.tabs)
        self.root_layoutH.addWidget(self.sub_window_widgets.animation_control)
        self.root_layoutV.addWidget(self.status_bar)
        self.central_widget.setLayout(self.root_layoutV)