"""The window for controlling the animation."""
from PyQt6 import QtCore
from PyQt6 import QtWidgets

class AnimationControlWindowWidgets():
    """The widgets for the animation control window.
    """
    def __init__(self) -> None:
        """Initialize each widget within the window.
        """
        self.next_button = QtWidgets.QPushButton()
        self.prev_button = QtWidgets.QPushButton()
        self.play_button = QtWidgets.QPushButton()
        self.to_x_block = QtWidgets.QPushButton()
        self.which_bloch = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.array_title = QtWidgets.QLabel()

class AnimationControl(QtWidgets.QFrame):
    """The frame for controlling the animation.
    """
    def __init__(self) -> None:
        """Initialize the animation control window.
        """
        super().__init__()
        self.main_layout = QtWidgets.QGridLayout()
        self.widgets = AnimationControlWindowWidgets()

        self.setLayout(self.main_layout)
        self.widgets.next_button.setText("next")
        self.widgets.prev_button.setText("prev")
        self.widgets.to_x_block.setText("go to x")
        self.main_layout.addWidget(
            self.widgets.next_button,
            0, 
            0,
            alignment=QtCore.Qt.AlignmentFlag.AlignCenter
        )
        self.main_layout.addWidget(
            self.widgets.prev_button,
            0, 
            1,
            alignment=QtCore.Qt.AlignmentFlag.AlignCenter
        )
        self.main_layout.addWidget(self.widgets.to_x_block, 0, 2, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(self.widgets.which_bloch, 1, 1, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(self.widgets.array_title, 2, 1, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
