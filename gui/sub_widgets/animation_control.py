from PyQt6 import QtCore
from PyQt6 import QtWidgets

class BlochWindowWidgets():
    def __init__(self):
        self.next_button = QtWidgets.QPushButton()
        self.prev_button = QtWidgets.QPushButton()
        self.play_button = QtWidgets.QPushButton()
        self.to_x_block = QtWidgets.QPushButton()
        self.which_bloch = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.array_title = QtWidgets.QLabel()

class AnimationControl(QtWidgets.QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.main_layout = QtWidgets.QGridLayout()
        self.widgets = BlochWindowWidgets()

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
