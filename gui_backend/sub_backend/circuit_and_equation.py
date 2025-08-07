"""Handles the entanglement matrix backend."""
import numpy as np
from typing import List, Optional
from gui_backend.helpers import DisplayProperties
import qiskit
from qiskit.quantum_info import Statevector
from gui_backend.sub_backend.display import GenericDisplay, AnimationBlock
from copy import deepcopy
from gui.sub_widgets.circuit_and_equation import CircuitWindow
from PyQt6.QtWidgets import QGraphicsOpacityEffect
from gui.helpers import background_color, clicked_color, border_color

MPL_STYLE = {
    "backgroundcolor": background_color,
    "textcolor": clicked_color,
    "gate_face_color": border_color,
    "gate_border_color": "#000000",
    "linecolor": clicked_color,
    "fontsize": 16,
    "displaycolor": {
        "x": ("#ff6666", "#000000"),  # red X gate
        "h": ("#66ccff", "#000000"),  # blue H gate
        "cx": ("#ccffcc", "#000000"),  # green CX gate
        "ccx": ("#76ff76", "#000000"),  # green CX gate
        "mcx": ("#26ff26", "#000000"),  # green CX gate
    }
}

class AnimationBlockCircuit(AnimationBlock):
    """A block of information for the animation steps to use.
    """
    def __init__(self, quantum_circuit: qiskit.QuantumCircuit, state_vector_equation: str, display_properties: DisplayProperties) -> None:
        """Initialize the animation block.

        Args:
            entanglement_values: The matrix of this step.
            display_properties: The properties to display for this step.
        """
        self.state_vector_equation = state_vector_equation
        self.quantum_circuit: qiskit.QuantumCircuit = deepcopy(quantum_circuit)
        self.matrix_color: str = display_properties.entanglement_colors
        self.display_properties: Optional[DisplayProperties] = display_properties
        

class CircuitVisualization(GenericDisplay):
    """The properties of each qubits plot."""
    
    def __init__(self, sub_window: CircuitWindow, information_input: qiskit.QuantumCircuit, display_properties: Optional[DisplayProperties]) -> None:
        """Initialize the qubits visualization.

        Args:
            subset_axes: The axis that is being manipulated.
            initial_matrix: The initial mixed state.
            display_properties: The initial display properties.
        """
        self._sub_window = sub_window
        self._effect = QGraphicsOpacityEffect()
        self._sub_window.widgets.equation.setGraphicsEffect(self._effect)
        self._sub_window.widgets.circuit_visual_widget.setGraphicsEffect(self._effect)
        self._changed_visual = False
        super().__init__(None, information_input, display_properties, None)
        self._animation_blocks: List[AnimationBlockCircuit]
        self._sub_window.widgets.circuit_slider_vertical.valueChanged.connect(self._update_visual)
        self._sub_window.widgets.circuit_slider_horizontal.valueChanged.connect(self._update_visual)
        self._sub_window.widgets.circuit_slider_zoom_horizontal.valueChanged.connect(self._update_visual)
        self._sub_window.widgets.circuit_slider_zoom_vertical.valueChanged.connect(self._update_visual)


    def __len__(self) -> int:
        """The length of the animation.

        Returns:
            The length of the animation blocks.
        """
        return len(self._animation_blocks)
      
    def initialize_plot(self, information_input: qiskit.QuantumCircuit, display_properties: Optional[DisplayProperties] = None) -> None:
        """Create the matplotlib visualization for the bloch sphere.
        """
        del display_properties
        self.draw_circuit(information_input)
        self._sub_window.set_elided_text(self._animation_blocks[0].state_vector_equation)

    def draw_circuit(self, quantum_circuit: qiskit.QuantumCircuit):
        quantum_circuit.draw("mpl", ax=self._sub_window.widgets.axis, style=MPL_STYLE, fold=-1)
        # Now manually change the full figure background
        self._sub_window.widgets.figure.patch.set_facecolor(background_color)  # outer (figure) background

        # Also ensure each axis matches
        for ax in self._sub_window.widgets.figure.axes:
            ax.set_facecolor(background_color)  # axes background again (optional but safe)

        for text in ax.texts:
            text.set_fontsize(16)

        self.x_limits = self._sub_window.widgets.axis.get_xlim()
        self.y_limits = self._sub_window.widgets.axis.get_ylim()


    def update_plot(self, interpolation_ratio: float, current_index: int, next_index: int) -> None:
        """Update the plot of the qubit.

        Args:
            interpolation_ratio: The current point of interpolation between this matrix and the next.
        """
        opacity = np.abs(0.5 - interpolation_ratio)
        if interpolation_ratio > 0.5 and not self._changed_visual:
            self._sub_window.widgets.axis.clear()
            self.draw_circuit(self._animation_blocks[next_index].quantum_circuit)
            self._sub_window.set_elided_text(self._animation_blocks[next_index].state_vector_equation)
            self._changed_visual = True
            
        self._effect.setOpacity(opacity)

    def _update_visual(self):
        x_total = self.x_limits[1] - self.x_limits[0]
        y_total = self.y_limits[1] - self.y_limits[0]
        percent_size_x = self._sub_window.widgets.circuit_slider_zoom_horizontal.value() / 100
        percent_size_y = self._sub_window.widgets.circuit_slider_zoom_vertical.value() / 100
        x_view_size = x_total * percent_size_x
        x_movement_area = x_total - x_view_size
        
        x_location = self.x_limits[0] + (x_view_size / 2) + (self._sub_window.widgets.circuit_slider_horizontal.value() / 100) * x_movement_area
        self._sub_window.widgets.axis.set_xlim(x_location - (x_view_size / 2), x_location + (x_view_size / 2))
        
        y_view_size = y_total * percent_size_y
        y_movement_area = y_total - y_view_size
        
        y_location = self.y_limits[0] + (y_view_size / 2) + (self._sub_window.widgets.circuit_slider_vertical.value() / 100) * y_movement_area
        self._sub_window.widgets.axis.set_ylim(y_location - (y_view_size / 2), y_location + (y_view_size / 2))
        self._sub_window.widgets.circuit_visual_widget.draw()

    def append_block(
        self, 
        information_input: qiskit.QuantumCircuit, 
        bloch_properties: Optional[DisplayProperties] = None, 
    ):
        """Append the x, y and z coordinates state to animate.

        Args:
            quantum_circuit: The circuit to visualize.
            display_properties: The initial display properties.
        """
        # immediately garbage collect statevector
        state_vector = Statevector.from_instruction(information_input)
        num_qubits = len(information_input.qubits)
        labels = [format(qubit, f'0{num_qubits}b') for qubit in range(2**num_qubits)]
        terms = []
        first_item = True
        for amplitude, basis in zip(state_vector.data, labels):
            term_string = ""
            some_amplitude = False
            use_paranthesis = False
            if not first_item:
                term_string += " + "

            if amplitude.real <= -0.01 or amplitude.real >= 0.01:
                some_amplitude = True
                term_string += f"{amplitude.real:.2f}"
                
            if some_amplitude and amplitude.imag >= 0.01:
                term_string += " + "
                use_paranthesis = True
            if some_amplitude and amplitude.imag <= -0.01:
                term_string += " - "
                use_paranthesis = True
            elif amplitude.imag <= -0.01 or amplitude.imag >= 0.01:
                some_amplitude = True
                term_string += f"{amplitude.imag:.2f}i"

            if some_amplitude:
                first_item = False
                terms.append(f"{"(" if use_paranthesis else ""}{term_string}{")" if use_paranthesis else ""}|{basis}⟩")
        self._animation_blocks.append(AnimationBlockCircuit(information_input, terms, bloch_properties))

        
    def next_animation_block(self, current_index: int, next_index: int) -> None:
        """Jump to the next block for animation.

        Args:
            next_index: The index of the animation block to jump to.

        Returns:
            Whether it was able to jump to this index.
        """
        self._changed_visual = False
