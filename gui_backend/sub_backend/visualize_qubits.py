"""The wrapper around visualizing the quantum circuit."""
import numpy as np
from typing import List, Dict, Optional, Literal
import matplotlib
from gui_backend.helpers import DisplayProperties
import matplotlib.gridspec as gridspec
import qiskit
import qiskit.circuit
from qiskit.quantum_info import Statevector, partial_trace
from qiskit_aer import AerSimulator
from functools import partial
from gui.main_window import MainWindow
from gui_backend.sub_backend.entanglement import EntanglementMatrix
from gui_backend.sub_backend.bloch_sphere import PerQubitVisualization
from gui_backend.sub_backend.circuit_and_equation import CircuitVisualization
from PyQt6.QtCore import QTimer
from gui.helpers import background_color
import random
import string
matplotlib.use("TkAgg")

class VisualizationWrapper():
    """A wrapper that contains the backend for the different visualizations. 
    """
    
    def __init__(self, main_window: MainWindow) -> None:
        """Initialize the wrapper with info from the main window gui.

        Args:
            main_window: The main gui window.
        """
        self.main_window = main_window
        self.layout = main_window.sub_window_widgets.bloch_window
        self._to_update_method = self._move_to_next_block
        self.main_window.sub_window_widgets.animation_control.widgets.next_button.clicked.connect(partial(self.update_method_to_use, "next"))
        self.main_window.sub_window_widgets.animation_control.widgets.prev_button.clicked.connect(partial(self.update_method_to_use, "prev"))
        self.main_window.sub_window_widgets.animation_control.widgets.to_x_block.clicked.connect(partial(self.update_method_to_use, "x"))
        self._maintained_properties: List[DisplayProperties] = []
        # timer must be managed by the wrapper
        self._timer = QTimer()
        self.currently_displayed_index = 0
        self._next_index_to_display = 0    

    def setup_circuit(self, quantum_circuit: qiskit.QuantumCircuit, frames_per_animation: int, display_properties: Optional[DisplayProperties] = None) -> None:
        """Setup the initial circuit so the window knows how to structure things.

        Args:
            quantum_circuit: The initial circuit that will be displayed.
            frames_per_animation: How many frames to play per animation.
            display_properties: The properties of the initial display.
        """
        self._maintained_properties.append(display_properties)
        self.func_animation = None
        self.frames_per_animation = frames_per_animation
        self.entanglement = EntanglementMatrix(
            self.main_window.sub_window_widgets.entanglement_window.widgets.axis,
            quantum_circuit, 
            display_properties,
            self.main_window.sub_window_widgets.entanglement_window.widgets.figure, 
        )
        self.circuit_visual = CircuitVisualization(
            self.main_window.sub_window_widgets.circuit_window,
            quantum_circuit,
            display_properties,
        )
        
        num_quantum_registers = len(quantum_circuit.qregs)
        
        # we use grid spec to seperate registers
        self.main_grid_layout = gridspec.GridSpec(
            int(np.ceil(np.sqrt(num_quantum_registers))), 
            int(np.ceil(np.sqrt(num_quantum_registers))), 
            hspace=0.4
        )
        self.main_window.sub_window_widgets.bloch_window.widgets.figure.patch.set_facecolor(background_color)
        self._quantum_register_grid: Dict[qiskit.QuantumRegister, gridspec.GridSpecFromSubplotSpec] = {}
        self._qubit_subplots: Dict[qiskit.circuit.Qubit, PerQubitVisualization] = {}
        name = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(256))
        quantum_circuit.save_statevector(label=name)
        simulator = AerSimulator(method='statevector')

        compiled_circuit = qiskit.transpile(quantum_circuit, simulator)
        result = simulator.run(compiled_circuit).result()
        quantum_vector = result.data(0)[name]
        keep_indices = list(range(len(quantum_circuit.qubits)))
        # create a sub plot for each qubit depending on the register
        for register_number, quantum_register in enumerate(quantum_circuit.qregs):
            qubit_count = len(list(quantum_register))
            square_length = int(np.ceil(np.sqrt(qubit_count)))
            self._quantum_register_grid[quantum_register.name] = gridspec.GridSpecFromSubplotSpec(
                square_length, 
                square_length,  
                subplot_spec=self.main_grid_layout[register_number], 
                wspace=0.3
            )
            
            for qubit_number, qubit in enumerate(quantum_register):
                traced_out_indices = keep_indices.copy()
                traced_out_indices.remove(quantum_circuit.qubits.index(qubit))
                reduced_state_vector = partial_trace(quantum_vector, traced_out_indices).data
                self._qubit_subplots[quantum_circuit.qubits.index(qubit)] = PerQubitVisualization(
                    self.main_window.sub_window_widgets.bloch_window.widgets.figure.add_subplot(
                        self._quantum_register_grid[quantum_register.name][qubit_number%square_length, qubit_number//square_length], projection='3d',
                ), reduced_state_vector, display_properties)
        print("Succeeded Initializing Bloch Spheres")
        self.main_window.sub_window_widgets.bloch_window.widgets.figure.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        self._frame = 0
        self._total_frame_count = 0
        print("Finished Initialization")

    @property
    def qubit_subplots(self) -> Dict[qiskit.circuit.Qubit, PerQubitVisualization]:
        """Qubit subplots are private.

        Returns:
            The qubit sub plots.
        """
        return self._qubit_subplots
    
    def add_circuit_state(
        self, 
        quantum_circuit: qiskit.QuantumCircuit, 
        bloch_properties: DisplayProperties,
        fast_vector: bool = False,
        simulator: Optional[AerSimulator] = None,
    ):
        self._maintained_properties.append(bloch_properties)
        print("Attempting to Attain Statevector")
        if not fast_vector:
            quantum_vector = Statevector.from_instruction(quantum_circuit)
        else:
            name = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(256))
            quantum_circuit.save_statevector(label=name)
            simulator = AerSimulator(method='statevector')

            compiled_circuit = qiskit.transpile(quantum_circuit, simulator)
            result = simulator.run(compiled_circuit).result()
            quantum_vector = result.data(0)[name]
        keep_indices = list(range(len(quantum_circuit.qubits)))
        for qubit in quantum_circuit.qubits:
            traced_out_indices = keep_indices.copy()
            traced_out_indices.remove(quantum_circuit.qubits.index(qubit))
            print(traced_out_indices)
            reduced_state_vector = partial_trace(quantum_vector, traced_out_indices).data
            self._qubit_subplots[quantum_circuit.qubits.index(qubit)].append_block(
                reduced_state_vector, bloch_properties
            )
        self.entanglement.append_block(quantum_circuit, bloch_properties)
        self.circuit_visual.append_block(quantum_circuit, bloch_properties)

    def _update(self) -> None:
        """Update to the next interpolated visual.
        """
        end_animation = self._frame == self.frames_per_animation
        interpolation_ratio = self._frame / self.frames_per_animation
        if self.main_window.tabs.currentWidget() == self.main_window.sub_window_widgets.bloch_window or end_animation:
            for per_qubit_obj in self._qubit_subplots.values():
                per_qubit_obj.update_plot(interpolation_ratio, self.currently_displayed_index, self._next_index_to_display)
            self.main_window.sub_window_widgets.bloch_window.widgets.bloch_visual_widget.draw_idle()
        if self.main_window.tabs.currentWidget() == self.main_window.sub_window_widgets.entanglement_window or end_animation:
            self.entanglement.update_plot(interpolation_ratio, self.currently_displayed_index, self._next_index_to_display)
            self.main_window.sub_window_widgets.entanglement_window.widgets.entanglement_visual_widget.draw_idle()
        if self.main_window.tabs.currentWidget() == self.main_window.sub_window_widgets.circuit_window or end_animation:
            self.circuit_visual.update_plot(interpolation_ratio, self.currently_displayed_index, self._next_index_to_display)
            self.main_window.sub_window_widgets.circuit_window.widgets.circuit_visual_widget.draw_idle()

    def _move_to_next_block(self):
        """Move to next block.

        Returns:
            Whether it was able to jump to that block.
        """
        for per_qubit_obj in self._qubit_subplots.values():
            per_qubit_obj.next_animation_block(self.currently_displayed_index, self._next_index_to_display)
        self.entanglement.next_animation_block(self.currently_displayed_index, self._next_index_to_display)
        self.circuit_visual.next_animation_block(self.currently_displayed_index, self._next_index_to_display)


    def update_method_to_use(self, update_method: Literal["next", "prev", "x"]):
        """Update which block to jump to.

        Args:
            update_method: Which process to use to move animation blocks.
        """
        self.currently_displayed_index = self._next_index_to_display
        if update_method == "next":
            if self._next_index_to_display > self.circuit_visual.animation_block_length - 1:
                return
            self._next_index_to_display += 1
        elif update_method == "prev":
            if self._next_index_to_display < 0:
                return
            self._next_index_to_display -= 1
        elif update_method == "x":
            self._next_index_to_display = self.main_window.sub_window_widgets.animation_control.widgets.which_bloch.value()
        self.main_window.sub_window_widgets.animation_control.widgets.which_bloch.setValue(self._next_index_to_display)
        self._timer.start()
        
    def update_block_label(self, slider_value: int) -> None:
        """Update the slider label to show what block will be moved to.

        Args:
            slider_value: The current value of the slider
        """
        self.main_window.sub_window_widgets.animation_control.set_elided_text(self._maintained_properties[slider_value].plot_name)

    def setup_animation_process(self) -> None:
        """Create process for animating each display.
        """
        self.main_window.sub_window_widgets.animation_control.widgets.which_bloch.setMaximum(0)
        self.main_window.sub_window_widgets.animation_control.widgets.which_bloch.setMaximum(len(self._maintained_properties)-1)
        self.main_window.sub_window_widgets.animation_control.widgets.which_bloch.setValue(0)
        self.main_window.sub_window_widgets.animation_control.widgets.which_bloch.valueChanged.connect(self.update_block_label)

        def update() -> None:
            """Update to next frame of animation.
            """
            if self._frame == 0:
                self._move_to_next_block()
                
            if self._frame % 5 == 0:
                self.main_window.timeout_label.setText(str(self._frame))
            self._frame += 1

            self._update()

            if self._frame == self.frames_per_animation:
                self._timer.stop()
                self._frame = 0

        self._timer.setInterval(50)
        self._timer.timeout.connect(update)
        self.main_window.sub_window_widgets.entanglement_window.widgets.entanglement_visual_widget.draw_idle()
        self.main_window.sub_window_widgets.bloch_window.widgets.figure.canvas.draw_idle()
