import numpy as np
from typing import List, Dict, Optional, Literal
import matplotlib
from gui_backend.helpers import DisplayProperties
import matplotlib.gridspec as gridspec
import qiskit
import qiskit.circuit
from qiskit.quantum_info import Statevector, partial_trace
from functools import partial
from gui.main_window import MainWindow
from gui_backend.sub_backend.entanglement import EntanglementMatrix
from gui_backend.sub_backend.bloch_sphere import PerQubitVisualization
from PyQt6.QtCore import QTimer
from gui.helpers import background_color
matplotlib.use("TkAgg")

class VisualizationWrapper():
    def __init__(self, main_window: MainWindow):
        self.main_window = main_window
        self.layout = main_window.sub_window_widgets.bloch_window
        self._to_update_method = self._run_to_next_block
        self.main_window.sub_window_widgets.animation_control.widgets.next_button.clicked.connect(partial(self.update_method_to_use, "next"))
        self.main_window.sub_window_widgets.animation_control.widgets.prev_button.clicked.connect(partial(self.update_method_to_use, "prev"))
        self.main_window.sub_window_widgets.animation_control.widgets.to_x_block.clicked.connect(partial(self.update_method_to_use, "x"))
        self.maintained_properties: List[DisplayProperties] = []
        self.timer = QTimer()
    
    def setup_circuit(self, quantum_circuit: qiskit.QuantumCircuit, frames_per_animation: int, display_properties: Optional[DisplayProperties] = None):
        self.maintained_properties.append(display_properties)
        self.func_animation = None
        self.frames_per_animation = frames_per_animation
        self.entanglement = EntanglementMatrix(
            quantum_circuit, 
            self.main_window.sub_window_widgets.entanglement_window.widgets.figure, 
            self.main_window.sub_window_widgets.entanglement_window.widgets.axis,
            display_properties
        )
        
        
        num_quantum_registers = len(quantum_circuit.qregs)
        
        self.main_grid_layout = gridspec.GridSpec(
            int(np.ceil(np.sqrt(num_quantum_registers))), 
            int(np.ceil(np.sqrt(num_quantum_registers))), 
            height_ratios=[1, 1], 
            hspace=0.4
        )
        self.main_window.sub_window_widgets.bloch_window.widgets.figure.patch.set_facecolor(background_color)
        self._quantum_register_grid: Dict[qiskit.QuantumRegister, gridspec.GridSpecFromSubplotSpec] = {}
        self._qubit_subplots: Dict[qiskit.circuit.Qubit, PerQubitVisualization] = {}
        quantum_vector = Statevector.from_instruction(quantum_circuit)
        keep_indices = list(range(len(quantum_circuit.qubits)))
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
        self.main_window.sub_window_widgets.bloch_window.widgets.figure.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        self._frame = 0
        self._total_frame_count = 0

    @property
    def qubit_subplots(self) -> Dict[qiskit.circuit.Qubit, PerQubitVisualization]:
        return self._qubit_subplots
    
    def add_circuit_state(
        self, 
        quantum_circuit: qiskit.QuantumCircuit, 
        bloch_properties: DisplayProperties, 
    ):
        self.maintained_properties.append(bloch_properties)
        quantum_vector = Statevector.from_instruction(quantum_circuit)
        keep_indices = list(range(len(quantum_circuit.qubits)))
        for qubit in quantum_circuit.qubits:
            traced_out_indices = keep_indices.copy()
            traced_out_indices.remove(quantum_circuit.qubits.index(qubit))
            reduced_state_vector = partial_trace(quantum_vector, traced_out_indices).data
            self._qubit_subplots[quantum_circuit.qubits.index(qubit)]._append_block(
                reduced_state_vector, bloch_properties
            )
        self.entanglement.append_block(quantum_circuit, bloch_properties)

    def _update(self):
        interpolation_ratio = self._frame / self.frames_per_animation
        for per_qubit_obj in self._qubit_subplots.values():
            per_qubit_obj.update_plot(interpolation_ratio)
        self.entanglement.update_heatmap(interpolation_ratio)
        self.main_window.sub_window_widgets.bloch_window.widgets.bloch_visual_widget.draw_idle()
        self.main_window.sub_window_widgets.entanglement_window.widgets.entanglement_visual_widget.draw_idle()

    def _run_to_x_block(self):
        more_animation = False
        which_bloch_to = self.main_window.sub_window_widgets.animation_control.widgets.which_bloch.value()
        for per_qubit_obj in self._qubit_subplots.values():
            more_animation |= per_qubit_obj.to_x_block(which_bloch_to)
        self.entanglement.to_x_block(which_bloch_to)
        return more_animation 
        
    def _run_to_next_block(self):
        more_animation = False
        for per_qubit_obj in self._qubit_subplots.values():
            more_animation |= per_qubit_obj.to_next_block()
        self.entanglement.to_next_block()
        return more_animation

    def _run_to_previous_block(self):
        more_animation = False
        for per_qubit_obj in self._qubit_subplots.values():
            more_animation |= per_qubit_obj.to_prev_block()
        self.entanglement.to_prev_block()
        return more_animation 

    def update_method_to_use(self, update_method: Literal["next", "prev"]):
        if update_method == "next":
            print("run to next")
            self._to_update_method = self._run_to_next_block
        elif update_method == "prev":
            print("run to prev")
            self._to_update_method = self._run_to_previous_block   
        
        elif update_method == "x":
            print("run to x")
            self._to_update_method = self._run_to_x_block   
          
        self.start_next_block()

    def start_next_block(self):
        self.timer.start()
        
    def update_block_label(self, value):
        self.main_window.sub_window_widgets.animation_control.widgets.array_title.setText(self.maintained_properties[value].plot_name)

    def animate_bloch_sphere(self):
        self.main_window.sub_window_widgets.animation_control.widgets.which_bloch.setMaximum(0)
        self.main_window.sub_window_widgets.animation_control.widgets.which_bloch.setMaximum(len(self.maintained_properties)-1)
        self.main_window.sub_window_widgets.animation_control.widgets.which_bloch.setValue(0)
        self.main_window.sub_window_widgets.animation_control.widgets.which_bloch.valueChanged.connect(self.update_block_label)

        def update():
            if self._frame == 0:
                more_animation = self._to_update_method()
                
                if not more_animation:
                    print("stopped animation")
                    self.timer.stop()
            if self._frame % 5 == 0:
                self.main_window.timeout_label.setText(str(self._frame))
            self._frame += 1

            self._update()

            if self._frame == self.frames_per_animation:
                self.timer.stop()
                self._frame = 0

        self.timer.setInterval(33)
        self.timer.timeout.connect(update)
        self.main_window.sub_window_widgets.entanglement_window.widgets.entanglement_visual_widget.draw_idle()
        self.main_window.sub_window_widgets.bloch_window.widgets.figure.canvas.draw_idle()
