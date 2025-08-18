"""The main window wrapper that holds everything."""
from gui.main_window import start_application, MainWindow
from gui_backend.sub_backend.visualize_qubits import VisualizationWrapper
import qiskit
from qiskit_aer import AerSimulator
from typing import Optional
from gui_backend.helpers import DisplayProperties

class QuantumCircuitWindow:
    """Wrapper around the window, to handle data."""

    def __init__(self) -> None:
        """Initialize the wrapper around the window object for pyqt."""
        # lookup table for classification colors
        self.app = start_application()
        self.runner = None
        self.window = MainWindow()
        self.window.showMaximized()
        self.circuit_initialized = False
        self.bloch_backend = VisualizationWrapper(self.window)
        
    def initialize_circuit_properties(self, quantum_circuit: qiskit.QuantumCircuit, frames_per_animation: int, display_properies: Optional[DisplayProperties] = None) -> None:
        """Wrapper for initializing the displays for the circuits

        Args:
            quantum_circuit: The quantum circuit that will be worked on.
            frames_per_animation: The frames that will be used for each animation block.
            display_properties: The down line properties that can be edited.
        """
        self.bloch_backend.setup_circuit(quantum_circuit, frames_per_animation, display_properies)
        self.circuit_initialized = True
        
    def add_circuit_state(self, quantum_circuit: qiskit.QuantumCircuit, display_properies: DisplayProperties, fast_state: bool = False) -> None:
        """Add a new circuit state that the initial or previous states can move to.

        Args:
            quantum_circuit: The quantum circuit that will be worked on.
            display_properties: The down line properties that can be edited.
        """
        if not self.circuit_initialized:
            raise ValueError("Circuit not initialized, call initialize_circuit_properties.")
        self.bloch_backend.add_circuit_state(quantum_circuit, display_properies, fast_state)

    def animate_circuit(self) -> None:
        """Animate the different displays.
        """
        self.bloch_backend.setup_animation_process()
        
