"""Handles the entanglement matrix backend."""
import numpy as np
from typing import List, Optional
from gui_backend.helpers import DisplayProperties
from numpy.typing import NDArray
import qiskit
from qiskit.quantum_info import Statevector, partial_trace
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from gui.helpers import background_color

class AnimatedMatrix():
    """A generic matrix that can interpolate to another value.
    """
    
    def __init__(self, matrix_value: NDArray[np.float64]) -> None:
        """Initialize the matrix that can animate.

        Does not overload np.ndarray because that has intrinsic properties that must be followed.

        Args:
            matrix_value: The value of the matrix that should be animated.
        """
        self.current_matrix_value: NDArray = matrix_value

    def __array__(self) -> NDArray[np.float64]:
        """Returns the animated matrix so that the 

        Returns:
            The array value of the base matrix.
        """
        return self.current_matrix_value

    def acquire_interpolated_value(
        self, 
        interpolation_ratio: float, 
        to_transition_to_matrix: NDArray,
    ) -> NDArray[np.float64]:
        """Sinusoidal transition between the current matrix and the provided matrix.

        Args:
            interpolation_ratio: The current point of interpolation between this matrix and the next.
            to_transition_to_matrix: The matrix to transition to.

        Returns:
            The interpolated matrix.
        """
        alpha_sin = (1 - np.cos(np.pi * interpolation_ratio)) / 2
        transition_matrix = (1 - alpha_sin) * self.current_matrix_value + alpha_sin * to_transition_to_matrix
        
        return transition_matrix


class AnimationBlockEntanglement():
    """A block of information for the animation steps to use.
    """
    def __init__(self, entanglement_values: AnimatedMatrix, display_properties: DisplayProperties) -> None:
        """Initialize the animation block.

        Args:
            entanglement_values: The matrix of this step.
            display_properties: The properties to display for this step.
        """
        self.animated_matrix: AnimatedMatrix = entanglement_values
        self.matrix_color: str = display_properties.entanglement_colors
        self.display_properties: Optional[DisplayProperties] = display_properties
        

class EntanglementMatrix():
    """The matrix to view the entanglement of the quantum circuit.
    """
    
    def __init__(self, quantum_circuit: qiskit.QuantumCircuit, figure: Figure, axes: Axes, display_properties: Optional[DisplayProperties]) -> None:
        """Initialize the class attributes for the entanglement matrix.

        Args:
            quantum_circuit: The circuit to visualize.
            figure: The figure to be handled.
            axes: The axes of the figure to be handled.
            display_properties: The initial display properties.
        """
        self.figure = figure
        self.axes = axes
        self.image: Optional[AxesImage] = None
        self.image_text = []
        self.currently_displayed_index = 0
        self.next_index_to_display = 0
        
        self._animation_blocks: List[AnimationBlockEntanglement] = []
        self.append_block(quantum_circuit, display_properties)
        self._initialize_heatmap(quantum_circuit, display_properties)

    def _initialize_heatmap(self, quantum_circuit: qiskit.QuantumCircuit, display_properties: DisplayProperties) -> None:
        """Initialize the actual heatmap display properties.

        Args:
            quantum_circuit: The circuit to visualize.
            display_properties: The initial display properties.
        """

        self.figure.patch.set_facecolor(background_color)

        self.axes.set_xticks(np.arange(np.array(self._animation_blocks[0].animated_matrix).shape[1]))
        self.axes.set_yticks(np.arange(np.array(self._animation_blocks[0].animated_matrix).shape[0]))


        self.axes.set_xticklabels(
            [f"{quantum_register.name}: qubit {quantum_register.index(qubit)}" for quantum_register in quantum_circuit.qregs for qubit in quantum_register],
            rotation=45, color='white'
        )
        self.axes.set_yticklabels(
            [f"{quantum_register.name}: qubit {quantum_register.index(qubit)}" for quantum_register in quantum_circuit.qregs for qubit in quantum_register],
            color='white'
        )
        self.axes.set_xticks(np.arange(np.array(self._animation_blocks[0].animated_matrix).shape[1]+1)-0.5, minor=True)
        self.axes.set_yticks(np.arange(np.array(self._animation_blocks[0].animated_matrix).shape[0]+1)-0.5, minor=True)
        self.axes.grid(which="minor", color="w", linestyle='-', linewidth=2)
        self.axes.tick_params(which="minor", bottom=False, left=False)

        self.image = self.axes.imshow(np.array(self._animation_blocks[0].animated_matrix), cmap='viridis', origin='lower', vmin=0, vmax=1)
        color_bar = self.figure.colorbar(self.image, ax=self.axes)
        color_bar.set_label('Entanglement Scale', color='white')

        # Loop over data dimensions and create text annotations.
        for i in range(np.array(self._animation_blocks[0].animated_matrix).shape[0]):
            self.image_text.append([])
            for j in range(np.array(self._animation_blocks[0].animated_matrix).shape[1]):
                text = self.axes.text(j, i, f"{np.array(self._animation_blocks[0].animated_matrix)[i, j]:.2f}", ha='center', va='center', color='white')
                self.image_text[i].append(text)

    def append_block(
        self, 
        quantum_circuit: qiskit.QuantumCircuit, 
        display_properties: Optional[DisplayProperties] = None, 
    ) -> None:
        """Append a new quantum circuit state to animate.

        Args:
            quantum_circuit: The circuit to visualize.
            display_properties: The initial display properties.
        """
        matrix_to_add = self.mutual_entanglement(quantum_circuit)
        self._animation_blocks.append(AnimationBlockEntanglement(
            matrix_to_add, display_properties
        ))
        
    def update_heatmap(self, interpolation_ratio) -> None:
        """Update the heatmap to the next interpolation item.

        Args:
            interpolation_ratio: The current point of interpolation between this matrix and the next.
        """
        current_matrix = self._animation_blocks[self.currently_displayed_index].animated_matrix
        next_matrix = self._animation_blocks[self.next_index_to_display].animated_matrix

        transition_matrix = current_matrix.acquire_interpolated_value(interpolation_ratio, next_matrix)
        self.image.set_data(transition_matrix)

        for i in range(transition_matrix.shape[0]):
            for j in range(transition_matrix.shape[1]):
                self.image_text[i][j].set_text(f"{transition_matrix[i, j]:.2f}")


    def next_animation_block(self, to_transition_index: int) -> bool:
        """Jump to the next block for animation.

        Args:
            to_transition_index: The index of the animation block to jump to.

        Returns:
            Whether it was able to jump to this index.
        """
        if 0 > to_transition_index > len(self._animation_blocks):
            return False
        else:
            self.currently_displayed_index = self.next_index_to_display
            self.next_index_to_display = to_transition_index
            self.axes.set_title(self._animation_blocks[self.next_index_to_display].display_properties.plot_name)

            return True
        
    @staticmethod
    def von_neumann_entropy(qubit_trace: np.ndarray) -> float:
        """Calculate entropy for a single qubit trace based on Von Neumanns entropy equation.

        Args:
            qubit_trace: The qubit trace to use for calculating entropy.

        Returns:
            The current entropy.
        """
        qubit_trace = (qubit_trace + qubit_trace.conj().T) / 2
        qubit_trace = qubit_trace / np.trace(qubit_trace)
        eigen_values = np.linalg.eigvalsh(qubit_trace)

        # filter out zeros
        eigen_values = eigen_values[eigen_values > 1e-12]
        return -np.sum(eigen_values * np.log2(eigen_values))
        
    def von_neumman_mutual_information(self, qubit_trace_1, qubit_trace_2, qubit_trace_1_2):
        # compute von neumon entropy for one qubit, another qubit and their joint state
        entropy_qubit_1 = self.von_neumann_entropy(qubit_trace_1)
        entropy_qubit_2 = self.von_neumann_entropy(qubit_trace_2)
        entropy_qubits_state = self.von_neumann_entropy(qubit_trace_1_2)

        # calculate mutual info based on the purity of state 1 and 2, minus the difference
        mutal_information = (entropy_qubit_1 + entropy_qubit_2) - entropy_qubits_state
        return mutal_information

    @staticmethod
    def purity(qubit_trace: np.ndarray) -> float:
        """Calculate the purity of the single qubit.

        Args:
            qubit_trace: The qubit trace to find the purity of.

        Returns:
            The purity of the qubit.
        """
        return np.real(np.trace(qubit_trace @ qubit_trace))

    def mutual_entanglement(self, quantum_circuit: qiskit.QuantumCircuit) -> AnimatedMatrix:
        """The entanglement of the entire quantum circuit, between each qubit.

        Args:
            quantum_circuit: The quantum circuit to find the entanglement of.

        Returns:
            The matrix that will be used in the animation block.
        """
        combination_matrices = np.empty((len(quantum_circuit.qubits), len(quantum_circuit.qubits)), dtype=np.float64)
        entanglement_matrix = AnimatedMatrix(combination_matrices)
        quantum_vector = Statevector.from_instruction(quantum_circuit)
        qubit_list = list(range(len(quantum_circuit.qubits)))
        qubit_traces: List[NDArray[np.complex128]] = []
        # find the single qubit traces
        for qubit_number in qubit_list:
            qubit_trace = qubit_list.copy()
            qubit_trace.remove(qubit_number)
            qubit_traces.append(partial_trace(quantum_vector, qubit_trace).data)
            combination_matrices[qubit_number, qubit_number] = self.purity(qubit_traces[-1])
            
        # find multi qubit state and von neumman
        for qubit_number_1 in qubit_list:
            for qubit_number_2 in qubit_list:
                if qubit_number_1 == qubit_number_2:
                    continue
                combination_qubit = qubit_list.copy()
                combination_qubit.remove(qubit_number_1)
                combination_qubit.remove(qubit_number_2)
                von_neuman = self.von_neumman_mutual_information(
                    qubit_traces[qubit_number_1], 
                    qubit_traces[qubit_number_2],
                    partial_trace(quantum_vector, combination_qubit).data, 
                )
                combination_matrices[qubit_number_1, qubit_number_2] = np.round(von_neuman, 3) / 2
                
        entanglement_matrix.current_matrix_value = combination_matrices
        return entanglement_matrix

    def to_x_block(self, block_to_jump_to: int) -> bool:
        """Move to some x block.

        Args:
            block_to_jump_to: The animation block to jump to.

        Returns:
            Whether it was able to jump to that block.
        """
        return self.next_animation_block(block_to_jump_to)

    def to_next_block(self) -> bool:
        """Move to next block.

        Returns:
            Whether it was able to jump to that block.
        """
        return self.next_animation_block(self.next_index_to_display+1)

    def to_prev_block(self) -> bool:
        """Move to previous block.

        Returns:
            Whether it was able to jump to that block.
        """
        return self.next_animation_block(self.next_index_to_display-1)