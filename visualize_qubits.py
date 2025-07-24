import numpy as np
from typing import List, Iterable, Dict, Optional, Generator
import matplotlib.pyplot as plt
from matplotlib.quiver import Quiver
from matplotlib.axes import Axes
import matplotlib.gridspec as gridspec
from numpy.typing import NDArray
import qiskit
import qiskit.circuit
from qiskit.quantum_info import Statevector, entropy, partial_trace, DensityMatrix
from scipy.spatial.transform import Rotation as R, Slerp
from dataclasses import dataclass, fields
from copy import deepcopy
from matplotlib.animation import FuncAnimation

_QUBIT_HILBERT_SPACE = 2

class ToAnimateMatrix():
    def __init__(self, matrix_value):
        self.current_matrix_value: NDArray = matrix_value
        self.previous_nonzero_rotation = None

    def __array__(self) -> NDArray[np.float64]:
        return self.current_matrix_value

    @staticmethod
    def normalize(matrix):
        return matrix / np.linalg.norm(matrix)
    
    def rotation_slerp(self, start_matrix, end_matrix, t):
        start_matrix = start_matrix / np.linalg.norm(start_matrix)
        end_matrix = end_matrix / np.linalg.norm(end_matrix)
        dot = np.dot(start_matrix, end_matrix)
        # if we are near the original, return our original
        if np.isclose(dot, 1.0):
            return start_matrix 
        # if we oppose the original
        elif np.isclose(dot, -1.0):
            axis = np.cross(start_matrix, np.array([1, 0, 0]))
            # if there is a minute difference in the y and the z, then instead we use x and z
            if np.linalg.norm(axis) < 1e-6:
                axis = np.cross(start_matrix, np.array([0, 1, 0]))
            # normalize
            axis = self.normalize(axis)
            rot = R.from_rotvec(np.pi * axis)
        else:
            axis = np.cross(start_matrix, end_matrix)
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(dot)
            rot = R.from_rotvec(angle * axis)
        slerp = Slerp([0, 1], R.concatenate([R.identity(), rot]))
        rot_t = slerp([t])[0]
        return rot_t.apply(start_matrix)

    def acquire_interpolated_value(
        self, 
        interpolation_ratio: float, 
        to_transition_to_matrix: NDArray,
    ):
        if (np.linalg.norm(self.current_matrix_value) == 0.0 and np.linalg.norm(to_transition_to_matrix) != 0.0):
            current_matrix = self.current_matrix_value + self.previous_nonzero_rotation*0.01
        elif (np.linalg.norm(self.current_matrix_value) != 0.0 and np.linalg.norm(to_transition_to_matrix) == 0.0):
            to_transition_to_matrix = to_transition_to_matrix + self.previous_nonzero_rotation*0.01
            current_matrix = self.current_matrix_value
        else:
            current_matrix = self.current_matrix_value 

        transition_matrix = self.rotation_slerp(
            self.normalize(current_matrix), 
            self.normalize(to_transition_to_matrix), 
            interpolation_ratio
        )
            
        # print(f"post_rotation {qubit_number}: {transition_matrix}")
        transition_matrix = transition_matrix * (
            np.linalg.norm(self.current_matrix_value) 
            + (np.linalg.norm(to_transition_to_matrix) - np.linalg.norm(to_transition_to_matrix)) 
            * interpolation_ratio
        )
        
        return transition_matrix

@dataclass
class BlochProperties():
    plot_name: str = ""
    display_on_all: bool = False
    quiver_state_color: str = "black"
    quiver_state_alpha: float = 1.0
    quiver_mixed_1_color: str = "cyan"
    quiver_mixed_1_alpha: float = 0.2
    quiver_mixed_2_color: str = "purple"
    quiver_mixed_2_alpha: float = 0.2

@dataclass
class ValueSet():
    animated_matrix: ToAnimateMatrix
    quiver_color: str
    quiver_alpha: float

class AnimationBlock():
    def __init__(self, state_matrices, mixed_matrices_1, mixed_matrices_2, bloch_properties: BlochProperties):
        self.state_value_set: ValueSet = ValueSet(
            animated_matrix=state_matrices,
            quiver_color=bloch_properties.quiver_state_color,
            quiver_alpha=bloch_properties.quiver_state_alpha
        )
        self.mixed_1_value_set: ValueSet = ValueSet(
            animated_matrix=mixed_matrices_1,
            quiver_color=bloch_properties.quiver_mixed_1_color,
            quiver_alpha=bloch_properties.quiver_mixed_1_alpha
        )
        self.mixed_2_value_set: ValueSet = ValueSet(
            animated_matrix=mixed_matrices_2,
            quiver_color=bloch_properties.quiver_mixed_2_color,
            quiver_alpha=bloch_properties.quiver_mixed_2_alpha
        )
        self.bloch_properties: Optional[BlochProperties] = bloch_properties
        
    def __iter__(self) -> Iterable[ToAnimateMatrix]:
        return iter()

    def state_quiver_mix(self) -> Generator[ValueSet, None, None]:
        value_sets = [self.state_value_set, self.mixed_1_value_set, self.mixed_2_value_set]
        for value_set in value_sets:
            yield value_set

class PerQubitProperties():
    def __init__(self, subset_axes: Axes, initial_matrix: NDArray, bloch_properties: Optional[BlochProperties]):
        self.currently_displayed_index = 0
        self.next_index_to_display = 0
        self._animation_blocks: List[AnimationBlock] = []
        self._subset_axes: Axes = subset_axes
        self._initialize_bloch_sphere()
        self._append_block(initial_matrix, bloch_properties)

        # ugly, but this is the best way to get around locally scoped self issues in iterables
        self.quiver_dict = {
            "state_quiver": None,
            "mixed_1_quiver": None,
            "mixed_2_quiver": None,
        }

        for quiver_name, quiver, value_set in zip(
            self.quiver_dict.keys(), 
            self.quiver_dict.values(), 
            self._animation_blocks[self.currently_displayed_index].state_quiver_mix()
        ):
            self.update_quiver(
                quiver_name,
                quiver, 
                np.array(value_set.animated_matrix), 
                value_set.quiver_color, 
                value_set.quiver_alpha
            )

    def _initialize_bloch_sphere(self):
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))
        self._subset_axes.plot_wireframe(x, y, z, color='lightblue', alpha=0.1, zorder=3)
        
        self._subset_axes.quiver(0, 0, 0, 0.77, 0, 0, color='r', arrow_length_ratio=0.1, alpha=0.3, zorder=2)
        self._subset_axes.quiver(0, 0, 0, 0, 0.77, 0, color='g', arrow_length_ratio=0.1, alpha=0.3, zorder=2)
        self._subset_axes.quiver(0, 0, 0, 0, 0, 0.77, color='b', arrow_length_ratio=0.1, alpha=0.3, zorder=2)

        # Settings
        self._subset_axes.set_xlim([-1, 1])
        self._subset_axes.set_ylim([-1, 1])
        self._subset_axes.set_zlim([-1, 1])
        self._subset_axes.set_box_aspect([1,1,1])
        self._subset_axes.axis('off')
                
        self._subset_axes.text(x=0.0, y=0.0, z=1.2, s='|0⟩', color='black', fontsize=8, zorder=1)
        self._subset_axes.text(x=0.0, y=0.0, z=-1.4, s='|1⟩', color='black', fontsize=8, zorder=1)
        self._subset_axes.text(x=0.0, y=1.1, z=0.0, s='y', color='black', fontsize=8, zorder=1)
        self._subset_axes.text(x=1.1, y=0.0, z=0.0, s='x', color='black', fontsize=8, zorder=1)
        
    def update_quiver(self, quiver_name, quiver, transition_matrix, quiver_color: str, quiver_alpha: float):
        if quiver is not None:
            quiver.remove()
        self.quiver_dict[quiver_name] = self._subset_axes.quiver(
            0, 
            0, 
            0, 
            transition_matrix[0], 
            transition_matrix[1], 
            transition_matrix[2], 
            color=quiver_color, 
            arrow_length_ratio=0.1, 
            alpha = quiver_alpha
        )
        
    def update_plots(self, interpolation_ratio):
        for value_set, value_set_transition, quiver_name, plot_quiver in zip(
            self._animation_blocks[self.currently_displayed_index].state_quiver_mix(), 
            self._animation_blocks[self.next_index_to_display].state_quiver_mix(),
            self.quiver_dict.keys(),
            self.quiver_dict.values(),
        ):
            if (
                not np.all(np.isclose(
                    np.array(value_set.animated_matrix),
                    np.array(value_set_transition.animated_matrix)
                ))
            ):
                state_interpolated = value_set.animated_matrix.acquire_interpolated_value(
                    interpolation_ratio,
                    np.array(value_set_transition.animated_matrix)
                )
                self.update_quiver(
                    quiver_name,
                    plot_quiver,
                    state_interpolated,
                    value_set_transition.quiver_color,
                    value_set_transition.quiver_alpha
                )

    @staticmethod
    def get_x_y_z(matrix):
        x = 2 * np.real(matrix[0, 1])
        y = 2 * np.imag(matrix[0, 1])
        z = np.real(matrix[0, 0] - matrix[1, 1])
        return np.array([x, y, z])

    @staticmethod
    def get_eigen_bloch_vector(eigen_vectors):
        bloch_vectors = []
        for i in range(_QUBIT_HILBERT_SPACE):
            eigen_vector = eigen_vectors[:, i]

            function = Statevector(eigen_vector)
            
            operator = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])

            bloch_vector = np.real(function.data.conj().T @ operator @ function.data)
            bloch_vectors.append(bloch_vector)
        return bloch_vectors

    def _append_block(
        self, 
        matrix_to_add: NDArray, 
        bloch_properties: Optional[BlochProperties] = None, 
    ):
            
        next_matrix = ToAnimateMatrix(self.get_x_y_z(matrix_to_add))
        
        percentiles, mixed_vectors = np.linalg.eigh(matrix_to_add)
        bloch_vectors = self.get_eigen_bloch_vector(mixed_vectors)
        next_mixed_matrix_1 = ToAnimateMatrix(bloch_vectors[0])
        next_mixed_matrix_2 = ToAnimateMatrix(bloch_vectors[1])
        if np.all(next_matrix == 0.0) and len(self._animation_blocks) != 0:
            next_matrix.previous_nonzero_rotation = self._animation_blocks[-1].state_value_set.animated_matrix.previous_nonzero_rotation
        else:
            next_matrix.previous_nonzero_rotation = np.array(next_matrix) / np.linalg.norm(np.array(next_matrix))
        if bloch_properties is None:
            bloch_properties = BlochProperties()
        self._animation_blocks.append(AnimationBlock(
            next_matrix, next_mixed_matrix_1, next_mixed_matrix_2, bloch_properties
        ))
        
    def next_animation_block(self, to_transition_index: int) -> bool:
        if to_transition_index > len(self._animation_blocks):
            return False
        else:
            self.currently_displayed_index = self.next_index_to_display
            self.next_index_to_display = to_transition_index
            
            change_in_matrix = False
            for value_set, value_set_transition in zip(
                self._animation_blocks[self.currently_displayed_index].state_quiver_mix(), 
                self._animation_blocks[self.next_index_to_display].state_quiver_mix(),
            ):
                if (
                    not np.all(np.isclose(
                        np.array(value_set.animated_matrix),
                        np.array(value_set_transition.animated_matrix)
                    ))
                ):
                    change_in_matrix = True
            if change_in_matrix:
                self._subset_axes.set_title(self._animation_blocks[self.currently_displayed_index+1].bloch_properties.plot_name)

            return True
    
    def to_next_block(self) -> bool:
        return self.next_animation_block(self.currently_displayed_index+1)

class BlochSpheres():
    def __init__(self, quantum_circuit: qiskit.QuantumCircuit, frames_per_animation: int, bloch_properties: Optional[BlochProperties] = None):
        self.figure = plt.figure()
        self.frames_per_animation = frames_per_animation
        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()
        
        num_quantum_registers = len(quantum_circuit.qregs)
        
        self.main_grid_layout = gridspec.GridSpec(
            int(np.ceil(np.sqrt(num_quantum_registers))), 
            int(np.ceil(np.sqrt(num_quantum_registers))), 
            height_ratios=[1, 1], 
            hspace=0.4
        )
    
        self._quantum_register_grid: Dict[qiskit.QuantumRegister, gridspec.GridSpecFromSubplotSpec] = {}
        self._qubit_subplots: Dict[qiskit.circuit.Qubit, PerQubitProperties] = {}
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
                self._qubit_subplots[quantum_circuit.qubits.index(qubit)] = PerQubitProperties(self.figure.add_subplot(
                    self._quantum_register_grid[quantum_register.name][qubit_number%square_length, qubit_number//square_length], projection='3d'
                ), reduced_state_vector, bloch_properties)
            
        self._frame = 0
        self._total_frame_count = 0

    @property
    def qubit_subplots(self) -> Dict[qiskit.circuit.Qubit, PerQubitProperties]:
        return self._qubit_subplots
    
    def add_circuit_state(
        self, 
        quantum_circuit: qiskit.QuantumCircuit, 
        bloch_properties: BlochProperties, 
    ):
        quantum_vector = Statevector.from_instruction(quantum_circuit)
        keep_indices = list(range(len(quantum_circuit.qubits)))
        for qubit in quantum_circuit.qubits:
            traced_out_indices = keep_indices.copy()
            traced_out_indices.remove(quantum_circuit.qubits.index(qubit))
            reduced_state_vector = partial_trace(quantum_vector, traced_out_indices).data
            self._qubit_subplots[quantum_circuit.qubits.index(qubit)]._append_block(
                reduced_state_vector, bloch_properties
            )

    def _update(self):
        interpolation_ratio = self._frame / self.frames_per_animation
        for per_qubit_obj in self._qubit_subplots.values():
            per_qubit_obj.update_plots(interpolation_ratio)
            
    def _run_next_animation_blocks(self):
        for per_qubit_obj in self._qubit_subplots.values():
            per_qubit_obj.to_next_block()
        
    def animate_bloch_sphere(self):
        self._run_next_animation_blocks()
        def update(frame):
            self._update()

            self._frame += 1
            if self._frame == self.frames_per_animation:
                self._frame = 0
                more_animation = self._run_next_animation_blocks()
                
                if not more_animation:
                    func_animation.event_source.stop()
                
        func_animation = FuncAnimation(self.figure, update, interval=5, blit=False, repeat=False)
        plt.show()           
 

def von_neuman_info(qubit_trace_1, qubit_trace_2, qubit_trace_1_2):
    # compute von neumon entropy
    entropy_qubit_1 = entropy(qubit_trace_1, base=2)
    entropy_qubit_2 = entropy(qubit_trace_2, base=2)
    entropy_qubits_state = entropy(qubit_trace_1_2, base=2)

    # calculate mutal info
    mutal_information = (entropy_qubit_1 + entropy_qubit_2) - entropy_qubits_state
    return mutal_information

def mutual_entanglement(quantum_circuit: qiskit.QuantumCircuit):
    combination_matrices = np.empty((len(quantum_circuit.qregs), len(quantum_circuit.qregs)), dtype=np.float64)
    density_matrix = DensityMatrix.from_instruction(quantum_circuit).data
    for qubit_number_1 in range(len(quantum_circuit)):
        combination_qubit = list(range(len(quantum_circuit)))
        qubit_trace_1 = list(range(len(quantum_circuit)))
        qubit_trace_1.remove(qubit_number_1)
        combination_qubit.remove(qubit_number_1)
        for qubit_number_2 in range(len(quantum_circuit)):
            if qubit_number_1 == qubit_number_2:
                continue
            qubit_trace_2 = list(range(len(quantum_circuit)))
            qubit_trace_2.remove(qubit_number_2)
            combination_qubit.remove(qubit_number_2)
            von_neuman = von_neuman_info(
                partial_trace(density_matrix, qubit_trace_1).data, 
                partial_trace(density_matrix, qubit_trace_2).data, 
                partial_trace(density_matrix, combination_qubit).data, 
            )
            combination_matrices[qubit_number_1, qubit_number_2] = von_neuman
    print(combination_matrices)


