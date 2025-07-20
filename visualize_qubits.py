import numpy as np
from typing import List, Union
import matplotlib.pyplot as plt
from matplotlib.quiver import Quiver
import qiskit
from numpy.typing import NDArray
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace, entropy
from qiskit_aer import Aer
from qiskit.visualization import plot_bloch_vector
from functools import partial
from matplotlib.animation import FuncAnimation
import timeit
from scipy.spatial.transform import Rotation as R, Slerp
from dataclasses import dataclass, fields
from copy import deepcopy

_NUMBER_OF_FRAMES = 100
_QUBIT_HILBERT_SPACE = 2

@dataclass
class QubitMatrices():
    state_matrices: List[List[NDArray[np.float64]]]
    mixed_matrices_1: List[List[NDArray[np.float64]]]
    mixed_matrices_2: List[List[NDArray[np.float64]]]

@dataclass
class Quivers():
    state_matrices: List[Quiver]
    mixed_matrices_1: List[Quiver]
    mixed_matrices_2: List[Quiver]
    state_matrices_color: str = "black"
    mixed_matrices_1_color: str = "cyan"
    mixed_matrices_2_color: str = "purple"
    state_matrices_alpha: float = 1.0
    mixed_matrices_1_alpha: float = 0.3
    mixed_matrices_2_alpha: float = 0.3


def get_x_y_z(matrix):
    x = 2 * np.real(matrix[0, 1])
    y = 2 * np.imag(matrix[0, 1])
    z = np.real(matrix[0, 0] - matrix[1, 1])
    return np.array([x, y, z])

def get_eigen_bloch_vector(eigen_vectors):
    bloch_vectors = []
    for i in range(_QUBIT_HILBERT_SPACE):
        eigen_vector = eigen_vectors[:, i]

        function = Statevector(eigen_vector)
        
        operator = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])

        bloch_vector = np.real(function.data.conj().T @ operator @ function.data)
        bloch_vectors.append(bloch_vector)
    return bloch_vectors

def append_matrices(matrices: QubitMatrices, matrix_to_add, qubit_number):
    matrices.state_matrices[qubit_number].append(get_x_y_z(matrix_to_add))

    eigen_values, eigen_vectors = np.linalg.eigh(matrix_to_add)
    bloch_vectors = get_eigen_bloch_vector(eigen_vectors)
    matrices.mixed_matrices_1[qubit_number].append(bloch_vectors[0])
    matrices.mixed_matrices_2[qubit_number].append(bloch_vectors[1])

class bloch_spheres():
    def __init__(self, init_matrices: List[NDArray[np.float64]], number_of_qubits: int):
        self.number_of_qubits = number_of_qubits
        self.figure = plt.figure()
        self.axes = []
        self.bloch_vectors = Quivers([], [], [])
        empty_list = [[] for _ in range(self.number_of_qubits)]
        self.matrices = QubitMatrices(deepcopy(empty_list), deepcopy(empty_list), deepcopy(empty_list))
        self.current_matrix_index = 0
        self.last_known_rotation = QubitMatrices(deepcopy(empty_list), deepcopy(empty_list), deepcopy(empty_list))
        for qubit_number in range(self.number_of_qubits):
            self.axes.append(
                self.figure.add_subplot(int(np.ceil(np.sqrt(self.number_of_qubits))), int(np.ceil(np.sqrt(self.number_of_qubits))), qubit_number+1, projection='3d')
            )
            append_matrices(self.matrices, init_matrices[qubit_number], qubit_number)
            for field in fields(self.matrices):
                name = field.name
                matrices = getattr(self.matrices, name)
                last_known_rot_matrix = getattr(self.last_known_rotation, name)
                last_known_rot_matrix[qubit_number] = matrices[qubit_number][0] / np.linalg.norm(matrices[qubit_number][0])
                
        self.create_bloch_sphere()

    @staticmethod
    def normalize(v):
        return v / np.linalg.norm(v)

    def rotation_slerp(self, start_matrix, end_matrix, t):
        # written by chatGPT, needs to be reworked to utilize a half transform for correct rotations
        start_matrix = start_matrix / np.linalg.norm(start_matrix)
        end_matrix = end_matrix / np.linalg.norm(end_matrix)
        dot = np.dot(start_matrix, end_matrix)

        if np.isclose(dot, 1.0):
            return start_matrix 
        elif np.isclose(dot, -1.0):
            axis = np.cross(start_matrix, np.array([1, 0, 0]))
            if np.linalg.norm(axis) < 1e-6:
                axis = np.cross(start_matrix, np.array([0, 1, 0]))
            axis = axis / np.linalg.norm(axis)
            rot = R.from_rotvec(np.pi * axis)
        else:
            axis = np.cross(start_matrix, end_matrix)
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(dot)
            rot = R.from_rotvec(angle * axis)
        slerp = Slerp([0, 1], R.concatenate([R.identity(), rot]))
        rot_t = slerp([t])[0]
        return rot_t.apply(start_matrix)

    def update(self, frame: int):
        for qubit_number in range(self.number_of_qubits):
            for field in fields(self.matrices):
                name = field.name
                matrices = getattr(self.matrices, name)
                last_known_rot_matrix = getattr(self.last_known_rotation, name)
                bloch_quiver = getattr(self.bloch_vectors, name)
                current_matrix = matrices[qubit_number][self.current_matrix_index]
                next_matrix = matrices[qubit_number][self.current_matrix_index+1]
                interpolation_ratio = (frame+1) / _NUMBER_OF_FRAMES
                if (np.linalg.norm(current_matrix) == 0.0 and np.linalg.norm(next_matrix) != 0.0):
                    current_matrix += last_known_rot_matrix[qubit_number]*0.01
                elif (np.linalg.norm(current_matrix) != 0.0 and np.linalg.norm(next_matrix) == 0.0):
                    next_matrix += last_known_rot_matrix[qubit_number]*0.01
                    
                transition_matrix = self.rotation_slerp(current_matrix/np.linalg.norm(current_matrix), next_matrix/np.linalg.norm(next_matrix), interpolation_ratio)
                    
                # print(f"post_rotation {qubit_number}: {transition_matrix}")
                transition_matrix = transition_matrix * (np.linalg.norm(current_matrix) + (np.linalg.norm(next_matrix) - np.linalg.norm(current_matrix)) * interpolation_ratio)
                # print(f"post_magnitude {qubit_number}: {transition_matrix}")
                quiver_to_remove = bloch_quiver[qubit_number]
                quiver_to_remove.remove()
                
                bloch_quiver[qubit_number] = self.axes[qubit_number].quiver(
                    0, 0, 0, transition_matrix[0], transition_matrix[1], transition_matrix[2], color=getattr(self.bloch_vectors, f"{name}_color"), arrow_length_ratio=0.1, alpha = getattr(self.bloch_vectors, f"{name}_alpha")
                )
        if frame == (_NUMBER_OF_FRAMES) - 1:
            self.current_matrix_index += 1
            for field in fields(self.matrices):
                name = field.name
                last_known_rot_matrix = getattr(self.last_known_rotation, name)
                last_known_rot_matrix[qubit_number] = next_matrix / np.linalg.norm(next_matrix)
            
    def create_bloch_sphere(self):
        for qubit_number in range(self.number_of_qubits):
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones_like(u), np.cos(v))
            self.axes[qubit_number].plot_wireframe(x, y, z, color='lightblue', alpha=0.1, zorder=3)
            
            self.axes[qubit_number].quiver(0, 0, 0, 0.77, 0, 0, color='r', arrow_length_ratio=0.1, alpha=0.3, zorder=2)
            self.axes[qubit_number].quiver(0, 0, 0, 0, 0.77, 0, color='g', arrow_length_ratio=0.1, alpha=0.3, zorder=2)
            self.axes[qubit_number].quiver(0, 0, 0, 0, 0, 0.77, color='b', arrow_length_ratio=0.1, alpha=0.3, zorder=2)

            # Settings
            self.axes[qubit_number].set_xlim([-1, 1])
            self.axes[qubit_number].set_ylim([-1, 1])
            self.axes[qubit_number].set_zlim([-1, 1])
            self.axes[qubit_number].set_box_aspect([1,1,1])
            self.axes[qubit_number].axis('off')
            
            self.draw_quivers(0, qubit_number)
            
            self.axes[qubit_number].text(x=0.0, y=0.0, z=1.2, s='|0⟩', color='black', fontsize=8, zorder=1)
            self.axes[qubit_number].text(x=0.0, y=0.0, z=-1.4, s='|1⟩', color='black', fontsize=8, zorder=1)
            self.axes[qubit_number].text(x=0.0, y=1.1, z=0.0, s='y', color='black', fontsize=8, zorder=1)
            self.axes[qubit_number].text(x=1.1, y=0.0, z=0.0, s='x', color='black', fontsize=8, zorder=1)


    def draw_quivers(self, matrix_number, qubit_number):
        for field in fields(self.matrices):
            name = field.name
            matrices = getattr(self.matrices, name)
            bloch_quiver = getattr(self.bloch_vectors, name)
            # Initial Bloch vector
            bloch_quiver.append(
                self.axes[qubit_number].quiver(
                    0, 
                    0, 
                    0, 
                    matrices[qubit_number][matrix_number][0], 
                    matrices[qubit_number][matrix_number][1], 
                    matrices[qubit_number][matrix_number][2], 
                    color=getattr(self.bloch_vectors, f"{name}_color"), 
                    arrow_length_ratio=0.1,
                    alpha = getattr(self.bloch_vectors, f"{name}_alpha")
                )
            )            
