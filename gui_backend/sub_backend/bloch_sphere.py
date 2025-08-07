"""Handles the bloch spheres backend."""
import numpy as np
from typing import List, Iterable, Optional, Generator
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.quiver import Quiver
from gui_backend.helpers import DisplayProperties
from numpy.typing import NDArray
from qiskit.quantum_info import Statevector
from scipy.spatial.transform import Rotation as R, Slerp
from dataclasses import dataclass
from gui_backend.helpers import _QUBIT_HILBERT_SPACE
from gui.helpers import background_color
from gui_backend.sub_backend.display import GenericDisplay, AnimationBlock

class SphericalAnimatedMatrix():
    """A matrix that can be interpolated around a spherical grid.
    """
    def __init__(self, matrix_value: NDArray[np.float64]) -> None:
        """Initialize the matrix that can animate.

        Does not overload np.ndarray because that has intrinsic properties that must be followed.

        Args:
            matrix_value: The value of the matrix that should be animated.
        """
        self.current_matrix_value: NDArray = matrix_value
        self.previous_nonzero_rotation = None

    def __array__(self) -> NDArray[np.float64]:
        """Returns the animated matrix so that the 

        Returns:
            The array value of the base matrix.
        """
        return self.current_matrix_value

    @staticmethod
    def normalize(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        """Normalizes a matrix.

        Args:
            matrix: The matrix to normalize

        Returns:
            The normalized matrix.
        """
        return matrix / np.linalg.norm(matrix)
    
    def rotation_slerp(self, start_matrix: NDArray[np.float64], to_transition_to_matrix: NDArray[np.float64], interpolation_ratio: float) -> NDArray[np.float64]:
        """Rotate the start matrix to some location between the two matrices.

        Args:
            start_matrix: The original start matrix.
            to_transition_to_matrix: The matrix to rotate to.
            interpolation_ratio: The current point of interpolation between this matrix and the next.

        Returns:
            The rotated matrix.
        """
        start_matrix = start_matrix / np.linalg.norm(start_matrix)
        to_transition_to_matrix = to_transition_to_matrix / np.linalg.norm(to_transition_to_matrix)
        orthogonality = np.dot(start_matrix, to_transition_to_matrix)
        # if we are near the original, return our original
        if np.isclose(orthogonality, 1.0):
            return start_matrix 
        # if we oppose the original
        elif np.isclose(orthogonality, -1.0):
            axis = np.cross(start_matrix, np.array([1, 0, 0]))
            # if there is a minute difference in the y and the z, then instead we use x and z
            if np.linalg.norm(axis) < 1e-6:
                axis = np.cross(start_matrix, np.array([0, 1, 0]))
            # normalize
            axis = self.normalize(axis)
            rotation_vector = R.from_rotvec(np.pi * axis)
        else:
            axis = np.cross(start_matrix, to_transition_to_matrix)
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(orthogonality)
            rotation_vector = R.from_rotvec(angle * axis)
        slerp = Slerp([0, 1], R.concatenate([R.identity(), rotation_vector]))
        rotation_transformation = slerp([interpolation_ratio])[0]
        return rotation_transformation.apply(start_matrix)

    def acquire_interpolated_value(
        self, 
        interpolation_ratio: float, 
        to_transition_to_matrix: NDArray,
    ) -> NDArray[np.float64]:
        """Find the rotational transition between two matrices.

        Args:
            interpolation_ratio: The current point of interpolation between this matrix and the next.
            to_transition_to_matrix: The matrix to transition to.

        Returns:
            The interpolated matrix.
        """
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
            
        transition_matrix = transition_matrix * (
            np.linalg.norm(self.current_matrix_value) 
            + (np.linalg.norm(to_transition_to_matrix) - np.linalg.norm(to_transition_to_matrix)) 
            * interpolation_ratio
        )
        
        return transition_matrix

@dataclass
class ValueSet():
    """The value set for a particular quiver.
    """
    animated_matrix: SphericalAnimatedMatrix
    quiver_color: str
    quiver_alpha: float

class AnimationBlockSphere(AnimationBlock):
    """A block of information the animation step can use.
    """
    def __init__(
        self, 
        mixed_matrix: SphericalAnimatedMatrix, 
        state_1_matrix: SphericalAnimatedMatrix, 
        state_2_matrix: SphericalAnimatedMatrix, 
        display_properties: DisplayProperties
    ) -> None:
        """Initialize the animation block.

        Args:
            mixed_matrix: The matrix reqpresenting the total state.
            state_1_matrix: The matrix representing the first state it can be in.
            state_2_matrix: The matrix representing the second state it can be in.
            display_properties: The properties to display for this step.
        """
        self.mixed_state_value_set: ValueSet = ValueSet(
            animated_matrix=mixed_matrix,
            quiver_color=display_properties.quiver_state_color,
            quiver_alpha=display_properties.quiver_state_alpha
        )
        self.state_1_value_set: ValueSet = ValueSet(
            animated_matrix=state_1_matrix,
            quiver_color=display_properties.quiver_mixed_1_color,
            quiver_alpha=display_properties.quiver_mixed_1_alpha
        )
        self.state_2_value_set: ValueSet = ValueSet(
            animated_matrix=state_2_matrix,
            quiver_color=display_properties.quiver_mixed_2_color,
            quiver_alpha=display_properties.quiver_mixed_2_alpha
        )
        self.bloch_properties: Optional[DisplayProperties] = display_properties

    def state_quiver_mix(self) -> Generator[ValueSet, None, None]:
        """A generator which returns both the mixed state and the two probabilities.

        Yields:
            One of the states.
        """
        value_sets = [self.mixed_state_value_set, self.state_1_value_set, self.state_2_value_set]
        for value_set in value_sets:
            yield value_set

class PerQubitVisualization(GenericDisplay):
    """The properties of each qubits plot."""
    
    def __init__(self, axes: Axes, information_input: NDArray, display_properties: Optional[DisplayProperties], figure: Optional[Figure] = None) -> None:
        """Initialize the qubits visualization.

        Args:
            subset_axes: The axis that is being manipulated.
            initial_matrix: The initial mixed state.
            display_properties: The initial display properties.
        """
        super().__init__(axes, information_input, display_properties, figure)
        self._animation_blocks: List[AnimationBlockSphere]
        
        # ugly, but this is the best way to get around locally scoped self issues in iterables
        self.quiver_dict = {
            "state_quiver": None,
            "mixed_1_quiver": None,
            "mixed_2_quiver": None,
        }

        for quiver_name, quiver, value_set in zip(
            self.quiver_dict.keys(), 
            self.quiver_dict.values(), 
            self._animation_blocks[0].state_quiver_mix()
        ):
            self.update_quiver(
                quiver_name,
                quiver, 
                np.array(value_set.animated_matrix), 
                value_set.quiver_color, 
                value_set.quiver_alpha
            )

    def __len__(self) -> int:
        """The length of the animation.

        Returns:
            The length of the animation blocks.
        """
        return len(self._animation_blocks)
        
    def initialize_plot(self, information_input: NDArray, display_properties: Optional[DisplayProperties] = None) -> None:
        """Create the matplotlib visualization for the bloch sphere.
        """
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))
        self.axes.plot_wireframe(x, y, z, color='lightblue', alpha=0.1, zorder=3)
        self.axes.patch.set_facecolor(background_color)
        self.axes.quiver(0, 0, 0, 0.77, 0, 0, color='r', arrow_length_ratio=0.1, alpha=0.5, zorder=2)
        self.axes.quiver(0, 0, 0, 0, 0.77, 0, color='g', arrow_length_ratio=0.1, alpha=0.5, zorder=2)
        self.axes.quiver(0, 0, 0, 0, 0, 0.77, color='b', arrow_length_ratio=0.1, alpha=0.5, zorder=2)

        # Settings
        self.axes.set_xlim([-0.7, 0.7])
        self.axes.set_ylim([-0.7, 0.7])
        self.axes.set_zlim([-0.7, 0.7])
        self.axes.set_box_aspect([1,1,1])
        self.axes.axis('off')
                
        self.axes.text(x=0.0, y=0.0, z=1.2, s='|0⟩', color='white', fontsize=8, zorder=1)
        self.axes.text(x=0.0, y=0.0, z=-1.4, s='|1⟩', color='white', fontsize=8, zorder=1)
        self.axes.text(x=0.0, y=1.1, z=0.0, s='y', color='white', fontsize=8, zorder=1)
        self.axes.text(x=1.1, y=0.0, z=0.0, s='x', color='white', fontsize=8, zorder=1)
        
    def update_quiver(self, quiver_name: str, quiver: Quiver, transition_matrix: NDArray[np.float64], quiver_color: str, quiver_alpha: float):
        """Update the quiver position.

        Args:
            quiver_name: The name of the quiver we are changing.
            quiver: The quiver to remove.
            transition_matrix: the location the quiver should move its arrow to.
            quiver_color: The color of the quiver.
            quiver_alpha: The alpha for the quiver.
        """
        if quiver is not None:
            quiver.remove()
        self.quiver_dict[quiver_name] = self.axes.quiver(
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
        
    def update_plot(self, interpolation_ratio: float, current_index: int, next_index: int) -> None:
        """Update the plot of the qubit.

        Args:
            interpolation_ratio: The current point of interpolation between this matrix and the next.
        """
        for value_set, value_set_transition, quiver_name, plot_quiver in zip(
            self._animation_blocks[current_index].state_quiver_mix(), 
            self._animation_blocks[next_index].state_quiver_mix(),
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
    def transform_complex_to_xyz(matrix: NDArray[np.complex128]) -> NDArray[np.float64]:
        """Translate the complex array into x, y and z coordinates.

        Args:
            matrix: The complex array we are translating.

        Returns:
            The x, y and z coordinates.
        """
        x_coordinate = 2 * np.real(matrix[0, 1])
        y_coordinate = 2 * np.imag(matrix[0, 1])
        z_coordinate = np.real(matrix[0, 0] - matrix[1, 1])
        return np.array([x_coordinate, y_coordinate, z_coordinate])

    @staticmethod
    def transform_vector_to_xyz(eigen_vectors: NDArray[np.float64]) -> NDArray[np.float64]:
        """Translate an eigen vector into x, y and z coordinates.

        Args:
            eigen_vectors: The eigen vectors

        Returns:
            The x, y and z coordinates of the eigen vector.
        """
        bloch_vectors = []
        for i in range(_QUBIT_HILBERT_SPACE):
            eigen_vector = eigen_vectors[:, i]

            function = Statevector(eigen_vector)
            
            operator = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])

            bloch_vector = np.real(function.data.conj().T @ operator @ function.data)
            bloch_vectors.append(bloch_vector)
        return bloch_vectors

    def append_block(
        self, 
        information_input: NDArray, 
        bloch_properties: Optional[DisplayProperties] = None, 
    ):
        """Append the x, y and z coordinates state to animate.

        Args:
            quantum_circuit: The circuit to visualize.
            display_properties: The initial display properties.
        """
        next_matrix = SphericalAnimatedMatrix(self.transform_complex_to_xyz(information_input))
        
        percentiles, mixed_vectors = np.linalg.eigh(information_input)
        bloch_vectors = self.transform_vector_to_xyz(mixed_vectors)
        next_mixed_matrix_1 = SphericalAnimatedMatrix(bloch_vectors[0])
        next_mixed_matrix_2 = SphericalAnimatedMatrix(bloch_vectors[1])
        if np.all(np.allclose(next_matrix, 0.0)) and len(self._animation_blocks) != 0:
            next_matrix.previous_nonzero_rotation = self._animation_blocks[-1].mixed_state_value_set.animated_matrix.previous_nonzero_rotation
        else:
            next_matrix.previous_nonzero_rotation = np.array(next_matrix) / np.linalg.norm(np.array(next_matrix))
        if bloch_properties is None:
            bloch_properties = DisplayProperties()
        self._animation_blocks.append(AnimationBlockSphere(
            next_matrix, next_mixed_matrix_1, next_mixed_matrix_2, bloch_properties
        ))
        
    def next_animation_block(self, current_index: int, next_index: int) -> None:
        """Jump to the next block for animation.

        Args:
            next_index: The index of the animation block to jump to.

        Returns:
            Whether it was able to jump to this index.
        """

        change_in_matrix = False
        for value_set, value_set_transition in zip(
            self._animation_blocks[current_index].state_quiver_mix(), 
            self._animation_blocks[next_index].state_quiver_mix(),
        ):
            if (
                not np.all(np.isclose(
                    np.array(value_set.animated_matrix),
                    np.array(value_set_transition.animated_matrix)
                ))
            ):
                change_in_matrix = True
        if change_in_matrix:
            self.axes.set_title(self._animation_blocks[next_index].bloch_properties.plot_name, color='white')
        else:
            self.axes.set_title("", color='white')
