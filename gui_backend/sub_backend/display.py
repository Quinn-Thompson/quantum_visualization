from abc import abstractmethod
from gui_backend.helpers import DisplayProperties
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import List, Optional, Any
import qiskit


class AnimationBlock():
    def __init__(self, display_properties: DisplayProperties):
        self.display_properties = display_properties

class GenericDisplay():
    def __init__(self, axes: Axes, information_input: Any, display_properties: Optional[DisplayProperties], figure: Optional[Figure] = None):
        self.axes = axes
        self.figure = figure
        self._animation_blocks: List[AnimationBlock] = []
        self.append_block(information_input, display_properties)
        self.initialize_plot(information_input, display_properties)

    @abstractmethod
    def initialize_plot(
        self, 
        information_input: Any,
        display_properties: Optional[DisplayProperties] = None, 
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def update_plot(self, interpolation_ratio: float, current_index: int, next_index: int) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def append_block(
        self, 
        information_input: Any,
        display_properties: Optional[DisplayProperties] = None, 
    ) -> None:
        raise NotImplementedError
        
    @abstractmethod
    def next_animation_block(self, current_index: int, next_index: int) -> None:
        raise NotImplementedError