"""Some helpers that all the backend will use."""
from dataclasses import dataclass

_QUBIT_HILBERT_SPACE = 2

@dataclass
class DisplayProperties():
    """Properties the down line can edit to change how the visuals are displayed.
    """
    plot_name: str = ""
    display_on_all: bool = False
    quiver_state_color: str = "white"
    quiver_state_alpha: float = 1.0
    quiver_mixed_1_color: str = "cyan"
    quiver_mixed_1_alpha: float = 0.1
    quiver_mixed_2_color: str = "yellow"
    quiver_mixed_2_alpha: float = 0.1
    entanglement_colors: str = "red"
