"""Default configuration for the simulation."""

from dataclasses import dataclass


@dataclass
class SimConfig:
    """Physical and numerical parameters of the simulation.

    Units: G = 1, disk scale length = 1, disk mass ~ 1.
    """

    n: int = 20_000
    dt: float = 0.02
    G: float = 1.0
    softening: float = 0.15
    seed: int = 7

    # Merger geometry
    separation: float = 10.5
    impact_parameter: float = 2.2
    relative_speed: float = 0.5

    def __post_init__(self) -> None:
        if self.n < 2:
            raise ValueError("n must be at least 2")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.softening <= 0:
            raise ValueError("softening must be positive")
