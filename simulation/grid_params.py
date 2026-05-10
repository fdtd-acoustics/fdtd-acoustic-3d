from dataclasses import dataclass

@dataclass(frozen=True)
class GridParams:
    dx: float
    dt: float
    Nx: int
    Ny: int
    Nz: int

    def __post_init__(self):
        #todo: walidacja
        pass