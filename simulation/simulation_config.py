from dataclasses import dataclass, field
from pathlib import Path
import config

@dataclass
class SimulationConfig:
    obj_filepath: str

    pml_thick: int
    alpha_max: float
    sound_speed: float = config.SOUND_SPEED
    nodes_per_wavelength: int = config.NODES_PER_WAVELENGTH
    dim: int = config.DIM
    safety_factor: float = config.SAFETY_FACTOR

    _scenes_out_dir: Path = config.SCENES_OUT_DIR
    _npz_filepath: Path | None = None

    @property
    def npz_filepath(self) -> Path:
        if self._npz_filepath is not None:
            return self._npz_filepath
        return self._scenes_out_dir / (Path(self.obj_filepath).stem + ".npz")


    @classmethod
    def from_dict(cls, cfg: dict) -> 'SimulationConfig':
        """Factory method to construct this class from GUI"""
        #todo: tutaj bedzie trzeba ustawiac npz file jak sie w gui wybierze gotowca
        return cls(
            obj_filepath=cfg['obj_file'],
            pml_thick = cfg['pml_thickness'],
            alpha_max = cfg['alpha_max'],
        )
