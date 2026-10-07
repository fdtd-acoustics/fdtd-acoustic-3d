import taichi as ti
import numpy as np

from . import waveform_factory


@ti.data_oriented
class SourceManager:
    def __init__(self, max_sources: int, max_steps: int, dim: int = 3):
        self.max_sources = max_sources
        self.max_steps = max_steps

        self.count = ti.field(ti.i32, shape=())
        self.count[None] = 0

        self.names = []

        self.pos = ti.Vector.field(dim, dtype=ti.i32, shape=self.max_sources)

        self.waveforms = ti.field(dtype=ti.f32, shape=(max_sources, max_steps))

    @ti.kernel
    def _copy_waveform(self, dest: ti.template(), src: ti.types.ndarray(), source_idx: ti.i32, steps: ti.i32):
        for t in range(steps):
            dest[source_idx, t] = src[t]

    def add_source(self, name: str, waveform_array: np.ndarray, x: int, y: int, z: int | None = None):
        idx = self.count[None]

        if idx < self.max_sources:
            if z is None:
                self.pos[idx] = [x, y]
            else:
                self.pos[idx] = [x, y, z]

            self._copy_waveform(self.waveforms, waveform_array, idx, self.max_steps)

            self.names.append(name)
            self.count[None] += 1
        else:
            print(f"Error: Tried to add source '{name}' but reached max_sources limit")

    def set_pos(self, idx: int, x: int, y: int, z: int | None = None):
        if 0 <= idx < self.count[None]:
            if z is None:
                self.pos[idx] = [int(x), int(y)]
            else:
                self.pos[idx] = [int(x), int(y), int(z)]
        else:
            print(f"ERROR: Invalid index {idx}")

    @ti.kernel
    def update_sources(self, p_field: ti.template(), step:ti.i32):
        for i in range(self.count[None]):
            curr_pos = self.pos[i]
            val = self.waveforms[i, step]
            p_field[curr_pos] += val

    @classmethod
    def build_source_manager(cls, sources: list[dict], dt: float, pml_thick: int, dim: int = 3) -> 'SourceManager':
        max_time = max(source.get('time', 1.0) for source in sources)
        max_steps = int(np.ceil(max_time / dt))

        manager = cls(max_sources=len(sources), max_steps=max_steps, dim=dim)
        for idx, source in enumerate(sources):
            waveform = waveform_factory.synthesize(source, dt, max_steps)
            waveform = np.ascontiguousarray(waveform, dtype=np.float32)

            coords = source.get('coords')
            if coords is None or len(coords) != dim:
                raise ValueError(f"Missing or invalid coordinates for source at index {idx}. Expected {dim} coords, got {coords}")

            src_x = int(coords[0]) + pml_thick
            src_y = int(coords[1]) + pml_thick
            src_z = int(coords[2]) + pml_thick if dim == 3 else None

            manager.add_source(
                name=source.get('name', f"src_{idx}"),
                waveform_array=waveform,
                x=src_x,
                y=src_y,
                z=src_z,
            )

        return manager

    @classmethod
    def get_highest_frequency(cls, sources: list[dict]) -> float:
        return max((waveform_factory.get_max_frequency(s) for s in sources), default=0.0)

