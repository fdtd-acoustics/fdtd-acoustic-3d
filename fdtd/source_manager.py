import taichi as ti
import numpy as np

from . import waveform_factory


@ti.data_oriented
class SourceManager:
    def __init__(self, max_sources: int, max_steps: int):
        self.max_sources = max_sources
        self.max_steps = max_steps

        self.count = ti.field(ti.i32, shape=())
        self.count[None] = 0

        self.names = []

        self.pos = ti.Vector.field(3, dtype=ti.i32, shape=self.max_sources)

        self.waveforms = ti.field(dtype=ti.f32, shape=(max_sources, max_steps))

    @ti.kernel
    def _copy_waveform(self, dest: ti.template(), src: ti.types.ndarray(), source_idx: ti.i32, steps: ti.i32):
        for t in range(steps):
            dest[source_idx, t] = src[t]

    def get_waveform_numpy(self, source_idx: int) -> np.ndarray:
        if source_idx >= self.count[None]:
            return None

        return self.waveforms.to_numpy()[source_idx, :]

    def add_source(self, name: str,x:int, y:int, z:int, waveform_array: np.ndarray):
        idx = self.count[None]

        if idx < self.max_sources:
            self.pos[idx] = [x, y, z]

            self._copy_waveform(self.waveforms, waveform_array, idx, self.max_steps)

            self.names.append(name)
            self.count[None] += 1
        else:
            print(f"Error: Tried to add source '{name}' but reached max_sources limit")

    def set_pos(self, idx, x, y, z):
        if 0 <= idx < self.count[None]:
            self.pos[idx] = [int(x), int(y), int(z)]
        else:
            print(f"ERROR:  {idx}")

    @ti.kernel
    def update_sources(self, p_field: ti.template(), step:ti.i32):
        for i in range(self.count[None]):
            curr_pos = self.pos[i]
            val = self.waveforms[i, step]
            p_field[curr_pos[0], curr_pos[1], curr_pos[2]] += val


    @classmethod
    def build_source_manager(cls, sources: list[dict], dt: float, pml_thick: int) -> 'SourceManager':
        max_time = max(source.get('time', 1.0) for source in sources)
        max_steps = int(np.ceil(max_time / dt))

        manager = cls(max_sources=len(sources), max_steps=max_steps)
        for idx, source in enumerate(sources):
            waveform = waveform_factory.synthesize(source, dt, max_steps)
            waveform = np.ascontiguousarray(waveform, dtype=np.float32)

            coords = source.get('coords')
            if coords is None or len(coords) < 3:
                raise ValueError(f"Missing or invalid coordinates for source at index {idx}.")

            x, y, z = coords
            src_x = int(x) + pml_thick
            src_y = int(y) + pml_thick
            src_z = int(z) + pml_thick

            manager.add_source(
                name=source.get('name', f"src_{idx}"),
                x=src_x,
                y=src_y,
                z=src_z,
                waveform_array=waveform
            )



        return manager

    @classmethod
    def get_highest_frequency(cls, sources: list[dict]) -> float:
        return max((waveform_factory.get_max_frequency(s) for s in sources), default=0.0)

