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
    def build_source_manager(cls, sources: list[dict], dt: float) -> 'SourceManager':
        max_time = max(source.get('time', 1.0) for source in sources)
        max_steps = int(np.ceil(max_time / dt))

        manager = cls(max_sources=len(sources), max_steps=max_steps)
        for idx, source in enumerate(sources):
            waveform = waveform_factory.synthesize(source, dt, max_steps)
            waveform = np.ascontiguousarray(waveform, dtype=np.float32)

            x, y, z = source.get('coords')

            manager.add_source(
                name=source.get('name', f"src_{idx}"),
                x=int(x),
                y=int(y),
                z=int(z),
                waveform_array=waveform
            )

            if 'coords' in source:
                x, y, z = source['coords']
                manager.set_pos(idx, int(x), int(y), int(z))


        return manager

    @classmethod
    def get_highest_frequency(cls, sources: list[dict]) -> float:
        return max((waveform_factory.get_max_frequency(s) for s in sources), default=0.0)

