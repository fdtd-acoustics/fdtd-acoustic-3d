import math
import taichi as ti
from scipy.io import wavfile
import numpy as np
from scipy.interpolate import interp1d


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

    def add_source(self, name: str, waveform_array: np.ndarray):
        idx = self.count[None]

        if idx < self.max_sources:
            self.pos[idx] = [30, 30, 30]  # tymczasowo

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
    def build_source_manager(cls, sources: list[dict], max_steps: int, dt: float) -> 'SourceManager':
        manager = cls(max_sources=len(sources), max_steps=max_steps)
        for source in sources:
            waveform = cls._calculate_waveform(source, dt, max_steps)

            waveform = np.ascontiguousarray(waveform, dtype=np.float32)

            manager.add_source(
                name=source.get('name'),
                waveform_array=waveform
            )
        return manager

    @staticmethod
    def _calculate_waveform(source: dict, dt: float, max_steps: int) -> np.ndarray:
        waveform = np.zeros(max_steps, dtype=np.float32)
        t = np.arange(max_steps) * dt

        if source['type'] == 'Gauss':
            freq = source.get('freq')
            amp = source.get('amp')
            sigma = np.sqrt(2 * np.log(2)) / (2 * np.pi * freq)
            delay = 4 * sigma
            waveform = amp * np.exp(-((t - delay)**2) / (2 * sigma**2))

        elif source['type'] == 'Custom':
            filepath = source['filepath']
            sample_rate, data = wavfile.read(filepath)
            if len(data.shape) > 1:
                data = data[:, 0]

            data = data.astype(np.float32) / (np.max(np.abs(data)) + 1e-9) # normalizacja

            duration = len(data) / sample_rate   # trzeba dopasowac do dt
            old_t = np.linspace(0, duration, len(data))

            interpolator = interp1d(old_t, data, kind='linear', bounds_error=False, fill_value=0.0)

            waveform = interpolator(t).astype(np.float32)

        return waveform



    @classmethod
    def get_highest_frequency(cls, sources: list[dict]) -> float:
        max_freq = 0.0
        for source in sources:
            current_freq = 0.0

            if source['type'] == 'Gauss':
                current_freq = float(source.get('freq', 0.0))

            elif source['type'] == 'Custom':
                file_path = source.get('filepath')
                if file_path:
                    current_freq = cls._analyze_wav_freq_max(file_path, threshold=0.01) # narazie tak na sztywno

            if current_freq > max_freq:
                max_freq = current_freq

        return max_freq


    @classmethod
    def _analyze_wav_freq_max(cls, file_path, threshold: float) -> float:
        try:
            sample_rate, data = wavfile.read(file_path)
            if len(data.shape) > 1:
                data = data[:, 0]

            n = len(data) # ilosc próbek

            fft_values = np.abs(np.fft.rfft(data)) # to wartosci liniowe(nie decybele)
            freqs = np.fft.rfftfreq(n, d=1 / sample_rate)

            limit = threshold * np.max(fft_values) # pozbywamy sie najcichszych dzwiekow(slabo slyszalne a moze sie tam trafic wysoka czestotliowsc)
            # wysoka czestotliowsc moze doprowadzic do strasznie malego dx
            significant_freqs = freqs[fft_values > limit] # tablica czestotliowsci

            if len(significant_freqs) > 0:
                return float(significant_freqs[-1]) # zwraca najwyzsza czestotliowsc
            return 0.0
        except Exception as e:
            print(f"File parsing error {file_path}: {e}")
            return 0.0

