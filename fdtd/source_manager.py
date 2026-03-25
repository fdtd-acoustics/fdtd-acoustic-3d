import math
import taichi as ti
from scipy.io import wavfile
import numpy as np
import config


@ti.data_oriented
class SourceManager:
    def __init__(self, max_sources: int):
        self.max_sources = max_sources
        self.count = ti.field(ti.i32, shape=())
        self.count[None] = 0

        self.names = []

        self.data = ti.Struct.field({
            "pos": ti.types.vector(3, ti.i32),
            "amp": ti.f32,
            "freq": ti.f32,
            "sigma": ti.f32,
            "delay_t": ti.f32,
        }, shape=max_sources)

    def add_source(self, name, freq, amplitude):
        idx = self.count[None]

        if idx < self.max_sources:
            sigma = math.sqrt(2 * math.log(2)) / (2 * math.pi * freq)
            #self.data[idx].pos = [x + config.PML_THICK, y + PML_THICK, z + PML_THICK] # uwzgledniamy warstwe pml
            self.data[idx].pos = [60,60,60] # domyslne
            self.names.append(name)

            self.data[idx].amp = amplitude
            self.data[idx].freq = freq
            self.data[idx].sigma = sigma
            self.data[idx].delay_t = 4 * sigma
            self.count[None] += 1


    # def add_source(self, name, source_type, **kwargs):
    #     idx = self.count[None]
    #     if idx >= self.max_sources:
    #         return


    def set_pos(self, idx, x, y, z):
        if 0 <= idx < self.count[None]:
            self.data[idx].pos = [int(x), int(y), int(z)]
        else:
            print(f"blad {idx}")

    @ti.kernel
    def update_sources(self, p_field: ti.template(), steps: ti.i32, dt:ti.f32):
        if steps < 10: #tymczasowo dodalem, tylko przez 10 krokow zrodlo dziala
            for i in range(self.count[None]):
                s = self.data[i]
                val = s.amp * ti.exp(- ((dt * steps - dt * s.delay_t) ** 2) / (2 * s.sigma ** 2))
                p_field[s.pos[0],s.pos[1], s.pos[2]] += val

    def get_max_freq(self) -> float:
        max_f = 0.0
        for i in range(self.count[None]):
            current_f = self.data[i].freq
            if current_f > max_f:
                max_f = current_f
        return max_f

    @classmethod
    def from_dict(cls, sources: list[dict]) -> 'SourceManager':
        manager = cls(len(sources))
        for source in sources:
            if source['type'] == 'Gauss':
                manager.add_source(
                    name='gauss_source',
                    freq=source['freq'],
                    amplitude=source['amp'],
                )
            elif source['type'] == 'Custom':
                manager.add_source(
                    name='custom_source',
                    freq=None,
                    amplitude=None,
                )
        return manager


    @classmethod
    def get_highest_frequency(cls, sources: list[dict]) -> float:
        max_freq = 0.0
        for source in sources:
            current_freq = 0.0

            if source['type'] == 'Gauss':
                current_freq = float(source.get('freq', 0.0))

            elif source['type'] == 'Custom':
                file_path = source.get('file_path')
                if file_path:
                    current_freq = cls._analyze_wav_freq_max(file_path)

            if current_freq > max_freq:
                max_freq = current_freq

        return max_freq


    @classmethod
    def _analyze_wav_freq_max(cls, file_path, threshold: float) -> float:
        try:
            samplerate, data = wavfile.read(file_path)
            if len(data.shape) > 1:
                data = data[:, 0]

            n = len(data) # ilosc próbek
            fft_values = np.abs(np.fft.fft(data)) # to wartosci liniowe(nie decybele)
            freqs = np.fft.rfftfreq(n, d=1 / samplerate)

            limit = threshold * np.max(fft_values) # pozbywamy sie najcichszych dzwiekow(slabo slyszalne a moze sie tam trafic wysoka czestotliowsc)
            # wysoka czestotliowsc moze doprowadzic do strasznie malego dx
            significant_freqs = freqs[fft_values > limit] # tablica czestotliowsci

            if len(significant_freqs) > 0:
                return float(significant_freqs[-1]) # zwraca najwyzsza czestotliowsc
            return 0.0
        except Exception as e:
            print(f"File parsing error {file_path}: {e}")
            return 0.0




