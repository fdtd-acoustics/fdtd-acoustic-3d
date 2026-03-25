import math
import taichi as ti
from scipy.io import wavfile
import numpy as np
import config


@ti.data_oriented
class SourceManager:
    def __init__(self, max_sources: int, max_steps: int):
        self.max_sources = max_sources
        self.max_steps = max_steps

        self.count = ti.field(ti.i32, shape=())
        self.count[None] = 0

        self.names = []

        self.info = ti.Struct.field({
            "pos": ti.types.vector(3, ti.i32),
            "base_amp": ti.f32,   # mozna podglosnic/sciszyc zrodlo
        }, shape=self.max_sources)

        self.waveforms = ti.field(dtype=ti.f32, shape=(max_sources, max_steps))

    @ti.kernel
    def _copy_waveform(self, dest: ti.template(), src: ti.template(), source_idx: ti.i32, steps: ti.i32):
        for t in range(steps):
            dest[source_idx, t] = src[t]

    def add_source(self, name: str, waveform_array: np.ndarray, base_amp:ti.f32):
        idx = self.count[None]

        if idx < self.max_sources:
            self.info[idx].pos =  [0,0,0] #TODO tu bedziemy ustawiac z argumentow
            self.info[idx].base_amp = base_amp

            temp_field = ti.field(dtype=ti.f32, shape=self.max_steps)
            temp_field.from_numpy(waveform_array)

            self._copy_waveform(self.waveforms, temp_field, idx, self.max_steps)

            print("JUZ PO KOPIOWANIU NA GPU:")
            for i in range(10):
                val = self.waveforms[idx, i]
                print(f"Krok {i}: {val:.10f}")

            self.names.append(name)
            self.count[None] +=1
        else:
            print(f"Error: Tried to add source '{name}' but reached max_sources limit")

    def set_pos(self, idx, x, y, z):
        if 0 <= idx < self.count[None]:
            self.info[idx].pos = [int(x), int(y), int(z)]
        else:
            print(f"ERROR:  {idx}")

    @ti.kernel
    def update_sources(self, p_field: ti.template(), step:ti.i32):
        for i in range(self.count[None]):
            curr_pos = self.info[i].pos
            val = self.waveforms[i, step]

            p_field[curr_pos[0], curr_pos[1], curr_pos[2]] += val
            print("| ", p_field[curr_pos[0], curr_pos[1], curr_pos[2]])


    # def get_max_freq(self) -> float:
    #     max_f = 0.0
    #     for i in range(self.count[None]):
    #         current_f = self.data[i].freq
    #         if current_f > max_f:
    #             max_f = current_f
    #     return max_f

    @classmethod
    def from_dict(cls, sources: list[dict], max_steps: int, dt: float) -> 'SourceManager':
        manager = cls(max_sources = len(sources), max_steps = max_steps)
        for source in sources:
            waveform = cls._calculate_waveform(source, dt, max_steps)  # tu liczymy tablice cisnien
            manager.add_source(
                name=source.get('name'),
                base_amp = 1.0,    # TODO to jest tymczasowo, pozniej do gui dodamy poziom glosnosci
                waveform_array=waveform
            )
        return manager

    @staticmethod
    def _calculate_waveform(source: dict, dt: float, max_steps: int) -> np.ndarray:
        waveform = np.zeros(max_steps, dtype=np.float32)
        t = np.arange(max_steps) * dt

        if source['type'] == 'Gauss':
            freq = source.get('freq')
            sigma = np.sqrt(2 * np.log(2)) / (2 * np.pi * freq)
            delay = 4 * sigma
            waveform = np.exp(-((t - delay)**2) / (2 * sigma**2))

        #TODO trzeba zrobic tez z wav
        #elif source['type'] == 'Custom':
            #trzeba zaladowac wav
            #zrobic resample
            #przyciac do dlugosci

        for i in range(10):
            print(f"Krok {i} (t={t[i]:.2e}): {waveform[i]:.10f}")


        return waveform



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




