from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.interpolate import interp1d


def synthesize(spec: dict, dt: float, total_steps: int) -> np.ndarray:
    waveform_type = spec.get("type")
    if waveform_type is None:
        raise ValueError("waveform.type is required")

    handler = _HANDLERS.get(waveform_type)
    if handler is None:
        raise ValueError(
            f"unknown waveform.type '{waveform_type}'. "
            f"valid types: {sorted(_HANDLERS.keys())}"
        )

    return handler(spec, dt, total_steps)


def get_max_frequency(spec: dict) -> float:
    w_type = spec.get("type")
    if w_type == 'Gauss':
        return float(spec.get('freq', 0.0))
    if w_type == 'Custom':
        return _analyze_wav_max_freq(spec.get('filepath'))
    if w_type == 'ContinuousTone':
        return float(spec.get('freq', 0.0))
    return 0.0

def _generate_gauss(spec: dict, dt: float, total_steps: int) -> np.ndarray:
    freq = float(spec['freq'])
    amp = float(spec.get('amp', 1.0))
    vol = float(spec.get('vol', 1.0))
    t = np.arange(total_steps) * dt

    sigma = np.sqrt(2 * np.log(2)) / (2 * np.pi * freq)
    delay = 4 * sigma
    waveform = amp * np.exp(-((t - delay) ** 2) / (2 * sigma ** 2))
    return (waveform * vol).astype(np.float32)

def _generate_continuous_tone(spec: dict, dt: float, total_steps: int) -> np.ndarray:
    freq = float(spec['freq'])
    amp = float(spec.get('amp', 1.0))
    vol = float(spec.get('vol', 1.0))
    t = np.arange(total_steps) * dt

    waveform = amp * np.sin(2 * np.pi * freq * t)

    return (waveform * vol).astype(np.float32)



def _load_wav(spec: dict, dt: float, total_steps: int) -> np.ndarray:
    path = Path(spec.get('filepath', spec.get('path', '')))
    if not path.is_file():
        raise FileNotFoundError(f"waveform file not found: {path}")

    vol = float(spec.get('vol', 1.0))
    t = np.arange(total_steps) * dt

    sample_rate, data = wavfile.read(path)
    if len(data.shape) > 1:
        data = data[:, 0]

    data = data.astype(np.float32) / (np.max(np.abs(data)) + 1e-9)  # normalizacja

    duration = len(data) / sample_rate  # trzeba dopasowac do dt
    old_t = np.linspace(0, duration, len(data))

    interpolator = interp1d(old_t, data, kind='linear', bounds_error=False, fill_value=0.0)

    waveform = interpolator(t).astype(np.float32)

    waveform -= np.mean(waveform)

    return (waveform * vol).astype(np.float32)

def _analyze_wav_max_freq(path: str, threshold: float = 0.01) -> float:
    if not path: return 0.0
    sample_rate, data = wavfile.read(path)
    if data.ndim > 1: data = data[:, 0]

    n = len(data)  # ilosc próbek

    fft_vals = np.abs(np.fft.rfft(data))
    freqs = np.fft.rfftfreq(n, d=1 / sample_rate)

    limit = threshold * np.max(fft_vals)
    significant = freqs[fft_vals > limit]
    return float(significant[-1]) if significant.size > 0 else 0.0

_HANDLERS = {
    "Gauss": _generate_gauss,
    "Custom": _load_wav,
    "ContinuousTone": _generate_continuous_tone,
}
