import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def save_plot(filename: str, pressure_data: np.ndarray,dt: float,
              directory: Path, global_max: float = None):
    os.makedirs(directory, exist_ok=True)

    if not filename.endswith(".png"):
        filename += ".png"
    file_path = os.path.join(directory, filename)

    pressure = pressure_data.copy()
    pressure -= np.mean(pressure)

    max = global_max if global_max is not None else np.max(np.abs(pressure))
    if max > 0:
        pressure = pressure / max

    time_axis = np.arange(len(pressure)) * dt

    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, pressure, color="#1f77b4", linewidth=1)

    mode = "Globally" if global_max is not None else "Locally"
    plt.title(f"Pressure Signal - {filename} ({mode} Normalized)")

    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.ylim(-1.1, 1.1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(file_path, dpi=150)
    plt.close()

    print(f"PLOT SAVED: {file_path} (Mode: {mode})")

def save_fft_plot( filename: str, pressure_data: np.ndarray, dt: float,
              directory: Path,):
    os.makedirs(directory, exist_ok=True)

    fft_filename = filename.replace('.png', '_fft.png') if filename.endswith('.png') else f"{filename}_fft.png"
    file_path = os.path.join(directory, fft_filename)

    fs = int(1.0 / dt)
    n = len(pressure_data)

    data = pressure_data - np.mean(pressure_data)
    yf = np.fft.rfft(data)   # real fast fourier transform - zmiana dziedziny czasu na dziedzinę czestotliowsci
    xf = np.fft.rfftfreq(n, dt)

    magnitude = 20 * np.log10(np.abs(yf) + 1e-10) # tu zdobywamy względne dB danych częstotliwości

    plt.figure(figsize=(10, 4))
    plt.semilogx(xf, magnitude, color="#ff7f0e", linewidth=1)
    plt.title(f"Frequency Spectrum (FFT) - {filename}")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude [dB]")
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.xlim(20, fs / 2)  # pasmo słyszalne
    plt.tight_layout()
    plt.savefig(file_path, dpi=150)
    plt.close()
    print(f"FFT PLOT SAVED: {file_path}")

def save_spl_plot(filename: str, pressure_data: np.ndarray,dt: float,
              directory: Path, global_max: float = None):
    os.makedirs(directory, exist_ok=True)

    spl_filename = filename.replace('.png', '_spl.png') if filename.endswith('.png') else f"{filename}_spl.png"
    file_path = os.path.join(directory, spl_filename)

    pressure = pressure_data.copy()
    pressure -= np.mean(pressure)

    max = global_max if global_max is not None else np.max(np.abs(pressure))
    if max > 0:
        pressure = pressure / max

    pressure_db = 20 * np.log10 \
        (np.abs(pressure) + 1e-10) # tutaj nie dzielimy ciśnienia, ponieważ chcemy żeby wynik był względny
    # ponieważ nie jesteśmy w stanie określić fizycznej wartości ciśnienia w Pascalach
    pressure_db = np.maximum(pressure_db, -80)

    time_axis = np.arange(len(pressure)) * dt

    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, pressure_db, color="#d62728", linewidth=1)

    mode = "Globally" if global_max is not None else "Locally"
    plt.title(f"Sound Pressure Level (SPL) - {filename} ({mode} Normalized)")

    plt.xlabel("Time [s]")
    plt.ylabel("Magnitude [dB relative to peak]")
    plt.ylim(-80, 5)
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()
    plt.savefig(file_path, dpi=150)
    plt.close()

    print(f"SPL PLOT SAVED: {file_path} (Mode: {mode})")