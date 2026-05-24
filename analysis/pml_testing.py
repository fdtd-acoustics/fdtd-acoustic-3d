import os
import numpy as np
import matplotlib.pyplot as plt
import json

def calculate_pml_error(pressure_array, split_index):
    part_A = pressure_array[:split_index]
    part_B = pressure_array[split_index:]

    ms_A = np.mean(part_A ** 2)
    ms_B = np.mean(part_B ** 2)

    if ms_A == 0:
        return float('inf'), float('inf')

    error_ratio = ms_B / ms_A
    error_db = 10 * np.log10(error_ratio) if error_ratio > 0 else float('-inf')

    return error_ratio, error_db


def calculate_split_index(pressure_array, original_waveform):
    max_amp = np.max(np.abs(original_waveform))

    if max_amp == 0:
        raise ValueError("The original signal is empty")

    # próg 5% dla źródła - jak ustawimy mniej zacznie lapac duzo szumu i tego ogona ktory jest po gaussie
    threshold = max_amp * 0.05
    active_indices = np.where(np.abs(original_waveform) > threshold)[0]
    actual_pulse_length = active_indices[-1] - active_indices[0] + 1
    safety_margin = int(actual_pulse_length * 0.1)

    # moment zakończenia generowania fali w źródle
    source_split_index = active_indices[-1] + safety_margin

    # moment zakończenia przelatywania fali przez mikrofon
    peak_index = np.argmax(np.abs(pressure_array))
    mic_split_index = peak_index + (actual_pulse_length // 2) + safety_margin
    mic_split_index = min(mic_split_index, len(pressure_array) - 1)

    return mic_split_index, source_split_index


def plot_pml_evaluation(pressure_array, original_waveform, mic_split, source_split, dt, error_db=None,
                        save_plots=False, output_dir=None, filename="mic"):
    time_mic = np.arange(len(pressure_array)) * dt
    time_src = np.arange(len(original_waveform)) * dt

    mic_split_time = mic_split * dt
    source_split_time = source_split * dt

    # mikrofon
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(time_mic, pressure_array, color='blue', label='Microphone signal')

    ax1.axvline(x=mic_split_time, color='red', linestyle='--', linewidth=2,
                label=f'Wave end ({mic_split_time:.4f} s)')

    ax1.axvspan(0, mic_split_time, color='green', alpha=0.1, label='Direct wave (A)')
    ax1.axvspan(mic_split_time, time_mic[-1], color='red', alpha=0.1, label='PML reflections (B)')

    title1 = 'PML Reflection Analysis at Microphone'
    if error_db is not None:
        title1 += f' (Error: {error_db:.2f} dB)'

    ax1.set_title(title1, fontweight='bold')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Pressure amplitude')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.7)

    fig1.tight_layout()

    if save_plots and output_dir is not None:
        pml_filename = filename.replace('.png', '_pml.png') if filename.endswith('.png') else f"{filename}_pml.png"
        file_path1 = os.path.join(output_dir, pml_filename)

        fig1.savefig(file_path1)
        plt.close(fig1)
    else:
        plt.show()

    # źródło
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(time_src, original_waveform, color='orange', label='Source signal (Gauss)')

    ax2.axvline(x=source_split_time, color='purple', linestyle='--', linewidth=2,
                label=f'Source end ({source_split_time:.4f} s)')

    ax2.axvspan(0, source_split_time, color='purple', alpha=0.1, label='Signal generation time')

    ax2.set_title('Source Generated Signal', fontweight='bold')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Source amplitude')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle=':', alpha=0.7)

    fig2.tight_layout()

    if save_plots and output_dir is not None:
        file_path2 = os.path.join(output_dir, "original_src_pml.png")
        fig2.savefig(file_path2)
        plt.close(fig2)
    else:
        plt.show()


def save_pml_results_to_json(output_dir, mic_name, error_ratio, error_db, split_index, dt):

    if output_dir is None:
        return
    base_dir = os.path.dirname(output_dir) if str(output_dir).endswith('plots') else output_dir
    json_path = os.path.join(base_dir, "pml_results.json")

    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}

    data[mic_name] = {
        "error_ratio": float(error_ratio),
        "error_db": float(error_db),
        "split_index": int(split_index),
        "split_time_s": float(split_index * dt)
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


def evaluate_pml(pressure_array, original_waveform, filename: str, dt: float, save_plots=False, output_dir=None):
    mic_split_index, source_split_index = calculate_split_index(pressure_array, original_waveform)

    error_ratio, error_db = calculate_pml_error(pressure_array, mic_split_index)

    if save_plots and output_dir is not None:
        plot_pml_evaluation(
            pressure_array, original_waveform,
            mic_split_index, source_split_index,
            dt, error_db, save_plots, output_dir, filename
        )

    if output_dir is not None:
        save_pml_results_to_json(output_dir, filename, error_ratio, error_db, mic_split_index, dt)

    return {
        'split_index': mic_split_index,
        'error_ratio': error_ratio,
        'error_db': error_db
    }