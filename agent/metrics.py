import math

import numpy as np

ONSET_THRESHOLD_DB = -30.0  # ISO 3382-1: onset more than 20 dB below max

def c50(pressures: np.ndarray, sample_rate: float) -> float:
    """C50 clarity [dB] of a single microphone impulse response (ISO 3382-1)."""
    pressures = pressures.astype(np.float64)
    energy = pressures ** 2

    magnitude = np.abs(pressures)
    peak = magnitude.max()
    if peak == 0.0:
        raise ValueError("Impulse response is silent")

    threshold = peak * 10 ** (ONSET_THRESHOLD_DB / 20)
    # get first value above threshold
    start = int(np.argmax(magnitude >= threshold))

    #calculate number of samples for 50 ms
    boundary = start + int(0.05 * sample_rate)

    early = np.sum(energy[start:boundary])
    late = np.sum(energy[boundary:])
    if late == 0.0:
        raise ValueError("No energy after 50 ms - recording too short")

    return 10 * math.log10(early / late)


def mean_c50(pressures: np.ndarray, sample_rate: float) -> float:
    """Mean C50 over all mics"""
    return float(np.mean([c50(mic, sample_rate) for mic in pressures]))