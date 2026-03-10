import taichi as ti
import math

C = 343.0   # speed of sound in air [m/s]
FREQ_MAX = 1000.0 # maximum frequency of the wave [Hz]
SIZE_M = 20.0  # domain size [m]
DIM = 2  # dimensions (2D)

NODES_PER_WAVELENGTH = 10   # number of grid points per wavelength
WAVELENGTH = C / FREQ_MAX    # wavelength [m]
DX = WAVELENGTH / NODES_PER_WAVELENGTH   # spatial step [m]
N = int(SIZE_M / DX)    # number of points in one dimension

SAFETY_FACTOR = 0.99 # stability safety factor
COURANT = (1.0 / math.sqrt(DIM)) * SAFETY_FACTOR # courant number
COURANT_SQ = COURANT**2 # squared Courant number

# Calculates the maximum stable time step (DT) to satisfy the CFL condition and prevent simulation instability
def get_time_step(dimensions, dx, speed, safety_factor):
    courant_limit = 1.0 / math.sqrt(dimensions)
    return (dx / speed) * courant_limit * safety_factor

DT = get_time_step(DIM, DX, C, SAFETY_FACTOR) # time step [s]


def print_config():
    print("-" * 30)
    print(f"FDTD SIMULATION CONFIGURATION {DIM}D")
    print("-" * 30)
    print(f"Grid Size:  {N} x {N} nodes")
    print(f"Spatial Step (DX):         {DX*1000:.2f} mm")
    print(f"Time Step (DT):         {DT*1e6:.2f} us")
    print(f"Courant Number: {COURANT:.4f} (Limit: {1/math.sqrt(DIM):.4f})")
    print(f"Max Frequency:     {FREQ_MAX} Hz")
    print("-" * 30)

if __name__ == "__main__":
    print_config()