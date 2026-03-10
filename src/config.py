import taichi as ti
import math

C = 343.0   # speed of sound in air [m/s]
FREQ_MAX = 1000.0 # maximum frequency of the wave [Hz]
SIZE_M = 20.0  # domain size [m]
DIM = 2  # dimensions (2D)

AMPLITUDE = 10.0 # peak amplitude of the acoustic pressure wave source

# Gaussian Impulse Parameter
SIGMA = math.sqrt(2 * math.log(2)) / (2 * math.pi * FREQ_MAX) # standard deviation (SIGMA) calculated to fit the maximum frequency
DELAY_GAUSS = 4 * SIGMA  # delay to ensure the pulse starts smoothly from near-zero amplitude

NODES_PER_WAVELENGTH = 10    # number of grid points per wavelength
WAVELENGTH = C / FREQ_MAX    # wavelength [m]
DX = WAVELENGTH / NODES_PER_WAVELENGTH   # spatial step [m]
N = int(SIZE_M / DX)     # interior grid points in one dimension
N_2 = N + 2 # grid size including a 1-node boundary layer on all sides for calculations

# source position
SRC_X = N_2 // 2
SRC_Y = N_2 // 2

SAFETY_FACTOR = 0.99 # stability safety factor
COURANT = (1.0 / math.sqrt(DIM)) * SAFETY_FACTOR # Courant number
COURANT_SQ = COURANT**2 # squared Courant number

# calculates the maximum stable time step (DT) to satisfy the CFL condition and prevent simulation instability
def get_time_step(dimensions, dx, speed, safety_factor):
    courant_limit = 1.0 / math.sqrt(dimensions)
    return (dx / speed) * courant_limit * safety_factor

DT = get_time_step(DIM, DX, C, SAFETY_FACTOR) # time step [s]

# material absorption coefficients at 500 Hz
# https://www.acoustic-supplies.com/absorption-coefficient-chart/
MATERIALS = {
    "brick": 0.03, # brick(natural)
    "wooden_bench": 0.76 # Benches (wooden, fully occupied)
}

frame_duration = 0.02 # fixed time interval for GUI visualization frames [s]


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
