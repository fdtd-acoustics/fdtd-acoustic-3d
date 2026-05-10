PML_THICKNESS = 30 # pml thickness
ALPHA_MAX = 0.30 # maximum absorption coefficient

SIZE_M = 20.0  # domain size [m]
DIM = 2  # dimensions (2D)
SOUND_SPEED = 343.0   # speed of sound in air [m/s]

FREQ_MAX = 1000.0 # maximum frequency of the wave [Hz]
AMPLITUDE = 100.0 # peak amplitude of the acoustic pressure wave source

NODES_PER_WAVELENGTH = 10    # number of grid points per wavelength

# source position
SRC_X = 350
SRC_Y = 350

#receiver
REC_X = 30
REC_Y = 30
MAX_STEPS = 10000
RECEIVER_NAME = "mic1"

SAFETY_FACTOR = 0.99 # stability safety factor

# material absorption coefficients and densities at 500 Hz
# https://www.acoustic-supplies.com/absorption-coefficient-chart/
MATERIAL_PROPS = {
    "air" : (0.0, 1.21),
    "brick": (0.03,1800.0), # brick(natural)
    "wooden_bench": (0.76, 950.0) # Benches (wooden, fully occupied)
}
DEFAULT_ALPHA, DEFAULT_DENSITY = MATERIAL_PROPS["air"]

FRAME_DURATION = 0.0002 # fixed time interval for GUI visualization frames [s]




