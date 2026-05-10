#TODO: Zostawiam narazie jakby sie okazalo ze czegos zapomnialem wrzucic
#wszystko jest teraz config.py w glownym pliku i vis_config.py w visualization

## Simulation settings
VOXEL_WIDTH = 1
VOXEL_DISTANCE = 1
DELTA_TIME = 0.1
PARTICLE_RADIUS = 1
PRESSURE_TRESHOLD = 0.25 # Noise Gate, nie wyświetlamy powietrza z ciśnieniem poniżej abs(0.25)

# Window settings
SCREEN_RESOLUTION = (800, 600)
FPS_LIMIT = 60
MEMORY_LIMIT_GB = 2

# Camera settings
CAMERA_LOOKAT_X, CAMERA_LOOKAT_Y, CAMERA_LOOKAT_Z = 0, 0, 0
CAMERA_SPEED = 0.5

# Colors
CUBE_COLOR = (0.5, 0.5, 0.5)
BG_COLOR = (0.95, 0.95, 0.95)
LIGHT_COLOR = (1, 1, 1)
AMBIENT_COLOR = (1.0, 1.0, 1.0)

# Temporary (just for test visualisation)
#SOURCE_POS = (N/2 * VOXEL_DISTANCE, N/2 * VOXEL_DISTANCE, N/2 * VOXEL_DISTANCE)

# SOURCE_POS = (N//2, N//2, N//2 )
#-------------- FDTD -----------------------------
# FREQ_MAX = 1000.0 # maksymalna czestotliwpsc fali [Hz]
# AMPLITUDE = 1000.0
# C = 343.0
# NODES_PER_WAVELENGTH = 10
# DIM = 3  # 3D
# SAFETY_FACTOR = 0.99
# --------------------------------------------
FRAME_DURATION = 0.02  # jeden krok trwa 0.02 sekundy a jeden krok to np. 0.00005s czasu rzeczywistego
#czyli liczac 60s / 0.02s to daje 3000 krokow(tyle zobaczymy przez minute),  3000 * 0.00005 to 0.171s czyli przez minute jak ogladamy widzimy 0.171 sekundy prawdziwego czasu

# DEFAULT_ALPHA = 0.0
# DEFAULT_DENSITY = 1.21
#
# X_METERS = 5.0
# Y_METERS = 5.0
# Z_METERS = 5.0

# --- paths ---
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
SCENES_DIR = ROOT_DIR / "scenes"
SCENES_MODELS_DIR = SCENES_DIR / "models"
SCENES_OUT_DIR = SCENES_DIR / "output"


