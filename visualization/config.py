#TODO:Maybe zmiana na yaml?

## Simulation settings
N = 185  # nieuzywane
MAX_VOXELS = N**3   #zmienanie w main
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
CAMERA_POS_X, CAMERA_POS_Y, CAMERA_POS_Z = 2,2,2 # tu zmienile,
CAMERA_LOOKAT_X, CAMERA_LOOKAT_Y, CAMERA_LOOKAT_Z = 0, 0, 0
CAMERA_SPEED = 0.5

# Colors
CUBE_COLOR = (0.5, 0.5, 0.5)
BG_COLOR = (0.95, 0.95, 0.95)
LIGHT_COLOR = (1, 1, 1)
AMBIENT_COLOR = (1.0, 1.0, 1.0)

# Temporary (just for test visualisation)
#SOURCE_POS = (N/2 * VOXEL_DISTANCE, N/2 * VOXEL_DISTANCE, N/2 * VOXEL_DISTANCE)

SOURCE_POS = (N//2, N//2, N//2 )
#-------------- FDTD -----------------------------
FREQ_MAX = 1000.0 # maksymalna czestotliwpsc fali [Hz]
AMPLITUDE = 1000.0
C = 343.0
PML_THICK = 20
NODES_PER_WAVELENGTH = 10
alpha_max = 0.15 # maksymalne tlumienie wystepujace na samym brzegu domeny obliczeniowej
DIM = 3  # 2D
SAFETY_FACTOR = 0.99
# --------------------------------------------
FRAME_DURATION = 0.02  # jeden krok trwa 0.02 sekundy a jeden krok to np. 0.00005s czasu rzeczywistego
#czyli liczac 60s / 0.02s to daje 3000 krokow(tyle zobaczymy przez minute),  3000 * 0.00005 to 0.171s czyli przez minute jak ogladamy widzimy 0.171 sekundy prawdziwego czasu

DEFAULT_ALPHA = 0.0
DEFAULT_DENSITY = 1.21

X_METERS = 5.0
Y_METERS = 5.0
Z_METERS = 5.0

# --- paths ---
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
SCENES_DIR = ROOT_DIR / "scenes"
SCENES_IN_DIR = SCENES_DIR / "input"
SCENES_OUT_DIR = SCENES_DIR / "output"
