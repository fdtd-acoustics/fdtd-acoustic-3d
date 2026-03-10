## Simulation settings
N = 256
MAX_VOXELS = N**3
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
CAMERA_POS_X, CAMERA_POS_Y, CAMERA_POS_Z = N*1.5, N*1.5, N*1.5
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
AMPLITUDE = 10.0
C = 343.0
SIZE_M = 5.0
PML_THICK = 20
NODES_PER_WAVELENGTH = 10
alpha_max = 0.15 # maksymalne tlumienie wystepujace na samym brzegu domeny obliczeniowej
DIM = 3  # 2D
safety_factor = 0.99
# --------------------------------------------