"""
Window and rendering settings
"""

# === Window ===
SCREEN_RESOLUTION = (1280, 720)
FPS_LIMIT = 500
MEMORY_LIMIT_GB = 4

#temp
#todo:poprawka w geometry i sumulation i do usuniecia
N               = 185
MAX_VOXELS      = N ** 3
VOXEL_WIDTH     = 1
VOXEL_DISTANCE  = 1
PARTICLE_RADIUS = 1

# === CAMERA ===
CAMERA_POS_X, CAMERA_POS_Y, CAMERA_POS_Z = N*1.5, N*1.5, N*1.5 # tu zmienile,
CAMERA_SPEED = 0.5

# === COLORS
BG_COLOR      = (0.95, 0.95, 0.95)
LIGHT_COLOR   = (1.0,  1.0,  1.0)
AMBIENT_COLOR = (0.6,  0.6,  0.6)
CUBE_COLOR    = (0.5,  0.5,  0.5)



