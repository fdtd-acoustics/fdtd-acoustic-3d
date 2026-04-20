"""
rule : if something directly changes simulation it goes here
       if something changes only visualization - it goes to visualization/vis_config.py
"""

from pathlib import Path


# === physics ===
SOUND_SPEED = 343.0
NODES_PER_WAVELENGTH = 10
DIM = 3
SAFETY_FACTOR = 0.95


# === Materials ===
#TODO:: to jest to o czym gadalismy - prawdopodobnie bedzie do zmiany

MATERIAL_MAP: dict[int, dict] = {
    0: {"name": "air",   "alpha": 0.0, "density": 1.225},
    1: {"name": "wall",  "alpha": 0.0, "density": 1.225},  # takie byly poprzednie wartosc: 0.5 #1000.0
    2: {"name": "metal", "alpha": 0.0, "density": 1.225}, # takie byly poprzednie wartosc: 0.1 500.0
}
DEFAULT_MATERIAL_ID = 1
DEFAULT_ALPHA       = MATERIAL_MAP[0]["alpha"]
DEFAULT_DENSITY     = MATERIAL_MAP[0]["density"]


# === Paths ===
ROOT_DIR = Path(__file__).parent

ASSETS_DIR = ROOT_DIR / "assets"
MODELS_DIR = ASSETS_DIR / "models"   # modele
AUDIO_DIR  = ASSETS_DIR / "audio"  # pliki audio

DATA_DIR = ROOT_DIR / "data"
VOXELS_DIR = DATA_DIR / "voxels"     # pliki .npz po voxelizacji
PROJECTS_DIR = DATA_DIR / "projects"  # pełne zapisy konfiguracji

OUTPUT_DIR = ROOT_DIR / "output"
WAV_DIR  = OUTPUT_DIR / "signals"    # pliki .wav
PLOT_DIR = OUTPUT_DIR / "plots"     # wykresy