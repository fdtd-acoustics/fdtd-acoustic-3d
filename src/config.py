import taichi as ti
import math

def get_time_step(dimensions, dx, speed, safety_factor):
    courant_limit = 1.0 / math.sqrt(dimensions)
    return (dx / speed) * courant_limit * safety_factor

#PML
PML_THICK= 20 # grubosc sciany pml
ALPHA_MAX = 0.15 # maksymalne tlumienie wystepujace na samym brzegu domeny obliczeniowej

# Parametry srodowiska
SIZE_M = 20.0  # rozmiar [m]
DIM = 2  # 2D
C = 343.0   # predkosc [m/s]

# Parametry zrodla
FREQ_MAX = 1000.0 # maksymalna czestotliwpsc fali [Hz]
AMPLITUDE = 10.0

#Parametry czasowe impulsu
SIGMA = math.sqrt(2 * math.log(2)) / (2 * math.pi * FREQ_MAX) # sigma dla impulsu Gaussa
DELAY_GAUSS = 4 * SIGMA # opóźnienie startu sygnału

# Dyskretyzacja przestrzenna
NODES_PER_WAVELENGTH = 10    # ilsoc punktow na dlugosc fali
WAVELENGTH = C / FREQ_MAX    # dlugosc fali [m]
DX = WAVELENGTH / NODES_PER_WAVELENGTH   # krok odleglosciowy [m]

# Obliczanie liczby wezlow
N = int(SIZE_M / DX) + PML_THICK*2    # ilosc punktow w jednym wymiarze
#N_2 = N + 2 # dodajemy po jednym punckie na kazda strone do obliczen

# Polozenie zrodla
SRC_X = N // 2
SRC_Y = N // 2

# Wspolczynnik bezpieczenstwa
SAFETY_FACTOR = 0.99

# Warunek Couranta
COURANT = (1.0 / math.sqrt(DIM)) * SAFETY_FACTOR
COURANT_SQ = COURANT**2

# Obliczanie kroku czasowego
DT = get_time_step(DIM, DX, C, SAFETY_FACTOR)

# wspolczynniki alfa dla materiałów przy fali o czestotliowsci 500 HZ
# oraz gestosci osrodkow
# https://www.acoustic-supplies.com/absorption-coefficient-chart/
MATERIAL_PROPS = {
    "air" : (0.0, 1.21),
    "brick": (0.03,1800.0), # brick(natural)
    "wooden_bench": (0.76, 950.0) # Benches (wooden, fully occupied)
}

# Parametry animacji
FRAME_DURATION = 0.02 # czas trwania jednej klatki wizualizacji [s]



def print_config():
    print("-" * 30)
    print(f"SYMULACJA FDTD {DIM}D")
    print("-" * 30)
    print(f"Rozmiar siatki:  {N} x {N}")
    print(f"Krok DX:         {DX*1000:.2f} mm")
    print(f"Krok DT:         {DT*1e6:.2f} us")
    print(f"Liczba Couranta: {COURANT:.4f} (Max: {1/math.sqrt(DIM):.4f})")
    print("-" * 30)
