import time

import taichi as ti
import math
from visualization import Simulation, SceneRenderer, config, MAX_VOXELS
from visualization import PlaneGeometry
from fdtd import FDTD_Simulation, ReceiverManager, SourceManager
import test


# do liczenia dt
def get_time_step(dimensions, dx, speed, safety_factor):
    courant_limit = 1.0 / math.sqrt(dimensions)
    return (dx / speed) * courant_limit * safety_factor

def print_information(dt, Nx, Ny, Nz, dx, receiver_steps):
    print("Simulation parameters:")
    print(f"dt: {dt}")
    print(f"Nx: {Nx}")
    print(f"Ny: {Ny}")
    print(f"Nz: {Nz}")
    print(f"dx: {dx}")
    print(f"receiver steps: {receiver_steps}")




def main():
    ti.init(arch=ti.gpu, device_memory_GB=config.MEMORY_LIMIT_GB)

    # ustawiamy zrodla
    source_manager = SourceManager(5)
    source_manager.add_source(70, 70,70 ,config.FREQ_MAX, config.AMPLITUDE)

    max_freq = source_manager.get_max_freq()
    c = config.C # na razie z configa

    wavelength = c / max_freq
    dx = wavelength / config.NODES_PER_WAVELENGTH

    Nx = int(config.X_METERS / dx)
    Ny = int(config.Y_METERS / dx)
    Nz = int(config.Z_METERS / dx)

    MAX_VOXELS = Nx**3  # traktujac ze Nx = Ny = Nz

    dt = get_time_step(config.DIM, dx, c, config.SAFETY_FACTOR)

    #tu bedziemy ustawiac ile maksymalnie czasu chcemy nagrywac( rzeczywistych sekund, nie tyle ile bedzie trwala symulacja)
    # chodzi o to ze bedziemy mogli symulacje odtworzyc 10 razy wolniej niz w rzeczywistosci
    receiver_seconds = 5.0
    receiver_steps = int(math.ceil(receiver_seconds / dt)) # musi byc int

    # ustawiamy mikrofony
    receiver_manager = ReceiverManager(5,receiver_steps, dt)
    receiver_manager.add_receiver(110,110,110,"mikrofon1")

    material_core = ti.Vector.field(2, dtype=ti.f32, shape=(Nx, Ny, Nz))
    test.generate_simulation_map(material_core) # to otrzymamy z wokselizacji
    # dokladnie to bedziemy musieli za pomoca slownika stworzyc tablice { alpha, density}

    print_information(dt, Nx, Ny, Nz, dx, receiver_steps)

    fdtd_sim = FDTD_Simulation(c, dx, dt, source_manager, receiver_manager, material_core )

    sim = Simulation()
    renderer = SceneRenderer()
    #geo = CubeGeometry()
    plane_geo = PlaneGeometry()

    slice_y = config.N // 2
    slice_z = config.N // 2

    prev_time = time.time()
    accumulator = 0

    while renderer.is_running:
        gui = renderer.window.get_gui()
        with gui.sub_window("2D Slices", 0.05, 0.05, 0.5, 0.2):
            slice_y = gui.slider_int("Horizontal Slice", slice_y, 0, config.N - 1)
            slice_z = gui.slider_int("Vertical Slice", slice_z, 0, config.N - 1)

        # now = time.time()
        # delta_real_time = now - prev_time
        # prev_time = now
        # accumulator += delta_real_time
        # if (accumulator >= config.FRAME_DURATION):
        fdtd_sim.update()
            # accumulator = 0

        current_pressure = fdtd_sim.get_current_pressure()
        #sim.update_wave()

        sim.update_planes(slice_y, slice_z, current_pressure)
        renderer.render_frame(simulation=sim, plane_geo=plane_geo)

if __name__ == "__main__":
    main()