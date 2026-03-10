import time

import taichi as ti

from visualization import Simulation, SceneRenderer,config
from visualization import PlaneGeometry
from fdtd import FDTD_Simulation, Source

def main():
    ti.init(arch=ti.gpu, device_memory_GB=config.MEMORY_LIMIT_GB)

    # to pozniej do wywalenia bedzie
    wavelength = config.C / config.FREQ_MAX
    dx = wavelength / config.NODES_PER_WAVELENGTH

    # ustawiam N zalezne od rozmiaru pomieszczenia(lepiej zmieniac SIZE_M niz N)
    config.N = int(config.SIZE_M / dx) + config.PML_THICK*2

    SOURCE_POS = (config.N // 2, config.N // 2, config.N // 2)
    source = Source(SOURCE_POS[0], SOURCE_POS[1], SOURCE_POS[2], config.FREQ_MAX, config.AMPLITUDE)
    fdtd_sim = FDTD_Simulation(dx, config.N, config.C, source)

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

        now = time.time()
        delta_real_time = now - prev_time
        prev_time = now
        accumulator += delta_real_time
        if (accumulator >= config.FRAME_DURATION):
            fdtd_sim.update()
            accumulator = 0

        current_pressure = fdtd_sim.get_current_pressure()
        #sim.update_wave()

        sim.update_planes(slice_y, slice_z, current_pressure)
        renderer.render_frame(simulation=sim, plane_geo=plane_geo)

if __name__ == "__main__":
    main()