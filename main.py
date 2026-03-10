import taichi as ti
from visualization import Simulation, SceneRenderer,config
from visualization import PlaneGeometry

def main():
    ti.init(arch=ti.gpu, device_memory_GB=config.MEMORY_LIMIT_GB)

    sim = Simulation()
    renderer = SceneRenderer()
    #geo = CubeGeometry()
    plane_geo = PlaneGeometry()

    slice_y = config.N // 2
    slice_z = config.N // 2

    while renderer.is_running:
        gui = renderer.window.get_gui()
        with gui.sub_window("2D Slices", 0.05, 0.05, 0.5, 0.2):
            slice_y = gui.slider_int("Horizontal Slice", slice_y, 0, config.N - 1)
            slice_z = gui.slider_int("Vertical Slice", slice_z, 0, config.N - 1)

        sim.update_wave()
        sim.update_planes(slice_y, slice_z)
        renderer.render_frame(simulation=sim, plane_geo=plane_geo)

if __name__ == "__main__":
    main()