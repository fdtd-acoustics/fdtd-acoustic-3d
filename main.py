import taichi as ti
from gui import MainMenuWindow
from visualization.vis_config import MEMORY_LIMIT_GB


def main():
    ti.init(arch=ti.gpu, device_memory_GB=MEMORY_LIMIT_GB)
    app = MainMenuWindow()
    app.mainloop()


if __name__ == "__main__":
    main()
