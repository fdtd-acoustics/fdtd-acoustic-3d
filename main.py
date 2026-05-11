import sys


def main():
    if len(sys.argv) > 1:
        from cli.main import main as cli_main

        raise SystemExit(cli_main(sys.argv[1:]))

    import taichi as ti

    from gui import MainMenuWindow
    from visualization.vis_config import MEMORY_LIMIT_GB

    ti.init(arch=ti.gpu, device_memory_GB=MEMORY_LIMIT_GB)
    app = MainMenuWindow()
    app.mainloop()


if __name__ == "__main__":
    main()
