# Changelog

All notable changes to this project will be documented in this file.

## [v0.1.0] - 2026-05-10

### Added

- **3D FDTD engine** – Full 3D acoustic wave propagation built on top of the 2D prototypes, accelerated with Taichi.
- **PML boundary conditions** – Perfectly Matched Layer absorbing boundaries in 3D with configurable thickness and `alpha_max`.
- **Voxelizer with `.obj` scene loading** – Convert arbitrary triangulated rooms into a voxel grid, auto-centre the mesh, and map per-face materials.
- **Material Library** – CSV-backed catalogue of acoustic materials (absorption, density, colour) with a dedicated GUI for adding, editing, and removing entries.
- **Multi-source / multi-receiver setup** – Place an arbitrary number of sources and microphones, each with its own name, position, duration, and volume.
- **Custom waveforms** – Sources can emit either parametric Gaussian pulses or arbitrary user-supplied WAV files, with automatic frequency analysis driving grid resolution.
- **Interactive 3D placement** – Click-to-place workflow for positioning sources and microphones inside the loaded scene.
- **Tkinter GUI** – Main menu, new-simulation wizard, post-setup dialog, and materials window covering the full configuration flow.
- **Real-time 3D visualisation** – Taichi viewer rendering the room mesh together with XY / XZ / YZ pressure slices, with log-scale display, pause, simulation timer, and step counter.
- **Project save / load (`.npz`)** – Persist the full simulation configuration alongside the voxel cache; reload it later in the GUI for editing, with frequency-vs-grid validation on load.
- **WAV audio output** – Per-receiver recordings written to disk as WAV files together with pressure-over-time plots.
- **Test scenes** – Bundled Blender-authored `.obj` rooms for quickly trying different geometries and material setups.
- **Project structure** – Reorganised the repository into clear `assets/`, `data/`, and `output/` directories.

## [v0.0.1] - 2026-03-11

### Added

- **2D FDTD prototypes** – Implemented simple 2D Finite-Difference Time-Domain simulations to experiment with acoustic wave propagation and understand the method before moving to the full 3D implementation. Two prototype variants were created: one using **Dirichlet boundary conditions** and another using **Neumann boundary conditions**.
- **CFL stability check** – Added automatic calculation of the maximum stable time step based on the Courant–Friedrichs–Lewy condition to ensure the simulation remains stable.
- **2D visualization prototype** – Added a basic real-time visualization of the acoustic pressure field using the Taichi GUI to help observe how the simulation behaves.