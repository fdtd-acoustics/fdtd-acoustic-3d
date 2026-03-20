# Acoustic Wave Propagation Simulator (3D FDTD)

<div align="center">
  <img src="https://img.shields.io/github/v/release/mateushhh/fdtd-acoustic-3d?color=success&logo=github" alt="GitHub Release" />
  <img src="https://img.shields.io/badge/python-%3E%3D3.10-3776AB?logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/Taichi_Lang-%3E%3D1.7.0-4A90E2" alt="Taichi Lang" />
</div>

## Description
This project is a computer application that simulates and visualizes how sound waves travel in a 3D space. It calculates how sound moves by solving the acoustic wave equation using a method called FDTD (Finite Difference Time Domain).

## Technologies Used
* **Python 3.x**: Chosen for its broad support for scientific libraries and ease of development.
* **Taichi Lang**: The core technology powering both the computational engine and result visualization.

## Main Features
* **3D FDTD Engine**: Simulates how sound pressure changes and moves through a 3D space.
* **PML Perfectly Matched Layers**: Uses special layers to simulate open, borderless spaces, and calculates how sound bounces off different obstacles.
* **Environment Voxelizer**: Takes complex 3D scenes (like .obj files made with Blender) and turns them into a grid of materials with specific physical properties like sound absorption so the engine can process them..
* **Real-time Visualization**: Uses Taichi's GUI system to render 2D slices of the 3D acoustic field dynamically.

## Roadmap & Future Works
* **Audio I/O Integration**: Adding support for `.wav` files to allow importing custom source signals and exporting audio "recordings" after the simulation completes.
* **Advanced Pre-processing**: Better synchronization with Blender and other 3D modeling software for easier scene creation.
* **Voxel Optimization**: Further optimization of the voxel processing pipeline to improve memory efficiency and initialization speed.

<!--

## Demo

<div align="center">
  <img src="assets/simulation_demo.gif" alt="Wave Propagation Simulation" width="600"/>
  <p><i>Real-time 2D slice visualization of 3D acoustic wave propagation.</i></p>
</div>

<div align="center">
  <img src="assets/voxelizer_preview.png" alt="Voxelized Environment" width="600"/>
  <p><i>3D environment (.obj) voxelized into particles.</i></p>
</div>

-->

## Setup

1. Clone the repository.
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. Run the simulation
```bash
python main.py
```

## License
Distributed under the MIT License.

---

<div align="center">
 <a href="https://github.com/mateushhh">
 <img src="https://img.shields.io/badge/Author-Mateusz_Grzonka-8A2BE2?logo=github" alt="Mateusz Grzonka">
 </a>
 
 <a href="https://github.com/pomiano">
 <img src="https://img.shields.io/badge/Author-Miłosz_Pomianowski-8A2BE2?logo=github" alt="Miłosz Pomianowski">
 </a>
 <a href="https://github.com/maciejblawat">
 <img src="https://img.shields.io/badge/Author-Maciej_Bławat-8A2BE2?logo=github" alt="Maciej Bławat">
 </a>
</div>
