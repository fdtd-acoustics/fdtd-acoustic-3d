# Changelog

All notable changes to this project will be documented in this file.

## [v0.0.1] - 2026-03-11

### Added

- **2D FDTD prototypes** – Implemented simple 2D Finite-Difference Time-Domain simulations to experiment with acoustic wave propagation and understand the method before moving to the full 3D implementation. Two prototype variants were created: one using **Dirichlet boundary conditions** and another using **Neumann boundary conditions**.
- **CFL stability check** – Added automatic calculation of the maximum stable time step based on the Courant–Friedrichs–Lewy condition to ensure the simulation remains stable.
- **2D visualization prototype** – Added a basic real-time visualization of the acoustic pressure field using the Taichi GUI to help observe how the simulation behaves.