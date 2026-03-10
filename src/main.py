import taichi as ti
from config import N,COURANT_SQ, DT, FREQ_MAX
import math
import numpy as np

ti.init(arch=ti.gpu)

# pressure fields for time stepping (p_old and p_curr)
p_old = ti.field(ti.f32, shape=(N, N))
p_curr = ti.field(ti.f32, shape=(N, N))

mask = ti.field(dtype=ti.i32, shape=(N, N)) # boundary and obstacle mask (0 = air, 1 = wall)

@ti.kernel
def setup_mask():
    for i, j in mask:
        # boundary walls (dirichlet boundary conditions)
        if i <= 1 or i >= N - 2 or j <= 1 or j >= N - 2:
            mask[i, j] = 1
        # internal scattering object (example obstacle)
        if N // 4 < i < N // 2 and N // 3 < j < N // 3 + 10:
            mask[i, j] = 1

@ti.kernel
def step(p_prev: ti.template(), p_now: ti.template(), time_elapsed: ti.f32):
    for i,j in p_now:
        if 0 < i < N - 1 and 0 < j < N - 1:
            if mask[i, j] == 1:
                # wall condition: pressure is zero (Dirichlet)
                p_prev[i, j] = 0
            else:
                # finite difference approximation of the Laplacian
                laplacian = p_now[i + 1, j] + p_now[i - 1, j] + \
                            p_now[i, j + 1] + p_now[i, j - 1] - 4 * p_now[i, j]
                # update pressure for the next time ste
                p_prev[i, j] = COURANT_SQ * laplacian + 2 * p_now[i, j] - p_prev[i, j]

    # Acoustic Wave Source
    center = N // 2
    # apply a sinusoidal soft source for the initial duration
    if time_elapsed < 0.05:
        source_val = ti.sin(2.0 * math.pi * FREQ_MAX * time_elapsed)
        p_prev[center, center] += source_val



def main():
    gui = ti.GUI("2D FDTD Simulation - Dirichlet", res=(N, N))
    setup_mask()

    buffers = [p_old, p_curr]
    print(f"Grid Resolution: {N}x{N}")

    frame = 0
    while gui.running:
        # To save memory, we use only two matrices instead of three by swapping their roles
        # This approach reuses the memory of the "old" buffer to store the "future" step
        b_old = frame % 2
        b_curr = (frame + 1) % 2

        # compute next time step
        step(buffers[b_old], buffers[b_curr], frame * DT)

        # transfers the pressure field from GPU to CPU and
        p_np = buffers[b_old].to_numpy()
        # calculates its maximum absolute value to normalize the wave's amplitude for visualization
        max_val = np.abs(p_np).max()

        # normalize pressure to [0, 1] range for grayscale display
        p_normalized = p_np / (max_val + 1e-8)
        p_np_norm = p_normalized * 0.5 + 0.5

        # render mask as pure white and pressure as grayscale
        mask_np = mask.to_numpy()
        img = np.where(mask_np > 0, 1.0, p_np_norm)

        gui.set_image(img)

        gui.show()
        frame += 1

if __name__ == "__main__":
    main()