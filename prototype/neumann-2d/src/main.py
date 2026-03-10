import taichi as ti
from config import COURANT_SQ, N_2, MATERIALS,COURANT, SIGMA, AMPLITUDE, SRC_X, SRC_Y, DT, frame_duration, DELAY_GAUSS, print_config
import numpy as np
import time

ti.init(arch=ti.gpu)

# pressure fields for time stepping (p_old and p_curr)
p_old = ti.field(ti.f32, shape=(N_2, N_2))
p_curr = ti.field(ti.f32, shape=(N_2, N_2))

# matrix representing the count of neighboring nodes for each grid point
k_field = ti.field(dtype=ti.i32, shape=(N_2, N_2))

# matrix field for local absorption coefficients used to implement material losses
alpha_field = ti.field(dtype=ti.f32, shape=(N_2, N_2))

# matrix of BK values used for simplified boundary condition calculations
bk_field = ti.field(dtype=ti.f32, shape=(N_2, N_2))


@ti.kernel
def generation_symulation_map():
    for i,j in k_field:
        k_field[i , j] = 4 # default 4 neighbours
        alpha_field [i, j] = 0.0 # air alpha =0

    # set boundary nodes to 0 neighbors to define the computational domain limits
    for i,j in k_field:
        if i == 0 or j == 0 or i == N_2 - 1 or j == N_2 - 1:
            k_field[i, j] = 0
            alpha_field[i, j] = 0.0

    # walls
    for i,j in k_field:
        # corner nodes: only 2 adjacent neighbors within the domain.
        if((i==1 and j ==1) or (i==1 and j==N_2-2) or (i==N_2-2 and j==1) or (i==N_2-2 and j==N_2-2)):
            k_field[i, j] = 2
            alpha_field[i, j] = MATERIALS["brick"]

        # vertical walls: 3 neighbors
        elif (i == 1 or i == N_2 - 2) and (j > 0 and j < N_2 -1):
            k_field[i, j] = 3
            alpha_field[i, j] = MATERIALS["brick"]

        # horizontal walls: 3 neighbors
        elif (j == 1 or j == N_2 - 2) and (i > 0 and i < N_2 -1):
            k_field[i, j] = 3
            alpha_field[i, j] = MATERIALS["brick"]

    # Internal Obstacle
    # coordinates defining the boundaries of the obstacle
    r_x1, r_x2 = 100, 200
    r_y1, r_y2 = 100, 200

    for i,j in k_field:
        # interior of the obstacle: No wave propagation
        if i > r_x1 and i < r_x2 and j > r_y1 and j < r_y2:
            k_field[i, j] = 0
            alpha_field[i, j] = 0.0

        # obstacle corners: 4 neighbors
        elif (i == r_x1 and j == r_y1) or \
                (i == r_x1 and j == r_y2) or \
                (i == r_x2 and j == r_y1) or \
                (i == r_x2 and j == r_y2):
            k_field[i, j] = 4
            alpha_field[i, j] = MATERIALS["wooden_bench"]

        # obstacle vertical edges: 3 neighbors
        elif (i == r_x1 or i == r_x2) and (j > r_y1 and j < r_y2):
            k_field[i, j] = 3
            alpha_field[i, j] = MATERIALS["wooden_bench"]

        # obstacle horizontal edges: 3 neighbors
        elif (j == r_y1 or j == r_y2) and (i > r_x1 and i < r_x2):
            k_field[i, j] = 3
            alpha_field[i, j] = MATERIALS["wooden_bench"]

    # precalculate the Bk matrix used for boundary conditions and material absorption
    for i,j in bk_field:
        k_val = k_field[i, j]
        beta_val = beta_from_alpha(alpha_field[i, j])
        bk_field[i, j] = (4.0 - float(k_val)) * COURANT * beta_val / 2.0


@ti.func
def beta_from_alpha(alpha):
    # converts the absorption coefficient (alpha) to the reflection-based beta factor
    Beta = 0.0
    if alpha >= 1.0:  # open space (total absorption)
        Beta = 1.0
    elif alpha <= 0.0:    # ideal wall (total reflection)
        Beta = 0.0
    else:
        R = ti.sqrt(1.0 - alpha)
        Beta = (1.0 - R) / (1.0 + R)

    return Beta

@ti.kernel
def step(p_prev: ti.template(), p_now: ti.template(), steps: int):
    for i,j in p_now:
        if k_field[i , j] > 0:
            k_val = k_field[i,j]
            bk_val = bk_field[i,j]

            # discretized acoustic wave equation optimized for GPU computation
            current_term = (2.0 - float(k_val) * COURANT_SQ) * p_now[i, j]
            loss_term = (bk_val - 1.0) * p_prev[i, j]
            laplacian_term = COURANT_SQ * (p_now[i + 1, j] + p_now[i - 1, j] + p_now[i, j + 1] + p_now[i, j - 1])

            # update pressure for the next time step
            p_prev[i, j] = (current_term + loss_term + laplacian_term) / (1 + bk_val)
        else:
            # nodes with 0 neighbors remain at zero
            p_prev[i, j] = 0

    # Acoustic Source
    # gaussian pulse calculation for a smooth, band-limited signal
    pulse = AMPLITUDE * ti.exp(- ((DT*steps - DT*DELAY_GAUSS)**2) / (2 * SIGMA**2))
    p_prev[SRC_X, SRC_Y] += pulse # soft source


def main():
    gui = ti.GUI("2D FDTD Simulation - Neumann", res=(N_2, N_2))
    print_config()
    generation_symulation_map() # initialize grid geometry and material fields

    # set up a buffer swap mechanism to alternate between past and current pressure fields
    buffers = [p_old, p_curr]

    prev_time = time.time()
    steps = 0
    accumulator = 0

    # create a boolean mask for visualization to highlight wall/obstacle locations
    alpha_map = alpha_field.to_numpy()
    is_wall = alpha_map > 0.01

    is_paused = False
    while gui.running:
        # event handling for pausing the simulation with the SPACE key
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == ti.GUI.SPACE:
                is_paused = not is_paused

        # calculate elapsed real time to maintain a constant frame rate
        now = time.time()
        delta_real_time = now - prev_time
        prev_time = now

        accumulator+=delta_real_time

        # only update the simulation and GUI if the frame duration interval has passed
        if(accumulator >= frame_duration):
            if not is_paused:
                steps += 1

                # To save memory, we use only two matrices instead of three by swapping their roles
                # This approach reuses the memory of the "old" buffer to store the "future" step
                b_old = steps % 2
                b_curr = (steps + 1) % 2

                # compute next time step
                step(buffers[b_old], buffers[b_curr], steps)

            p_np = buffers[b_old].to_numpy()
            max_val = np.abs(p_np).max()

            # Normalize pressure values to the range [-1, 1]
            p_normalized = p_np / (max_val + 1e-8)
            # normalize pressure to [0, 1] range for grayscale display
            gray_val = p_normalized * 0.5 + 0.5

            gray_val = np.clip(gray_val, 0.0, 1.0)
            img = np.dstack((gray_val, gray_val, gray_val))

            # colorize walls
            img[is_wall, 1] *= 0.3 # green 30%
            img[is_wall, 2] *= 0.3 # blue 30%

            gui.set_image(img)
            gui.show()

            # reset the timer accumulator for the next frame
            accumulator = 0


if __name__ == "__main__":
    main()