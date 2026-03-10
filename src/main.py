import taichi as ti
from config import N,COURANT_SQ, MATERIAL_PROPS,C,COURANT, SIGMA, AMPLITUDE, SRC_X, SRC_Y, DT, FRAME_DURATION, DELAY_GAUSS, print_config, ALPHA_MAX, PML_THICK
import numpy as np
import time

ti.init(arch=ti.gpu)

#macierze
p_old = ti.field(ti.f32, shape=(N, N)) # macierz ciśnień, do niej wsadzane są najnowsze wartosci
p_curr = ti.field(ti.f32, shape=(N, N)) # macierz ciśnień

k_field = ti.field(dtype=ti.i32, shape=(N, N)) # macierz sasiadow
bk_field = ti.field(dtype=ti.f32, shape=(N, N)) # macierz lokalnej admintacji brzegowej

alpha_field = ti.field(dtype=ti.f32, shape=(N, N)) # macierz wspólczynnikow pochlaniania materialow (potrzebne do wyznaczenia alpha_A oraz alpha_B)
alpha_A = ti.field(dtype=ti.f32, shape=(N , N)) # macierz tlumienia stratnego
alpha_B = ti.field(dtype=ti.f32, shape=(N , N)) # macierz tlumienia lepkiego

# materialy
brick_alpha, brick_density = MATERIAL_PROPS["brick"]
air_alpha, air_density = MATERIAL_PROPS["air"]

@ti.func
def calculate_alpha_A(c, alpha, density):
    return alpha * (density * c**2 + 1.0 / density)

@ti.func
def calculate_alpha_B(c, alpha):
    return (c**2) * (alpha**2)

@ti.func
def beta_from_alpha(alpha):
    Beta = 0.0
    if alpha >= 1.0:  # otwarta przestrzeń
        Beta = 1.0
    elif alpha <= 0.0:    # idealna ściana
        Beta = 0.0
    else:
        R = ti.sqrt(1.0 - alpha)
        Beta = (1.0 - R) / (1.0 + R)

    return Beta


@ti.kernel
def generate_symulation_map():
    for i,j in alpha_field: # domyslnie wszedzie 4 sąsiadów i otwarta przestrzen
        alpha_field[i, j] = 0.0
        alpha_A[i,j] = 0.0
        alpha_B[i, j] = 0.0

        count = 0
        if i > 1: count += 1
        if i < N - 2: count += 1
        if j > 1: count += 1
        if j < N - 2: count += 1
        k_field[i, j] = count

    for i,j in alpha_A: # sciany
        if i == N - PML_THICK - 1 or j == PML_THICK or j == N - PML_THICK - 1:
            alpha_field[i,j] = 0.0 #brick_alpha
            alpha_A[i,j] = 0.0 #calculate_alpha_A(C, brick_alpha, brick_density)
            alpha_B[i,j] = 0.0 # calculate_alpha_B(C, brick_alpha)
        if i == PML_THICK:
            alpha_field[i, j] = brick_alpha
            alpha_A[i, j] = calculate_alpha_A(C, brick_alpha, brick_density)
            alpha_B[i, j] = calculate_alpha_B(C, brick_alpha)


    # tworzymy sciany pml
    for i, j in alpha_A:
        dist_x = 0.0
        if i < PML_THICK:
            dist_x = ti.cast(PML_THICK - i, ti.f32) / PML_THICK
        elif i >= N - PML_THICK:
            dist_x = ti.cast(i - (N - PML_THICK) + 1, ti.f32) / PML_THICK

        dist_y = 0.0
        if j < PML_THICK:
            dist_y = ti.cast(PML_THICK - j, ti.f32) / PML_THICK
        elif j >= N - PML_THICK:
            dist_y = ti.cast(j - (N - PML_THICK) + 1, ti.f32) / PML_THICK

        dist_final = ti.max(dist_x, dist_y)
        if dist_final > 0.0:
            alpha_val = ALPHA_MAX * (dist_final ** 3)
            alpha_field[i,j] = alpha_val
            alpha_A[i, j] = calculate_alpha_A(C, alpha_val, air_density)
            alpha_B[i, j] = calculate_alpha_B(C,alpha_val)


    for i,j in bk_field:
        k_val = k_field[i, j]
        beta_val = beta_from_alpha(alpha_field[i, j])
        bk_field[i, j] = (4.0 - float(k_val)) * COURANT * beta_val
        if i == 0 or i == N - 1 or j == 0 or j == N - 1:
            p_curr[i,j] = 0.0
            p_old[i,j] = 0.0




@ti.kernel
def step(p_prev: ti.template(), p_now: ti.template(), steps: int):
    for i,j in ti.ndrange((1, N - 1), (1, N - 1)):
        k_val = k_field[i,j]
        bk_val = bk_field[i,j]
        alpha_a = alpha_A[i,j]
        alpha_b = alpha_B[i,j]

        w1 = 1.0 + bk_val + alpha_a/2 * DT + alpha_b * (DT**2)
        w2 = COURANT_SQ * (p_now[i + 1, j] + p_now[i - 1, j] + p_now[i, j + 1] + p_now[i, j - 1])
        w3 = (2.0 - k_val * COURANT_SQ) * p_now[i,j]
        w4 = (bk_val - 1.0 - (alpha_a / 2.0) * DT) * p_prev[i,j]

        p_prev[i, j] = (1.0 / w1) * (w2 + w3 + w4)

    #ZRODLO
    pulse = AMPLITUDE * ti.exp(- ((DT*steps - DT*DELAY_GAUSS)**2) / (2 * SIGMA**2))
    p_prev[SRC_X, SRC_Y] += pulse # soft source


def main():
    gui = ti.GUI("2d - fdtd (pml & neumann)", res=(N, N))
    print_config()
    generate_symulation_map()
    buffers = [p_old, p_curr]

    prev_time = time.time()

    steps = 0
    accumulator = 0

    alpha_map = alpha_field.to_numpy()
    is_wall = alpha_map > 0.001

    is_paused = False
    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == ti.GUI.SPACE:
                is_paused = not is_paused

        now = time.time()
        delta_real_time = now - prev_time
        prev_time = now

        accumulator+=delta_real_time
        if(accumulator >= FRAME_DURATION):
            if not is_paused:
                steps += 1
                b_old = steps % 2
                b_curr = (steps + 1) % 2
                step(buffers[b_old], buffers[b_curr], steps)

            p_np = buffers[b_old].to_numpy()
            max_val = np.abs(p_np).max()
            fixed_max = AMPLITUDE * 0.5
            #p_normalized = p_np / (max_val + 1e-8) # normalizujemy poniewaz chcemy uzyskac wartosci od -1 do 1
            p_normalized = p_np / fixed_max
            gray_val = p_normalized * 0.5 + 0.5 # uzyskujemy wartoci od 0 do 1

            gray_val = np.clip(gray_val, 0.0, 1.0) # zabezpieczenie choc i tak juz wczesniej znormalizowane
            img = np.dstack((gray_val, gray_val, gray_val))
            img[is_wall, 1] *= 0.3 # zielony na 30%
            img[is_wall, 2] *= 0.3 # niebieski na 30%

            gui.set_image(img)
            gui.show()

            accumulator = 0


if __name__ == "__main__":
    main()