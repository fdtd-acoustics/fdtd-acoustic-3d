import taichi as ti
from visualization import config, PML_THICK


@ti.data_oriented
class ReceiverManager:
    def __init__(self,max_receivers: int, max_steps: int):
        self.max_receivers = max_receivers
        self.max_steps = max_steps

        self.count = ti.field(ti.i32, shape=())
        self.count[None] = 0

        self.names = []
        self.pos = ti.Vector.field(3, dtype=ti.i32, shape=max_receivers)
        self.history = ti.field(dtype=ti.f32, shape=(max_receivers, max_steps))

    def add_receiver(self, x, y, z, name):
        idx = self.count[None]
        if idx < self.max_receivers:
            self.pos[idx] = [x + PML_THICK, y + PML_THICK, z+ PML_THICK]
            self.names.append(name)
            self.count[None] += 1

    def update_receivers(self, p_field: ti.template(), steps: ti.i32):
        if steps < self.max_steps:
            for i in range(self.count[None]):
                p_idx = self.pos[i]
                self.history[i, steps] = p_field[p_idx[0], p_idx[1], p_idx[2]]