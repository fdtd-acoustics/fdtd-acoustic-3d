import taichi as ti
import numpy as np

@ti.data_oriented
class ColorScaleHUD:
    def __init__(self, x_start=0.2, x_end=0.8, y_start=0.05, y_end=0.1, segments=100):
        self.segments = segments
        self.num_vertices = (segments + 1) * 2
        self.num_indices = segments * 6

        self.vertices = ti.Vector.field(2, dtype=ti.f32, shape=self.num_vertices)
        self.colors = ti.Vector.field(3, dtype=ti.f32, shape=self.num_vertices)
        self.indices = ti.field(dtype=ti.i32, shape=self.num_indices)

        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end

        self.init_mesh()
        self.update_colors()

    @ti.kernel
    def init_mesh(self):
        for i in range(self.segments + 1):
            x = self.x_start + (self.x_end - self.x_start) * (i / self.segments)
            self.vertices[i * 2] = ti.Vector([x, self.y_start])
            self.vertices[i * 2 + 1] = ti.Vector([x, self.y_end])

        for i in range(self.segments):
            idx = i * 6
            self.indices[idx + 0] = i * 2
            self.indices[idx + 1] = i * 2 + 1
            self.indices[idx + 2] = (i + 1) * 2
            self.indices[idx + 3] = (i + 1) * 2
            self.indices[idx + 4] = i * 2 + 1
            self.indices[idx + 5] = (i + 1) * 2 + 1

    @ti.kernel
    def update_colors(self):
        for i in range(self.segments + 1):
            val = -1.0 + 2.0 * (i / self.segments)

            color = ti.Vector([1.0, 1.0, 1.0])

            if val > 0:
                color = ti.Vector([1.0, 1.0 - val, 1.0 - val])
            else:
                color = ti.Vector([1.0 + val, 1.0 + val, 1.0])

            self.colors[i * 2] = color
            self.colors[i * 2 + 1] = color

    def draw(self, canvas):
        canvas.triangles(self.vertices, color=(1, 1, 1), indices=self.indices, per_vertex_color=self.colors)

@ti.data_oriented
class VectorTextHUD:
    def __init__(self, max_chars=30):
        # Mapowanie, które linie zapalić dla jakiego znaku (jak w starych zegarkach elektronicznych)
        self.segment_map = {
            '0': [0, 1, 2, 3, 4, 5], '1': [1, 2], '2': [0, 1, 6, 4, 3],
            '3': [0, 1, 6, 2, 3], '4': [5, 6, 1, 2], '5': [0, 5, 6, 2, 3],
            '6': [0, 5, 4, 3, 2, 6], '7': [0, 1, 2], '8': [0, 1, 2, 3, 4, 5, 6],
            '9': [0, 1, 2, 3, 5, 6], '-': [6], ' ': []
        }
        self.max_lines = max_chars * 7
        self.vertices = ti.Vector.field(2, dtype=ti.f32, shape=self.max_lines * 2)

    def set_text(self, texts_with_positions, size=0.03):
        verts = []
        w = size * 0.5  # proporcja szerokości do wysokości znaku
        h = size

        # Współrzędne 7 segmentów: [start_x, start_y, koniec_x, koniec_y]
        lines = [
            (0, h, w, h),       # 0: góra poziom
            (w, h, w, h/2),     # 1: prawy-góra pion
            (w, h/2, w, 0),     # 2: prawy-dół pion
            (0, 0, w, 0),       # 3: dół poziom
            (0, 0, 0, h/2),     # 4: lewy-dół pion
            (0, h/2, 0, h),     # 5: lewy-góra pion
            (0, h/2, w, h/2)    # 6: środek poziom
        ]

        for text, start_x, start_y in texts_with_positions:
            x_offset = start_x
            for char in str(text):
                if char in self.segment_map:
                    for seg in self.segment_map[char]:
                        l = lines[seg]
                        verts.append([x_offset + l[0], start_y + l[1]])
                        verts.append([x_offset + l[2], start_y + l[3]])
                x_offset += w + size * 0.3 # Odstęp do kolejnego znaku

        # Wypychamy niewykorzystane wektory poza ekran, żeby ich nie rysować
        while len(verts) < self.max_lines * 2:
            verts.append([-1.0, -1.0])

        self.vertices.from_numpy(np.array(verts, dtype=np.float32))

    def draw(self, canvas, color=(0.0, 0.0, 0.0)):
        canvas.lines(self.vertices, width=0.005, color=color)