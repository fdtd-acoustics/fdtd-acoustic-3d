import csv
import tkinter as tk
from tkinter import ttk, messagebox, colorchooser
import os
import config


class MaterialLibraryWindow(tk.Tk):
    def __init__(self, on_close=None):
        super().__init__()
        self.title("Material Library")
        self.geometry("900x500")
        self.materials = []

        self.on_close = on_close
        self.create_widgets()
        self.initialize_data()


    def create_widgets(self):
        main_frame = tk.Frame(self, padx=15, pady=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill=tk.X, pady=5)

        ttk.Label(input_frame, text="ID:").grid(row=0, column=0, sticky="w", padx=2)
        self.entry_id = ttk.Entry(input_frame, width=3)
        self.entry_id.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(input_frame, text="Name:").grid(row=0, column=2, sticky="w", padx=2)
        self.entry_name = ttk.Entry(input_frame, width=10)
        self.entry_name.grid(row=0, column=3, padx=5, pady=2)

        ttk.Label(input_frame, text="Alpha:").grid(row=0, column=4, sticky="w", padx=(5, 2))
        self.entry_alpha = ttk.Entry(input_frame, width=5)
        self.entry_alpha.grid(row=0, column=5, padx=5)

        ttk.Label(input_frame, text="Density:").grid(row=0, column=6, sticky="w", padx=(5, 2))
        self.entry_density = ttk.Entry(input_frame, width=5)
        self.entry_density.grid(row=0, column=7, padx=5)

        ttk.Label(input_frame, text="Color:").grid(row=0, column=8, sticky="w", padx=(5, 2))
        self.color_preview = tk.Label(input_frame, width=4, relief="sunken", bg="white")
        self.color_preview.grid(row=0, column=9, padx=5)

        ttk.Button(input_frame, text="Choose color", command=self.choose_color).grid(
            row=0, column=10, padx=10
        )

        ttk.Button(input_frame, text="Add source", command=self.add_material).grid(
            row=0, column=11, padx=10
        )
        ttk.Button(input_frame, text="Remove selected", command=self.remove_material).grid(
            row=0, column=12, padx=5
        )

        columns = ("id","name", "alpha", "density")
        self.tree = ttk.Treeview(main_frame, columns=columns, show="headings", height=50)

        self.tree.heading("id", text="ID")
        self.tree.column("id", width=40, anchor="center")

        self.tree.heading("name", text="Name")
        self.tree.column("name", width=100, anchor="center")

        self.tree.heading("alpha", text="Alpha")
        self.tree.column("alpha", width=50, anchor="center")

        self.tree.heading("density", text="Density")
        self.tree.column("density", width=80, anchor="center")

        nav_frame = ttk.Frame(main_frame)
        nav_frame.pack(fill=tk.X, side=tk.BOTTOM, pady = (10,0))

        ttk.Button(nav_frame,text = "Back", command=self.on_close).pack(side = tk.LEFT)

        self.tree.pack(fill=tk.X,expand=True, pady=5)


    def add_material(self):
        id = self.entry_id.get().strip()
        name = self.entry_name.get().strip()
        alpha = self.entry_alpha.get().strip()
        density = self.entry_density.get().strip()

        hex_color = self.color_preview.cget("bg")

        if not name or not alpha or not density:
            print("All fields must be completed!")
            return

        f_alpha = float(alpha)
        f_density = float(density)

        new_material = [id, name, f_alpha, f_density, hex_color]
        self.materials.append(new_material)

        img = self.create_color_icon(hex_color)
        if not hasattr(self, 'color_images'):
            self.color_images = []
        self.color_images.append(img)

        self.tree.insert("", tk.END, values=(id, name, alpha, density), tags=(hex_color,))

        self.tree.tag_configure(hex_color, background=hex_color)

        self.entry_id.delete(0, tk.END)
        self.entry_name.delete(0, tk.END)
        self.entry_alpha.delete(0, tk.END)
        self.entry_density.delete(0, tk.END)
        self.color_preview.config(bg="white")
        self.save_materials()

    def remove_material(self) -> None:
        selected = self.tree.selection()
        if not selected:
            messagebox.showinfo("Info", "Please select a source to remove.")
            return

        for item in selected:
            index = self.tree.index(item)
            self.tree.delete(item)
            del self.materials[index]

        self.save_materials()


    def initialize_data(self):
        config.MAIN_MATERIAL_LIBRARY.parent.mkdir(parents=True, exist_ok=True)
        if not config.MAIN_MATERIAL_LIBRARY.exists():
            path = os.path.join(config.MATERIAL_LIBRARY_DIR, "materials.csv")
            with open(config.MAIN_MATERIAL_LIBRARY, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'name', 'alpha', 'density', 'color'])

        self.load_materials(config.MAIN_MATERIAL_LIBRARY)

    def load_materials(self, path):
        if not os.path.exists(path):
            print(f"Note: File {path} does not exist.")
            return

        self.materials.clear()
        if hasattr(self, 'color_images'):
            self.color_images.clear()
        else:
            self.color_images = []

        for item in self.tree.get_children():
            self.tree.delete(item)

        with open(path, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')

            try:
                next(spamreader)
            except StopIteration:
                return

            for row in spamreader:
                if not row: continue

                converted_row = [
                    int(row[0]),   # id
                    str(row[1]),   # name
                    float(row[2]),  # alpha
                    float(row[3]),   # density
                    str(row[4])   # color HEX
                ]

                self.materials.append(converted_row)

                hex_color = converted_row[4]
                img = self.create_color_icon(hex_color)
                self.color_images.append(img)
                self.tree.insert("", tk.END, values=(converted_row[0],converted_row[1], converted_row[2], converted_row[3]), tags=(hex_color,))

                self.tree.tag_configure(hex_color, background=hex_color)


    def save_materials(self):
        config.MAIN_MATERIAL_LIBRARY.parent.mkdir(parents=True, exist_ok=True)

        with open(config.MAIN_MATERIAL_LIBRARY, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

            writer.writerow(['id', 'name', 'alpha', 'density', 'color'])

            for row in self.materials:
                writer.writerow(row)


    def choose_color(self):
        color_code = colorchooser.askcolor(title = "Select material color")

        if color_code[1]:
            self.color_preview.config(bg=color_code[1])


    def create_color_icon(self, hex_color, size=(16, 16)):
        img = tk.PhotoImage(width=size[0], height=size[1], master=self)
        img.put(hex_color, to=(0, 0, size[0], size[1]))
        return img

