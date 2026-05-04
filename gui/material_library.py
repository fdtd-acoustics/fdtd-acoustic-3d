from dataclasses import dataclass
import csv
import tkinter as tk
from tkinter import ttk, messagebox
import os
import config

@dataclass
class Record:
    id: int
    name: str
    alpha: float
    density: float


class MaterialLibraryWindow(tk.Tk):
    def __init__(self, on_close=None):
        super().__init__()
        self.title("Material Library")
        self.geometry("700x300")
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

        ttk.Button(input_frame, text="Add source", command=self.add_material).grid(
            row=0, column=8, padx=10
        )
        ttk.Button(input_frame, text="Remove selected", command=self.remove_material).grid(
            row=0, column=9, padx=5
        )

        columns = ("id","name", "alpha", "density")
        self.tree = ttk.Treeview(main_frame, columns=columns, show="headings", height=4)

        self.tree.heading("id", text="ID")
        self.tree.column("id", width=40, anchor="center")

        self.tree.heading("name", text="Name")
        self.tree.column("name", width=100, anchor="center")

        self.tree.heading("alpha", text="Alpha")
        self.tree.column("alpha", width=50, anchor="center")

        self.tree.heading("density", text="Density")
        self.tree.column("density", width=80, anchor="center")

        self.tree.pack(fill=tk.X, pady=5)

        nav_frame = ttk.Frame(main_frame)
        nav_frame.pack(fill=tk.X, side=tk.BOTTOM, pady = (10,0))

        ttk.Button(nav_frame,text = "Back", command=self.on_close).pack(side = tk.LEFT)


    def add_material(self):
        id = self.entry_id.get().strip()
        name = self.entry_name.get().strip()
        alpha = self.entry_alpha.get().strip()
        density = self.entry_density.get().strip()

        if not name or not alpha or not density:
            print("All fields must be completed!")
            return

        f_alpha = float(alpha)
        f_density = float(density)

        new_material = [id, name, f_alpha, f_density]

        self.materials.append(new_material)

        self.tree.insert("", tk.END, values=(id, name, alpha, density))

        self.entry_id.delete(0, tk.END)
        self.entry_name.delete(0, tk.END)
        self.entry_alpha.delete(0, tk.END)
        self.entry_density.delete(0, tk.END)
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
        os.makedirs(config.MATERIAL_LIBRARY_DIR, exist_ok=True)
        path = os.path.join(config.MATERIAL_LIBRARY_DIR, "materials.csv")
        if not os.path.exists(path):
            with open(path, 'w', newline='') as f:
                pass
        self.load_materials(path)

    def load_materials(self, path):
        if not os.path.exists(path):
            print(f"Note: File {path} does not exist.")
            return

        self.materials.clear()
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
                    int(row[0]),
                    str(row[1]),
                    float(row[2]),
                    float(row[3])
                ]

                self.materials.append(converted_row)
                self.tree.insert("", tk.END, values=(converted_row[0],converted_row[1], converted_row[2], converted_row[3]))


    def save_materials(self):
        if not os.path.exists(config.MATERIAL_LIBRARY_DIR):
            os.makedirs(config.MATERIAL_LIBRARY_DIR)

        path = os.path.join(config.MATERIAL_LIBRARY_DIR, "materials.csv")
        with open(path, 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

            spamwriter.writerow(['id', 'name', 'alpha', 'density'])

            for row in self.materials:
                spamwriter.writerow(row)

