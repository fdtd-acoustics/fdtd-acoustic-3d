import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

#Todo: oprocz tych poprawek co gadalismy to mozna zrobic dark theme
class NewSimulationWindow(tk.Toplevel):
    def __init__(self, on_start=None, loaded_data: dict | None = None ) -> None:
        super().__init__()
        self.title("FDTD simulation configuration menu")
        self.geometry("950x600")
        self.on_start = on_start

        self.loaded_data = loaded_data
        self.is_loaded = loaded_data is not None

        self.obj_filepath: str | None = None
        self.sources_data: list[dict] = []
        self.current_wav_path: str | None = None

        self.create_widgets()

    def create_widgets(self) -> None:
        main_frame = tk.Frame(self, padx=15, pady=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.obj_widgets(main_frame)
        ttk.Separator(main_frame, orient="horizontal").pack(fill=tk.X, pady=15)

        self.sources_widgets(main_frame)
        ttk.Separator(main_frame, orient="horizontal").pack(fill=tk.X, pady=15)

        self.receivers_widgets(main_frame)
        ttk.Separator(main_frame, orient="horizontal").pack(fill=tk.X, pady=15)

        self.advanced_widgets(main_frame)
        ttk.Separator(main_frame, orient="horizontal").pack(fill=tk.X, pady=20)

        self.start_btn = tk.Button(
            main_frame,
            text="Prepare and run simulation",
            command=self.start_simulation,
            font=("Arial", 12, "bold"),
            relief="flat",
            bd=0,
        )
        self.start_btn.pack(fill=tk.X, ipady=12, pady=10)

        if self.is_loaded:
            self.fill_loaded_data()

    def obj_widgets(self, parent: tk.Frame) -> None:
        header_text = "1. Room geometry (Locked from .npz)" if self.is_loaded else "1. Room geometry (.obj)"
        header_color = "#28a745" if self.is_loaded else "black"

        ttk.Label(
            parent,
            text="1. Room geometry (.obj)",
            font=("Arial", 10, "bold"),
            foreground=header_color
        ).pack(anchor="w", pady=(0, 5))

        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X)

        self.lbl_obj_path = ttk.Label(frame, text="No file selected...", padding=(0,5))
        self.lbl_obj_path.pack(side=tk.LEFT, fill=tk.X, expand=True)

        if self.is_loaded:
            lock_label = ttk.Label(
                frame,
                text="🔒",
                foreground="gray",
                font=("Arial", 9, "bold")
            )
            lock_label.pack(side=tk.RIGHT, padx=5)

            self.lbl_obj_path.configure(font=("Arial", 9, "italic"))
        else:
            ttk.Button(frame, text="Choose .obj", command=self.load_obj).pack(side=tk.RIGHT)

    def load_obj(self) -> None:
        filepath = filedialog.askopenfilename(
            filetypes=[("OBJ Files", "*.obj"), ("All files", "*.*")]
        )
        if filepath:
            self.obj_filepath = filepath
            self.lbl_obj_path.config(text=os.path.basename(filepath), foreground="black")

    def sources_widgets(self, parent: tk.Frame) -> None:
        ttk.Label(
            parent,
            text="2. Sound source configuration",
            font=("Arial", 10, "bold"),
        ).pack(anchor="w", pady=(0, 5))

        input_frame = ttk.Frame(parent)
        input_frame.pack(fill=tk.X, pady=5)

        # grid columns 0-1: Name
        ttk.Label(input_frame, text="Name:").grid(row=0, column=0, sticky="w", padx=2)
        self.entry_name = ttk.Entry(input_frame, width=8)
        initial_name = f"src_{len(getattr(self, 'sources_data', [])) + 1}"
        self.entry_name.insert(0, initial_name)
        self.entry_name.grid(row=0, column=1, padx=5, pady=2)

        # grid columns 2-3: Type
        ttk.Label(input_frame, text="Type:").grid(row=0, column=2, sticky="w", padx=2)
        self.combo_type = ttk.Combobox(
            input_frame, values=["Gauss", "Custom"], state="readonly", width=10
        )
        self.combo_type.current(0)
        self.combo_type.grid(row=0, column=3, padx=5, pady=2)
        self.combo_type.bind("<<ComboboxSelected>>", self.on_source_type_change)

        # grid column 4: Params
        self.params_container = ttk.Frame(input_frame)
        self.params_container.grid(row=0, column=4, sticky="w")

        # gauss panel
        self.frame_gauss = ttk.Frame(self.params_container)
        ttk.Label(self.frame_gauss, text="Amp:").pack(side=tk.LEFT, padx=(5, 2))
        self.entry_amp = ttk.Entry(self.frame_gauss, width=6)
        self.entry_amp.insert(0, "1000.0")
        self.entry_amp.pack(side=tk.LEFT)
        ttk.Label(self.frame_gauss, text="Freq:").pack(side=tk.LEFT, padx=(10, 2))
        self.entry_freq = ttk.Entry(self.frame_gauss, width=6)
        self.entry_freq.insert(0, "1000")
        self.entry_freq.pack(side=tk.LEFT)

        # custom panel
        self.frame_custom = ttk.Frame(self.params_container)
        self.btn_wav = ttk.Button(self.frame_custom, text="Choose .wav", command=self.load_wav)
        self.btn_wav.pack(side=tk.LEFT, padx=5)
        self.lbl_wav_path = ttk.Label(self.frame_custom, text="No file", width=15, foreground="gray")
        self.lbl_wav_path.pack(side=tk.LEFT)

        self.frame_gauss.pack(side=tk.LEFT)

        # grid columns 5-6: Time
        ttk.Label(input_frame, text="Time (s):").grid(row=0, column=5, sticky="w", padx=(10, 2))
        self.entry_time = ttk.Entry(input_frame, width=6)
        self.entry_time.insert(0, "5.0")
        self.entry_time.grid(row=0, column=6, padx=5)

        ttk.Label(input_frame, text="Vol:").grid(row=0, column=7, sticky="w", padx=(5, 2))
        self.entry_vol = ttk.Entry(input_frame, width=5)
        self.entry_vol.insert(0, "1.0")  # Domyślnie 1.0 (100%)
        self.entry_vol.grid(row=0, column=8, padx=5)

        # grid columns 7-8: Buttons
        ttk.Button(input_frame, text="Add source", command=self.add_source).grid(
            row=0, column=9, padx=10
        )
        ttk.Button(input_frame, text="Remove selected", command=self.remove_source).grid(
            row=0, column=10, padx=5
        )

        # --- Treeview Order: Type, Name, Time, Details ---
        columns = ("type", "name", "time","vol", "details")
        self.tree = ttk.Treeview(parent, columns=columns, show="headings", height=4)

        self.tree.heading("type", text="Type")
        self.tree.column("type", width=80, anchor="center")

        self.tree.heading("name", text="Name")
        self.tree.column("name", width=80, anchor="center")

        self.tree.heading("vol", text="Vol")
        self.tree.column("vol", width=50, anchor="center")

        self.tree.heading("time", text="Time (s)")
        self.tree.column("time", width=80, anchor="center")

        self.tree.heading("details", text="Details (Parameters / File)")
        self.tree.column("details", width=350)

        self.tree.pack(fill=tk.X, pady=5)

        ttk.Label(
            parent,
            text="Note: Source positions will be set in the 3D view.",
            font=("Arial", 8, "italic"),
            foreground="gray"
        ).pack(anchor="w", pady=(2, 0))

    def receivers_widgets(self, parent) -> None:
        ttk.Label(
            parent,
            text="3. Microphones",
            font=("Arial", 10, "bold"),
        ).pack(anchor="w", pady=(0, 5))

        rec_frame = ttk.Frame(parent)
        rec_frame.pack(fill=tk.X)

        ttk.Label(rec_frame, text="Recording Duration (s):").grid(row=0, column=0, sticky="w", pady=2)

        self.rec_time_var = tk.DoubleVar(value=2.0)
        self.entry_rec_time = ttk.Entry(rec_frame, textvariable=self.rec_time_var, width=10)
        self.entry_rec_time.grid(row=0, column=1, padx=10, pady=2)

        ttk.Label(
            parent,
            text="Note: Microphone positions will be set in the 3D view.",
            font=("Arial", 8, "italic"),
            foreground="gray"
        ).pack(anchor="w", pady=(2, 0))


    def on_source_type_change(self, event: tk.Event) -> None:
        s_type = self.combo_type.get()
        if s_type == "Gauss":
            self.frame_custom.pack_forget()
            self.frame_gauss.pack(side=tk.LEFT)
        elif s_type == "Custom":
            self.frame_gauss.pack_forget()
            self.frame_custom.pack(side=tk.LEFT)

    def load_wav(self) -> None:
        filepath = filedialog.askopenfilename(filetypes=[("WAV Audio", "*.wav")])
        if filepath:
            if not os.path.exists(filepath):
                messagebox.showerror("Error", "Selected .wav file does not exist!")
                return
            self.current_wav_path = filepath
            self.lbl_wav_path.config(text=os.path.basename(filepath), foreground="black")

    def add_source(self) -> None:
        s_type = self.combo_type.get()
        default_coords = (60.0, 60.0, 60.0)

        user_name = self.entry_name.get().strip() if hasattr(self, 'entry_name') else ""
        source_name = user_name if user_name else f"src_{len(self.sources_data) + 1}"

        try:
            sim_time = float(self.entry_time.get())
            sim_vol = float(self.entry_vol.get())

            if s_type == "Gauss":
                amp = float(self.entry_amp.get())
                freq = float(self.entry_freq.get())
                details = f"Amp: {amp}, Freq: {freq} Hz"
                source_info: dict = {
                    "name": source_name,
                    "type": "Gauss",
                    "amp": amp,
                    "freq": freq,
                    "time": sim_time,
                    "vol": sim_vol,
                    "coords": default_coords,
                }

            elif s_type == "Custom":
                if not self.current_wav_path:
                    messagebox.showwarning("No file", "Please select a .wav file for the Custom source!")
                    return
                if not os.path.exists(self.current_wav_path):
                    messagebox.showerror(
                        "Error", "Selected .wav file does not exist or has been deleted!"
                    )
                    return

                filename = os.path.basename(self.current_wav_path)
                details = f"File: {filename}"

                source_info = {
                    "name": source_name,
                    "type": "Custom",
                    "filepath": self.current_wav_path,
                    "time": sim_time,
                    "vol": sim_vol,
                    "coords": default_coords,
                }
                self.current_wav_path = None
                self.lbl_wav_path.config(text="No file", foreground="gray")

            self.sources_data.append(source_info)

            self.tree.insert("", tk.END, values=(s_type, source_name, sim_time, sim_vol, details))

            self.entry_name.delete(0, tk.END)
            self.entry_name.insert(0, f"src_{len(self.sources_data) + 1}")

        except ValueError:
            messagebox.showerror("Error", "Amplitude, Frequency and Time must be valid numbers!")

    def remove_source(self) -> None:
        selected = self.tree.selection()
        if not selected:
            messagebox.showinfo("Info", "Please select a source to remove.")
            return

        for item in selected:
            index = self.tree.index(item)
            self.tree.delete(item)
            del self.sources_data[index]

    def advanced_widgets(self, parent: tk.Frame) -> None:
        ttk.Label(
            parent,
            text="4. Advanced FDTD settings",
            font=("Arial", 10, "bold"),
        ).pack(anchor="w", pady=(0, 5))

        adv_frame = ttk.Frame(parent)
        adv_frame.pack(fill=tk.X)

        ttk.Label(adv_frame, text="PML thickness (layers):").grid(row=0, column=0, sticky="w", pady=2)
        self.entry_pml = ttk.Entry(adv_frame, width=10)
        self.entry_pml.insert(0, "20")
        self.entry_pml.grid(row=0, column=1, padx=10, pady=2)

        ttk.Label(adv_frame, text="Alpha max (attenuation):").grid(row=1, column=0, sticky="w", pady=2)
        self.entry_alpha = ttk.Entry(adv_frame, width=10)
        self.entry_alpha.insert(0, "0.15")
        self.entry_alpha.grid(row=1, column=1, padx=10, pady=2)

    def start_simulation(self) -> None:
        try:
            pml_thick = int(self.entry_pml.get())
            alpha = float(self.entry_alpha.get())
            record_time = float(self.rec_time_var.get())

        except ValueError:
            messagebox.showerror("Error", "Check the validity of the entered parameters.")
            return

        if not self.is_loaded:
            if not self.obj_filepath:
                messagebox.showwarning("Warning", "No .obj geometry file selected!")
                return

            if not os.path.exists(self.obj_filepath):
                messagebox.showerror("Error", "The .obj file does not exist or has been deleted!")
                return

        if not self.sources_data:
            messagebox.showwarning("Warning", "No sound sources have been added!")
            return

        for i, src in enumerate(self.sources_data):
            if src["type"] == "Custom" and not os.path.exists(src["filepath"]):
                messagebox.showerror(
                    "Error",
                    f"The .wav file for source #{i + 1} does not exist or has been deleted:\n{src['filepath']}",
                )
                return

        config = {
            "obj_file": self.obj_filepath,
            "sources": self.sources_data,
            "pml_thickness": pml_thick,
            "alpha_max": alpha,
            "record_time": record_time,
        }

        print("--- Configuration accepted ---")
        print(f"OBJ file:      {config['obj_file']}")
        print(f"Sources:       {config['sources']}")
        print(f"PML:           {config['pml_thickness']},  Alpha max: {config['alpha_max']}")
        if self.on_start:
            self.on_start(config, self.loaded_data)
            self.destroy()

    def fill_loaded_data(self) -> None:
        if not self.is_loaded:
            return

        d = self.loaded_data
        from pathlib import Path

        if 'obj_filepath' in d:
            full_path = str(d['obj_filepath'])
            self.obj_filepath = full_path
            display_name = Path(full_path).name
        else:
            self.obj_filepath = "Embedded in .npz (legacy file)"
            display_name = self.obj_filepath

        if hasattr(self, 'lbl_obj_path'):
            self.lbl_obj_path.config(text=f"Room geometry: {display_name}")

        if 'pml_thick' in d and hasattr(self, 'entry_pml'):
            self.entry_pml.delete(0, tk.END)
            self.entry_pml.insert(0, str(int(d['pml_thick'])))

        if 'alpha_max' in d and hasattr(self, 'entry_alpha'):
            self.entry_alpha.delete(0, tk.END)
            self.entry_alpha.insert(0, str(float(d['alpha_max'])))

        if 'record_time' in d:
            if hasattr(self, 'rec_time_var'):
                self.rec_time_var.set(float(d['record_time']))
            elif hasattr(self, 'entry_rec_time'):
                self.entry_rec_time.delete(0, tk.END)
                self.entry_rec_time.insert(0, str(float(d['record_time'])))

        if 'sources' in d:
            sources = d['sources']
            self.sources_data = sources.tolist() if hasattr(sources, 'tolist') else sources

            if hasattr(self, 'tree'):
                for item in self.tree.get_children():
                    self.tree.delete(item)

                for src in self.sources_data:
                    if src['type'] == "Gauss":
                        details = f"Amp: {src['amp']}, Freq: {src['freq']} Hz"
                    else:
                        import os
                        fname = os.path.basename(src.get('wav_path', 'unknown.wav'))
                        details = f"File: {fname}"

                    self.tree.insert(
                        "",
                        tk.END,
                        values=(
                            src['type'],
                            src['name'],
                            src['time'],
                            src.get('vol', "1.0"),
                            details
                        )
                    )

        print(f"GUI populated with data from: {display_name}")