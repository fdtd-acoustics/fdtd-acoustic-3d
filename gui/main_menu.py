import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

#Todo: oprocz tych poprawek co gadalismy to mozna zrobic dark theme
class MainMenuWindow(tk.Tk):
    def __init__(self, on_start=None) -> None:
        super().__init__()
        self.title("FDTD simulation configuration menu")
        self.geometry("780x540")
        self.on_start = on_start

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

    def obj_widgets(self, parent: tk.Frame) -> None:
        ttk.Label(
            parent,
            text="1. Room geometry (.obj)",
            font=("Arial", 10, "bold"),
        ).pack(anchor="w", pady=(0, 5))

        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X)

        self.lbl_obj_path = ttk.Label(frame, text="No file selected...", padding=(0,5))
        self.lbl_obj_path.pack(side=tk.LEFT, fill=tk.X, expand=True)

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

        ttk.Label(input_frame, text="Type:").grid(row=0, column=0, sticky="w", padx=2)
        self.combo_type = ttk.Combobox(
            input_frame, values=["Gauss", "Custom"], state="readonly", width=10
        )
        self.combo_type.current(0)
        self.combo_type.grid(row=0, column=1, padx=5, pady=2)
        self.combo_type.bind("<<ComboboxSelected>>", self.on_source_type_change)

        self.params_container = ttk.Frame(input_frame)
        self.params_container.grid(row=0, column=2, sticky="w")

        # gauss params panel
        self.frame_gauss = ttk.Frame(self.params_container)
        ttk.Label(self.frame_gauss, text="Amplitude:").pack(side=tk.LEFT, padx=(5, 2))
        self.entry_amp = ttk.Entry(self.frame_gauss, width=6)
        self.entry_amp.insert(0, "1000.0")
        self.entry_amp.pack(side=tk.LEFT)
        ttk.Label(self.frame_gauss, text="Freq (Hz):").pack(side=tk.LEFT, padx=(10, 2))
        self.entry_freq = ttk.Entry(self.frame_gauss, width=6)
        self.entry_freq.insert(0, "1000")
        self.entry_freq.pack(side=tk.LEFT)

        # wav params panel
        self.frame_custom = ttk.Frame(self.params_container)
        self.btn_wav = ttk.Button(self.frame_custom, text="Choose .wav file", command=self.load_wav)
        self.btn_wav.pack(side=tk.LEFT, padx=5)
        self.lbl_wav_path = ttk.Label(
            self.frame_custom, text="No file", width=15, foreground="gray"
        )
        self.lbl_wav_path.pack(side=tk.LEFT)

        self.frame_gauss.pack(side=tk.LEFT)

        ttk.Button(input_frame, text="Add source", command=self.add_source).grid(
            row=0, column=3, padx=10
        )
        ttk.Button(input_frame, text="Remove selected", command=self.remove_source).grid(
            row=0, column=4, padx=5
        )

        # sources
        columns = ("type", "details")
        self.tree = ttk.Treeview(parent, columns=columns, show="headings", height=4)
        self.tree.heading("type", text="Type")
        self.tree.column("type", width=80, anchor="center")
        self.tree.heading("details", text="Details (Parameters / File)")
        self.tree.column("details", width=350)
        self.tree.pack(fill=tk.X, pady=5)

    def on_source_type_change(self, event: tk.Event) -> None:
        """Switch parameter panel based on selected source type."""
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

        try:
            if s_type == "Gauss":
                amp = float(self.entry_amp.get())
                freq = float(self.entry_freq.get())
                details = f"Amp: {amp}, Freq: {freq} Hz"
                source_info: dict = {
                    "type": "Gauss",
                    "amp": amp,
                    "freq": freq,
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
                    "type": "Custom",
                    "filepath": self.current_wav_path,
                    "coords": default_coords,
                }
                self.current_wav_path = None
                self.lbl_wav_path.config(text="No file", foreground="gray")

            self.sources_data.append(source_info)
            self.tree.insert("", tk.END, values=(s_type, details))

        except ValueError:
            messagebox.showerror("Error", "Amplitude and frequency must be numbers!")

    def remove_source(self) -> None:
        """Remove selected rows from the sources list and the backing data."""
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
            text="3. Advanced FDTD settings",
            font=("Arial", 10, "bold"),
        ).pack(anchor="w", pady=(0, 5))

        adv_frame = ttk.Frame(parent)
        adv_frame.pack(fill=tk.X)

        ttk.Label(adv_frame, text="PML thickness (layers):").grid(row=0, column=0, sticky="w", pady=2)
        self.entry_pml = ttk.Entry(adv_frame, width=10)
        self.entry_pml.insert(0, "10")
        self.entry_pml.grid(row=0, column=1, padx=10, pady=2)

        ttk.Label(adv_frame, text="Alpha max (attenuation):").grid(row=1, column=0, sticky="w", pady=2)
        self.entry_alpha = ttk.Entry(adv_frame, width=10)
        self.entry_alpha.insert(0, "0.01")
        self.entry_alpha.grid(row=1, column=1, padx=10, pady=2)

    def start_simulation(self) -> None:
        try:
            pml = int(self.entry_pml.get())
            alpha = float(self.entry_alpha.get())
        except ValueError:
            messagebox.showerror("Error", "Check the validity of the entered parameters.")
            return

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
            "pml_thickness": pml,
            "alpha_max": alpha,
        }

        print("--- Configuration accepted ---")
        print(f"OBJ file:      {config['obj_file']}")
        print(f"Sources:       {config['sources']}")
        print(f"PML:           {config['pml_thickness']},  Alpha max: {config['alpha_max']}")
        if self.on_start:
            self.on_start(config)
            self.destroy()

if __name__ == "__main__":
    app = MainMenuWindow()
    app.mainloop()
