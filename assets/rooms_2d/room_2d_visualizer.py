import os
import numpy as np
import matplotlib.pyplot as plt

RASTERIZED_FOLDER = 'rasterized_rooms'
IMAGES_FOLDER = 'room_images'


def visualize_and_save_rooms():
    if not os.path.exists(RASTERIZED_FOLDER):
        print(f"Folder {RASTERIZED_FOLDER} does not exist. Generate rooms first.")
        return

    os.makedirs(IMAGES_FOLDER, exist_ok=True)
    npz_files = [f for f in os.listdir(RASTERIZED_FOLDER) if f.endswith('.npz')]

    if not npz_files:
        print(f"No NPZ files in {RASTERIZED_FOLDER}.")
        return

    for filename in npz_files:
        filepath = os.path.join(RASTERIZED_FOLDER, filename)

        data = np.load(filepath, allow_pickle=True)
        grid = data['material_core']
        dx = data['dx'].item()

        src_data = data['sources'] if 'sources' in data else []
        sources = src_data.tolist() if isinstance(src_data, np.ndarray) else src_data

        recv_data = data['receivers'] if 'receivers' in data else []
        receivers = recv_data.tolist() if isinstance(recv_data, np.ndarray) else recv_data

        slot_data = data['foam_slots'] if 'foam_slots' in data else []
        foam_slots = slot_data.tolist() if isinstance(slot_data, np.ndarray) else slot_data

        plt.figure(figsize=(6, 6))
        plt.imshow(grid, cmap='Set1', origin='lower')

        # Speakers
        if sources:
            src = sources[0]
            src_x = src['position'][0] / dx
            src_y = src['position'][1] / dx
            plt.scatter(src_x, src_y, color='yellow', marker='*', s=128, label='Source', zorder=5)

        # Microphones
        if receivers:
            rec_x = [r['position'][0] / dx for r in receivers]
            rec_y = [r['position'][1] / dx for r in receivers]
            plt.scatter(rec_x, rec_y, color='blue', marker='o', s=64, label='Receivers', zorder=4)

            # Microphone numbers
            for i, (x, y) in enumerate(zip(rec_x, rec_y)):
                plt.annotate(str(i), (x, y),
                             textcoords="offset points",
                             xytext=(0,0),
                             ha='center',
                             va='center',
                             fontsize=6,
                             color='white',
                             fontweight='bold',
                             zorder=6)

        # Foam Slots
        if foam_slots:
            slot_x = [s['grid_x'] for s in foam_slots]
            slot_y = [s['grid_y'] for s in foam_slots]
            plt.scatter(slot_x, slot_y, color='green', marker='s', s=40, label='Foam Slots', zorder=3)

        plt.title(f"{filename}")
        plt.xlabel("X Axis")
        plt.ylabel("Y Axis")
        plt.colorbar(label="Material ID")

        base_name = os.path.splitext(filename)[0]
        img_path = os.path.join(IMAGES_FOLDER, f"{base_name}.png")

        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Generated room image: {img_path}")

    print(f"Visualized {len(npz_files)} rooms.")

if __name__ == "__main__":
    print("Visualizing rooms...")
    visualize_and_save_rooms()