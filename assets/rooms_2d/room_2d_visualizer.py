import os
import numpy as np
import matplotlib.pyplot as plt

RASTERIZED_FOLDER = 'rasterized_rooms'
IMAGES_FOLDER = 'visualized_rooms'

# Example use:
# visualize_room_data(rasterized_room_data, title=f"Room (seed={room_seed})", save_path=None ,show=True)

def visualize_room_data(data, title="Room Preview", save_path=None, show=False):

    grid = data['material_core']
    dx = data['dx'].item() if isinstance(data['dx'], np.ndarray) else data['dx']
    nx, ny = grid.shape

    src_data = data['sources'] if 'sources' in data else []
    sources = src_data.tolist() if isinstance(src_data, np.ndarray) else src_data

    recv_data = data['receivers'] if 'receivers' in data else []
    receivers = recv_data.tolist() if isinstance(recv_data, np.ndarray) else recv_data

    slot_data = data['foam_slots'] if 'foam_slots' in data else []
    foam_slots = slot_data.tolist() if isinstance(slot_data, np.ndarray) else slot_data

    plt.figure(figsize=(6, 6))
    plt.imshow(grid.T, cmap='Set1', origin='lower')

    grid_center_x = (nx - 1.0) / 2.0
    grid_center_y = (ny - 1.0) / 2.0

    # Speakers
    if sources:
        src = sources[0]
        coords_src_x = src['coords'][0]
        coords_src_y = src['coords'][1]

        src_x = coords_src_x / dx + grid_center_x
        src_y = coords_src_y / dx + grid_center_y

        plt.scatter(src_x, src_y, color='yellow', marker='*', s=128, label='Source', zorder=5)

    # Microphones
    if receivers:
        rec_x = []
        rec_y = []
        for r in receivers:
            coords_rec_x = r['coords'][0]
            coords_rec_y = r['coords'][1]

            rx = coords_rec_x / dx + grid_center_x
            ry = coords_rec_y / dx + grid_center_y

            rec_x.append(rx)
            rec_y.append(ry)

        plt.scatter(rec_x, rec_y, color='blue', marker='o', s=64, label='Receivers', zorder=4)

        # Microphone numbers
        for i, (x, y) in enumerate(zip(rec_x, rec_y)):
            plt.annotate(str(i), (x, y),
                         textcoords="offset points",
                         xytext=(0, 0),
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

    plt.title(title)
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plt.colorbar(label="Material ID")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Generated room image: {save_path}")

    if show:
        plt.show()

    plt.close()


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

        base_name = os.path.splitext(filename)[0]
        img_path = os.path.join(IMAGES_FOLDER, f"{base_name}.png")

        visualize_room_data(data, title=filename, save_path=img_path, show=False)

    print(f"Visualized {len(npz_files)} rooms.")


if __name__ == "__main__":
    print("Visualizing rooms...")
    visualize_and_save_rooms()