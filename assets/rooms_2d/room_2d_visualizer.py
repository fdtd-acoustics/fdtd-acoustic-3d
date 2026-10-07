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

        data = np.load(filepath)
        grid = data['grid']

        plt.figure(figsize=(6, 6))
        plt.imshow(grid, cmap='Set1', origin='lower')

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