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
        grid = data['grid']

        speaker = data['speaker'].item() if 'speaker' in data else None

        mic_data = data['microphones'] if 'microphones' in data else []
        microphones = mic_data.tolist() if isinstance(mic_data, np.ndarray) else mic_data

        slot_data = data['foam_slots'] if 'foam_slots' in data else []
        foam_slots = slot_data.tolist() if isinstance(slot_data, np.ndarray) else slot_data

        plt.figure(figsize=(6, 6))
        plt.imshow(grid, cmap='Set1', origin='lower')

        if speaker:
            plt.scatter(speaker['grid_x'], speaker['grid_y'], color='yellow', marker='*', s=150, label='Speaker', zorder=5)

        if microphones:
            mic_x = [m['grid_x'] for m in microphones]
            mic_y = [m['grid_y'] for m in microphones]
            plt.scatter(mic_x, mic_y, color='blue', marker='o', s=80, label='Microphones', zorder=4)

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