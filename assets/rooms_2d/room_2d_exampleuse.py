from room_2d_generator import generate_room
from room_2d_rasterizer import rasterize_room_data
from room_2d_visualizer import visualize_room_data

ROOM_COUNT = 1
MICROPHONE_COUNT = 3
FOAM_SIZE = 0.25    # [m]
MAX_FREQ = 1000     # [Hz]
VISUALISE = True

for room_seed in range(ROOM_COUNT):
    generated_room_data = generate_room(
        material_id=1,
        foam_size=FOAM_SIZE,
        num_mics=MICROPHONE_COUNT,
        seed=room_seed,
        shape_type=None
    )

    rasterized_room_data = rasterize_room_data(
        generated_room_data,
        max_freq=MAX_FREQ
    )

    print(f"[Seed {room_seed}] Shape: {generated_room_data['room_shape']}")
    print(f"[Seed {room_seed}] Generated room grid shape: {rasterized_room_data['material_core'].shape}")
    print(f"\tFoam slots count: {len(rasterized_room_data['foam_slots'])}")

    if VISUALISE:
        visualize_room_data(rasterized_room_data,
                            title=f"Room {generated_room_data['room_shape']} (seed={room_seed})",
                            save_path=None,
                            show=False,
                            use_meters=False)