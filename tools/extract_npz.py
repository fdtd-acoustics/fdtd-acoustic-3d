import sys
import json
from pathlib import Path
import numpy as np

def convert_to_serializable(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

def extract_and_display(filepath: str | Path, export_json: bool = False):
    path = Path(filepath)
    if not path.is_file():
        print(f"[ERROR] File not found: {path}")
        return

    print(f"\n==================================================")
    print(f" EXTRACTING NPZ FILE: {path.name}")
    print(f" Path: {path.resolve()}")
    print(f"==================================================")

    extracted_data = {}

    with np.load(path, allow_pickle=True) as data:
        for key in data.files:
            val = data[key]

            if isinstance(val, np.ndarray) and val.ndim == 0:
                try:
                    val = val.item()
                except Exception:
                    pass

            elif isinstance(val, np.ndarray) and val.dtype == object:
                try:
                    val = val.tolist()
                    val = [item.item() if isinstance(item, np.ndarray) else item for item in val]
                except Exception:
                    pass

            extracted_data[key] = val

    for key, val in extracted_data.items():
        print(f"\n• Key: '{key}'")
        if isinstance(val, np.ndarray):
            print(f"  |- Type: NumPy Array (Shape: {val.shape}, Dtype: {val.dtype})")
            if val.size < 20:
                print(f"  |- Values:\n{val}")
        elif isinstance(val, list):
            print(f"  |- Type: Python List (Items: {len(val)})")
            for idx, item in enumerate(val):
                print(f"     [{idx}]: {item}")
        else:
            print(f"  |- Value: {val} (Type: {type(val).__name__})")

    if export_json:
        json_path = path.with_suffix('.json')
        json_export_dict = {}
        for k, v in extracted_data.items():
            if isinstance(v, np.ndarray) and v.ndim > 1:
                json_export_dict[k] = f"<NumPy Array of shape {v.shape}>"
            else:
                json_export_dict[k] = convert_to_serializable(v)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_export_dict, f, indent=4, ensure_ascii=False)
        print(f"\n[Info] Exported metadata and parameters to: {json_path.name}")

    print("==================================================\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_npz.py <path_to_file.npz> [--json]")
        print(f"No path provided")
        sys.exit(1)
    else:
        file_arg = sys.argv[1]
        should_export = "--json" in sys.argv
        extract_and_display(file_arg, export_json=should_export)