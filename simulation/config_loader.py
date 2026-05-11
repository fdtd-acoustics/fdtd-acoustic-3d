"""Load and validate single-run YAML configs.
"""

from pathlib import Path

import yaml

import config as global_config

from fdtd.waveform_factory import _HANDLERS as _WAVEFORM_HANDLERS

_VALID_WAVEFORM_TYPES = set(_WAVEFORM_HANDLERS.keys())


class ConfigError(ValueError):
    pass


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open("r") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping, got {type(raw).__name__}")
    return validate_config(raw, base_dir=path.parent)


def validate_config(raw: dict, base_dir: Path | None = None) -> dict:
    base_dir = Path(base_dir) if base_dir is not None else Path.cwd()

    scene = _validate_scene(raw.get("scene", {}), base_dir)
    physics = _validate_physics(raw.get("physics", {}))
    sources = _validate_sources(raw.get("sources", []), base_dir, scene["record_time"])
    receivers = _validate_receivers(raw.get("receivers", []))
    run = _validate_run(raw.get("run", {}))

    if not sources:
        raise ConfigError("at least one source is required")
    if not receivers:
        raise ConfigError("at least one receiver is required")

    return {
        "scene": scene,
        "physics": physics,
        "sources": sources,
        "receivers": receivers,
        "run": run,
    }


def _validate_scene(raw: dict, base_dir: Path) -> dict:
    if "obj_file" not in raw:
        raise ConfigError("scene.obj_file is required")
    obj_path = (base_dir / raw["obj_file"]).resolve() if not Path(raw["obj_file"]).is_absolute() else Path(raw["obj_file"])
    if not obj_path.is_file():
        raise ConfigError(f"scene.obj_file not found: {obj_path}")

    pml_thickness = int(raw.get("pml_thick", raw.get("pml_thickness", 10)))
    alpha_max = float(raw.get("alpha_max", 0.5))
    record_time = float(raw.get("record_time", 0.05))

    if pml_thickness <= 0:
        raise ConfigError("scene.pml_thick must be > 0")
    if alpha_max < 0:
        raise ConfigError("scene.alpha_max must be >= 0")
    if record_time <= 0:
        raise ConfigError("scene.record_time must be > 0")

    return {
        "obj_file": str(obj_path),
        "pml_thickness": pml_thickness,
        "alpha_max": alpha_max,
        "record_time": record_time,
    }


def _validate_physics(raw: dict) -> dict:
    sound_speed = float(raw.get("sound_speed", global_config.SOUND_SPEED))
    nodes_per_wavelength = int(raw.get("nodes_per_wavelength", global_config.NODES_PER_WAVELENGTH))
    safety_factor = float(raw.get("safety_factor", global_config.SAFETY_FACTOR))

    if sound_speed <= 0:
        raise ConfigError("physics.sound_speed must be > 0")
    if nodes_per_wavelength <= 0:
        raise ConfigError("physics.nodes_per_wavelength must be > 0")
    if not (0 < safety_factor <= 1.0):
        raise ConfigError("physics.safety_factor must be in (0, 1]")

    return {
        "sound_speed": sound_speed,
        "nodes_per_wavelength": nodes_per_wavelength,
        "safety_factor": safety_factor,
    }


def _validate_sources(raw: list, base_dir: Path, record_time: float) -> list[dict]:
    if not isinstance(raw, list):
        raise ConfigError("sources must be a list")
    out = []
    for i, src in enumerate(raw):
        if not isinstance(src, dict):
            raise ConfigError(f"sources[{i}] must be a mapping")
        position = _validate_position(src.get("position"), f"sources[{i}].position")
        waveform = _validate_waveform(src.get("waveform"), base_dir, f"sources[{i}].waveform")
        emit_time = float(src.get("time", record_time))
        if emit_time <= 0:
            raise ConfigError(f"sources[{i}].time must be > 0")
        if emit_time > record_time:
            raise ConfigError(
                f"sources[{i}].time ({emit_time}s) exceeds scene.record_time ({record_time}s)"
            )
        out.append({"position": position, "waveform": waveform, "time": emit_time})
    return out


def _validate_waveform(raw, base_dir: Path, ctx: str) -> dict:
    if not isinstance(raw, dict):
        raise ConfigError(f"{ctx} must be a mapping")
    waveform_type = raw.get("type")
    if waveform_type not in _VALID_WAVEFORM_TYPES:
        raise ConfigError(
            f"{ctx}.type must be one of {sorted(_VALID_WAVEFORM_TYPES)}, got {waveform_type!r}"
        )

    spec = dict(raw)
    if waveform_type == "Gauss":
        if "freq" not in spec:
            raise ConfigError(f"{ctx}.freq required for type=Gauss")
        if float(spec["freq"]) <= 0:
            raise ConfigError(f"{ctx}.freq must be > 0")
        spec.setdefault("amp", 1.0)
        spec.setdefault("vol", 1.0)
    elif waveform_type == "Custom":
        path_key = "filepath" if "filepath" in spec else "path"
        if path_key not in spec:
            raise ConfigError(f"{ctx}.filepath required for type=Custom")
        wav_path = Path(spec[path_key])
        if not wav_path.is_absolute():
            wav_path = (base_dir / wav_path).resolve()
        if not wav_path.is_file():
            raise ConfigError(f"{ctx}.filepath not found: {wav_path}")
        spec["filepath"] = str(wav_path)
        spec.pop("path", None)
        spec.setdefault("vol", 1.0)

    return spec


def _validate_receivers(raw: list) -> list[dict]:
    if not isinstance(raw, list):
        raise ConfigError("receivers must be a list")
    seen_names = set()
    out = []
    for i, rec in enumerate(raw):
        if not isinstance(rec, dict):
            raise ConfigError(f"receivers[{i}] must be a mapping")
        name = rec.get("name")
        if not name or not isinstance(name, str):
            raise ConfigError(f"receivers[{i}].name is required and must be a non-empty string")
        if name in seen_names:
            raise ConfigError(f"receivers[{i}].name '{name}' duplicates an earlier receiver")
        seen_names.add(name)
        position = _validate_position(rec.get("position"), f"receivers[{i}].position")
        out.append({"name": name, "position": position})
    return out


def _validate_position(raw, ctx: str) -> list[float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ConfigError(f"{ctx} must be a 3-element list [x, y, z]")
    try:
        return [float(c) for c in raw]
    except (TypeError, ValueError):
        raise ConfigError(f"{ctx} must contain numeric values")


def _validate_run(raw: dict) -> dict:
    return {
        "output": str(raw.get("output", "output/runs/")),
        "save_plots": bool(raw.get("save_plots", True)),
    }
