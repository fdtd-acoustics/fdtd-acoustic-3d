#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON       = str(PROJECT_ROOT / ".venv" / "bin" / "python")
if not Path(PYTHON).is_file():
    PYTHON = sys.executable

MODEL_NAME = "Room_GoldenRatio"
OBJ_FILE   = str(PROJECT_ROOT / "assets" / "models" / f"{MODEL_NAME}.obj")

PML_EXPONENTS    = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
ALPHA_MAX_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
PML_THICKNESSES  = [10, 20, 30, 40, 50]
GAUSS_FREQS      = [500, 1000, 1500, 2000, 2500, 3000]

RECORD_TIME           = 0.8
SOUND_SPEED           = 343.0
NODES_PER_WAVELENGTH  = 10
SAFETY_FACTOR         = 0.95

SOURCE_POSITION = [-0.0772, -0.4151, -3.0410]
SOURCE_AMP      = 100.0
SOURCE_VOL      = 1.0
SOURCE_TIME     = 0.1

RECEIVERS = [
    {"name": "mic_1", "position": [-0.0772, -0.4151, -3.0410]},
    {"name": "mic_2", "position": [-0.5213, -0.4730,  2.9058]},
]

BATCH_DIR    = PROJECT_ROOT / "output" / "batch" / f"{MODEL_NAME}_sweep"
RESULTS_CSV  = BATCH_DIR / "results.csv"
RUN_TIMEOUT  = 600

SKIP_PLOTS = False



def combo_label(pml_exp: float, alpha: float, pml_thick: int, freq: int) -> str:
    return (
        f"exp{pml_exp:.1f}"
        f"_alpha{alpha:.2f}"
        f"_pml{pml_thick:02d}"
        f"_f{freq:04d}"
    )


def generate_yaml(out_path: Path, pml_exp: float, pml_thick: int,
                  alpha_max: float, freq: int) -> None:
    cfg = {
        "scene": {
            "obj_file":      OBJ_FILE,
            "pml_thickness": pml_thick,
            "alpha_max":     alpha_max,
            "pml_exponent":  pml_exp,
            "record_time":   RECORD_TIME,
        },
        "physics": {
            "sound_speed":          SOUND_SPEED,
            "nodes_per_wavelength": NODES_PER_WAVELENGTH,
            "safety_factor":        SAFETY_FACTOR,
        },
        "sources": [{
            "position": SOURCE_POSITION,
            "time":     SOURCE_TIME,
            "waveform": {
                "type": "Gauss",
                "freq": float(freq),
                "amp":  SOURCE_AMP,
                "vol":  SOURCE_VOL,
            },
        }],
        "receivers": RECEIVERS,
        "run":      {"save_plots": not SKIP_PLOTS},
    }
    with out_path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def fieldnames_for_csv() -> list[str]:
    cols = [
        "combo_label",
        "pml_exponent", "alpha_max", "pml_thickness", "gauss_freq",
    ]
    for rec in RECEIVERS:
        n = rec["name"]
        cols += [f"{n}_error_db", f"{n}_error_ratio"]
    cols += ["status"]
    return cols


def load_done_labels(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    done = set()
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            lbl = row.get("combo_label", "")
            status = row.get("status", "")
            if lbl and status == "OK":
                done.add(lbl)
    return done


def parse_pml_results(run_dir: Path) -> dict:
    p = run_dir / "pml_results.json"
    if not p.is_file():
        return {}
    with p.open() as f:
        return json.load(f)


def main() -> int:
    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    combos = list(product(
        PML_EXPONENTS, ALPHA_MAX_VALUES, PML_THICKNESSES, GAUSS_FREQS
    ))
    total = len(combos)

    done = load_done_labels(RESULTS_CSV)
    remaining = total - len(done)

    print(f"Model:          {MODEL_NAME}")
    print(f"Total combos:   {total}")
    print(f"Already done:   {len(done)}")
    print(f"Remaining:      {remaining}")
    print(f"Batch dir:      {BATCH_DIR}")
    print(f"CSV:            {RESULTS_CSV}")
    print()

    fields = fieldnames_for_csv()
    is_new = not RESULTS_CSV.exists()
    csv_f = RESULTS_CSV.open("a", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=fields)
    if is_new:
        writer.writeheader()
        csv_f.flush()

    yaml_path = BATCH_DIR / "_current.yaml"
    sweep_start = time.time()

    for idx, (pml_exp, alpha, pml_thick, freq) in enumerate(combos, 1):
        label = combo_label(pml_exp, alpha, pml_thick, freq)

        if label in done:
            print(f"[{idx}/{total}] {label}  SKIP (cached)")
            continue

        print(f"[{idx}/{total}] {label}")

        run_out = BATCH_DIR / label
        if run_out.exists():
            shutil.rmtree(run_out)
        run_out.mkdir(parents=True)

        generate_yaml(yaml_path, pml_exp, pml_thick, alpha, freq)

        cmd = [PYTHON, "main.py", "run", str(yaml_path),
               "--output", str(run_out), "--quiet"]
        if SKIP_PLOTS:
            cmd.append("--no-plots")

        t0 = time.time()
        status = "OK"
        stderr_tail = ""
        try:
            proc = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=RUN_TIMEOUT,
            )
            if proc.returncode != 0:
                status = f"RC_{proc.returncode}"
                stderr_tail = (proc.stderr or "")[-400:]
        except subprocess.TimeoutExpired:
            status = "TIMEOUT"
        wall = time.time() - t0

        row = {
            "combo_label":   label,
            "pml_exponent":  pml_exp,
            "alpha_max":     alpha,
            "pml_thickness": pml_thick,
            "gauss_freq":    freq,
            "status":        status,
        }
        for rec in RECEIVERS:
            n = rec["name"]
            row[f"{n}_error_db"]    = ""
            row[f"{n}_error_ratio"] = ""

        if status == "OK":
            pml_data = parse_pml_results(run_out)
            for rec in RECEIVERS:
                n = rec["name"]
                mic = pml_data.get(n, {})
                row[f"{n}_error_db"]    = mic.get("error_db", "")
                row[f"{n}_error_ratio"] = mic.get("error_ratio", "")
            mic_parts = []
            for rec in RECEIVERS:
                n = rec["name"]
                mic_parts.append(f"{n}={row[n + '_error_db']} dB")
            print(f"  OK  ({wall:5.1f}s)  " + ", ".join(mic_parts))
        else:
            print(f"  {status}  ({wall:5.1f}s)")
            if stderr_tail:
                print(f"  stderr: ...{stderr_tail.rstrip()}")

        writer.writerow(row)
        csv_f.flush()

    csv_f.close()
    elapsed = time.time() - sweep_start
    print()
    print(f"Done. Total wall clock: {elapsed/60:.1f} min")
    print(f"Results CSV: {RESULTS_CSV}")
    print(f"Plot folders: {BATCH_DIR}/<combo_label>/plots/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
