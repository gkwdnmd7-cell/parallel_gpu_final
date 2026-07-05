#!/usr/bin/env python3
"""Benchmark serial JacobiSetComputation against the CUDA hybrid version."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path

from compare_jacobi import read_jacobi_edges


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
JS_DIR = PROJECT_ROOT / "CODE" / "JacobiSetComputation-master"
GPU_EXE = SCRIPT_DIR / "build" / "jacobi_cuda"
CPU_EXE = JS_DIR / "build" / "JacobiSetComputation"
MESH_MAKE = JS_DIR / "trimesh2" / "bin.Linux64" / "mesh_make"
WORK_DIR = SCRIPT_DIR / "benchmark_meshes"


def parse_sizes(value: str) -> list[tuple[int, int]]:
    sizes: list[tuple[int, int]] = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        match = re.fullmatch(r"(\d+)x(\d+)", item)
        if not match:
            raise argparse.ArgumentTypeError(f"invalid size {item!r}; expected e.g. 32x16")
        sizes.append((int(match.group(1)), int(match.group(2))))
    if not sizes:
        raise argparse.ArgumentTypeError("at least one size is required")
    return sizes


def run_command(args: list[str], cwd: Path | None = None) -> tuple[str, float]:
    start = time.perf_counter()
    proc = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if proc.returncode != 0:
        print(proc.stdout, file=sys.stderr)
        raise RuntimeError(f"command failed with exit code {proc.returncode}: {' '.join(args)}")
    return proc.stdout, elapsed_ms


def parse_gpu_output(output: str) -> dict[str, float | int]:
    fields: dict[str, float | int] = {}
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in {"jacobi_edges", "degenerate_edges", "sos_fallback_edges"}:
            fields[key] = int(value)
        elif key in {"gpu_kernel_ms", "gpu_total_ms"}:
            fields[key] = float(value)
    missing = {"jacobi_edges", "degenerate_edges", "sos_fallback_edges", "gpu_kernel_ms", "gpu_total_ms"} - fields.keys()
    if missing:
        raise ValueError(f"missing GPU output fields: {sorted(missing)}")
    return fields


def ensure_inputs() -> None:
    for path in (GPU_EXE, CPU_EXE, MESH_MAKE):
        if not path.exists():
            raise FileNotFoundError(f"required executable not found: {path}")
    WORK_DIR.mkdir(parents=True, exist_ok=True)


def benchmark_size(u: int, v: int) -> dict[str, object]:
    mesh = WORK_DIR / f"torus_{u}x{v}.obj"
    cpu_out = WORK_DIR / f"torus_{u}x{v}_jacobi.txt"
    gpu_out = WORK_DIR / f"torus_{u}x{v}_gpu_jacobi.txt"

    run_command([str(MESH_MAKE), "torus", str(u), str(v), str(mesh)], cwd=JS_DIR)
    cpu_output, cpu_wall_ms = run_command([str(CPU_EXE), str(mesh)], cwd=JS_DIR / "build")
    gpu_output, gpu_wall_ms = run_command([str(GPU_EXE), str(mesh), "--output", str(gpu_out)], cwd=SCRIPT_DIR)

    gpu_fields = parse_gpu_output(gpu_output)
    ref_edges = read_jacobi_edges(cpu_out)
    cand_edges = read_jacobi_edges(gpu_out)
    match = ref_edges == cand_edges

    vertices = u * v
    edges = len(ref_edges)
    speedup = cpu_wall_ms / gpu_wall_ms if gpu_wall_ms > 0.0 else 0.0

    print(
        f"{u}x{v}: cpu_wall={cpu_wall_ms:.3f} ms, "
        f"gpu_wall={gpu_wall_ms:.3f} ms, speedup={speedup:.3f}x, match={'yes' if match else 'no'}"
    )

    return {
        "u": u,
        "v": v,
        "vertices": vertices,
        "edges": 3 * u * v,
        "jacobi_edges": gpu_fields["jacobi_edges"],
        "degenerate_edges": gpu_fields["degenerate_edges"],
        "sos_fallback_edges": gpu_fields["sos_fallback_edges"],
        "cpu_wall_ms": f"{cpu_wall_ms:.3f}",
        "gpu_wall_ms": f"{gpu_wall_ms:.3f}",
        "gpu_kernel_ms": f"{gpu_fields['gpu_kernel_ms']:.6f}",
        "gpu_total_ms": f"{gpu_fields['gpu_total_ms']:.6f}",
        "speedup_wall": f"{speedup:.3f}",
        "match": "yes" if match else "no",
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=parse_sizes, default=parse_sizes("32x16,64x32,128x64"))
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "benchmark_results.csv")
    args = parser.parse_args(argv[1:])

    ensure_inputs()
    rows = [benchmark_size(u, v) for u, v in args.sizes]

    fieldnames = [
        "u",
        "v",
        "vertices",
        "edges",
        "jacobi_edges",
        "degenerate_edges",
        "sos_fallback_edges",
        "cpu_wall_ms",
        "gpu_wall_ms",
        "gpu_kernel_ms",
        "gpu_total_ms",
        "speedup_wall",
        "match",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
