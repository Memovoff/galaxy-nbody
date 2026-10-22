"""Headless runs: preview images, frame sequences, energy diagnostics, benchmarks."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import taichi as ti

from .config import SimConfig
from .core import Simulation
from .render import Renderer


def _save_energy_plot(
    times: list[float], energies: list[float], path: Path, n: int
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times_arr = np.asarray(times)
    energies_arr = np.asarray(energies)
    drift = (energies_arr - energies_arr[0]) / max(abs(energies_arr[0]), 1e-12)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9.0, 5.0), dpi=160)
    ax.plot(times_arr, drift * 100.0, color="#7cc7ff", linewidth=1.8)
    ax.axhline(0.0, color="#444a55", linewidth=1.0, linestyle="--")
    ax.set_xlabel("simulation time")
    ax.set_ylabel("relative energy drift, %")
    ax.set_title(f"Energy conservation (N = {n:,})")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def run_headless(
    cfg: SimConfig | None = None,
    steps: int = 1200,
    width: int = 1280,
    height: int = 720,
    scale: float = 36.0,
    exposure: float = 1.10,
    out: str | Path | None = "media/preview.png",
    frames_dir: str | Path | None = None,
    frame_every: int = 0,
    energy_csv: str | Path | None = None,
    diag_every: int = 50,
    azimuth_deg: float = -28.0,
    elevation_deg: float = 38.0,
) -> dict[str, float]:
    """Run without a window and optionally save frames and diagnostics."""
    import math

    cfg = cfg or SimConfig()
    sim = Simulation(cfg)
    renderer = Renderer(width, height, scale=scale, exposure=exposure)
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)

    frames_path = Path(frames_dir) if frames_dir else None
    if frames_path is not None:
        frames_path.mkdir(parents=True, exist_ok=True)

    def save_frame(step: int) -> None:
        if frames_path is None:
            return
        renderer.render(sim, azimuth, elevation)
        renderer.save(frames_path / f"frame_{step:05d}.png")

    energies: list[float] = []
    times: list[float] = []
    if energy_csv is not None:
        energies.append(sim.total_energy())
        times.append(0.0)

    if frames_path is not None:
        save_frame(0)

    start = time.perf_counter()
    for step in range(1, steps + 1):
        sim.step()
        if (
            energy_csv is not None
            and diag_every > 0
            and step % diag_every == 0
        ):
            energies.append(sim.total_energy())
            times.append(sim.time)
        if frames_path is not None and frame_every > 0 and step % frame_every == 0:
            save_frame(step)
    elapsed = time.perf_counter() - start

    renderer.render(sim, azimuth, elevation)
    if out is not None:
        renderer.save(out)

    if energy_csv is not None:
        energy_path = Path(energy_csv)
        energy_path.parent.mkdir(parents=True, exist_ok=True)
        with energy_path.open("w", encoding="utf-8") as handle:
            handle.write("time,energy,relative_drift\n")
            e0 = energies[0]
            for t, e in zip(times, energies):
                drift = (e - e0) / max(abs(e0), 1e-12)
                handle.write(f"{t:.8f},{e:.10e},{drift:.10e}\n")
        _save_energy_plot(times, energies, energy_path.with_suffix(".png"), cfg.n)

    steps_per_second = steps / elapsed if elapsed > 0 else float("inf")
    drift = 0.0
    if energies:
        drift = (energies[-1] - energies[0]) / max(abs(energies[0]), 1e-12)

    stats = {
        "steps": float(steps),
        "sim_time": sim.time,
        "wall_time": elapsed,
        "steps_per_second": steps_per_second,
        "energy_drift": drift,
        "n": float(cfg.n),
    }

    print(
        f"n={cfg.n:,}  steps={steps}  t={sim.time:.2f}  "
        f"wall={elapsed:.2f}s  rate={steps_per_second:.1f} steps/s"
    )
    if energies:
        print(f"relative energy drift: {drift * 100.0:+.4f}%")
    if out is not None:
        print(f"saved preview: {out}")
    if frames_path is not None:
        print(f"saved frames:  {frames_path}")
    if energy_csv is not None:
        print(f"saved energy:  {energy_csv}")

    return stats


def run_benchmark(
    sizes: list[int],
    steps: int = 30,
    base_cfg: SimConfig | None = None,
    warmup: int = 3,
) -> list[dict[str, float]]:
    """Measure raw stepping throughput for several particle counts."""
    base_cfg = base_cfg or SimConfig()
    results: list[dict[str, float]] = []

    print(f"{'N':>10}  {'steps/s':>10}  {'Mpair/s':>10}")
    print("-" * 36)
    for n in sizes:
        cfg = replace(base_cfg, n=n)
        sim = Simulation(cfg)
        for _ in range(warmup):
            sim.step()
        ti.sync()

        start = time.perf_counter()
        for _ in range(steps):
            sim.step()
        ti.sync()
        elapsed = time.perf_counter() - start

        rate = steps / elapsed if elapsed > 0 else float("inf")
        pair_rate = n * n * rate / 1e6
        results.append(
            {
                "n": float(n),
                "steps_per_second": rate,
                "mpair_per_second": pair_rate,
                "wall_time": elapsed,
            }
        )
        print(f"{n:>10,}  {rate:>10.2f}  {pair_rate:>10.1f}")

    return results
