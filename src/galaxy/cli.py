"""Command line entry point."""

from __future__ import annotations

import argparse

import taichi as ti

from .app import run_interactive
from .config import SimConfig
from .headless import run_benchmark, run_headless

_ARCHES = {
    "gpu": ti.gpu,
    "cuda": ti.cuda,
    "vulkan": ti.vulkan,
    "cpu": ti.cpu,
}


def _init_taichi(arch_name: str) -> None:
    arch = _ARCHES[arch_name]
    try:
        ti.init(arch=arch, default_fp=ti.f32, kernel_profiler=False)
    except Exception:
        if arch_name != "gpu":
            raise
        print("GPU auto-detect failed, falling back to Vulkan")
        ti.init(arch=ti.vulkan, default_fp=ti.f32, kernel_profiler=False)
    backend = ti.lang.impl.current_cfg().arch
    print(f"Taichi backend: {backend}")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n", type=int, default=20_000, help="number of particles")
    parser.add_argument("--dt", type=float, default=0.02, help="time step")
    parser.add_argument("--softening", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--separation", type=float, default=10.5)
    parser.add_argument("--impact", type=float, default=2.2, help="impact parameter")
    parser.add_argument("--speed", type=float, default=0.5, help="relative speed")
    parser.add_argument(
        "--arch",
        choices=sorted(_ARCHES),
        default="gpu",
        help="Taichi backend (default: auto GPU)",
    )


def _add_render(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--scale", type=float, default=36.0, help="pixels per unit")
    parser.add_argument("--exposure", type=float, default=1.10)


def _config_from_args(args: argparse.Namespace) -> SimConfig:
    return SimConfig(
        n=args.n,
        dt=args.dt,
        softening=args.softening,
        seed=args.seed,
        separation=args.separation,
        impact_parameter=args.impact,
        relative_speed=args.speed,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="galaxy",
        description="GPU N-body galaxy merger simulation",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="open the interactive window")
    _add_common(run)
    _add_render(run)
    run.add_argument("--no-rotate", action="store_true", help="disable auto camera orbit")

    headless = sub.add_parser("headless", help="run without a window")
    _add_common(headless)
    _add_render(headless)
    headless.add_argument("--steps", type=int, default=1200)
    headless.add_argument("--out", type=str, default="media/preview.png")
    headless.add_argument("--no-preview", action="store_true")
    headless.add_argument("--frames-dir", type=str, default=None)
    headless.add_argument("--frame-every", type=int, default=0)
    headless.add_argument("--energy-csv", type=str, default=None)
    headless.add_argument("--diag-every", type=int, default=50)
    headless.add_argument("--azimuth", type=float, default=-28.0, help="degrees")
    headless.add_argument("--elevation", type=float, default=38.0, help="degrees")

    bench = sub.add_parser("bench", help="measure raw GPU throughput")
    bench.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[5_000, 20_000, 50_000],
        help="particle counts to benchmark",
    )
    bench.add_argument("--steps", type=int, default=20)
    bench.add_argument("--dt", type=float, default=0.02)
    bench.add_argument("--softening", type=float, default=0.15)
    bench.add_argument("--seed", type=int, default=7)
    bench.add_argument(
        "--arch",
        choices=sorted(_ARCHES),
        default="gpu",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    _init_taichi(args.arch)

    if args.command == "run":
        run_interactive(
            _config_from_args(args),
            width=args.width,
            height=args.height,
            scale=args.scale,
            exposure=args.exposure,
            auto_rotate=not args.no_rotate,
        )
        return

    if args.command == "headless":
        run_headless(
            _config_from_args(args),
            steps=args.steps,
            width=args.width,
            height=args.height,
            scale=args.scale,
            exposure=args.exposure,
            out=None if args.no_preview else args.out,
            frames_dir=args.frames_dir,
            frame_every=args.frame_every,
            energy_csv=args.energy_csv,
            diag_every=args.diag_every,
            azimuth_deg=args.azimuth,
            elevation_deg=args.elevation,
        )
        return

    if args.command == "bench":
        base = SimConfig(dt=args.dt, softening=args.softening, seed=args.seed)
        run_benchmark(args.sizes, steps=args.steps, base_cfg=base)
        return

    parser.error(f"unknown command: {args.command}")
