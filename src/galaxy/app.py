"""Interactive Taichi UI for the galaxy merger.

Uses Taichi's legacy ``ti.GUI`` window: it works with non-ASCII project
paths on Windows, where the newer Vulkan-based ``ti.ui.Window`` fails to
load its shaders. The fast GUI path can display a ``ti.Vector.field``
directly, so no per-frame NumPy copy is needed.
"""

import math
import time

import taichi as ti

from .config import SimConfig
from .core import Simulation
from .render import Renderer


def run_interactive(
    cfg: SimConfig | None = None,
    width: int = 1280,
    height: int = 720,
    scale: float = 36.0,
    exposure: float = 1.10,
    auto_rotate: bool = True,
) -> None:
    """Open the simulation window.

    Controls:
        SPACE      pause / resume
        R / ENTER  reset
        arrows     rotate camera
        mouse drag rotate camera
        + / -      simulation speed (steps per frame)
        ESC        quit
    """
    cfg = cfg or SimConfig()
    sim = Simulation(cfg)
    renderer = Renderer(width, height, scale=scale, exposure=exposure)

    gui = ti.GUI(
        "Galaxy Merger - GPU N-body",
        res=(width, height),
        fast_gui=True,
    )

    azimuth = math.radians(-28.0)
    elevation = math.radians(38.0)
    paused = False
    speed = 1
    previous_cursor: tuple[float, float] | None = None

    frames = 0
    last_report = time.perf_counter()

    while gui.running:
        for event in gui.get_events(ti.GUI.PRESS):
            key = event.key
            if key == ti.GUI.ESCAPE:
                gui.running = False
            elif key == ti.GUI.SPACE:
                paused = not paused
            elif key == ti.GUI.RETURN or key in ("r", "R"):
                sim.reset()
            elif key == ti.GUI.LEFT:
                azimuth -= 0.06
            elif key == ti.GUI.RIGHT:
                azimuth += 0.06
            elif key == ti.GUI.UP:
                elevation = min(elevation + 0.04, 1.50)
            elif key == ti.GUI.DOWN:
                elevation = max(elevation - 0.04, 0.05)
            elif key in ("+", "="):
                speed = min(speed + 1, 8)
            elif key in ("-", "_"):
                speed = max(speed - 1, 1)

        if gui.is_pressed(ti.GUI.LMB):
            cursor = gui.get_cursor_pos()
            if previous_cursor is not None:
                dx = cursor[0] - previous_cursor[0]
                dy = cursor[1] - previous_cursor[1]
                azimuth += dx * 6.0
                elevation = min(max(elevation + dy * 4.0, 0.05), 1.50)
            previous_cursor = cursor
        else:
            previous_cursor = None
            if auto_rotate and not paused:
                azimuth += 0.0018

        if not paused:
            for _ in range(speed):
                sim.step()

        renderer.render(sim, azimuth, elevation)
        gui.set_image(renderer.pixels)
        gui.show()

        frames += 1
        now = time.perf_counter()
        if now - last_report >= 1.0:
            fps = frames / (now - last_report)
            print(
                f"t={sim.time:7.2f}  steps={sim.steps:6d}  "
                f"fps={fps:5.1f}  speed={speed}x",
                end="\r",
                flush=True,
            )
            frames = 0
            last_report = now

    print()
