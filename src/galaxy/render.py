"""Additive particle renderer for the galaxy simulation."""

import numpy as np
import taichi as ti

from .core import Simulation

_PI = 3.141592653589793
_GAMMA = 1.0 / 2.2


@ti.kernel
def copy_background(pixels: ti.template(), background: ti.template()):
    """Start each frame from the static starfield background."""
    for i, j in pixels:
        pixels[i, j] = background[i, j]


@ti.kernel
def splat(
    pos: ti.template(),
    color: ti.template(),
    pixels: ti.template(),
    width: ti.i32,
    height: ti.i32,
    scale: ti.f32,
    azimuth: ti.f32,
    elevation: ti.f32,
):
    """Project particles and additively splat them into the framebuffer.

    The disk lies in the xy plane. ``elevation`` is measured from the disk
    plane: 90 degrees is face-on, 0 is edge-on.
    """
    ca = ti.cos(azimuth)
    sa = ti.sin(azimuth)
    tilt = _PI * 0.5 - elevation
    ct = ti.cos(tilt)
    st = ti.sin(tilt)

    for i in range(pos.shape[0]):
        p = pos[i]

        # Rotate around z, then tilt around x.
        x1 = ca * p.x + sa * p.y
        y1 = -sa * p.x + ca * p.y
        screen_y = ct * y1 - st * p.z

        px = width * 0.5 + x1 * scale
        py = height * 0.5 + screen_y * scale
        ix = ti.cast(px, ti.i32)
        iy = ti.cast(py, ti.i32)
        c = color[i]

        # 5x5 Gaussian sprite, softly glowing. Weights are normalized.
        for dy in ti.static(range(-2, 3)):
            for dx in ti.static(range(-2, 3)):
                w = ti.exp(-0.5 * (dx * dx + dy * dy)) / 7.2
                jx = ix + dx
                jy = iy + dy
                if 0 <= jx and jx < width and 0 <= jy and jy < height:
                    ti.atomic_add(pixels[jx, jy], c * w)


@ti.kernel
def extract_bright(src: ti.template(), dst: ti.template(), threshold: ti.f32):
    """Keep only the HDR part above a threshold for the bloom pass."""
    for i, j in src:
        c = src[i, j] - threshold
        dst[i, j] = ti.Vector(
            [ti.max(c.x, 0.0), ti.max(c.y, 0.0), ti.max(c.z, 0.0)]
        )


@ti.kernel
def blur_horizontal(
    src: ti.template(), dst: ti.template(), width: ti.i32, height: ti.i32
):
    for i, j in src:
        acc = ti.Vector([0.0, 0.0, 0.0])
        wsum = 0.0
        for k in ti.static(range(-5, 6)):
            x = i + k
            if 0 <= x and x < width:
                w = ti.exp(-0.5 * (k / 2.2) ** 2)
                acc += src[x, j] * w
                wsum += w
        dst[i, j] = acc / wsum


@ti.kernel
def blur_vertical(
    src: ti.template(), dst: ti.template(), width: ti.i32, height: ti.i32
):
    for i, j in src:
        acc = ti.Vector([0.0, 0.0, 0.0])
        wsum = 0.0
        for k in ti.static(range(-5, 6)):
            y = j + k
            if 0 <= y and y < height:
                w = ti.exp(-0.5 * (k / 2.2) ** 2)
                acc += src[i, y] * w
                wsum += w
        dst[i, j] = acc / wsum


@ti.kernel
def composite(
    pixels: ti.template(), bloom: ti.template(), strength: ti.f32
):
    for i, j in pixels:
        pixels[i, j] += bloom[i, j] * strength


@ti.kernel
def tonemap(pixels: ti.template(), exposure: ti.f32):
    """Exponential tone mapping followed by gamma correction."""
    for i, j in pixels:
        c = pixels[i, j] * exposure
        c = ti.Vector(
            [
                1.0 - ti.exp(-c.x),
                1.0 - ti.exp(-c.y),
                1.0 - ti.exp(-c.z),
            ]
        )
        pixels[i, j] = ti.Vector(
            [
                ti.pow(c.x, _GAMMA),
                ti.pow(c.y, _GAMMA),
                ti.pow(c.z, _GAMMA),
            ]
        )


class Renderer:
    """Holds the framebuffer and renders simulation frames."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        scale: float = 36.0,
        exposure: float = 1.10,
        bloom_strength: float = 0.9,
    ):
        self.width = width
        self.height = height
        self.scale = scale
        self.exposure = exposure
        self.bloom_strength = bloom_strength
        self.pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
        self.background = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
        self._bloom_a = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
        self._bloom_b = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
        self._build_background()

    def _build_background(self) -> None:
        """Generate a static, very dim starfield with a soft vignette."""
        rng = np.random.default_rng(2024)
        width, height = self.width, self.height

        x = np.linspace(-1.0, 1.0, width, dtype=np.float32)[:, None]
        y = np.linspace(-1.0, 1.0, height, dtype=np.float32)[None, :]
        vignette = np.clip(1.0 - 0.25 * (x * x + y * y), 0.0, 1.0)
        base = np.array([0.0025, 0.0045, 0.0090], dtype=np.float32)
        image = base[None, None, :] * vignette[:, :, None]

        n_stars = max(200, width * height // 400)
        xs = rng.integers(0, width, n_stars)
        ys = rng.integers(0, height, n_stars)
        brightness = (
            0.015 + 0.10 * rng.random(n_stars).astype(np.float32) ** 2.5
        )
        hue = rng.random(n_stars).astype(np.float32)
        colors = np.empty((n_stars, 3), dtype=np.float32)
        colors[:, 0] = 0.80 + 0.20 * hue
        colors[:, 1] = 0.86 + 0.10 * (1.0 - np.abs(hue - 0.5) * 2.0)
        colors[:, 2] = 1.00 - 0.25 * hue

        kernel = np.array(
            [[0.2, 0.5, 0.2], [0.5, 1.0, 0.5], [0.2, 0.5, 0.2]],
            dtype=np.float32,
        )
        kernel /= kernel.sum()
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                xx = np.clip(xs + dx, 0, width - 1)
                yy = np.clip(ys + dy, 0, height - 1)
                contribution = colors * (
                    brightness * kernel[dy + 1, dx + 1]
                )[:, None]
                np.add.at(image, (xx, yy), contribution)

        self.background.from_numpy(np.ascontiguousarray(image))

    def render(
        self, sim: Simulation, azimuth: float, elevation: float
    ) -> None:
        copy_background(self.pixels, self.background)
        splat(
            sim.pos,
            sim.color,
            self.pixels,
            self.width,
            self.height,
            self.scale,
            azimuth,
            elevation,
        )
        extract_bright(self.pixels, self._bloom_a, 0.18)
        blur_horizontal(self._bloom_a, self._bloom_b, self.width, self.height)
        blur_vertical(self._bloom_b, self._bloom_a, self.width, self.height)
        composite(self.pixels, self._bloom_a, self.bloom_strength)
        tonemap(self.pixels, self.exposure)

    def image(self) -> np.ndarray:
        """Return the current frame as an (H, W, 3) top-left NumPy image."""
        raw = self.pixels.to_numpy()
        return np.flipud(np.transpose(raw, (1, 0, 2)))

    def save(self, path: str) -> None:
        """Save the current frame as an 8-bit PNG."""
        from pathlib import Path

        from PIL import Image

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        image = np.clip(self.image(), 0.0, 1.0)
        Image.fromarray((image * 255.0 + 0.5).astype(np.uint8)).save(path)
