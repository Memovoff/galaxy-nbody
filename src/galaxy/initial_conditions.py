"""Galaxy initial conditions.

Two exponential disks with a central bulge are sampled, given circular
velocities, tilted in 3D, and placed on a bound collision orbit.

Everything is generated with NumPy; the result is copied into Taichi fields
by :class:`galaxy.core.Simulation`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray

# Fraction of the disk particles placed into spiral arms.
_ARM_FRACTION = 0.55
_ARM_PITCH = 2.4
_ARM_SCATTER = 0.28
_DISK_CUTOFF = 5.0


@dataclass(frozen=True)
class GalaxySpec:
    """Parameters of a single model galaxy."""

    center: tuple[float, float, float]
    velocity: tuple[float, float, float]
    spin: float = 1.0
    tilt: tuple[float, float, float] = (0.0, 0.0, 0.0)

    disk_mass: float = 1.0
    bulge_mass: float = 0.22
    scale_length: float = 1.0
    bulge_radius: float = 0.35
    thickness: float = 0.09
    bulge_fraction: float = 0.16

    color_inner: tuple[float, float, float] = (1.0, 0.85, 0.60)
    color_outer: tuple[float, float, float] = (0.50, 0.70, 1.00)


def _sample_exponential_disk(
    rng: np.random.Generator, n: int, scale_length: float, thickness: float
) -> tuple[Array, Array, Array]:
    """Sample an exponential disk with two logarithmic spiral arms.

    Returns local positions, cylindrical radius and azimuth.
    """
    radius = np.empty(n, dtype=np.float64)
    filled = 0
    cutoff = _DISK_CUTOFF * scale_length
    while filled < n:
        remaining = n - filled
        # Radial PDF is p(r) ~ r * exp(-r / h), i.e. Gamma(shape=2).
        candidates = -scale_length * (
            np.log(rng.random(remaining * 2)) + np.log(rng.random(remaining * 2))
        )
        candidates = candidates[candidates < cutoff]
        if candidates.size == 0:
            continue
        take = min(candidates.size, remaining)
        radius[filled : filled + take] = candidates[:take]
        filled += take

    base_theta = rng.uniform(0.0, 2.0 * np.pi, n)
    arm_id = rng.integers(0, 2, n)
    spiral_phase = _ARM_PITCH * np.log(np.maximum(radius, 0.05) / scale_length + 1.0)
    arm_theta = spiral_phase + arm_id * np.pi + rng.normal(0.0, _ARM_SCATTER, n)
    in_arm = rng.random(n) < _ARM_FRACTION
    theta = np.where(in_arm, arm_theta, base_theta)

    z = rng.normal(0.0, thickness, n)
    pos = np.column_stack(
        [radius * np.cos(theta), radius * np.sin(theta), z]
    )
    return pos, radius, theta


def _sample_plummer(
    rng: np.random.Generator, n: int, scale_radius: float
) -> Array:
    """Sample an isotropic Plummer sphere."""
    u = np.clip(rng.random(n), 1e-6, 1.0 - 1e-6)
    radius = scale_radius / np.sqrt(u ** (-2.0 / 3.0) - 1.0)
    radius = np.minimum(radius, 10.0 * scale_radius)

    z = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack(
        [
            radius * sin_theta * np.cos(phi),
            radius * sin_theta * np.sin(phi),
            radius * z,
        ]
    )


def _rotation_matrix(tilt: tuple[float, float, float]) -> Array:
    """Rotation matrix Rz @ Ry @ Rx (angles in radians)."""
    tx, ty, tz = tilt
    cx, sx = np.cos(tx), np.sin(tx)
    cy, sy = np.cos(ty), np.sin(ty)
    cz, sz = np.cos(tz), np.sin(tz)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def _make_galaxy(
    rng: np.random.Generator, n: int, spec: GalaxySpec, G: float
) -> tuple[Array, Array, Array, Array]:
    """Build positions, velocities, masses and colors of one galaxy."""
    n_bulge = int(round(n * spec.bulge_fraction))
    n_bulge = min(max(n_bulge, 0), n)
    n_disk = n - n_bulge

    pos_parts: list[Array] = []
    vel_parts: list[Array] = []
    mass_parts: list[Array] = []
    color_parts: list[Array] = []

    if n_disk > 0:
        disk_pos, radius, theta = _sample_exponential_disk(
            rng, n_disk, spec.scale_length, spec.thickness
        )

        # Circular velocity from the mass enclosed by the disk and bulge.
        h = spec.scale_length
        enclosed_disk = spec.disk_mass * (
            1.0 - (1.0 + radius / h) * np.exp(-radius / h)
        )
        enclosed_bulge = (
            spec.bulge_mass
            * radius**3
            / (radius**2 + spec.bulge_radius**2) ** 1.5
        )
        v_circ = np.sqrt(
            G * (enclosed_disk + enclosed_bulge) / np.maximum(radius, 1e-4)
        )

        vx = -np.sin(theta) * v_circ * spec.spin
        vy = np.cos(theta) * v_circ * spec.spin
        vel = np.column_stack([vx, vy, np.zeros(n_disk)])

        # A small isotropic dispersion keeps the disk from being razor thin.
        sigma = 0.06 * v_circ + 1e-3
        vel += rng.normal(0.0, 1.0, (n_disk, 3)) * sigma[:, None]

        t = np.clip(radius / (4.5 * h), 0.0, 1.0)[:, None]
        inner = np.asarray(spec.color_inner, dtype=np.float64)
        outer = np.asarray(spec.color_outer, dtype=np.float64)
        color = inner * (1.0 - t) + outer * t
        color *= rng.uniform(0.7, 1.3, (n_disk, 1))

        pos_parts.append(disk_pos)
        vel_parts.append(vel)
        mass_parts.append(np.full(n_disk, spec.disk_mass / n_disk))
        color_parts.append(color)

    if n_bulge > 0:
        bulge_pos = _sample_plummer(rng, n_bulge, spec.bulge_radius)
        # Crude virial estimate, enough to keep the bulge visually stable.
        sigma = 0.9 * np.sqrt(G * spec.bulge_mass / (6.0 * spec.bulge_radius))
        bulge_vel = rng.normal(0.0, sigma, (n_bulge, 3))

        bulge_color = np.asarray(spec.color_inner, dtype=np.float64)
        bulge_color = bulge_color * rng.uniform(0.5, 0.8, (n_bulge, 1))

        pos_parts.append(bulge_pos)
        vel_parts.append(bulge_vel)
        mass_parts.append(np.full(n_bulge, spec.bulge_mass / n_bulge))
        color_parts.append(bulge_color)

    pos = np.vstack(pos_parts)
    vel = np.vstack(vel_parts)
    mass = np.concatenate(mass_parts)
    color = np.vstack(color_parts)

    rot = _rotation_matrix(spec.tilt)
    pos = pos @ rot.T + np.asarray(spec.center, dtype=np.float64)
    vel = vel @ rot.T + np.asarray(spec.velocity, dtype=np.float64)

    return pos, vel, mass, color


def make_merger(
    n: int = 20_000,
    seed: int = 7,
    separation: float = 10.5,
    impact_parameter: float = 2.2,
    relative_speed: float = 0.5,
    G: float = 1.0,
) -> dict[str, Array]:
    """Create two tilted spiral galaxies on a bound collision orbit.

    Galaxy A is cool/blue and galaxy B is warm/orange, which makes the
    mixing during the merger easy to see.
    """
    if n < 2:
        raise ValueError("n must be at least 2")

    rng = np.random.default_rng(seed)
    n_a = n // 2
    n_b = n - n_a

    half = separation / 2.0
    offset = impact_parameter / 2.0
    v_half = relative_speed / 2.0

    spec_a = GalaxySpec(
        center=(-half, -offset, 0.0),
        velocity=(v_half, 0.0, 0.02),
        spin=1.0,
        tilt=(
            np.radians(12.0),
            np.radians(-18.0),
            np.radians(8.0),
        ),
        color_inner=(1.00, 0.82, 0.55),
        color_outer=(0.35, 0.58, 1.00),
    )
    spec_b = GalaxySpec(
        center=(half, offset, 0.0),
        velocity=(-v_half, 0.0, -0.02),
        spin=-1.0,
        tilt=(
            np.radians(-14.0),
            np.radians(22.0),
            np.radians(-6.0),
        ),
        color_inner=(1.00, 0.72, 0.40),
        color_outer=(1.00, 0.42, 0.18),
    )

    pos_a, vel_a, mass_a, color_a = _make_galaxy(rng, n_a, spec_a, G=G)
    pos_b, vel_b, mass_b, color_b = _make_galaxy(rng, n_b, spec_b, G=G)

    pos = np.vstack([pos_a, pos_b])
    vel = np.vstack([vel_a, vel_b])
    mass = np.concatenate([mass_a, mass_b])
    color = np.vstack([color_a, color_b])

    # Center the system and remove net momentum so it stays in frame.
    total_mass = mass.sum()
    com = (pos * mass[:, None]).sum(axis=0) / total_mass
    com_vel = (vel * mass[:, None]).sum(axis=0) / total_mass
    pos -= com
    vel -= com_vel

    return {
        "pos": np.ascontiguousarray(pos, dtype=np.float32),
        "vel": np.ascontiguousarray(vel, dtype=np.float32),
        "mass": np.ascontiguousarray(mass, dtype=np.float32),
        "color": np.ascontiguousarray(color, dtype=np.float32),
    }
