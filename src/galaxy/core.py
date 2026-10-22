"""Taichi simulation core: softened gravity and a leapfrog integrator."""

import numpy as np
import taichi as ti

from .config import SimConfig
from .initial_conditions import make_merger


@ti.kernel
def compute_acc(
    pos: ti.template(),
    mass: ti.template(),
    acc: ti.template(),
    n: ti.i32,
    G: ti.f32,
    eps2: ti.f32,
):
    """Direct O(N^2) gravitational acceleration with Plummer softening."""
    for i in range(n):
        p_i = pos[i]
        a = ti.Vector([0.0, 0.0, 0.0])
        for j in range(n):
            d = pos[j] - p_i
            r2 = d.dot(d) + eps2
            inv_r3 = 1.0 / (r2 * ti.sqrt(r2))
            a += (G * mass[j] * inv_r3) * d
        acc[i] = a


@ti.kernel
def kick(vel: ti.template(), acc: ti.template(), half_dt: ti.f32, n: ti.i32):
    for i in range(n):
        vel[i] += acc[i] * half_dt


@ti.kernel
def drift(pos: ti.template(), vel: ti.template(), dt: ti.f32, n: ti.i32):
    for i in range(n):
        pos[i] += vel[i] * dt


@ti.kernel
def kinetic_energy(vel: ti.template(), mass: ti.template(), n: ti.i32) -> ti.f32:
    energy = 0.0
    for i in range(n):
        energy += 0.5 * mass[i] * vel[i].dot(vel[i])
    return energy


@ti.kernel
def potential_energy(
    pos: ti.template(),
    mass: ti.template(),
    n: ti.i32,
    G: ti.f32,
    eps2: ti.f32,
) -> ti.f32:
    energy = 0.0
    for i in range(n):
        p_i = pos[i]
        for j in range(i + 1, n):
            d = p_i - pos[j]
            r = ti.sqrt(d.dot(d) + eps2)
            energy -= G * mass[i] * mass[j] / r
    return energy


class Simulation:
    """State and time stepping for the galaxy merger."""

    def __init__(self, cfg: SimConfig | None = None, seed: int | None = None):
        self.cfg = cfg or SimConfig()
        self.n = self.cfg.n

        self.pos = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        self.vel = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        self.acc = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        self.mass = ti.field(dtype=ti.f32, shape=self.n)
        self.color = ti.Vector.field(3, dtype=ti.f32, shape=self.n)

        self.time = 0.0
        self.steps = 0
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        """Re-sample the initial conditions (positions, colors, time)."""
        seed = self.cfg.seed if seed is None else seed
        data = make_merger(
            n=self.n,
            seed=seed,
            separation=self.cfg.separation,
            impact_parameter=self.cfg.impact_parameter,
            relative_speed=self.cfg.relative_speed,
            G=self.cfg.G,
        )
        self.pos.from_numpy(data["pos"])
        self.vel.from_numpy(data["vel"])
        self.mass.from_numpy(data["mass"])
        self.color.from_numpy(data["color"])
        self.time = 0.0
        self.steps = 0

        self._compute_acc()

    def _compute_acc(self) -> None:
        compute_acc(
            self.pos,
            self.mass,
            self.acc,
            self.n,
            self.cfg.G,
            self.cfg.softening**2,
        )

    def step(self) -> None:
        """One kick-drift-kick leapfrog step."""
        half_dt = 0.5 * self.cfg.dt
        kick(self.vel, self.acc, half_dt, self.n)
        drift(self.pos, self.vel, self.cfg.dt, self.n)
        self._compute_acc()
        kick(self.vel, self.acc, half_dt, self.n)

        self.time += self.cfg.dt
        self.steps += 1

    def kinetic_energy(self) -> float:
        return float(kinetic_energy(self.vel, self.mass, self.n))

    def potential_energy(self) -> float:
        return float(
            potential_energy(
                self.pos,
                self.mass,
                self.n,
                self.cfg.G,
                self.cfg.softening**2,
            )
        )

    def total_energy(self) -> float:
        return self.kinetic_energy() + self.potential_energy()

    def total_momentum(self) -> np.ndarray:
        vel = self.vel.to_numpy()
        mass = self.mass.to_numpy()
        return (vel * mass[:, None]).sum(axis=0)

    def state(self) -> dict[str, np.ndarray]:
        """Copy the full state to NumPy (useful for tests and analysis)."""
        return {
            "pos": self.pos.to_numpy(),
            "vel": self.vel.to_numpy(),
            "mass": self.mass.to_numpy(),
            "color": self.color.to_numpy(),
        }
