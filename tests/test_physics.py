import numpy as np
import taichi as ti

from galaxy.config import SimConfig
from galaxy.core import Simulation, compute_acc
from galaxy.initial_conditions import make_merger


def test_force_matches_numpy_reference():
    n = 64
    rng = np.random.default_rng(123)
    pos_np = rng.normal(0.0, 2.0, (n, 3)).astype(np.float32)
    mass_np = rng.uniform(0.5, 1.5, n).astype(np.float32)
    eps2 = 0.01

    pos = ti.Vector.field(3, ti.f32, shape=n)
    mass = ti.field(ti.f32, shape=n)
    acc = ti.Vector.field(3, ti.f32, shape=n)
    pos.from_numpy(pos_np)
    mass.from_numpy(mass_np)

    compute_acc(pos, mass, acc, n, 1.0, eps2)
    acc_np = acc.to_numpy()

    diff = pos_np[None, :, :] - pos_np[:, None, :]
    r2 = (diff**2).sum(axis=-1) + eps2
    inv_r3 = 1.0 / (r2 * np.sqrt(r2))
    np.fill_diagonal(inv_r3, 0.0)
    acc_ref = (
        mass_np[None, :, None] * diff * inv_r3[:, :, None]
    ).sum(axis=1)

    np.testing.assert_allclose(acc_np, acc_ref, rtol=3e-3, atol=1e-4)


def test_energy_is_conserved():
    cfg = SimConfig(n=384, dt=0.005, softening=0.15, seed=11)
    sim = Simulation(cfg)

    e0 = sim.total_energy()
    for _ in range(100):
        sim.step()
    e1 = sim.total_energy()

    drift = abs((e1 - e0) / e0)
    assert np.isfinite(e0)
    assert drift < 0.01, f"energy drift too large: {drift:.4%}"


def test_momentum_is_conserved():
    cfg = SimConfig(n=384, dt=0.005, softening=0.15, seed=11)
    sim = Simulation(cfg)

    p0 = sim.total_momentum()
    for _ in range(20):
        sim.step()
    p1 = sim.total_momentum()

    assert np.linalg.norm(p1 - p0) < 1e-3


def test_initial_conditions_are_sane():
    data = make_merger(n=2000, seed=3)

    assert data["pos"].shape == (2000, 3)
    assert data["vel"].shape == (2000, 3)
    assert data["mass"].shape == (2000,)
    assert data["color"].shape == (2000, 3)
    assert np.isfinite(data["pos"]).all()
    assert np.isfinite(data["vel"]).all()

    mass = data["mass"].astype(np.float64)
    com = (data["pos"] * mass[:, None]).sum(axis=0) / mass.sum()
    momentum = (data["vel"] * mass[:, None]).sum(axis=0)
    assert np.linalg.norm(com) < 1e-3
    assert np.linalg.norm(momentum) < 1e-3
    assert np.all(data["color"] >= 0.0)
