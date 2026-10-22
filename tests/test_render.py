import numpy as np

from galaxy.config import SimConfig
from galaxy.core import Simulation
from galaxy.render import Renderer


def test_renderer_produces_an_image():
    sim = Simulation(SimConfig(n=128))
    renderer = Renderer(width=320, height=180, scale=20.0)
    renderer.render(sim, azimuth=0.3, elevation=0.65)

    image = renderer.image()
    assert image.shape == (180, 320, 3)
    assert np.isfinite(image).all()
    assert image.min() >= 0.0
    assert image.max() <= 1.0
