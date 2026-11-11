import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from diffusion_portfolio.model import ConditionalUNet, timestep_embedding
from diffusion_portfolio.ddpm import Diffusion


def test_embeddings():
    assert timestep_embedding(torch.tensor([0, 1]), 9).shape == (2, 9)


def test_model_shapes():
    model = ConditionalUNet(channels=1, classes=10, base=8, time_width=16)
    x = torch.randn(2, 1, 32, 32)
    result = model(x, torch.tensor([0, 3]), torch.tensor([-1, 5]))
    assert result.shape == x.shape
    assert torch.isfinite(result).all()


def test_forward_and_reverse():
    ddpm = Diffusion(steps=5)
    x = torch.randn(2, 3, 8, 8)
    t = torch.tensor([0, 4])
    xt, epsilon = ddpm.forward_noise(x, t)
    assert xt.shape == epsilon.shape == x.shape
    assert torch.isfinite(ddpm.reverse_step(xt, t, epsilon)).all()


def test_short_generation():
    model = ConditionalUNet(channels=1, classes=2, base=8, time_width=16).eval()
    ddpm = Diffusion(steps=3)
    with torch.no_grad():
        samples, frames = ddpm.generate(model, (2, 1, 8, 8), torch.tensor([0, 1]), snapshots=3, seed=1)
    assert samples.shape == (2, 1, 8, 8)
    assert len(frames) >= 2
