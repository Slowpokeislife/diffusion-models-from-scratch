"""Illustrate the analytical forward DDPM process on real training images."""
import argparse
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from diffusion_portfolio.data import get_loader
from diffusion_portfolio.ddpm import Diffusion
from diffusion_portfolio.media import export_frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['fashion', 'mnist', 'flowers'], default='fashion')
    p.add_argument('--data-root', default='data')
    p.add_argument('--output', default='demo/forward_fashion')
    p.add_argument('--samples', type=int, default=30)
    p.add_argument('--steps', type=int, default=150)
    p.add_argument('--fps', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    loader, _ = get_loader(args.dataset, args.data_root, args.samples)
    x, _ = next(iter(loader)); x = x[:args.samples]
    d = Diffusion(steps=args.steps)
    # Shared epsilon across times yields a smooth visualization of the closed-form marginal.
    fixed_noise = torch.randn_like(x)
    timesteps = sorted(set(torch.linspace(0, args.steps - 1, 60).long().tolist()))
    frames = [d.forward_noise(x, torch.full((len(x),), step, dtype=torch.long), fixed_noise)[0]
              for step in timesteps]
    print('Saved', export_frames(frames, args.output, min(10, args.samples), args.fps, tile=3))


if __name__ == '__main__': main()
