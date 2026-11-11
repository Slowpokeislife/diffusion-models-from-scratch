"""Train a conditional DDPM. Example: python scripts/train.py --dataset fashion --epochs 15"""
import argparse
from pathlib import Path
import random
import sys
import numpy as np
import torch
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from diffusion_portfolio.data import get_loader
from diffusion_portfolio.ddpm import Diffusion
from diffusion_portfolio.model import ConditionalUNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['fashion', 'mnist', 'flowers'], default='fashion')
    p.add_argument('--data-root', default='data')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--steps', type=int, default=150)
    p.add_argument('--base', type=int, default=32)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--drop-prob', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-samples', type=int)
    p.add_argument('--device', default='auto')
    p.add_argument('--out', default='checkpoints')
    args = p.parse_args()
    if not 0 <= args.drop_prob <= 1:
        p.error('--drop-prob must lie within [0,1]')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device)
    loader, ds = get_loader(args.dataset, args.data_root, args.batch_size, 32,
                            max_samples=args.max_samples)
    classes = len(ds.dataset.classes) if isinstance(ds, torch.utils.data.Subset) and hasattr(ds.dataset, 'classes') else (len(ds.classes) if hasattr(ds, 'classes') else (3 if args.dataset == 'flowers' else 10))
    channels = 3 if args.dataset == 'flowers' else 1
    model = ConditionalUNet(channels=channels, classes=classes, base=args.base).to(device)
    diffusion = Diffusion(steps=args.steps, device=device)
    opt = AdamW(model.parameters(), lr=args.lr)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f'Device={device}, dataset={args.dataset}, classes={classes}, train samples={len(ds)}')
    for epoch in range(1, args.epochs + 1):
        model.train(); running = 0.; total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            t = torch.randint(0, args.steps, (x.shape[0],), device=device)
            y = y.masked_fill(torch.rand(x.shape[0], device=device) < args.drop_prob, -1)
            loss = diffusion.loss(model, x, t, y)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            running += loss.item() * x.shape[0]; total += x.shape[0]
        print(f'Epoch {epoch:03d}/{args.epochs}: noise-prediction MSE={running / total:.6f}', flush=True)
        torch.save({'weights': model.state_dict(), 'config': vars(args), 'classes': classes,
                    'channels': channels, 'epoch': epoch, 'loss': running / total},
                   out / f'{args.dataset}.pt')
    print(f'Checkpoint saved to {out / (args.dataset + ".pt")}')


if __name__ == '__main__': main()
