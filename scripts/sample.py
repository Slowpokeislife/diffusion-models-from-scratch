"""Generate class-guided images, reverse trajectory animation and final image."""
import argparse
from pathlib import Path
import sys
import torch
from torchvision.utils import save_image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from diffusion_portfolio.ddpm import Diffusion
from diffusion_portfolio.model import ConditionalUNet
from diffusion_portfolio.media import export_frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', default='demo/generated')
    p.add_argument('--samples', type=int, default=30)
    p.add_argument('--guidance', type=float, default=2.0)
    p.add_argument('--snapshots', type=int, default=50)
    p.add_argument('--fps', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='auto')
    args = p.parse_args()
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    cfg = ckpt['config']
    model = ConditionalUNet(ckpt['channels'], ckpt['classes'], cfg['base']).to(device)
    model.load_state_dict(ckpt['weights']); model.eval()
    diffuser = Diffusion(cfg['steps'], device=device)
    labels = torch.arange(args.samples, device=device) % ckpt['classes']
    result, trajectory = diffuser.generate(model, (args.samples, ckpt['channels'], 32, 32),
                                            labels, args.guidance, args.snapshots, args.seed)
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True)
    columns = min(10, args.samples)
    save_image(((result.cpu().clamp(-1, 1) + 1) / 2), destination.with_suffix('.png'), nrow=columns)
    gif, mp4 = export_frames(trajectory, destination, columns=columns, fps=args.fps,
                             tile=3 if ckpt['channels'] == 3 else 4)
    print(f'Saved {destination.with_suffix(".png")}, {gif}, {mp4}')


if __name__ == '__main__': main()
