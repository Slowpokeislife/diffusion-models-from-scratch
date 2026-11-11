"""DDPM forward and reverse processes; original beta posterior variance."""
import torch
import torch.nn.functional as F


class Diffusion:
    def __init__(self, steps: int = 150, beta_start: float = 1e-4, beta_end: float = 0.02, device: str = 'cpu'):
        if steps < 2:
            raise ValueError('steps must be >= 2')
        self.steps = steps
        self.betas = torch.linspace(beta_start, beta_end, steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.alpha_bars_prev = torch.cat([torch.ones_like(self.alpha_bars[:1]), self.alpha_bars[:-1]])

    @staticmethod
    def extract(values: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
        return values.gather(0, t.long()).reshape(-1, *([1] * (ndim - 1)))

    def forward_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None):
        noise = torch.randn_like(x0) if noise is None else noise
        alpha_bar = self.extract(self.alpha_bars, t, x0.ndim)
        xt = alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * noise
        return xt, noise

    def loss(self, model, x0: torch.Tensor, t: torch.Tensor, labels: torch.Tensor | None = None):
        xt, noise = self.forward_noise(x0, t)
        return F.mse_loss(model(xt, t, labels), noise)

    @torch.no_grad()
    def reverse_step(self, x: torch.Tensor, t: torch.Tensor, epsilon: torch.Tensor):
        beta = self.extract(self.betas, t, x.ndim)
        alpha = self.extract(self.alphas, t, x.ndim)
        alpha_bar = self.extract(self.alpha_bars, t, x.ndim)
        alpha_bar_prev = self.extract(self.alpha_bars_prev, t, x.ndim)
        mean = (x - (beta / (1 - alpha_bar).sqrt()) * epsilon) / alpha.sqrt()
        posterior_variance = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
        mask = (t > 0).reshape(-1, *([1] * (x.ndim - 1)))
        return mean + mask * posterior_variance.clamp(min=0).sqrt() * torch.randn_like(x)

    @torch.no_grad()
    def generate(self, model, shape: tuple[int, ...], labels: torch.Tensor | None = None,
                 guidance: float = 2.0, snapshots: int = 32, seed: int | None = None):
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        device = self.betas.device
        x = torch.randn(shape, device=device)
        if labels is not None and labels.shape[0] != shape[0]:
            raise ValueError('labels must match batch size')
        frames = [x.detach().cpu().clone()]
        every = max(1, self.steps // max(1, snapshots))
        for step in reversed(range(self.steps)):
            t = torch.full((shape[0],), step, device=device, dtype=torch.long)
            if labels is None:
                eps = model(x, t, None)
            else:
                # Classifier-free guidance: epsilon_uncond + w * (epsilon_cond - epsilon_uncond)
                e_cond = model(x, t, labels)
                e_uncond = model(x, t, None)
                eps = e_uncond + guidance * (e_cond - e_uncond)
            x = self.reverse_step(x, t, eps)
            if step % every == 0 or step == 0:
                frames.append(x.detach().cpu().clone())
        return x, frames
