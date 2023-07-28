import torch
import torch.nn.functional

from torchsde import sdeint

from typing import Callable
from functools import partial


def triangle_func(t: torch.tensor, a: float = 1e-9, b: float = 1.3e-4, *args, **kwargs) -> torch.tensor:
    """
        defines triangle function with min_beta(0) = a, min_beta(1) = a, max_beta(0.5) = b
        correctly works on t in [0, 1]
    """
    beta = torch.empty_like(t, device=t.device)
    beta[t <= 0.5] = a + 2 * t[t <= 0.5] * (b - a)
    beta[t > 0.5] = 2 * (1 - t[t > 0.5]) * b + (2 * t[t > 0.5] - 1) * a
    return beta


def constant_func(t: torch.Tensor, epsilon: float = 1e-4, *args, **kwargs) -> torch.tensor:
    return torch.empty_like(t, device=t.device).fill_(epsilon)


def get_beta_function(type: str = "triangle", *args, **kwargs) -> Callable:
    if type == "triangle":
        beta_func = partial(triangle_func, *args, **kwargs)
    else:
        beta_func = partial(constant_func, *args, **kwargs)
    return beta_func


class Diffusion:
    def __init__(self, beta_type: str, beta_min: float = 1e-9, beta_max: float = 1.3e-4):
        self.beta_type = beta_type
        self.beta_min, self.beta_max = beta_min, beta_max
        self.beta_func = get_beta_function(beta_type, a=beta_min, b=beta_max, epsilon=beta_max)

    def get_variance(self, t: torch.Tensor):
        sigma1 = torch.empty_like(t, device=t.device) # [0, t]
        sigma2 = torch.empty_like(t, device=t.device) # [t, 1]
        if self.beta_type == "triangle":
            # because beta is piecewise, define separately for each segment [0, 0.5] and [0.5, 1.0]
            sigma1[t <= 0.5] = (t[t <= 0.5] ** 2) * (self.beta_max - self.beta_min) + self.beta_min * t[t <= 0.5]
            sigma1[t > 0.5] = (self.beta_max - self.beta_min) * (2 * t[t > 0.5]  - t[t > 0.5] ** 2 - 0.75)
            sigma2[t <= 0.5] = self.beta_min * (1 - t[t <= 0.5]) + (1 - t[t <= 0.5] ** 2) * (self.beta_max - self.beta_min)
            sigma2[t > 0.5] = (self.beta_max - self.beta_min) * (t[t > 0.5] - 1) ** 2
        else: # assume constant
            sigma1 = self.beta_max * t
            sigma2 = self.beta_max * (1 - t)
        
        return sigma1, sigma2
    
    def get_interpolant(self, t, x0, x1):
        sigma1, sigma2 = self.get_variance(t)
        s = sigma1 + sigma2
        mu = x0 * sigma2 / s + x1 * sigma1  / s
        cov = sigma1 * sigma2 / s

        x_t = mu + cov * torch.randn_like(x0)
        return x_t, sigma2

    def generate(self, net, x0, n_steps: int = 100):
        t = torch.linspace(0., 1., steps=n_steps + 1).to(x0.device)
        
        class SDEWrapper(torch.nn.Module):
            """
                SDE model wrapper for torchsde.sdeint
            """
            def __init__(self, model: torch.nn.Module, beta_f: Callable, sde_type: str = "ito", noise_type: str = "general"):
                super().__init__()
                self.sde_type = sde_type
                self.noise_type = noise_type
                self.model = model
                self.model.eval()
                self.beta_f = beta_f
            
            def f(self, t, y):
                with torch.no_grad():
                    return self.model(y.unsqueeze(1), t).squeeze()

            def g(self, t, y):
                return torch.sqrt(self.beta_f(t.view(1, 1, 1).expand(y.size(0), y.size(-1), 1)))

        sde_model = SDEWrapper(net, self.beta_func)
        return sdeint(sde_model, x0.squeeze(), t, dt=1e-2, dt_min=1e-3)[-1]

    def __call__(self, net, x0, x1):
        """
            loss function
        """

        t = torch.rand(x0.size(0)).to(x0.device)
        while t.dim() < x0.dim():
            t = t[:, None]
        beta_schedule = self.beta_func(t)
        x_t, sigma2 = self.get_interpolant(t, x0, x1)
        pred_vf = net(x_t, t.view(-1))
        true_vf = beta_schedule / sigma2 * (x1 - x_t)

        return torch.nn.functional.mse_loss(pred_vf, true_vf)



if __name__ == "__main__":
    from omegaconf import OmegaConf
    from diffwave import DiffWave

    device = "cuda:0"

    params = OmegaConf.create(
        dict(
            residual_channels=512,
            residual_layers=27,
            n_mels=80,
            unconditional=True,
            dilation_cycle_length=12,
        )
    )
    model = DiffWave(params).to(device)
    print(sum(p.numel() for p in model.parameters()))

    #y = model(torch.randn(2, 32768).to(device), torch.rand(2).to(device))
    #print(y.shape)

    x = torch.randn(2, 1, 32768).to(device)
    y = torch.randn(2, 1, 32768).to(device)


    diffusion = Diffusion("triangle")
    loss = diffusion(model, x0=y, x1=x)
    print(loss)

    s = diffusion.generate(model, torch.randn(2, 1, 32768).to(device))
    print(s.shape)
