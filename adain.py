import torch
import torch.nn as nn

class AdaIn2d(nn.Module):
    def __init__(self, content: torch.Tensor, style: torch.Tensor):
        super(AdaIn2d, self).__init__()
        self.mu_content = content.mean((1, 2), keepdim=True)
        self.beta_style = style.mean((1, 2), keepdim=True)

        self.sigma_content = safe_std(content, (1, 2), keepdim=True)
        self.gamma_style = safe_std(style, (1, 2), keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sd = (x - self.mu_content) / self.sigma_content
        return self.gamma_style * sd + self.beta_style

class MyAdaIn(nn.Module):
    def __init__(self, content: torch.Tensor, style: torch.Tensor):
        super(MyAdaIn, self).__init__()
        self.mu_content = content.mean((0, 1), keepdim=True)
        self.beta_style = style.mean(1, keepdim=True)

        self.sigma_content = safe_std(content, (0, 1), keepdim=True)
        self.gamma_style = safe_std(style, 1, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sd = (x - self.mu_content) / self.sigma_content
        return self.gamma_style * sd + self.beta_style

def safe_std(
    x: torch.Tensor,
    dim: int,
    keepdim: bool = False,
    epsilon: float = 1e-5,
) -> torch.Tensor:

    return x.var(dim, keepdim=keepdim).add(epsilon).sqrt()
