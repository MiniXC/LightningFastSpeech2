import torch
from torch import nn


class LayerNorm2(nn.Module):
    """Layer norm for the 2nd dimension of the input using torch primitive.
    Args:
        channels (int): number of channels (2nd dimension) of the input.
        eps (float): to prevent 0 division
    Shapes:
        - input: (B, C, T)
        - output: (B, C, T)
    """

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = torch.nn.functional.layer_norm(
            x, (self.channels,), self.gamma, self.beta, self.eps
        )
        return x.transpose(1, -1)
