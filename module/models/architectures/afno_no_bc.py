import torch
import torch.nn as nn

from module.models.blocks import AFNO2DBlock
from module.models.normalization import ChannelStandardScaler


class TC_AFNO_NoBC(nn.Module):
    def __init__(
        self,
        num_vars=11,
        num_times=3,
        H=100,
        W=100,
        num_blocks=4,
        film_zdim=64,
        x_mean=None,
        x_std=None,
        y_mean=None,
        y_std=None,
        return_physical: bool = False,
        channels=64,
        hidden_factor=2,
        mlp_expansion_ratio=4,
        stem_channels=64,
    ):
        super().__init__()
        in_channels_main = num_vars * num_times

        self.num_vars = num_vars
        self.num_times = num_times
        self.film_zdim = film_zdim

        self.stem = nn.Conv2d(in_channels_main, stem_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [
                AFNO2DBlock(
                    channels=channels,
                    hidden_factor=hidden_factor,
                    mlp_expansion_ratio=mlp_expansion_ratio,
                )
                for _ in range(num_blocks)
            ]
        )
        self.out_conv = nn.Conv2d(channels, num_vars, kernel_size=1)

        self.x_scaler = None
        if (x_mean is not None) and (x_std is not None):
            self.x_scaler = ChannelStandardScaler(x_mean, x_std)

        self.return_physical = return_physical
        self.y_scaler = None
        if (y_mean is not None) and (y_std is not None):
            self.y_scaler = ChannelStandardScaler(y_mean, y_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, V, H, W]
        """
        B, T, V, H, W = x.shape
        assert T == self.num_times and V == self.num_vars, "Unexpected input shape."

        x = x.reshape(B, T * V, H, W)
        x = self.stem(x)

        for blk in self.blocks:
            x = blk(x)

        out = self.out_conv(x)

        if self.return_physical and (self.y_scaler is not None):
            out = self.y_scaler.denorm(out)

        return out
