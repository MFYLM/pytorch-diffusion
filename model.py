import math
from typing import Callable, Optional

import torch
import torch.functional as F
import torch.nn as nn
from einops import rearrange


class Residual(nn.Module):
    def __init__(self, fn) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x) + x


def UpSampling(dim: int):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)


def DownSampling(dim: int):
    return nn.Conv2d(dim, dim, 4, 2, 1)


class ConvBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, groups: int = 8, activation: Callable = nn.SiLU()) -> None:
        self.conv = nn.Conv2d(in_dim, out_dim, 3, padding=1)
        self.norm = nn.GroupNorm(groups, out_dim)
        self.act = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(
            half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResnetBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, *, time_embed_dim: Optional[int] = None, groups: int = 8):
        super().__init__()
        self.time_mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, out_dim))
            if time_embed_dim is not None
            else None
        )

        self.block1 = ConvBlock(in_dim, out_dim, groups=groups)
        self.block2 = ConvBlock(out_dim, out_dim, groups=groups)
        self.res_conv = nn.Conv2d(
            in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor, pos_embed: Optional[torch.Tensor] = None):
        y = self.block1(x)
        if self.time_mlp is not None and pos_embed is not None:
            time_embed = self.time_mlp(pos_embed)
            y = y + time_embed
        y = self.block2(y)
        return self.res_conv(x) + y


class ConvNetBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        invert_factor: int = 4, 
        time_embed_dim: Optional[int] = None, 
        norm: bool = True
    ) -> None:
        super().__init__()
        self.time_mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, out_dim))
            if time_embed_dim is not None
            else None
        )
        
        # patchify the stem cell with larger kernel
        self.path_conv = nn.Conv2d(in_dim, in_dim, 7, padding=3, groups=in_dim)
        
        # conv block with fewer activation layers
        self.net = nn.Sequential(
            nn.GroupNorm(1, in_dim) if norm else nn.Identity(),
            nn.Conv2d(in_dim, out_dim * invert_factor, 3, padding=1),
            nn.SiLU(),
            nn.GroupNorm(1, out_dim * invert_factor),
            nn.Conv2d(out_dim * invert_factor, out_dim, 3, padding=1),
        )
        
        # allow for residual connection
        self.res_conv = nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        
    def forward(self, x: torch.Tensor, time_embed=None):
        y = self.path_conv(x)
        if self.time_mlp is not None and time_embed is not None:
            time_embed = self.time_mlp(time_embed)
            print(f"shape checking: time embed -> {time_embed.shape}, y -> {y.shape}")
            y = y + time_embed
        y = self.net(y)
        return self.res_conv(x) + y
        

class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.heads = heads
        self.scale = self.heads**-0.5
        self.dim = dim
        hidden_dim = dim_head * heads
        self.to_q = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.to_k = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.to_v = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)
        
    def forward(self, x: torch.Tensor):
        # input shape: (b, c, h, w)
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q = q * self.scale
        raw_score = q * torch.transpose(k, -1, -2) / math.sqrt(self.dim)
        raw_score = raw_score - raw_score.amax(dim=-1, keepdim=True).detach()
        attn = raw_score.softmax(dim=-1)
        
        weighted_value = attn * v
        return self.to_out(weighted_value)



class LinearAttention(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        
        
        
class UNet(nn.Module):
    def __init__(self, ):
        super().__init__()
        
        
