"""
https://jakiw.com/diffusion_model_intro
https://huggingface.co/blog/annotated-diffusion
https://github.com/acids-ircam/diffusion_models
https://github.com/FilippoMB/Diffusion_models_tutorial
https://github.com/Ryota-Kawamura/How-Diffusion-Models-Work
https://varun-ml.github.io/posts/diffusion-models/diffusion-models-notebooks/
https://github.com/huggingface/diffusion-models-class/blob/main/unit1/01_introduction_to_diffusers.ipynb
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch import einsum
from einops import rearrange, reduce
from einops.layers.torch import Rearrange


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    
    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x
    
def Upsample(dim, dim_out=None):
    dim_out  = dim if dim_out is None else dim_out

    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, dim_out, 3, padding=1),
    )

def Downsample(dim, dim_out=None):
    dim_out  = dim if dim_out is None else dim_out

    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim *4, dim_out, 1) # 1x1 커널
    )

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)
    
    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t):
        """
        Inputs:
          t: Tensor of shape [B] (timestep indices)
        Outputs:
          emb: Tensor of shape [B, embed_dim] (sinusoidal embedding)
        """
        # Compute sinusoidal position embeddings for t
        half_dim = self.embed_dim // 2
        # [0, 1, 2, ..., half_dim-1]
        positions = torch.arange(half_dim, device=t.device).float()
        # Log scale frequencies as in Transformer
        inv_freq = 1.0 / (10000 ** (positions / (half_dim - 1)))
        # Outer product t and inv_freq -> shape [B, half_dim]
        sinusoid_inp = torch.outer(t.float(), inv_freq)  # shape [B, half_dim]
        emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=1)
        return emb  # [B, embed_dim]
    

class WeightStandardizedConv2d(nn.Conv2d):
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = weight.mean(dim=(1,2,3,), keepdim=True)
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

class Block(nn.Module):
    """
    Conv-GroupNorm-SiLU
    """
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x
    
class ResBlock(nn.Module):
    """
    time embedding scale shift Residual Module
    shape를 맞추기 위해 1x1 kernel 짜리 conv 적용
    """
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()

        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if time_emb_dim else None
        )
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if self.mlp and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)
        
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)
    

class SelfAttention2D(nn.Module):
    def __init__(self, dim, heads=4, dim_head=2):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        # spatial dimension을 serialization하도록 하자
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        # softmax 안정화 기법 (log-sum-exp trick)
        # 값을 shift 시켜도 분포는 같은데 분포가 달라지진 않음
        # amax :  특정 dim 방향으로 따져봤을 때 최대값을 뽑도록 함
        # detach : 방향 dim 방향의 최대값 뽑은 텐서는 상수로 쓰도록 함
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # softmax * value
        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)
    
class LinearSelfAttention2D(nn.Module):
    """
    Q(K^T V)를 계산하는데 Q, K 별도에 sfotmax, 결합법칙(associativity)을 적용
    큰 성능 손실이 없다고 알려짐
    길이가 큰 시퀀스에 대해 더 효율적
    """
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), 
                                    nn.GroupNorm(1, dim))
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        # 채널 방향
        q = q.softmax(dim=-2)
        # 데이터 포인트 방향
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)
    
class UNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, time_embed_dim=128):
        super().__init__()
        # Time embedding layer:
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim*4),
            nn.ReLU(),
            nn.Linear(time_embed_dim*4, time_embed_dim)
        )
        # Encoder: Downsample with residual blocks
        self.down1 = ResBlock(in_channels, base_channels)
        self.down2 = ResBlock(base_channels, base_channels*2)
        self.down3 = ResBlock(base_channels*2, base_channels*4)
        # (Optionally more downs and an attention block if needed)
        # Bottleneck
        self.bottleneck = ResBlock(base_channels*4, base_channels*4)
        # Decoder: Upsample with residual blocks, and skip connections
        self.up3 = ResBlock(base_channels*4 + base_channels*4, base_channels*2)  # skip connection adds channels
        self.up2 = ResBlock(base_channels*2 + base_channels*2, base_channels)
        self.up1 = ResBlock(base_channels + base_channels, base_channels)
        # Final output layer
        self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def forward(self, x, t):
        # x: [B, C, H, W], t: [B] (timestep indices)
        # Compute time embedding and expand to image spatial dims if needed
        t_embed = self.time_mlp(t)  # [B, time_embed_dim]
        # Add time embedding to features (could be done via addition in ResBlock)
        # Downsample (encoder)
        d1 = self.down1(x, t_embed)
        d2 = self.down2(d1, t_embed)
        d3 = self.down3(d2, t_embed)
        # Bottleneck
        b  = self.bottleneck(d3, t_embed)
        # Upsample (decoder) with skip connections
        u3 = self.up3(torch.cat([b, d3], dim=1), t_embed)
        u2 = self.up2(torch.cat([u3, d2], dim=1), t_embed)
        u1 = self.up1(torch.cat([u2, d1], dim=1), t_embed)
        # Output
        return self.out_conv(u1)
    

class UNet(nn.Module):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            self_condition=False,
            resnet_block_groups=4,
    ):
        super().__init__()

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        if init_dim is None:
            init_dim = dim
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_class = partial(ResBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_class(dim_in, dim_in, time_emb_dim=time_dim),
                        block_class(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearSelfAttention2D(dim_in))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_class(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, SelfAttention2D(mid_dim)))
        self.mid_block2 = block_class(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_class(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_class(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearSelfAttention2D(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )
        self.out_dim = out_dim if out_dim else channels

        self.final_res_block = block_class(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, time, x, x_self_cond=None):
        """
        time: [B] eg) torch.randint(0, 1, size=(5, ))
        x: [B, C, H, W]
        out: [B, C, H, W]
        """
        if self.self_condition:
            x_self_cond = x_self_cond if x_self_cond else torch.zeros_like(x)
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)
    

if __name__ == "__main__":
    model = Unet(32)
    x = torch.randn(size=(5, 3, 32, 32))
    t = torch.randint(0, 1, size=(5, ))
    y = model(t, x)
    print(y.shape)