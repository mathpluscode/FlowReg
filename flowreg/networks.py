"""FlowReg network architecture with flow matching."""

import math
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch.distributions.normal import Normal


class FlowReg(nn.Module):

    def __init__(self, in_chs=1, enc_chs=8, dec_chs=16, t_chs=64, use_checkpoint=True):
        super().__init__()
        self.encoder = ConvEncoder(in_chs, enc_chs, use_checkpoint)
        self.decoder = MLPDecoder(enc_chs, dec_chs, t_chs, use_checkpoint)
        self.time_embed = TimestepEmbedding(t_chs)

    def forward(self, fixed, moving, ddf, t):
        # fixed/moving: [B, 1, D, H, W], ddf: [B, 3, D, H, W], t: [B]
        x_fix = self.encoder(fixed)     # list of [B, C_i, D_i, H_i, W_i]
        x_mov = self.encoder(moving)
        t_emb = self.time_embed(t)         # [B, t_chs]
        flow = self.decoder(x_fix, x_mov, t_emb, ddf)  # [B, 3, D, H, W]
        return flow


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep encoding + MLP."""

    def __init__(self, out_dim, freq_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    @staticmethod
    def sinusoidal(t, dim, max_period=10000):
        # t: [B] -> [B, dim]
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
        args = t[:, None] * freqs[None]  # [B, half]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [B, dim]

    def forward(self, t):
        # t: [B] -> [B, out_dim]
        return self.mlp(self.sinusoidal(t, self.freq_dim))


class ConvEncoder(nn.Module):

    def __init__(self, in_chs, chs, use_checkpoint=False):
        super().__init__()
        self.block1 = ConvBlock(in_chs, chs, use_checkpoint)
        self.block2 = ConvBlock(chs, chs * 2, use_checkpoint)
        self.block3 = ConvBlock(chs * 2, chs * 4, use_checkpoint)
        self.block4 = ConvBlock(chs * 4, chs * 8, use_checkpoint)
        self.downsample = nn.AvgPool3d(2, stride=2)

    def forward(self, x):
        # x: [B, in_chs, D, H, W]
        x1 = self.block1(x)                    # [B, C, D, H, W]
        x2 = self.block2(self.downsample(x1))   # [B, 2C, D/2, H/2, W/2]
        x3 = self.block3(self.downsample(x2))   # [B, 4C, D/4, H/4, W/4]
        x4 = self.block4(self.downsample(x3))   # [B, 8C, D/8, H/8, W/8]
        return [x1, x2, x3, x4]



class MLPDecoder(nn.Module):

    def __init__(self, in_chs, chs, t_chs=64, use_checkpoint=False):
        super().__init__()
        # First-pass CMW blocks
        self.cmw1 = CMWBlock(in_chs, chs, use_checkpoint)
        self.cmw2 = CMWBlock(in_chs * 2, chs * 2, use_checkpoint)
        self.cmw3 = CMWBlock(in_chs * 4, chs * 4, use_checkpoint)
        self.cmw4 = CMWBlock(in_chs * 8, chs * 8, use_checkpoint)

        # Second-pass: merge with upsampled features
        self.merge1 = CMWBlock(chs, chs, use_checkpoint)
        self.merge2 = CMWBlock(chs * 2, chs * 2, use_checkpoint)
        self.merge3 = CMWBlock(chs * 4, chs * 4, use_checkpoint)

        # Time embedding projections (one per encoder scale)
        self.t_proj1 = nn.Sequential(nn.SiLU(), nn.Linear(t_chs, in_chs))
        self.t_proj2 = nn.Sequential(nn.SiLU(), nn.Linear(t_chs, in_chs * 2))
        self.t_proj3 = nn.Sequential(nn.SiLU(), nn.Linear(t_chs, in_chs * 4))
        self.t_proj4 = nn.Sequential(nn.SiLU(), nn.Linear(t_chs, in_chs * 8))

        self.up1 = PatchExpand(chs * 2)
        self.up2 = PatchExpand(chs * 4)
        self.up3 = PatchExpand(chs * 8)

        self.resize = ResizeTransformer(factor=2, mode='trilinear')
        self.resize_down = ResizeTransformer(factor=0.5, mode='trilinear')
        self.stn = SpatialTransformer(mode='bilinear')

        self.head1 = RegHead(chs)
        self.head2 = RegHead(chs * 2)
        self.head3 = RegHead(chs * 4)
        self.head4 = RegHead(chs * 8)

    def forward(self, x_fix, x_mov, t_emb, ddf_raw):
        # x_fix/x_mov: list of [B, C_i, D_i, H_i, W_i], t_emb: [B, t_chs]
        # ddf_raw: [B, 3, D, H, W] — raw input DDF
        x_fix1, x_fix2, x_fix3, x_fix4 = x_fix
        x_mov1, x_mov2, x_mov3, x_mov4 = x_mov

        # Add time embedding to moving features at each scale
        x_mov1 = x_mov1 + self.t_proj1(t_emb)[:, :, None, None, None]
        x_mov2 = x_mov2 + self.t_proj2(t_emb)[:, :, None, None, None]
        x_mov3 = x_mov3 + self.t_proj3(t_emb)[:, :, None, None, None]
        x_mov4 = x_mov4 + self.t_proj4(t_emb)[:, :, None, None, None]

        # Downsample raw DDF to each scale
        ddf1 = ddf_raw
        ddf2 = self.resize_down(ddf1)
        ddf3 = self.resize_down(ddf2)
        ddf4 = self.resize_down(ddf3)

        # Level 4 (coarsest): [B, 8C, D/8, H/8, W/8]
        x_mov4 = self.stn(x_mov4, ddf4)
        x4 = self.cmw4(x_fix4, x_mov4)
        flow = self.head4(x4) + ddf4  # [B, 3, D/8, H/8, W/8]

        # Level 3: [B, 4C, D/4, H/4, W/4]
        flow_up = self.resize(flow)  # [B, 3, D/4, H/4, W/4]
        x_mov3 = self.stn(x_mov3, flow_up)
        x3 = self.merge3(self.cmw3(x_fix3, x_mov3), self.up3(x4))
        flow = self.head3(x3) + flow_up

        # Level 2: [B, 2C, D/2, H/2, W/2]
        flow_up = self.resize(flow)
        x_mov2 = self.stn(x_mov2, flow_up)
        x2 = self.merge2(self.cmw2(x_fix2, x_mov2), self.up2(x3))
        flow = self.head2(x2) + flow_up

        # Level 1 (finest): [B, C, D, H, W]
        flow_up = self.resize(flow)
        x_mov1 = self.stn(x_mov1, flow_up)
        x1 = self.merge1(self.cmw1(x_fix1, x_mov1), self.up1(x2))
        flow = self.head1(x1) + flow_up  # [B, 3, D, H, W]

        return flow


class SpatialTransformer(nn.Module):

    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode

    def _get_grid(self, shape, device):
        if not hasattr(self, '_grid') or self._grid.shape[2:] != shape or self._grid.device != device:
            vectors = [torch.arange(0, s) for s in shape]
            grids = torch.meshgrid(vectors, indexing='ij')
            self._grid = torch.stack(grids).unsqueeze(0).float().to(device)  # [1, 3, D, H, W]
        return self._grid

    def forward(self, src, flow):
        # src: [B, C, D, H, W], flow: [B, 3, D, H, W]
        shape = flow.shape[2:]
        grid = self._get_grid(shape, flow.device)

        new_locs = grid + flow  # [B, 3, D, H, W]
        # Normalize to [-1, 1] without in-place ops (preserves autograd graph)
        scale = torch.tensor([2.0 / (s - 1) for s in shape], device=flow.device).view(1, 3, 1, 1, 1)
        new_locs = new_locs * scale - 1.0

        new_locs = new_locs.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]  # [B, D, H, W, 3]
        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class ResizeTransformer(nn.Module):

    def __init__(self, factor, mode='trilinear'):
        super().__init__()
        self.factor = factor
        self.mode = mode

    def forward(self, x):
        # x: [B, 3, D, H, W] -> [B, 3, D*f, H*f, W*f]
        if self.factor < 1:
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x
        elif self.factor > 1:
            x = self.factor * x
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
        return x


class ConvBlock(nn.Module):

    def __init__(self, in_chs, out_chs, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.conv1 = nn.Conv3d(in_chs, out_chs, kernel_size=3, padding='same')
        self.conv2 = nn.Conv3d(out_chs, out_chs, kernel_size=3, padding='same')
        self.norm1 = nn.InstanceNorm3d(out_chs)
        self.norm2 = nn.InstanceNorm3d(out_chs)
        self.lrelu = nn.LeakyReLU(0.2)

    def _forward(self, x):
        x = self.norm1(self.lrelu(self.conv1(x)))
        x = self.norm2(self.lrelu(self.conv2(x)))
        return x

    def forward(self, x):
        if self.use_checkpoint and x.requires_grad:
            return cp.checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


class RegHead(nn.Module):

    def __init__(self, in_chs):
        super().__init__()
        self.conv = nn.Conv3d(in_chs, 3, kernel_size=3, padding='same')
        self.conv.weight = nn.Parameter(Normal(0, 1e-5).sample(self.conv.weight.shape))
        self.conv.bias = nn.Parameter(torch.zeros(self.conv.bias.shape))

    def forward(self, x):
        # x: [B, C, D, H, W] -> [B, 3, D, H, W]
        return self.conv(x)


class PatchExpand(nn.Module):

    def __init__(self, chs):
        super().__init__()
        self.up = nn.ConvTranspose3d(chs, chs // 2, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(chs // 2)

    def forward(self, x):
        # x: [B, C, D, H, W] -> [B, C/2, 2D, 2H, 2W]
        x = self.up(x)
        x = einops.rearrange(x, 'b c d h w -> b d h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b d h w c -> b c d h w')


class CMWBlock(nn.Module):
    """Correlation-aware multi-window MLP block (same as CorrMLP)."""

    def __init__(self, in_chs, chs, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.corr = Correlation(max_disp=1)
        self.proj = nn.Conv3d(in_chs * 2 + 27, chs, kernel_size=3, padding='same')
        self.mlp = MultiWindowMLP(chs)
        self.rcab = RCAB(chs)

    def _forward(self, x1, x2):
        x = self.proj(torch.cat([x1, self.corr(x1, x2), x2], dim=1))  # [B, chs, D, H, W]
        shortcut = x
        x = x.permute(0, 2, 3, 4, 1)  # [B, D, H, W, C]
        x = self.mlp(x)
        x = self.rcab(x)
        x = x.permute(0, 4, 1, 2, 3)  # [B, C, D, H, W]
        return x + shortcut

    def forward(self, x1, x2):
        if self.use_checkpoint and x1.requires_grad:
            return cp.checkpoint(self._forward, x1, x2, use_reentrant=False)
        return self._forward(x1, x2)


class MultiWindowMLP(nn.Module):
    """Multi-window gated MLP with learned reweighting."""

    def __init__(self, chs):
        super().__init__()
        self.norm = nn.LayerNorm(chs)
        self.gmlp3 = WindowGMLP(win_size=3, chs=chs)
        self.gmlp5 = WindowGMLP(win_size=5, chs=chs)
        self.gmlp7 = WindowGMLP(win_size=7, chs=chs)
        self.reweight = FeedForward(chs, chs // 4, chs * 3)
        self.out_proj = nn.Linear(chs, chs)

    def forward(self, x):
        # x: [B, D, H, W, C]
        n, h, w, d, c = x.shape
        x_norm = self.norm(x)

        x3 = self.gmlp3(x_norm)  # [B, D, H, W, C]
        x5 = self.gmlp5(x_norm)
        x7 = self.gmlp7(x_norm)

        # Learned reweighting: pool -> predict 3 softmax weights
        a = (x3 + x5 + x7).permute(0, 4, 1, 2, 3).flatten(2).mean(2)  # [B, C]
        a = self.reweight(a).reshape(n, c, 3).permute(2, 0, 1).softmax(dim=0)  # [3, B, C]
        a = a.unsqueeze(2).unsqueeze(2).unsqueeze(2)  # [3, B, 1, 1, 1, C]

        out = x3 * a[0] + x5 * a[1] + x7 * a[2]  # [B, D, H, W, C]
        return self.out_proj(out) + x


class WindowGMLP(nn.Module):
    """Windowed gated MLP for local spatial mixing."""

    def __init__(self, win_size, chs, factor=2):
        super().__init__()
        self.win = win_size
        self.norm = nn.LayerNorm(chs)
        self.in_proj = nn.Linear(chs, chs * factor)
        self.gelu = nn.GELU()
        self.sgu = SpatialGatingUnit(chs * factor, n=win_size ** 3)
        self.out_proj = nn.Linear(chs * factor // 2, chs)

    def forward(self, x):
        # x: [B, D, H, W, C]
        _, h, w, d, _ = x.shape

        # Pad to multiple of window size
        pad_h = (self.win - h % self.win) % self.win
        pad_w = (self.win - w % self.win) % self.win
        pad_d = (self.win - d % self.win) % self.win
        x = F.pad(x, (0, 0, 0, pad_d, 0, pad_w, 0, pad_h))

        gh, gw, gd = x.shape[1] // self.win, x.shape[2] // self.win, x.shape[3] // self.win
        ps = (self.win, self.win, self.win)
        x = split_images(x, ps)  # [B, G, P, C] where G=gh*gw*gd, P=win^3

        shortcut = x
        x = self.out_proj(self.sgu(self.gelu(self.in_proj(self.norm(x)))))  # [B, G, P, C]
        x = x + shortcut

        x = unsplit_images(x, (gh, gw, gd), ps)  # [B, D', H', W', C]
        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            x = x[:, :h, :w, :d, :].contiguous()
        return x


class SpatialGatingUnit(nn.Module):

    def __init__(self, chs, n):
        super().__init__()
        self.linear = nn.Linear(n, n)
        self.norm = nn.LayerNorm(chs // 2)

    def forward(self, x):
        # x: [B, G, P, C]
        c = x.size(-1) // 2
        u, v = torch.split(x, c, dim=-1)  # [B, G, P, C/2] each
        v = self.norm(v)
        v = self.linear(v.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)  # mix along P dim
        return u * (v + 1.0)  # [B, G, P, C/2]


class RCAB(nn.Module):
    """Residual channel attention block."""

    def __init__(self, chs, reduction=4):
        super().__init__()
        self.norm = nn.LayerNorm(chs)
        self.conv1 = nn.Conv3d(chs, chs, kernel_size=3, padding='same')
        self.conv2 = nn.Conv3d(chs, chs, kernel_size=3, padding='same')
        self.lrelu = nn.LeakyReLU(0.2)
        self.se = ChannelAttention(chs, reduction)

    def forward(self, x):
        # x: [B, D, H, W, C]
        shortcut = x
        x = self.norm(x).permute(0, 4, 1, 2, 3)  # [B, C, D, H, W]
        x = self.lrelu(self.conv1(x))
        x = self.conv2(x).permute(0, 2, 3, 4, 1)  # [B, D, H, W, C]
        return self.se(x) + shortcut


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation channel attention."""

    def __init__(self, chs, reduction=4):
        super().__init__()
        self.fc1 = nn.Conv3d(chs, chs // reduction, kernel_size=1)
        self.fc2 = nn.Conv3d(chs // reduction, chs, kernel_size=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, D, H, W, C]
        w = x.permute(0, 4, 1, 2, 3)                          # [B, C, D, H, W]
        w = w.mean(dim=(2, 3, 4), keepdim=True)                # [B, C, 1, 1, 1]
        w = self.sigmoid(self.fc2(self.relu(self.fc1(w))))      # [B, C, 1, 1, 1]
        return x * w.permute(0, 2, 3, 4, 1)                    # [B, D, H, W, C]


class FeedForward(nn.Module):

    def __init__(self, in_features, hidden_features, out_features, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Correlation(nn.Module):

    def __init__(self, max_disp=1):
        super().__init__()
        self.max_disp = max_disp
        self.pad = nn.ConstantPad3d(max_disp, 0)

    def forward(self, x1, x2):
        # x1, x2: [B, C, D, H, W] -> [B, (2r+1)^3, D, H, W]
        x2 = self.pad(x2)
        r = self.max_disp
        offsets = torch.arange(0, 2 * r + 1)
        ox, oy, oz = torch.meshgrid(offsets, offsets, offsets, indexing='ij')
        _, _, D, H, W = x1.shape
        return torch.cat([
            torch.mean(x1 * x2[:, :, dx:dx+D, dy:dy+H, dz:dz+W], 1, keepdim=True)
            for dx, dy, dz in zip(ox.reshape(-1), oy.reshape(-1), oz.reshape(-1))
        ], 1)


def split_images(x, patch_size):
    """Partition volume into non-overlapping patches."""
    gh = x.shape[1] // patch_size[0]
    gw = x.shape[2] // patch_size[1]
    gd = x.shape[3] // patch_size[2]
    return einops.rearrange(
        x, 'n (gh fh) (gw fw) (gd fd) c -> n (gh gw gd) (fh fw fd) c',
        gh=gh, gw=gw, gd=gd, fh=patch_size[0], fw=patch_size[1], fd=patch_size[2],
    )


def unsplit_images(x, grid_size, patch_size):
    """Merge patches back into volume."""
    return einops.rearrange(
        x, 'n (gh gw gd) (fh fw fd) c -> n (gh fh) (gw fw) (gd fd) c',
        gh=grid_size[0], gw=grid_size[1], gd=grid_size[2],
        fh=patch_size[0], fw=patch_size[1], fd=patch_size[2],
    )
