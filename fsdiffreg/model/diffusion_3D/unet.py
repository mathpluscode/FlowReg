import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
from einops import rearrange

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# model
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) *
            (-math.log(10000) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)
        return pos_emb


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv3d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv3d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=4, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv3d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, dropout=0):
        super().__init__()
        self.mlp = nn.Sequential(
            Swish(),
            nn.Linear(time_emb_dim, dim_out)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out, dropout=dropout)
        self.res_conv = nn.Conv3d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        if exists(self.mlp):
            h += self.mlp(time_emb)[:, :, None, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=4):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(4, in_channel)
        self.qkv = nn.Conv3d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv3d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, depth, height, width = input.shape
        n_head = self.n_head

        norm = self.norm(input)
        # print(norm.shape)
        qkv = self.qkv(norm)
        # print(qkv.shape)
        q, k, v = rearrange(qkv, 'b (qkv heads c) d h w -> qkv b heads c (d h w)', heads=n_head, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (d h w) -> b (heads c) d h w', heads=n_head, h=height, w=width)
        out = self.out(out)
        return out + input


class SelfAttention_fuse(nn.Module):
    def __init__(self, in_channel, n_head=4):
        super().__init__()

        self.n_head = n_head
        self.norm = nn.GroupNorm(4, in_channel)
        self.out = nn.Conv3d(in_channel, in_channel, 1)
        self.defmgen=nn.Conv3d(in_channel,3,3,padding=1)
        self.nonlinear=nn.Conv3d(3, 3, 3, padding=1)

    def forward(self, q,k,v,size):
        batch, channel, depth, height, width = q.shape

        n_head = self.n_head
        residual=q
        norm_q = self.norm(q)
        norm_k = self.norm(k)
        norm_v = self.norm(v)
        

        qkv=torch.cat([norm_q,norm_k,norm_v],dim=1)
        q, k, v = rearrange(qkv, 'b (qkv heads c) d h w -> qkv b heads c (d h w)', heads=n_head, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (d h w) -> b (heads c) d h w', heads=n_head, h=height, w=width)
        out = self.out(out)
        out=self.defmgen(out+residual)
        out=F.upsample_nearest(out,size)
        out=self.nonlinear(out)
        return out


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, time_emb_dim, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=(8),
        res_blocks=3,
        dropout=0,
        with_time_emb=True,
        image_size=128,
        opt=None
    ):
        super().__init__()

        if with_time_emb:
            time_dim = inner_channel
            self.time_mlp = nn.Sequential(
                TimeEmbedding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            time_dim = None
            self.time_mlp = None
        self.opt=opt
        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size[1]
        downs = [nn.Conv3d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, time_emb_dim=time_dim, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, time_emb_dim=time_dim,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel,
                               time_emb_dim=time_dim, dropout=dropout, with_attn=False)
        ])


        ups_diff = []
        ups_regis = []
        ups_adapt=[]
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                feat_channel=feat_channels.pop()
                ups_diff.append(ResnetBlocWithAttn(
                    pre_channel+feat_channel, channel_mult, time_emb_dim=time_dim, dropout=dropout, with_attn=use_attn))
                regischannel=pre_channel+feat_channel+channel_mult
                ups_regis.append(ResnetBlocWithAttn(
                    regischannel, channel_mult, time_emb_dim=time_dim, dropout=dropout, with_attn=use_attn))
                ups_adapt.append(
                    SelfAttention_fuse(channel_mult)
                )
                pre_channel = channel_mult
            if not is_last:
                ups_adapt.append(nn.Identity())
                ups_diff.append(Upsample(pre_channel))
                ups_regis.append(Upsample(pre_channel))
                now_res = now_res*2

        self.ups_diff = nn.ModuleList(ups_diff)
        self.ups_regis = nn.ModuleList(ups_regis)
        self.ups_adapt=nn.ModuleList(ups_adapt)
        self.final_conv = Block(pre_channel, default(out_channel, in_channel))
        # self.final_attn=SelfAttention_fuse(1)
        self.final_conv_defm = Block(pre_channel+1,3,groups=3)

    def forward(self, x,x_m, time):
        input_size=(x.size(2),x.size(3),x.size(4))
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
        x_1=x
        x_2=x
        defm=[]
        # x_1vis=[]
        for layerd,layerr,layera in zip(self.ups_diff,self.ups_regis,self.ups_adapt):
            if isinstance(layerd, ResnetBlocWithAttn):
                feat=feats.pop()
                x_1 = layerd(torch.cat((x_1, feat), dim=1), t)
                x_2 = layerr(torch.cat((x_2, feat,x_1), dim=1), t)
                defm_=layera(x_2,x_1,x_1,input_size)
                defm.append(defm_)
            else:
                x_1 = layerd(x_1)
                x_2 = layerr(x_2)
        recon=self.final_conv(x_1)
        defm=torch.stack(defm,dim=1)
        defm=torch.cat([defm,self.final_conv_defm(torch.cat((x_2,recon), dim=1)).unsqueeze_(1)],dim=1)
        defm=torch.mean(defm,dim=1)
        return recon,defm

class Dense3DSpatialTransformer(nn.Module):
    """Spatial transformer using grid_sample (device agnostic)."""

    def __init__(self):
        super().__init__()

    def forward(self, src, flow):
        """Apply flow field to source image.

        Args:
            src: Source image [B, C, D, H, W] in [-1, 1]
            flow: Flow field [B, 3, D, H, W] (displacement in voxels)

        Returns:
            Warped image [B, C, D, H, W]
        """
        src = (src[:, :1] + 1) / 2.0  # to [0, 1]

        B, C, D, H, W = src.shape
        device = src.device

        # Create identity grid
        d = torch.linspace(-1, 1, D, device=device)
        h = torch.linspace(-1, 1, H, device=device)
        w = torch.linspace(-1, 1, W, device=device)
        grid_d, grid_h, grid_w = torch.meshgrid(d, h, w, indexing='ij')
        grid = torch.stack([grid_w, grid_h, grid_d], dim=-1)  # [D, H, W, 3]
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1, -1)  # [B, D, H, W, 3]

        # Convert flow from voxels to normalized coordinates
        # flow is [B, 3, D, H, W], need to convert to [B, D, H, W, 3]
        flow = flow.permute(0, 2, 3, 4, 1)  # [B, D, H, W, 3]
        flow_normalized = torch.zeros_like(flow)
        flow_normalized[..., 0] = flow[..., 2] / (W - 1) * 2  # x (width)
        flow_normalized[..., 1] = flow[..., 1] / (H - 1) * 2  # y (height)
        flow_normalized[..., 2] = flow[..., 0] / (D - 1) * 2  # z (depth)

        # Add flow to identity grid
        new_grid = grid + flow_normalized

        # Sample using grid_sample
        return F.grid_sample(src, new_grid, mode='bilinear', padding_mode='border', align_corners=True)