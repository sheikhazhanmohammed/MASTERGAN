import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from einops import rearrange
from DepthwiseSeparableConvolution import depthwise_separable_conv

class ERAM(nn.Module):
    def __init__(self, channel_begin, dimension):
        super().__init__()
        self.conv = nn.Conv2d(channel_begin, channel_begin, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(dimension)
        
        self.conv1 = nn.Conv2d(channel_begin, channel_begin//2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channel_begin//2, channel_begin, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(channel_begin, channel_begin, kernel_size=3, stride=1, padding=1)

        self.dconv = depthwise_separable_conv(channel_begin, channel_begin, kernel_size = 3, padding = 1, bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        si_ca = self.avgpool(x) + torch.var_mean(x, dim=(2,3))[0].unsqueeze(2).unsqueeze(2)
        mi_ca = self.conv2(self.relu(self.conv1(si_ca)))

        mi_sa = self.conv3(self.relu(self.dconv(x)))

        return self.sigmoid(mi_ca+mi_sa) * x
    
class SelfAttn(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False):
        super(SelfAttn, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj_out = nn.Linear(dim, dim)

    def forward(self, x):
        b, N, c = x.shape

        qkv = self.qkv(x).chunk(3, dim=-1)
        # [b, N, c] -> [b, N, head, c//head] -> [b, head, N, c//head]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)

        # [b, head, N, c//head] * [b, head, N, c//head] -> [b, head, N, N]
        attn = torch.einsum('bijc, bikc -> bijk', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        # [b, head, N, N] * [b, head, N, c//head] -> [b, head, N, c//head] -> [b, N, head, c//head]
        x = torch.einsum('bijk, bikc -> bijc', attn, v)
        x = rearrange(x, 'b i j c -> b j (i c)')
        x = self.proj_out(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, mlp_ratio=4):
        super(Mlp, self).__init__()
        hidden_features = in_features * mlp_ratio

        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, in_features)
        )

    def forward(self, x):
        return self.fc(x)
    
def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size
    Returns:
        windows: (num_windows*b, window_size, window_size, c) [non-overlap]
    """
    return rearrange(x, 'b (h s1) (w s2) c -> (b h w) s1 s2 c', s1=window_size, s2=window_size)

def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image
    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    return rearrange(windows, '(b h w) s1 s2 c -> b (h s1) (w s2) c', b=b, h=h // window_size, w=w // window_size)

class Transformer(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=8, mlp_ratio=4, qkv_bias=False):
        super(Transformer, self).__init__()
        self.window_size = window_size
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttn(dim, num_heads, qkv_bias)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = Mlp(dim, mlp_ratio)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = rearrange(x, 'b c h w -> b h w c')
        b, h, w, c = x.shape

        shortcut = x
        x = self.norm1(x)

        pad_l = pad_t = 0
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        x_windows = window_partition(x, self.window_size)  # nW*B, window_size, window_size, c
        x_windows = rearrange(x_windows, 'B s1 s2 c -> B (s1 s2) c', s1=self.window_size,
                              s2=self.window_size)  # nW*b, window_size*window_size, c

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)  # nW*b, window_size*window_size, c

        # merge windows
        attn_windows = rearrange(attn_windows, 'B (s1 s2) c -> B s1 s2 c', s1=self.window_size, s2=self.window_size)
        x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # b H' W' c

        # reverse cyclic shift
        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()

        x = x + shortcut
        x = x + self.mlp(self.norm2(x))
        return rearrange(x, 'b h w c -> b c h w')
    
class ResBlock(nn.Module):
    def __init__(self, in_features, ratio=4):
        super(ResBlock, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_features, in_features * ratio, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_features * ratio, in_features * ratio, 3, 1, 1, groups=in_features * ratio),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_features * ratio, in_features, 1, 1, 0),
        )

    def forward(self, x):
        return self.net(x) + x

class BaseBlock(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=8, ratios=[1, 2, 2, 4, 4], qkv_bias=False):
        super(BaseBlock, self).__init__()
        self.layers = nn.ModuleList([])
        for ratio in ratios:
            self.layers.append(nn.ModuleList([
                Transformer(dim, num_heads, window_size, ratio, qkv_bias),
                ResBlock(dim, ratio),
                ERAM(dim,128)
            ]))

    def forward(self, x):
        for tblock, rblock ,eram in self.layers:
            x = tblock(x)
            x = rblock(x)
            x = eram(x)
        return x
    
class MobileSR(nn.Module):
    def __init__(self, n_feats=40, n_heads=8, ratios=[4, 2, 2, 2, 4], upscaling_factor=2):
        super(MobileSR, self).__init__()
        self.scale = upscaling_factor
        self.head = nn.Conv2d(1, n_feats, 3, 1, 1)

        self.body = BaseBlock(n_feats, num_heads=n_heads, ratios=ratios)

        self.fuse = nn.Conv2d(n_feats * 2, n_feats, 3, 1, 1)

        if self.scale == 4:
            self.upsapling = nn.Sequential(
                nn.Conv2d(n_feats, n_feats * 4, 1, 1, 0),
                nn.PixelShuffle(2),
                nn.Conv2d(n_feats, n_feats * 4, 1, 1, 0),
                nn.PixelShuffle(2)
            )
        else:
            self.upsapling = nn.Sequential(
                nn.Conv2d(n_feats, n_feats * self.scale * self.scale, 1, 1, 0),
                nn.PixelShuffle(self.scale)
            )

        self.tail = nn.Conv2d(n_feats, 1, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x0 = self.head(x)
        x0 = self.fuse(torch.cat([x0, self.body(x0)], dim=1))
        x0 = self.upsapling(x0)
        x0 = self.tail(self.act(x0))
        x = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
        return (torch.tanh(x0 + x) +1.0)/2.0

class Blocks(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Blocks, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels, features):
        super(Discriminator, self).__init__()
        self.first_layer= nn.Sequential(
            nn.Conv2d(in_channels, features, 3, 2 ,1),
            nn.LeakyReLU(0.2),
        )
        self.Block1 = Blocks(features, features*2, stride=2)
        self.Block2 = Blocks(features*2, features*2, stride=1)
        self.Block3 = Blocks(features*2, features*4, stride=2)
        self.Block4 = Blocks(features*4, features*4, stride=1)
        self.Block5 = Blocks(features*4, features*8, stride=2)
        self.Block6 = Blocks(features*8, features*8, stride=1)
        self.Block7 = Blocks(features*8, features*8, stride=2)
        self.Block8 = Blocks(features*8, features*8, stride=2)
        self.Block9 = nn.Sequential(
            nn.Conv2d(features*8, features*4, 3, 2, 1),
        
            nn.LeakyReLU(0.2),
        )
        self.final_layer = nn.Sequential(
            nn.Linear(features*4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x =  self.first_layer(x)
        x =  self.Block1(x)
        x =  self.Block2(x)
        x =  self.Block3(x)
        x =  self.Block4(x)
        x =  self.Block5(x)
        x =  self.Block6(x)
        x =  self.Block7(x)
        x =  self.Block8(x)
        x = self.Block9(x)
        x = x.view(x.size(0), -1)
        return self.final_layer(x)