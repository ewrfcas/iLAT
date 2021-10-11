import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


def Normalize(in_channels, mode):
    if mode == 'IN':
        return nn.InstanceNorm2d(in_channels)
    elif mode == 'GN':
        return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    else:
        raise NotImplementedError


def nonlinearity(x, mode):
    if mode == 'RELU':
        return torch.relu(x)
    elif mode == 'SWISH':
        return x * torch.sigmoid(x)
    else:
        raise NotImplementedError


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout=0., norm='IN', act='RELU'):
        super().__init__()
        self.act = act
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=1, padding=1)
        self.norm2 = Normalize(out_channels, norm)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                               stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                              stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h, self.act)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h, self.act)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels, norm='IN'):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels, norm)
        self.q = nn.Conv2d(in_channels, in_channels,
                           kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels,
                           kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels,
                           kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels,
                                  kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=4,
                                        stride=2, padding=1)

    def forward(self, x):
        if self.with_conv:
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                        stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample_inout(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsample_inout(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_attention = config['use_attention']
        self.ch = config['ch']
        self.act = config['act']
        self.num_resolutions = len(config['ch_mult'])
        self.num_res_blocks = config['num_res_blocks']
        self.resolution = config['resolution']
        self.give_pre_end = False
        self.resamp_with_conv = True

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = self.ch * config['ch_mult'][-1]
        curr_res = self.resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, config['z_channels'], curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = nn.Conv2d(config['z_channels'], block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=config['dropout'],
                                       norm=config['norm'], act=config['act'])
        if self.use_attention:
            self.mid.attn_1 = AttnBlock(block_in, norm=config['norm'])
        else:
            self.mid.attn_1 = None
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=config['dropout'],
                                       norm=config['norm'], act=config['act'])

        # upsampling
        self.up = nn.ModuleList()
        # in :[512,512,256,256,128]
        # out:[512,256,256,128,128]
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = self.ch * config['ch_mult'][i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=config['dropout'],
                                         norm=config['norm'], act=config['act']))
                block_in = block_out
                if self.use_attention and curr_res in config['attn_resolutions']:
                    attn.append(AttnBlock(block_in, norm=config['norm']))
            up = nn.Module()
            up.block = block
            if self.use_attention:
                up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, self.resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in, config['norm'])
        self.conv_out = nn.Conv2d(block_in, config['out_ch'], kernel_size=3,
                                  stride=1, padding=1)

    def forward(self, z):
        self.last_z_shape = z.shape

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        if self.use_attention:
            h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if self.use_attention and len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h, self.act)
        h = self.conv_out(h)
        h = torch.tanh(h)
        return h
