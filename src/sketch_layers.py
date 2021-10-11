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


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ch = config['ch']
        self.act = config['act']
        self.num_resolutions = len(config['ch_mult'])
        self.num_res_blocks = config['num_res_blocks']
        self.resolution = config['resolution']
        self.in_channels = config['in_channels']
        self.resamp_with_conv = True

        self.conv_in = nn.Sequential(nn.ReflectionPad2d(6),
                                     nn.Conv2d(self.in_channels, self.ch, kernel_size=13, stride=1, padding=0))

        # in : [1,2,4,6]
        # out: [2,4,6,8]
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions - 1):
            block = nn.ModuleList()
            down = nn.Module()
            block_in = self.ch * config['ch_mult'][i_level]
            block_out = self.ch * config['ch_mult'][i_level + 1]
            down.downsample = Downsample_inout(block_in, block_out)
            block_in = block_out
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=config['dropout'],
                                         norm=config['norm'], act=config['act']))
            down.block = block
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=config['dropout'],
                                       norm=config['norm'], act=config['act'])
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=config['dropout'],
                                       norm=config['norm'], act=config['act'])

        # end
        self.norm_out = Normalize(block_in, config['norm'])
        self.conv_out = nn.Conv2d(block_in, 2 * config['z_channels'] if config['double_z'] else config['z_channels'],
                                  kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # downsampling
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions - 1):
            h = self.down[i_level].downsample(h)
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h, self.act)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
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
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=config['dropout'],
                                       norm=config['norm'], act=config['act'])

        # upsampling
        self.up = nn.ModuleList()
        for i_level in range(1, self.num_resolutions):
            block = nn.ModuleList()
            block_out = self.ch * config['ch_mult'][-i_level - 1]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=config['dropout'],
                                         norm=config['norm'], act=config['act']))
            up = nn.Module()
            up.block = block
            up.upsample = Upsample_inout(block_in, block_out)
            curr_res = curr_res * 2
            block_in = block_out
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
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions - 1)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
            h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h, self.act)
        h = self.conv_out(h)
        h = torch.tanh(h)
        return h
