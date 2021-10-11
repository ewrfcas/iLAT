import torch
import torch.nn as nn
import torch.nn.functional as F


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


def plm_conv(conv, x, mask=None, kernel_size=3, stride=1, padding=1):
    if mask is None:
        return conv(x)
    else:
        f1 = conv(x)  # original feature
        f2 = conv(x * (1 - mask))  # masked feature
        # get leak_area
        conv_weights = torch.ones(1, 1, kernel_size, kernel_size).to(dtype=x.dtype, device=x.device)
        leak_area = F.conv2d(mask, conv_weights, stride=stride, padding=padding)
        if stride > 1:
            mask = F.max_pool2d(mask, kernel_size=stride, stride=stride)
        leak_area[leak_area > 0] = 1
        leak_area = torch.clamp(leak_area - mask, 0, 1)
        # leak_area uses masked feature
        out = f1 * (1 - leak_area) + f2 * leak_area

        return out


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

    def forward(self, x, mask=None):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h, self.act)
        h = plm_conv(self.conv1, h, mask=mask, kernel_size=3, stride=1, padding=1)

        h = self.norm2(h)
        h = nonlinearity(h, self.act)
        h = self.dropout(h)
        h = plm_conv(self.conv2, h, mask=mask, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = plm_conv(self.conv_shortcut, x, mask=mask, kernel_size=3, stride=1, padding=1)
            else:
                x = plm_conv(self.nin_shortcut, x, mask=mask, kernel_size=1, stride=1, padding=0)

        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=4,
                                        stride=2, padding=1)

    def forward(self, x, mask=None):
        if self.with_conv:
            x = plm_conv(self.conv, x, mask=mask, kernel_size=4, stride=2, padding=1)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class Downsample_inout(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=4, stride=2, padding=1)

    def forward(self, x, mask):
        x = plm_conv(self.conv, x, mask=mask, kernel_size=4, stride=2, padding=1)
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

        self.conv_in = nn.Conv2d(self.in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = config['resolution']
        in_ch_mult = (1,) + tuple(config['ch_mult'])
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in = self.ch * in_ch_mult[i_level]
            block_out = self.ch * config['ch_mult'][i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=config['dropout'],
                                         norm=config['norm'], act=config['act']))
                block_in = block_out

            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, self.resamp_with_conv)
                curr_res = curr_res // 2
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

    def forward(self, x, mask=None):
        # downsampling
        h = plm_conv(self.conv_in, x, mask=mask, kernel_size=3, stride=1, padding=1)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h, mask)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h, mask)
            if mask is not None:
                mask = F.max_pool2d(mask, kernel_size=int(mask.shape[2] / h.shape[2]),
                                    stride=int(mask.shape[2] / h.shape[2]))

        # middle
        h = self.mid.block_1(h, mask)
        h = self.mid.block_2(h, mask)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h, self.act)
        h = plm_conv(self.conv_out, h, mask=mask, kernel_size=3, stride=1, padding=1)
        return h
