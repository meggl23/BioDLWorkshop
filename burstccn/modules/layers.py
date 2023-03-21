import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

import torch.distributions as tdist
from helpers import unravel_index

use_cudnn = False


class Flatten(nn.Module):
    def forward(self, x):
        self.input_size = x.size()
        return x.view(self.input_size[0], -1)

    def backward(self, b_input, b_input_t, b_input_bp, b_input_fa):
        return b_input.view(self.input_size), b_input_t.view(self.input_size), b_input_bp.view(self.input_size), b_input_fa.view(self.input_size)


class _ConvNd(nn.Module):
    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias', 'padding_mode']

    def __init__(self, in_channels, out_channels, in_size, device, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        if transposed:
            self.weight = torch.Tensor(
                in_channels, out_channels // groups, *kernel_size).to(device)
        else:
            self.weight = torch.Tensor(
                out_channels, in_channels // groups, *kernel_size).to(device)

        self.out_size = int((in_size - kernel_size[0] + 2*padding[0])/stride[0] + 1)

        if bias:
            self.bias = torch.Tensor(out_channels).to(device)
        else:
            self.register_parameter('bias', None)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)
