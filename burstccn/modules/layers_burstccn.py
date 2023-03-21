import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.utils import _pair


class BurstCCNBaseLayer(nn.Module):
    def to(self, device):
        super().to(device)


class BurstCCNLayerNormalisation(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.out_size = size
        self.normalisation_factor = 1.0

    def forward(self, input, feedforward_noise=None):
        self.normalisation_factor = 1.0 / torch.std(input.view(input.shape[0], -1), dim=1).view(-1, 1, 1, 1)
        # mean = torch.mean(input, dim=(1,2), keepdim=True)
        # var = torch.square(input - mean)

        return ((input - torch.mean(input.view(input.shape[0], -1))) * self.normalisation_factor + 0.5).view(input.shape)

    def backward(self, b_t_input, e_input, b_input_bp, next_delta_fa):
        return b_t_input * self.normalisation_factor, e_input * self.normalisation_factor, b_input_bp * self.normalisation_factor, next_delta_fa * self.normalisation_factor


class BurstCCNOutputLayer(nn.Module):
    def __init__(self, in_features, out_features, p_baseline):
        super(BurstCCNOutputLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.p_baseline = p_baseline
        self.p = self.p_baseline * torch.ones(self.out_features)

        self.weight = nn.Parameter(torch.zeros(out_features, in_features), requires_grad=False)
        self.weight_grad = nn.Parameter(torch.zeros(out_features, in_features), requires_grad=False)
        self.weight_grad_bp = nn.Parameter(torch.zeros(out_features, in_features), requires_grad=False)
        self.weight_grad_fa = nn.Parameter(torch.zeros(out_features, in_features), requires_grad=False)

        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        self.bias_grad = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        self.bias_grad_bp = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        self.bias_grad_fa = nn.Parameter(torch.zeros(out_features), requires_grad=False)

    def forward(self, input, feedforward_noise=None):
        if feedforward_noise is not None:
            assert feedforward_noise != 0.0
            self.input = input + self.forward_noise * torch.randn(input.shape, device=input.device)
        else:
            self.input = input
        self.e = torch.sigmoid(F.linear(self.input, self.weight, self.bias))
        return self.e

    def backward(self, b_input):
        if b_input is None:
            b_input = self.e

        self.p = self.p_baseline
        self.p_t = self.p_baseline * ((b_input - self.e) * (1 - self.e) + 1)

        self.b = self.p * self.e
        self.b_t = self.p_t * self.e

        self.delta = -(self.b_t - self.b)

        self.delta_bp = -(b_input - self.e) * self.e * (1 - self.e)
        self.delta_fa = -(b_input - self.e) * self.e * (1 - self.e)

        batch_size = self.delta.shape[0]

        self.weight_grad.data = self.delta.transpose(0, 1).mm(self.input) / batch_size
        self.bias_grad.data = torch.sum(self.delta, dim=0) / batch_size

        self.weight_grad_bp.data = self.delta_bp.transpose(0, 1).mm(self.input) / batch_size
        self.bias_grad_bp.data = torch.sum(self.delta_bp, dim=0) / batch_size

        self.weight_grad_fa.data = self.delta_fa.transpose(0, 1).mm(self.input) / batch_size
        self.bias_grad_fa.data = torch.sum(self.delta_fa, dim=0) / batch_size

        return self.b_t, self.e, self.delta_bp.mm(self.weight), self.delta_fa


class BurstCCNHiddenLayer(nn.Module):
    def __init__(self, in_features, out_features, next_features, p_baseline, Y_learning, Q_learning, use_layer_norm=False):
        super(BurstCCNHiddenLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.p_baseline = p_baseline

        self.weight = nn.Parameter(torch.zeros(out_features, in_features), requires_grad=False)
        self.weight_grad = nn.Parameter(torch.zeros(out_features, in_features), requires_grad=False)
        self.weight_grad_bp = nn.Parameter(torch.zeros(out_features, in_features), requires_grad=False)
        self.weight_grad_fa = nn.Parameter(torch.zeros(out_features, in_features), requires_grad=False)

        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        self.bias_grad = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        self.bias_grad_bp = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        self.bias_grad_fa = nn.Parameter(torch.zeros(out_features), requires_grad=False)

        self.weight_Y = nn.Parameter(torch.zeros(next_features, out_features), requires_grad=False)
        self.weight_Q = nn.Parameter(torch.zeros(next_features, out_features), requires_grad=False)

        self.Y_learning = Y_learning
        self.Q_learning = Q_learning

        if self.Y_learning:
            self.weight_Y_grad = nn.Parameter(torch.zeros(next_features, out_features), requires_grad=False)

        if self.Q_learning:
            self.weight_Q_grad = nn.Parameter(torch.zeros(next_features, out_features), requires_grad=False)

        self.use_layer_norm = use_layer_norm

    def forward(self, input, feedforward_noise=None):
        if feedforward_noise is not None:
            self.input = input + feedforward_noise * torch.randn(input.shape, device=input.device)
        else:
            self.input = input

        if self.use_layer_norm:
            pre_activation = F.linear(self.input, self.weight)
            self.divisive_factor = torch.std(pre_activation, dim=1, keepdim=True)
            mean = torch.mean(pre_activation, dim=1, keepdim=True)
            self.normalised_pre_activation = (pre_activation - mean) / (self.divisive_factor + 1e-6)
            self.e = torch.sigmoid(self.normalised_pre_activation)
        else:
            pre_activation = F.linear(self.input, self.weight, self.bias)
            self.e = torch.sigmoid(pre_activation)

        return self.e

    def backward(self, b_t_input, e_input, b_input_bp, next_delta_fa):
        self.Y_input = b_t_input.mm(self.weight_Y).detach().cpu()
        self.Q_input = e_input.mm(self.weight_Q).detach().cpu()

        self.apic = b_t_input.mm(self.weight_Y) - e_input.mm(self.weight_Q)

        if self.use_layer_norm:
            self.p_t = torch.sigmoid(4.0 * self.apic * (1 - self.e) / self.divisive_factor)
        else:
            self.p_t = torch.sigmoid(4.0 * self.apic * (1 - self.e))

        self.b = self.p_baseline * self.e
        self.b_t = self.p_t * self.e

        self.delta = -(self.b_t - self.b)

        if self.use_layer_norm:
            self.delta_bp = b_input_bp * self.e * (1 - self.e) / self.divisive_factor
            self.delta_fa = next_delta_fa.mm(self.weight_Y) * self.e * (1 - self.e) / self.divisive_factor
        else:
            self.delta_bp = b_input_bp * self.e * (1 - self.e)
            self.delta_fa = next_delta_fa.mm(self.weight_Y) * self.e * (1 - self.e)

        batch_size = self.delta.shape[0]

        self.weight_grad.data = self.delta.transpose(0, 1).mm(self.input) / batch_size
        self.bias_grad.data = torch.sum(self.delta, dim=0) / batch_size

        self.weight_grad_bp.data = self.delta_bp.transpose(0, 1).mm(self.input) / batch_size
        self.bias_grad_bp.data = torch.sum(self.delta_bp, dim=0) / batch_size

        self.weight_grad_fa.data = self.delta_fa.transpose(0, 1).mm(self.input) / batch_size
        self.bias_grad_fa.data = torch.sum(self.delta_fa, dim=0) / batch_size

        if self.Y_learning:
            self.weight_Y_grad.data = e_input.transpose(0, 1).mm(self.apic) / batch_size

        if self.Q_learning:
            self.weight_Q_grad.data = -e_input.transpose(0, 1).mm(self.apic) / batch_size

        return self.b_t, self.e, self.delta_bp.mm(self.weight), self.delta_fa


class BurstCCNConv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, in_size, kernel_size, stride, padding, dilation,
                 groups, use_bias, padding_mode, p_baseline, Y_learning, Q_learning, use_layer_norm):
        super(BurstCCNConv2dLayer, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        self.p_baseline = p_baseline

        self.Y_learning = Y_learning
        self.Q_learning = Q_learning
        self.use_layer_norm = use_layer_norm

        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels // groups, *kernel_size), requires_grad=False)
        self.weight_grad = nn.Parameter(torch.zeros(out_channels, in_channels // groups, *kernel_size), requires_grad=False)
        self.weight_grad_bp = nn.Parameter(torch.zeros(out_channels, in_channels // groups, *kernel_size), requires_grad=False)
        self.weight_grad_fa = nn.Parameter(torch.zeros(out_channels, in_channels // groups, *kernel_size), requires_grad=False)

        self.out_size = int((in_size - kernel_size[0] + 2 * padding[0]) / stride[0] + 1)

        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=False)
            self.bias_grad = nn.Parameter(torch.zeros(out_channels), requires_grad=False)
            self.bias_grad_bp = nn.Parameter(torch.zeros(out_channels), requires_grad=False)
            self.bias_grad_fa = nn.Parameter(torch.zeros(out_channels), requires_grad=False)

    def forward(self, input, feedforward_noise=None):
        self.input = input

        if self.use_layer_norm:
            pre_activation = F.conv2d(self.input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            self.divisive_factor = torch.std(pre_activation, dim=(1,2,3), keepdim=True)
            mean = torch.mean(pre_activation, dim=(1,2,3), keepdim=True)
            self.normalised_pre_activation = (pre_activation - mean) / (self.divisive_factor + 1e-6)
            self.e = torch.sigmoid(self.normalised_pre_activation)
        else:
            pre_activation = F.conv2d(self.input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            self.e = torch.sigmoid(pre_activation)

        return self.e


class BurstCCNHiddenConv2dLayer(BurstCCNConv2dLayer):
    def __init__(self, in_channels, out_channels, next_channels, in_size, kernel_size, next_kernel_size, stride,
                 next_stride, padding, dilation, groups, use_bias, padding_mode, p_baseline, Y_learning, Q_learning, use_layer_norm):
        super(BurstCCNHiddenConv2dLayer, self).__init__(in_channels, out_channels, in_size, kernel_size, stride,
                                                        padding, dilation,
                                                        groups, use_bias, padding_mode, p_baseline,
                                                        Y_learning, Q_learning, use_layer_norm)

        next_kernel_size = _pair(next_kernel_size)
        self.next_stride = _pair(next_stride)
        self.weight_Y = nn.Parameter(torch.Tensor(next_channels, out_channels // groups, *next_kernel_size), requires_grad=False)

        self.weight_Q = nn.Parameter(torch.Tensor(next_channels, out_channels // groups, *next_kernel_size), requires_grad=False)

        if self.Y_learning:
            self.weight_Y_grad = nn.Parameter(torch.Tensor(next_channels, out_channels // groups, *next_kernel_size), requires_grad=False)

        if self.Q_learning:
            self.weight_Q_grad = nn.Parameter(torch.Tensor(next_channels, out_channels // groups, *next_kernel_size), requires_grad=False)

    def backward(self, b_t_input, e_input, b_input_bp, next_delta_fa):
        Y_input = nn.grad.conv2d_input(self.e.shape, self.weight_Y, b_t_input, self.next_stride,
                                       self.padding, self.dilation, self.groups)
        Q_input = nn.grad.conv2d_input(self.e.shape, self.weight_Q, e_input, self.next_stride,
                                       self.padding, self.dilation, self.groups)

        self.apic = Y_input - Q_input

        if self.use_layer_norm:
            self.p_t = torch.sigmoid(4.0 * self.apic * (1 - self.e) / self.divisive_factor)
        else:
            self.p_t = torch.sigmoid(4.0 * self.apic * (1 - self.e))

        self.b = self.p_baseline * self.e
        self.b_t = self.p_t * self.e

        self.delta = -(self.b_t - self.b)

        if self.use_layer_norm:
            self.delta_bp = b_input_bp * self.e * (1 - self.e) / self.divisive_factor
            self.delta_fa = nn.grad.conv2d_input(self.e.shape, self.weight_Y, next_delta_fa, self.next_stride, self.padding, self.dilation, self.groups) * self.e * (1 - self.e) / self.divisive_factor
        else:
            self.delta_bp = b_input_bp * self.e * (1 - self.e)
            self.delta_fa = nn.grad.conv2d_input(self.e.shape, self.weight_Y, next_delta_fa, self.next_stride, self.padding,
                                                 self.dilation, self.groups) * self.e * (1 - self.e)

        batch_size = self.delta.shape[0]

        self.weight_grad.data = nn.grad.conv2d_weight(self.input, self.weight.shape, self.delta, self.stride,
                                                 self.padding, self.dilation, self.groups) / batch_size
        self.weight_grad_bp.data = nn.grad.conv2d_weight(self.input, self.weight.shape, self.delta_bp, self.stride,self.padding, self.dilation, self.groups) / batch_size
        self.weight_grad_fa.data = nn.grad.conv2d_weight(self.input, self.weight.shape, self.delta_fa, self.stride, self.padding, self.dilation, self.groups) / batch_size

        if self.use_bias:
            self.bias_grad.data = torch.sum(self.delta, dim=[0, 2, 3]) / batch_size
            self.bias_grad_bp.data = torch.sum(self.delta_bp, dim=[0, 2, 3]) / batch_size
            self.bias_grad_fa.data = torch.sum(self.delta_fa, dim=[0, 2, 3]) / batch_size

        if self.Y_learning:
            self.weight_Y_grad.data = e_input.transpose(0, 1).mm(self.apic) / batch_size

        if self.Q_learning:
            self.weight_Q_grad.data = -e_input.transpose(0, 1).mm(self.apic) / batch_size

        feedback_bp = nn.grad.conv2d_input(self.input.shape, self.weight, self.delta_bp, self.stride, self.padding,
                                           self.dilation, self.groups)

        return self.b_t, self.e, feedback_bp, self.delta_fa


class BurstCCNFinalConv2dLayer(BurstCCNConv2dLayer):
    def __init__(self, in_channels, out_channels, next_features, in_size, kernel_size, stride, padding, dilation,
                 groups, use_bias, padding_mode, p_baseline, Y_learning, Q_learning, use_layer_norm):
        super(BurstCCNFinalConv2dLayer, self).__init__(in_channels, out_channels, in_size, kernel_size, stride, padding,
                                                       dilation,
                                                       groups, use_bias, padding_mode, p_baseline, Y_learning, Q_learning, use_layer_norm)

        self.out_features = out_channels * self.out_size ** 2

        self.weight_Y = nn.Parameter(torch.Tensor(next_features, self.out_features), requires_grad=False)
        self.weight_Q = nn.Parameter(torch.Tensor(next_features, self.out_features), requires_grad=False)

        if self.Y_learning:
            self.weight_Y_grad = nn.Parameter(torch.Tensor(next_features, self.out_features), requires_grad=False)

        if self.Q_learning:
            self.weight_Q_grad = nn.Parameter(torch.Tensor(next_features, self.out_features), requires_grad=False)

    def forward(self, input, feedforward_noise=None):
        output = super(BurstCCNFinalConv2dLayer, self).forward(input)
        output_size = output.size()
        return output.view(output_size[0], -1)

    def backward(self, b_t_input, e_input, b_input_bp, next_delta_fa):
        b_input_bp = b_input_bp.view(self.e.shape)

        self.apic = (b_t_input.mm(self.weight_Y) - e_input.mm(self.weight_Q)).view(self.e.shape)

        if self.use_layer_norm:
            self.p_t = torch.sigmoid(4.0 * self.apic * (1 - self.e) / self.divisive_factor)
        else:
            self.p_t = torch.sigmoid(4.0 * self.apic * (1 - self.e))

        self.b = self.p_baseline * self.e
        self.b_t = self.p_t * self.e

        self.delta = -(self.b_t - self.b)

        if self.use_layer_norm:
            self.delta_bp = b_input_bp * self.e * (1 - self.e) / self.divisive_factor
            self.delta_fa = (next_delta_fa.mm(self.weight_Y)).view(self.e.shape) * self.e * (1 - self.e) / self.divisive_factor
        else:
            self.delta_bp = b_input_bp * self.e * (1 - self.e)
            self.delta_fa = (next_delta_fa.mm(self.weight_Y)).view(self.e.shape) * self.e * (1 - self.e)

        batch_size = self.delta.shape[0]

        self.weight_grad.data = nn.grad.conv2d_weight(self.input, self.weight.shape, self.delta, self.stride,
                                                 self.padding, self.dilation, self.groups) / batch_size
        self.weight_grad_bp.data = nn.grad.conv2d_weight(self.input, self.weight.shape, self.delta_bp, self.stride, self.padding, self.dilation, self.groups) / batch_size
        self.weight_grad_fa.data = nn.grad.conv2d_weight(self.input, self.weight.shape, self.delta_fa, self.stride, self.padding, self.dilation, self.groups) / batch_size

        if self.use_bias:
            self.bias_grad.data = torch.sum(self.delta, dim=[0, 2, 3]) / batch_size
            self.bias_grad_bp.data = torch.sum(self.delta_bp, dim=[0, 2, 3]) / batch_size
            self.bias_grad_fa.data = torch.sum(self.delta_fa, dim=[0, 2, 3]) / batch_size

        if self.Y_learning:
            self.weight_Y_grad.data = e_input.transpose(0, 1).mm(self.apic) / batch_size

        if self.Q_learning:
            self.weight_Q_grad.data = -e_input.transpose(0, 1).mm(self.apic) / batch_size

        feedback_bp = nn.grad.conv2d_input(self.input.shape, self.weight, self.delta_bp, self.stride, self.padding,
                                           self.dilation, self.groups)

        return self.b_t, self.e, feedback_bp, self.delta_fa
