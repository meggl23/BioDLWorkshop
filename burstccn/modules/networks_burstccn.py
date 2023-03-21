import copy
import math

import wandb

import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from helpers import similarity

from modules.layers import Flatten
from modules.layers_burstccn import BurstCCNHiddenLayer, BurstCCNOutputLayer, BurstCCNHiddenConv2dLayer, BurstCCNFinalConv2dLayer, BurstCCNConv2dLayer


class BaseBurstCCN(nn.Module):
    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def get_angle_list(self, index_range, get_matrix1, get_matrix2):
        angle_list = []
        for i in index_range:
            matrix1 = get_matrix1(i, self.classification_layers).flatten()
            matrix2 = get_matrix2(i, self.classification_layers).flatten()
            angle_list.append((180 / math.pi) * torch.acos(similarity(matrix1, matrix2)).item())

        return angle_list

    def get_global_angle(self, index_range, get_matrix1, get_matrix2):
        full_matrix1 = torch.cat([get_matrix1(i, self.classification_layers).flatten() for i in index_range])
        full_matrix2 = torch.cat([get_matrix2(i, self.classification_layers).flatten() for i in index_range])

        return (180 / math.pi) * torch.acos(similarity(full_matrix1, full_matrix2)).item()

    def weight_angles_W_Y(self, global_angle=False):
        index_range = range(1, len(self.classification_layers))
        get_matrix_W = lambda i, layers: layers[i].weight
        get_matrix_Y = lambda i, layers: layers[i-1].weight_Y
        return self.get_global_angle(index_range, get_matrix_W, get_matrix_Y) if global_angle else self.get_angle_list(index_range, get_matrix_W, get_matrix_Y)

    def weight_angles_Q_Y(self, global_angle=False):
        index_range = range(len(self.classification_layers) - 1)
        get_matrix_Y = lambda i, layers: layers[i].weight_Y
        get_matrix_Q = lambda i, layers: layers[i].weight_Q
        return self.get_global_angle(index_range, get_matrix_Y, get_matrix_Q) if global_angle else self.get_angle_list(index_range, get_matrix_Y, get_matrix_Q)

    def bp_angles(self, global_angle=False):
        index_range = range(len(self.classification_layers))
        get_matrix_grad = lambda i, layers: layers[i].weight_grad
        get_matrix_grad_bp = lambda i, layers: layers[i].weight_grad_bp
        return self.get_global_angle(index_range, get_matrix_grad, get_matrix_grad_bp) if global_angle else self.get_angle_list(index_range, get_matrix_grad, get_matrix_grad_bp)

    def fa_angles(self, global_angle=False):
        index_range = range(len(self.classification_layers))
        get_matrix_grad = lambda i, layers: layers[i].weight_grad
        get_matrix_grad_fa = lambda i, layers: layers[i].weight_grad_fa
        return self.get_global_angle(index_range, get_matrix_grad, get_matrix_grad_fa) if global_angle else self.get_angle_list(index_range, get_matrix_grad, get_matrix_grad_fa)

    def fa_to_bp_angles(self, global_angle=False):
        index_range = range(len(self.classification_layers))
        get_matrix_grad_bp = lambda i, layers: layers[i].weight_grad_bp
        get_matrix_grad_fa = lambda i, layers: layers[i].weight_grad_fa
        return self.get_global_angle(index_range, get_matrix_grad_bp, get_matrix_grad_fa) if global_angle else self.get_angle_list(index_range, get_matrix_grad_bp, get_matrix_grad_fa)

    def get_layer_update_magnitudes(self, get_updates):
        return [torch.mean(torch.abs(get_updates(layer))).item() for layer in self.classification_layers]

    def grad_magnitudes(self):
        get_grad = lambda layer: layer.weight_grad
        return self.get_layer_update_magnitudes(get_grad)

    def bp_grad_magnitudes(self):
        get_grad_bp = lambda layer: layer.weight_grad_bp
        return self.get_layer_update_magnitudes(get_grad_bp)

    def get_layer_normalisation_factors(self):
        normalisation_factors = []
        for layer in self.classification_layers:
            if isinstance(layer, (BurstCCNConv2dLayer, BurstCCNHiddenLayer)) and layer.use_layer_norm:
                normalisation_factors.append(layer.divisive_factor)
        return normalisation_factors


class BurstCCN(BaseBurstCCN):
    def __init__(self, n_inputs, n_outputs, p_baseline, n_hidden_layers, n_hidden_units, Y_learning, Q_learning, Y_mode,
                 Q_mode, Y_scale, Q_scale, use_layer_norm):
        super(BurstCCN, self).__init__()

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.p_baseline = p_baseline

        assert Y_mode in ['tied', 'symmetric_init', 'random_init']
        self.Y_mode = Y_mode
        self.Y_scale = Y_scale

        assert Q_mode in ['tied', 'symmetric_init', 'random_init', 'W_symmetric_init']
        self.Q_mode = Q_mode
        self.Q_scale = Q_scale

        self.use_layer_norm = use_layer_norm

        self.feature_layers = nn.ModuleList()
        self.feature_layers.append(Flatten())

        self.classification_layers = nn.ModuleList()

        if n_hidden_layers == 0:
            self.classification_layers.append(BurstCCNOutputLayer(n_inputs, n_outputs, p_baseline))
        elif n_hidden_layers == 1:
            self.classification_layers.append(
                BurstCCNHiddenLayer(n_inputs, n_hidden_units, n_outputs, p_baseline, Y_learning, Q_learning, use_layer_norm))
            self.classification_layers.append(BurstCCNOutputLayer(n_hidden_units, n_outputs, p_baseline))
        else:
            self.classification_layers.append(
                BurstCCNHiddenLayer(n_inputs, n_hidden_units, n_hidden_units, p_baseline, Y_learning, Q_learning, use_layer_norm))

            for i in range(1, n_hidden_layers - 1):
                self.classification_layers.append(
                    BurstCCNHiddenLayer(n_hidden_units, n_hidden_units, n_hidden_units, p_baseline, Y_learning,
                                        Q_learning, use_layer_norm))

            self.classification_layers.append(
                BurstCCNHiddenLayer(n_hidden_units, n_hidden_units, n_outputs, p_baseline, Y_learning, Q_learning, use_layer_norm))
            self.classification_layers.append(BurstCCNOutputLayer(n_hidden_units, n_outputs, p_baseline))

        self._initialize_weights()

    def forward(self, x, feedforward_noise=None):
        for layer in self.feature_layers:
            x = layer.forward(x)

        for layer in self.classification_layers:
            x = layer.forward(x, feedforward_noise=feedforward_noise)
        return x

    def backward(self, target):
        burst_rate, event_rate, feedback_bp, feedback_fa = self.classification_layers[-1].backward(target)
        for i in range(len(self.classification_layers) - 2, -1, -1):
            burst_rate, event_rate, feedback_bp, feedback_fa = self.classification_layers[i].backward(burst_rate,
                                                                                                      event_rate,
                                                                                                      feedback_bp,
                                                                                                      feedback_fa)

    def apply_weight_constraints(self):
        for i in range(len(self.classification_layers)):
            if i < len(self.classification_layers) - 1:
                if self.Y_mode == 'tied':
                    self.classification_layers[i].weight_Y.data = self.Y_scale * copy.deepcopy(
                        self.classification_layers[i + 1].weight)
                if self.Q_mode == 'tied':
                    self.classification_layers[i].weight_Q.data = self.p_baseline * copy.deepcopy(
                        self.classification_layers[i].weight_Y)

    def loss(self, output, target):
        return F.mse_loss(output, target)

    def log_inner_states(self):
        # Log hidden
        for i in range(len(self.classification_layers) - 1):
            layer = self.classification_layers[i]
            wandb.log({f"hidden{i}.event_rate": wandb.Histogram(layer.e.flatten().cpu().numpy()),
                       f"hidden{i}.apical": wandb.Histogram(layer.apic.flatten().cpu().numpy()),
                       f"hidden{i}.burst_prob": wandb.Histogram(layer.p_t.flatten().cpu().numpy()),
                       f"hidden{i}.burst_rate": wandb.Histogram(layer.b_t.flatten().cpu().numpy())}, commit=False)

        # Log output
        output_layer = self.classification_layers[-1]
        wandb.log({"output.event_rate": wandb.Histogram(output_layer.e.flatten().cpu().numpy()),
                   "output.burst_prob": wandb.Histogram(output_layer.p_t.flatten().cpu().numpy()),
                   "output.burst_rate": wandb.Histogram(output_layer.b_t.flatten().cpu().numpy())}, commit=False)

    def _initialize_weights(self):
        self._initialize_ff_weights()
        self._initialize_secondary_weights()

    def _initialize_ff_weights(self):
        module_list = list(self.modules())

        for module_index, m in enumerate(module_list):
            if isinstance(m, BurstCCNHiddenLayer) or isinstance(m, BurstCCNOutputLayer):
                nn.init.xavier_normal_(m.weight, gain=3.6)
                # nn.init.normal_(m.weight, 0.0, 0.1)
                nn.init.constant_(m.bias, 0)
                # m.weight.data.normal_(0, 0.1)
                # stdv = 1. / math.sqrt(m.weight.size(1))
                # m.bias.data.uniform_(-stdv, stdv)

    def _initialize_secondary_weights(self):
        layer_index = 0
        module_list = list(self.modules())

        for module_index, m in enumerate(module_list):
            if isinstance(m, BurstCCNHiddenLayer):
                if self.Y_mode == 'tied' or self.Y_mode == 'symmetric_init':
                    m.weight_Y.data = self.Y_scale * copy.deepcopy(module_list[module_index + 1].weight.detach())
                elif self.Y_mode == 'random_init':
                    init.normal_(m.weight_Y, 0, self.Y_scale)

                if self.Q_mode == 'tied' or self.Q_mode == 'symmetric_init':
                    assert self.p_baseline == 0.5
                    assert self.Q_scale == 1.0
                    m.weight_Q.data = self.p_baseline * copy.deepcopy(m.weight_Y.data)
                elif self.Q_mode == 'W_symmetric_init':
                    m.weight_Q.data = self.p_baseline * self.Q_scale * copy.deepcopy(
                        module_list[module_index + 1].weight.detach())
                elif self.Q_mode == 'random_init':
                    init.normal_(m.weight_Q, 0, self.Q_scale)

                layer_index += 1

    def _initialize_weights_from_list(self, ff_weights_list, ff_bias_list):
        layer_index = 0
        for m in self.modules():
            if isinstance(m, BurstCCNHiddenLayer) or isinstance(m, BurstCCNOutputLayer):
                m.weight.data = copy.deepcopy(ff_weights_list[layer_index].detach())
                m.bias.data = copy.deepcopy(ff_bias_list[layer_index].detach())
                layer_index += 1

        self._initialize_secondary_weights()
