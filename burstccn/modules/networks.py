from abc import abstractmethod
import math

import numpy as np

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from modules.layers import Flatten

from helpers import similarity

import copy


class ANN(nn.Module):
    def __init__(self, n_inputs, n_hidden_layers, n_hidden_units, n_outputs, device):
        super(ANN, self).__init__()

        self.linear_layers = nn.ModuleList()

        if n_hidden_layers == 0:
            self.linear_layers.append(nn.Linear(n_inputs, n_outputs))
            # self.classification_layers.append(nn.Sigmoid())
        else:
            # self.layers.append(ContinuousBurstCCNHiddenLayer(n_inputs, n_hidden_units, n_hidden_units, p_baseline, device))
            self.linear_layers.append(nn.Linear(n_inputs, n_hidden_units))
            # self.classification_layers.append(nn.Sigmoid())

            for i in range(1, n_hidden_layers):
                self.linear_layers.append(nn.Linear(n_hidden_units, n_hidden_units))
                # self.classification_layers.append(nn.Sigmoid())

            self.linear_layers.append(nn.Linear(n_hidden_units, n_outputs))
            # self.classification_layers.append(nn.Sigmoid())


        all_layers = []
        for l in self.linear_layers:
            all_layers.append(l)
            all_layers.append(nn.Sigmoid())

        for l in self.linear_layers:
            # 3x3
            # nn.init.xavier_normal_(l.weight, gain=3.6)
            # 5x5
            # nn.init.xavier_normal_(l.weight, gain=3.5)
            # CatCam
            nn.init.xavier_normal_(l.weight, gain=1.0)
            nn.init.constant_(l.bias, 0)
        # self.layers = nn.Sequential(*all_layers)
        # self.layers.to(device)

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        for layer in self.linear_layers:
            x = layer(x)
            # print(x)
            x = torch.sigmoid(x)
            layer.activity = x

        return x
        # return self.layers(x)

    def set_weights(self, weight_list, bias_list):
        for i, (weights, biases) in enumerate(zip(weight_list, bias_list)):
            self.linear_layers[i].weight.data = weights.detach().clone()
            self.linear_layers[i].bias.data = biases.detach().clone()


class BioNetwork(nn.Module):

    @abstractmethod
    def get_layer_states(self):
        pass

    @abstractmethod
    def get_weight_angles(self):
        pass

    @abstractmethod
    def get_gradient_magnitudes(self):
        pass