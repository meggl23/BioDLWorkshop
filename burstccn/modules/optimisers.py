from abc import ABC, abstractmethod

import torch


class Optimiser(ABC):
    def __init__(self, weight_parameters, bias_parameters, weight_update_parameters, bias_update_parameters):
        self.weight_parameters = weight_parameters
        self.bias_parameters = bias_parameters

        self.weight_grads = weight_update_parameters
        self.bias_grads = bias_update_parameters

    @abstractmethod
    def step(self):
        pass


def SGDOptimiser(weight_parameters, bias_parameters, weight_update_parameters, bias_update_parameters, lr, momentum=0.0, weight_decay=0.0):
    if momentum == 0.0:
        return SGDNoMomentum(weight_parameters, bias_parameters, weight_update_parameters, bias_update_parameters, lr, weight_decay)
    else:
        return SGDMomentum(weight_parameters, bias_parameters, weight_update_parameters, bias_update_parameters, lr, momentum, weight_decay)


class SGDNoMomentum(Optimiser):
    def __init__(self, weight_parameters, bias_parameters, weight_update_parameters, bias_update_parameters, lr, weight_decay):
        super().__init__(weight_parameters, bias_parameters, weight_update_parameters, bias_update_parameters)
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self):
        for weight, weight_grad in zip(self.weight_parameters, self.weight_grads):
            weight += -self.lr * weight_grad - self.weight_decay * weight

        if self.bias_parameters is not None:
            for bias, bias_grad in zip(self.bias_parameters, self.bias_grads):
                bias += -self.lr * bias_grad - self.weight_decay * bias

    def zero_grad(self):
        for weight_grad in self.weight_grads:
            weight_grad *= 0.0

        if self.bias_parameters is not None:
            for bias_grad in self.bias_grads:
                bias_grad *= 0.0


class SGDMomentum(Optimiser):
    def __init__(self, weight_parameters, bias_parameters, weight_update_parameters, bias_update_parameters, lr, momentum=0.0, weight_decay=0.0):
        super().__init__(weight_parameters, bias_parameters, weight_update_parameters, bias_update_parameters)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.weight_m_buffers = [torch.zeros(weight.shape, device=weight.device) for weight in weight_parameters]
        self.bias_m_buffers = [torch.zeros(bias.shape, device=bias.device) for bias in bias_parameters]

    def step(self):
        # Update weight buffers
        for weight, weight_m, weight_grad in zip(self.weight_parameters, self.weight_m_buffers, self.weight_grads):
            weight_m.mul_(self.momentum).add_(weight_grad, alpha=1.0)

        # Update weights
        for weight, weight_m in zip(self.weight_parameters, self.weight_m_buffers):
            weight += -self.lr * weight_m - self.weight_decay * weight

        if self.bias_parameters is not None:
            # Update bias buffer
            for bias, bias_m, bias_grad in zip(self.bias_parameters, self.bias_m_buffers, self.bias_grads):
                bias_m.mul_(self.momentum).add_(bias_grad, alpha=1.0)

            # Update biases
            for bias, bias_m in zip(self.bias_parameters, self.bias_m_buffers):
                bias += -self.lr * bias_m - self.weight_decay * bias

    def zero_grad(self):
        for weight_grad in self.weight_grads:
            weight_grad *= 0.0

        if self.bias_parameters is not None:
            for bias_grad in self.bias_grads:
                bias_grad *= 0.0