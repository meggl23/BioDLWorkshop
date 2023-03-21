from abc import abstractmethod, ABC

import os
import pickle

import numpy as np

import torch
import torch.nn.functional as F

from tqdm import tqdm
import wandb

from modules.networks import TinyImageNetConvNet
from modules.layers_burstccn import BurstCCNLayerNormalisation, BurstCCNConv2dLayer, BurstCCNHiddenLayer

from modules.optimisers import SGDOptimiser
from modules.networks_burstccn import BurstCCN, TinyImagenetConvBurstCCN

from helpers import topk_correct


class ModelTrainer(ABC):
    def __init__(self, device):
        self.device = device

        # Overwritten by parse_model_params
        self.model = None

        # Overwritten by setup
        self.config = None

    @abstractmethod
    def add_parser_model_params(self, parser, model_name):
        parser.add_argument("--lr", type=float, help="Learning rate for hidden layers", required=True)
        parser.add_argument("--momentum", type=float, help="Momentum", required=True)
        parser.add_argument("--weight_decay", type=float, help="Weight decay", required=True)

    @abstractmethod
    def setup(self, config):
        self.config = config

    @abstractmethod
    def update_model_weights(self):
        pass

    @abstractmethod
    def train(self, train_loader, curr_epoch):
        pass

    @abstractmethod
    def test(self, test_loader):
        pass

    @abstractmethod
    def get_metrics(self):
        return dict()

    @abstractmethod
    def get_inner_states(self):
        pass

    @abstractmethod
    def update_local_state_dict(self, batch_index, batch_inputs, batch_targets):
        pass

    def save_local_states(self, curr_epoch):
        with open(os.path.join(self.config.vars_save_dir, f"epoch_{curr_epoch}.pkl"), 'wb') as f:
            pickle.dump(self.local_state_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.local_state_dict = dict()


class PyTorchTrainer(ModelTrainer):
    def __init__(self, device):
        super().__init__(device)

    def add_parser_model_params(self, parser, model_name):
        super().add_parser_model_params(parser, model_name)

        if model_name in ['mnist_ann']:
            parser.add_argument("--n_hidden_layers", type=int,
                                help="Number of hidden layers",
                                required=True)
            parser.add_argument("--n_hidden_units", type=int,
                                help="Number of hidden units in each layer",
                                required=True)

    def setup(self, config):
        super().setup(config)

        if config.model_name == 'mnist_ann':
            raise NotImplementedError(f'Model {config.model_name} is not implemented.')
        elif config.model_name == 'mnist_conv':
            raise NotImplementedError(f'Model {config.model_name} is not implemented.')
        elif config.model_name == 'cifar10_conv':
            raise NotImplementedError(f'Model {config.model_name} is not implemented.')
        elif config.model_name == 'cifar100_conv':
            raise NotImplementedError(f'Model {config.model_name} is not implemented.')
        elif config.model_name == 'tinyimagenet_conv_relu':
            self.model = TinyImageNetConvNet(activation_function='relu')
        elif config.model_name == 'tinyimagenet_conv_sigmoid':
            self.model = TinyImageNetConvNet(activation_function='sigmoid')
        elif config.model_name == 'tinyimagenet_conv_sigmoid_layer_norm':
            self.model = TinyImageNetConvNet(activation_function='sigmoid', use_layer_norm=True)

            def hook_fn(model, input, output):
                model.output = output.detach()

            for layer in self.model.feature_layers:
                if isinstance(layer, torch.nn.ReLU) or isinstance(layer, torch.nn.Sigmoid):
                    layer.register_forward_hook(hook_fn)

        self.model.to(self.device)

        self.optimiser = torch.optim.SGD(self.model.parameters(), lr=config.lr, momentum=config.momentum,
                                         weight_decay=config.weight_decay)

    def update_model_weights(self):
        self.optimiser.step()
        self.optimiser.zero_grad()

    def train(self, train_loader, curr_epoch):
        self.model.train()

        train_loss = 0.0
        top1_correct, top5_correct, total = 0, 0, 0

        progress_bar = tqdm(train_loader)
        for batch_index, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            log_dict = dict()

            log_metrics_this_batch = self.config.metric_mode in ['all', 'only_metrics'] and batch_index % self.config.metric_frequency == 0
            log_inner_states_this_batch = self.config.metric_mode in ['all'] and batch_index % self.config.metric_frequency == 0
            save_local_state_this_batch = self.config.local_state_dir is not None and batch_index % self.config.local_state_frequency == 0

            outputs = self.model(inputs)

            one_hot_targets = F.one_hot(targets, num_classes=self.model.n_outputs).float()
            loss = self.model.loss(outputs, one_hot_targets)

            loss.backward()

            if log_metrics_this_batch:
                log_dict.update(self.get_metrics())

            if log_inner_states_this_batch:
                log_dict.update(self.get_inner_states())

            if log_metrics_this_batch or log_inner_states_this_batch:
                log_dict.update({'batch_index': batch_index})
                wandb.log(log_dict)

            if save_local_state_this_batch:
                self.update_local_state_dict(batch_index, inputs, targets)

            self.update_model_weights()

            train_loss += loss
            total += targets.size(0)

            topk_correct_result = topk_correct(outputs, targets, topk=(1, 5))
            top1_correct += topk_correct_result[0]
            top5_correct += topk_correct_result[1]

            progress_bar.set_description(
                "Train Loss: {:.3f} | Top1 Acc: {:.3f}% ({:d}/{:d}) | Top5 Acc: {:.3f}% ({:d}/{:d})".format(
                    train_loss / (batch_index + 1),
                    100 * top1_correct / total, top1_correct, total,
                    100 * top5_correct / total, top5_correct, total))

        if self.config.local_state_dir is not None:
            self.save_local_states(curr_epoch)

        return 100.0 * (1.0 - top1_correct / total), 100.0 * (1.0 - top5_correct / total), train_loss / len(
            train_loader)

    def test(self, test_loader):
        self.model.eval()

        print(wandb.config)

        test_loss = 0.0
        top1_correct, top5_correct, total = 0, 0, 0

        with torch.no_grad():
            progress_bar = tqdm(test_loader)
            for batch_index, (inputs, targets) in enumerate(progress_bar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                one_hot_targets = F.one_hot(targets, num_classes=self.model.n_outputs).float()
                loss = self.model.loss(outputs, one_hot_targets)

                test_loss += loss
                total += targets.size(0)

                topk_correct_result = topk_correct(outputs, targets, topk=(1, 5))
                top1_correct += topk_correct_result[0]
                top5_correct += topk_correct_result[1]

                progress_bar.set_description(
                    "Test Loss: {:.3f} | Top1 Acc: {:.3f}% ({:d}/{:d}) | Top5 Acc: {:.3f}% ({:d}/{:d})".format(
                        test_loss / (batch_index + 1),
                        100 * top1_correct / total, top1_correct, total,
                        100 * top5_correct / total, top5_correct, total))

        return 100.0 * (1.0 - top1_correct / total), 100.0 * (1.0 - top5_correct / total), test_loss / len(test_loader)

    def get_metrics(self):
        return dict()

    def get_inner_states(self, wandb_histogram_values=True):
        inner_states_dict = dict()
        if self.config.metric_mode in ['all']:
            # Get hidden layer states
            for i in range(len(self.model.feature_layers)-1):
                layer = self.model.feature_layers[i]
                if isinstance(layer, torch.nn.ReLU) or isinstance(layer, torch.nn.Sigmoid):
                    inner_states_dict[f"hidden{i}.output"] = layer.output.flatten().cpu().numpy()

            # Get output layer states
            # output_layer = self.model.feature_layers[-1]
            # inner_states_dict[f"output.output"] = output_layer.output.flatten().cpu().numpy()
            if wandb_histogram_values:
                inner_states_dict = {k: wandb.Histogram(v) for k, v in inner_states_dict.items()}

        return inner_states_dict

    def update_local_state_dict(self, batch_index, batch_inputs, batch_targets):
        pass


class BioModelTrainer(ModelTrainer, ABC):
    def __init__(self, device):
        super(BioModelTrainer, self).__init__(device)
        self.local_state_dict = dict()

    def add_parser_model_params(self, parser, model_name):
        super().add_parser_model_params(parser, model_name)

        parser.add_argument("--feedforward_noise", type=lambda x: None if str(x).lower() == 'none' else x)
        parser.add_argument("--no_teaching_signal", type=lambda x: (str(x).lower() == 'true'), required=True)

    def setup(self, config):
        super().setup(config)

        torch.autograd.set_grad_enabled(False)

    def train(self, train_loader, curr_epoch):
        self.model.train()

        # Reset local state
        self.local_state_dict = dict()

        train_loss = 0.0
        top1_correct, top5_correct, total = 0, 0, 0

        with torch.no_grad():
            progress_bar = tqdm(train_loader)
            for batch_index, (inputs, targets) in enumerate(progress_bar):
                inputs, targets = inputs.to(self.model.device), targets.to(self.model.device)

                log_dict = dict()

                one_hot_targets = F.one_hot(targets, num_classes=self.model.n_outputs).float()

                log_metrics_this_batch = self.config.metric_mode in ['all', 'only_metrics'] and batch_index % self.config.metric_frequency == 0
                log_inner_states_this_batch = self.config.metric_mode in ['all'] and batch_index % self.config.metric_frequency == 0
                save_local_state_this_batch = self.config.local_state_dir is not None and batch_index % self.config.local_state_frequency == 0

                # If computing angles or storing the inner state of the network then do a noiseless forward/backwards pass
                if log_metrics_this_batch or save_local_state_this_batch:
                    self.model.forward(inputs, feedforward_noise=None)
                    self.model.backward(one_hot_targets)

                    if log_metrics_this_batch: log_dict.update(self.get_metrics())
                    # if save_local_state_this_batch: self.save_local_states(curr_epoch)

                # If logging inner states of the network then need a separate noiseless forward/backward pass
                # to remove the teacher signal if it is not supposed to be present.
                if log_inner_states_this_batch:
                    self.model.forward(inputs, feedforward_noise=None)

                    if self.config.no_teaching_signal:
                        self.model.backward(None)
                    else:
                        self.model.backward(one_hot_targets)

                    log_dict.update(self.get_inner_states())

                if log_metrics_this_batch or log_inner_states_this_batch:
                    log_dict.update({'batch_index': batch_index})
                    wandb.log(log_dict)

                # Do the actual forward/backward passes to update
                outputs = self.model.forward(inputs,
                                             feedforward_noise=self.config.feedforward_noise if 'feedforward_noise' in self.config else None)
                if self.config.no_teaching_signal:
                    self.model.backward(None)
                else:
                    self.model.backward(one_hot_targets)

                loss = self.model.loss(outputs, one_hot_targets)

                self.update_model_weights()

                train_loss += loss
                total += targets.size(0)

                topk_correct_result = topk_correct(outputs, targets, topk=(1, 5))
                top1_correct += topk_correct_result[0]
                top5_correct += topk_correct_result[1]

                progress_bar.set_description(
                    "Train Loss: {:.3f} | Top1 Acc: {:.3f}% ({:d}/{:d}) | Top5 Acc: {:.3f}% ({:d}/{:d})".format(
                        train_loss / (batch_index + 1),
                        100 * top1_correct / total, top1_correct, total,
                        100 * top5_correct / total, top5_correct, total))

            if self.config.local_state_dir is not None:
                self.save_local_states(curr_epoch)

        return 100.0 * (1.0 - top1_correct / total), 100.0 * (1.0 - top5_correct / total), train_loss / len(
            train_loader)

    def test(self, test_loader):
        self.model.eval()

        test_loss = 0.0
        top1_correct, top5_correct, total = 0, 0, 0

        with torch.no_grad():
            progress_bar = tqdm(test_loader)
            for batch_index, (inputs, targets) in enumerate(progress_bar):
                inputs, targets = inputs.to(self.model.device), targets.to(self.model.device)

                one_hot_targets = F.one_hot(targets, num_classes=self.model.n_outputs).float()

                outputs = self.model.forward(inputs)
                loss = self.model.loss(outputs, one_hot_targets)

                test_loss += loss
                total += targets.size(0)

                topk_correct_result = topk_correct(outputs, targets, topk=(1, 5))
                top1_correct += topk_correct_result[0]
                top5_correct += topk_correct_result[1]

                progress_bar.set_description(
                    "Test Loss: {:.3f} | Top1 Acc: {:.3f}% ({:d}/{:d}) | Top5 Acc: {:.3f}% ({:d}/{:d})".format(
                        test_loss / (batch_index + 1),
                        100 * top1_correct / total, top1_correct, total,
                        100 * top5_correct / total, top5_correct, total))

        return 100.0 * (1.0 - top1_correct / total), 100.0 * (1.0 - top5_correct / total), test_loss / len(test_loader)


class BurstCCNTrainer(BioModelTrainer):
    def __init__(self, device):
        super().__init__(device)

    def add_parser_model_params(self, parser, model_name):
        super().add_parser_model_params(parser, model_name)
        parser.add_argument("--p_baseline", type=float, help="Baseline burst probability", required=True)
        parser.add_argument('--Y_learning', help="Whether to update Y feedback weights",
                            type=lambda x: (str(x).lower() == 'true'), required=True)
        parser.add_argument('--Q_learning', help="Whether to update Y feedback weights",
                            type=lambda x: (str(x).lower() == 'true'), required=True)

        parser.add_argument("--Y_lr", type=float, required=True)
        parser.add_argument("--Q_lr", type=float, required=True)

        parser.add_argument("--Y_mode", type=str, help="Must be 'tied', 'symmetric_init' or 'random_init",
                            required=True)
        parser.add_argument("--Y_scale", type=float, help="Scale of the feedback weights.", required=True)

        parser.add_argument("--Q_mode", type=str, help="Must be 'tied', 'symmetric_init' or 'random_init",
                            required=True)
        parser.add_argument("--Q_scale", type=float, help="", required=True)

        parser.add_argument("--use_backprop", type=lambda x: (str(x).lower() == 'true'), help="Use backprop update instead")
        parser.add_argument("--use_feedback_alignment", type=lambda x: (str(x).lower() == 'true'), help="Use_feedback_alignment update instead")

        if model_name in ['mnist_burstccn']:
            parser.add_argument("--n_hidden_layers", type=int, help="Number of hidden layers", required=True)
            parser.add_argument("--n_hidden_units", type=int, help="Number of hidden units in each layer",
                                required=True)

    def setup(self, config):
        super().setup(config)
        if config.model_name == 'mnist_burstccn':
            self.model = BurstCCN(n_inputs=784, n_outputs=10,
                                  p_baseline=config.p_baseline, n_hidden_layers=config.n_hidden_layers,
                                  n_hidden_units=config.n_hidden_units, Y_learning=config.Y_learning,
                                  Q_learning=config.Q_learning, Y_mode=config.Y_mode, Q_mode=config.Q_mode,
                                  Y_scale=config.Y_scale,
                                  Q_scale=config.Q_scale,
                                  use_layer_norm=False)
        elif config.model_name == 'mnist_burstccn_layer_norm':
            if config.model_name == 'mnist_burstccn':
                self.model = BurstCCN(n_inputs=784, n_outputs=10,
                                      p_baseline=config.p_baseline, n_hidden_layers=config.n_hidden_layers,
                                      n_hidden_units=config.n_hidden_units, Y_learning=config.Y_learning,
                                      Q_learning=config.Q_learning, Y_mode=config.Y_mode, Q_mode=config.Q_mode,
                                      Y_scale=config.Y_scale,
                                      Q_scale=config.Q_scale,
                                      use_layer_norm=True)
        elif config.model_name == 'tinyimagenet_burstccn_conv':
            self.model = TinyImagenetConvBurstCCN(p_baseline=config.p_baseline, Y_learning=config.Y_learning,
                                                  Q_learning=config.Q_learning, Y_mode=config.Y_mode,
                                                  Q_mode=config.Q_mode,
                                                  Y_scale=config.Y_scale,
                                                  Q_scale=config.Q_scale,
                                      use_layer_norm=False)
        elif config.model_name == 'tinyimagenet_burstccn_conv_layer_norm':
            self.model = TinyImagenetConvBurstCCN(p_baseline=config.p_baseline, Y_learning=config.Y_learning,
                                                  Q_learning=config.Q_learning, Y_mode=config.Y_mode,
                                                  Q_mode=config.Q_mode,
                                                  Y_scale=config.Y_scale,
                                                  Q_scale=config.Q_scale,
                                      use_layer_norm=True)
        else:
            raise NotImplementedError("Invalid model name")

        self.model.to(self.device)

        if config.use_backprop:
            weight_update_parameters = [layer.weight_grad_bp for layer in self.model.classification_layers if not isinstance(layer, BurstCCNLayerNormalisation)]
            bias_update_parameters = [layer.bias_grad_bp for layer in self.model.classification_layers if not isinstance(layer, BurstCCNLayerNormalisation)]
        elif config.use_feedback_alignment:
            weight_update_parameters = [layer.weight_grad_fa for layer in self.model.classification_layers if not isinstance(layer, BurstCCNLayerNormalisation)]
            bias_update_parameters = [layer.bias_grad_fa for layer in self.model.classification_layers if not isinstance(layer, BurstCCNLayerNormalisation)]
        else:
            weight_update_parameters = [layer.weight_grad for layer in self.model.classification_layers if not isinstance(layer, BurstCCNLayerNormalisation)]
            bias_update_parameters = [layer.bias_grad for layer in self.model.classification_layers if not isinstance(layer, BurstCCNLayerNormalisation)]

        self.optimiser = SGDOptimiser(weight_parameters=[layer.weight for layer in self.model.classification_layers if not isinstance(layer, BurstCCNLayerNormalisation)],
                                  bias_parameters=[layer.bias for layer in self.model.classification_layers if not isinstance(layer, BurstCCNLayerNormalisation)],
                                  weight_update_parameters=weight_update_parameters,
                                  bias_update_parameters=bias_update_parameters,
                                  lr=config.lr,
                                  momentum=config.momentum,
                                  weight_decay=config.weight_decay)

        if config.Y_learning:
            weight_Y_update_parameters = [layer.weight_Y_grad for layer in self.model.classification_layers[:-1] if not isinstance(layer, BurstCCNLayerNormalisation)]
            self.Y_optimiser = SGDOptimiser(
                weight_parameters=[layer.weight_Y for layer in self.model.classification_layers[:-1] if not isinstance(layer, BurstCCNLayerNormalisation)],
                weight_update_parameters=weight_Y_update_parameters,
                bias_parameters=None,
                bias_update_parameters=None,
                lr=config.Y_lr)

        if config.Q_learning:
            weight_Q_update_parameters = [layer.weight_Q_grad for layer in self.model.classification_layers[:-1] if not isinstance(layer, BurstCCNLayerNormalisation)]
            self.Q_optimiser = SGDOptimiser(
                weight_parameters=[layer.weight_Q for layer in self.model.classification_layers[:-1] if not isinstance(layer, BurstCCNLayerNormalisation)],
                weight_update_parameters=weight_Q_update_parameters,
                bias_parameters=None,
                bias_update_parameters=None,
                lr=config.Q_lr)

    def update_model_weights(self):
        self.optimiser.step()
        self.optimiser.zero_grad()

        if self.config.Y_learning:
            self.Y_optimiser.step()
            self.Y_optimiser.zero_grad()

        if self.config.Q_learning:
            self.Q_optimiser.step()
            self.Q_optimiser.zero_grad()

        self.model.apply_weight_constraints()

    def get_metrics(self):
        metric_dict = super().get_metrics()
        if self.config.metric_mode in ['all', 'no_layer_states']:
            layer_angles_dict = self.get_layer_angles()
            layer_update_magnitudes_dict = self.get_layer_update_magnitudes()
            loggable_update_magnitudes_dict = {k: wandb.Histogram(np_histogram=(v, np.array(range(len(v) + 1)))) for k, v in layer_update_magnitudes_dict.items()}

            metric_dict = metric_dict | layer_angles_dict | loggable_update_magnitudes_dict

        return metric_dict

    def get_inner_states(self, wandb_histogram_values=True):
        inner_states_dict = dict()
        if self.config.metric_mode in ['all']:
            # Get hidden layer states
            for i in range(len(self.model.classification_layers) - 1):
                layer = self.model.classification_layers[i]
                if not isinstance(layer, BurstCCNLayerNormalisation):
                    inner_states_dict[f"hidden{i + 1}.event_rate"] = layer.e.flatten().cpu().numpy()
                    inner_states_dict[f"hidden{i + 1}.burst_prob"] = layer.p_t.flatten().cpu().numpy()
                    inner_states_dict[f"hidden{i + 1}.burst_rate"] = layer.b_t.flatten().cpu().numpy()

                    if isinstance(layer, (BurstCCNConv2dLayer, BurstCCNHiddenLayer)) and layer.use_layer_norm:
                        inner_states_dict[f"hidden{i + 1}.normalisation_factor"] = layer.divisive_factor.flatten().cpu().numpy()

            # Get output layer states
            output_layer = self.model.classification_layers[-1]
            inner_states_dict["output.event_rate"] = output_layer.e.flatten().cpu().numpy()
            inner_states_dict["output.burst_prob"] = output_layer.p_t.flatten().cpu().numpy()
            inner_states_dict["output.burst_rate"] = output_layer.b_t.flatten().cpu().numpy()
            if wandb_histogram_values:
                inner_states_dict = {k: wandb.Histogram(v) for k, v in inner_states_dict.items()}

        return inner_states_dict

    def get_layer_angles(self):
        W_Y_angles = self.model.weight_angles_W_Y()
        W_Y_angles_dict = {f'angle_W_Y ({i})': W_Y_angles[i] for i in range(len(W_Y_angles))}

        bp_angles = self.model.bp_angles()
        bp_angles_dict = {f'angle_bp ({i})': bp_angles[i] for i in range(len(bp_angles))}

        fa_angles = self.model.fa_angles()
        fa_angles_dict = {f'angle_fa ({i})': fa_angles[i] for i in range(len(fa_angles))}

        fa_to_bp_angles = self.model.fa_to_bp_angles()
        fa_to_bp_angles_dict = {f'angle_fa_to_bp ({i})': fa_to_bp_angles[i] for i in range(len(fa_to_bp_angles))}

        global_bp_angle = self.model.bp_angles(global_angle=True)
        global_fa_angle = self.model.fa_angles(global_angle=True)
        global_fa_to_bp_angle = self.model.fa_to_bp_angles(global_angle=True)
        global_angles_dict = {'global_bp_angle': global_bp_angle,
                              'global_fa_angle': global_fa_angle,
                              'global_fa_to_bp_angle': global_fa_to_bp_angle}

        layer_angles_dict = W_Y_angles_dict | bp_angles_dict | fa_angles_dict | fa_to_bp_angles_dict | global_angles_dict
        return layer_angles_dict

    def get_layer_update_magnitudes(self):
        bp_grad_magnitudes = self.model.bp_grad_magnitudes()
        grad_magnitudes = self.model.grad_magnitudes()

        layer_update_magnitudes_dict = {'bp_grad_magnitudes': bp_grad_magnitudes,
                                        'grad_magnitudes': grad_magnitudes}
        return layer_update_magnitudes_dict

    def update_local_state_dict(self, batch_index, batch_inputs, batch_targets):
        pass
