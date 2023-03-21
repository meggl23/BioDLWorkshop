import os
import pprint
import random
import configargparse

import math

import numpy as np

import torch
import wandb

from datasets import get_dataset

from model_trainers import PyTorchTrainer, BurstCCNTrainer#, BurstPropTrainer, BurstCCNTrainer, EDNTrainer, NodePerturbationTrainer


def create_model_trainer(model_type, device):
    # if model_type == 'burstprop':
    #     model_trainer = BurstPropTrainer()
    if model_type == 'burstccn':
         model_trainer = BurstCCNTrainer(device)
    # elif model_type == 'edn':
    #     model_trainer = EDNTrainer()
    elif model_type == 'ann':
    #     # model_trainer = ANNTrainer()
        model_trainer = PyTorchTrainer(device)
    # elif model_type == 'node_perturbation':
    #     model_trainer = NodePerturbationTrainer()

    # else:
    #     raise NotImplementedError()

    return model_trainer


def train(parser=None):
    if wandb.config.force_gpu and not torch.cuda.is_available():
        print("GPU not available and require_gpu is True!")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_trainer = create_model_trainer(wandb.config.model_type, device)

    if parser is None:
        default_config = os.path.join(os.getcwd(), 'configs', 'default.cfg')
        model_config = os.path.join(os.getcwd(), wandb.config.config)
        parser = configargparse.ArgParser(default_config_files=[default_config, model_config])

    model_trainer.add_parser_model_params(parser, wandb.config.model_name)

    model_args, _ = parser.parse_known_args()

    pprint.pprint(vars(model_args))
    # config = argparse.Namespace(**(vars(model_args) | vars(config)))
    wandb.config.update(vars(model_args) | dict(wandb.config))

    torch.manual_seed(wandb.config.seed)
    torch.cuda.manual_seed(wandb.config.seed)
    torch.cuda.manual_seed_all(wandb.config.seed)
    random.seed(wandb.config.seed)
    np.random.seed(wandb.config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if wandb.config.local_state_mode != 'none':
        wandb.config.local_state_dir = os.path.join(os.getcwd(), 'results', wandb.config.run_name, 'variables')
    else:
        wandb.config.local_state_dir = None

    model_trainer.setup(wandb.config)

    train_data_loader, validation_data_loader, test_data_loader = get_dataset(dataset_name=wandb.config.dataset,
                                                                              working_directory=os.getcwd(),
                                                                              batch_size=wandb.config.batch_size,
                                                                              use_validation=wandb.config.use_validation)

    top1_test_error, top5_test_error, test_loss = model_trainer.test(test_data_loader)

    best_top1_test_error, best_top5_test_error = top1_test_error, top5_test_error
    best_top1_test_error_epoch, best_top5_test_error_epoch = 0, 0

    log_dict = {'top1_test_error': top1_test_error,
                'top5_test_error': top5_test_error,
                'test_loss': test_loss,
                'best_top1_test_error': best_top1_test_error,
                'best_top5_test_error': best_top5_test_error,
                'best_top1_test_error_epoch': best_top1_test_error_epoch,
                'best_top5_test_error_epoch': best_top5_test_error_epoch,
                'epoch': 0}

    if wandb.config.use_validation:
        top1_val_error, top5_val_error, val_loss = model_trainer.test(validation_data_loader)

        best_top1_val_error, best_top5_val_error = top1_val_error, top5_val_error
        best_top1_val_error_epoch, best_top5_val_error_epoch = 0, 0

        log_dict.update({'val_loss': val_loss,
                         'top1_val_error': top1_val_error,
                         'top5_val_error': top5_val_error,
                         'best_top1_val_error': best_top1_val_error,
                         'best_top1_val_error_epoch': best_top1_val_error_epoch,
                         'best_top5_val_error_epoch': best_top5_val_error_epoch})

    wandb.log(log_dict)

    for epoch in range(1, wandb.config.n_epochs + 1):
        print(f"\nEpoch {epoch}.")
        top1_train_error, top5_train_error, train_loss = model_trainer.train(train_data_loader, epoch)
        top1_test_error, top5_test_error, test_loss = model_trainer.test(test_data_loader)
        if wandb.config.use_validation:
            top1_val_error, top5_val_error, val_loss = model_trainer.test(validation_data_loader)

        # If model evaluation breaks then stop training
        if math.isnan(top1_test_error):
            break

        if top1_test_error < best_top1_test_error:
            best_top1_test_error = top1_test_error
            best_top1_test_error_epoch = epoch

        if top5_test_error < best_top5_test_error:
            best_top5_test_error = top5_test_error
            best_top5_test_error_epoch = epoch

        if wandb.config.max_stagnant_epochs != -1 and epoch > best_top1_test_error_epoch + wandb.config.max_stagnant_epochs:
            print(f"Test error has not improved for {wandb.config.max_stagnant_epochs} epochs. Stopping...")
            break

        log_dict = {'top1_train_error': top1_train_error,
                    'top5_train_error': top5_train_error,
                    'train_loss': train_loss,
                    'top1_test_error': top1_test_error,
                    'top5_test_error': top5_test_error,
                    'test_loss': test_loss,
                    'best_top1_test_error': best_top1_test_error,
                    'best_top5_test_error': best_top5_test_error,
                    'best_top1_test_error_epoch': best_top1_test_error_epoch,
                    'best_top5_test_error_epoch': best_top5_test_error_epoch,
                    'epoch': epoch}

        if wandb.config.use_validation:
            if top1_val_error < best_top1_val_error:
                best_top1_val_error = top1_val_error
                best_top1_val_error_epoch = epoch

            if top5_val_error < best_top5_val_error:
                best_top5_val_error = top5_val_error
                best_top5_val_error_epoch = epoch

            log_dict.update({'val_loss': val_loss,
                             'top1_val_error': top1_val_error,
                             'top5_val_error': top5_val_error,
                             'best_top1_val_error': best_top1_val_error,
                             'best_top5_val_error': best_top5_val_error,
                             'best_top1_val_error_epoch': best_top1_val_error_epoch,
                             'best_top5_val_error_epoch': best_top5_val_error_epoch})

        wandb.log(log_dict)


if __name__ == '__main__':
    # import pickle
    #
    # file = os.path.join(os.getcwd(), 'results/burstccn_SGD/variables/epoch_1.pkl')
    # with open(file, "rb") as input_file:
    #     a = pickle.load(input_file)
    # print(a)

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--run_name', type=str, help='Name of the run', required=True)
    # parser.add_argument('--model_type', type=str, help='Type of model to use', required=True)
    #
    # parser.add_argument('--seed', type=int, help='The seed number', default=1)
    # parser.add_argument('--working_directory', type=str, default=os.getcwd())
    # parser.add_argument('--dataset', type=str, default='mnist')
    #
    # parser.add_argument('--n_epochs', type=int, help='Number of epochs', default=250)
    # parser.add_argument('--batch_size', type=int, help='Batch size', default=32)
    # parser.add_argument('--use_validation', default=False, help='Whether to the validation set',
    #                     type=lambda x: (str(x).lower() == 'true'))
    #
    # parser.add_argument('--require_gpu', default=False, type=lambda x: (str(x).lower() == 'true'))
    #
    # parser.add_argument('--log_frequency', type=int, help='How often to log states', default=200)
    # parser.add_argument('--local_store_frequency', type=int, help='How often to store states', default=10)
    # parser.add_argument('--log_mode', type=str, default='all')
    # parser.add_argument('--max_stagnant_epochs', type=int, help='Number of epochs to run for with no improvement',
    #                     default=-1)
    # parser.add_argument('--save_local_vars', type=str, default='all')
    #
    # run_args, _ = parser.parse_known_args()
    #
    # wandb.init(project='burstccn', entity='burstyboys', name=run_args.run_name, config=run_args)
    #
    # train(parser)

    import configargparse

    os.chdir(os.path.dirname(__file__))
    os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = "1"

    default_config = os.path.join(os.getcwd(), 'configs', 'default.cfg')

    parser = configargparse.ArgParser(default_config_files=[default_config])
    parser.add_argument('--config', required=True, is_config_file=True, help='Config file path')

    parser.add_argument('--run_name', type=str, help='Name of the run', required=True)
    parser.add_argument('--model_type', type=str, help='Type of model to use', required=True)
    parser.add_argument("--model_name", type=str, help="Which model to use", required=True)

    parser.add_argument('--seed', type=int, help='The seed number', default=1)
    parser.add_argument('--dataset', type=str, help='The seed number', required=True)

    parser.add_argument('--n_epochs', type=int, help='Number of epochs', required=True)
    parser.add_argument('--batch_size', type=int, help='Batch size', required=True)
    parser.add_argument('--use_validation', help='Whether to the validation set', type=lambda x: (str(x).lower() == 'true'), required=True)

    parser.add_argument('--force_gpu', type=lambda x: (str(x).lower() == 'true'), required=True)

    parser.add_argument('--metric_mode', help='Mode for logging metrics.', type=str, required=True)
    parser.add_argument('--metric_frequency', type=int, help='How often to log states', required=True)
    parser.add_argument('--local_state_mode', help='Mode for storing network state locally.', type=str, required=True)
    parser.add_argument('--local_state_frequency', type=int, help='How often to store states.', required=True)
    parser.add_argument('--max_stagnant_epochs', type=int, help='Number of epochs to run for with no improvement', required=True)

    run_args, _ = parser.parse_known_args()

    wandb.init(project='burstccn', entity='burstyboys', name=run_args.run_name, config=run_args)
    # print(run_args)
    try:
        train(parser)
    except KeyboardInterrupt as e:
        print("Interrupted...")
        wandb.finish(exit_code=1)
