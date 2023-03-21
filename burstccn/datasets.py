import sys
import os

import math
import random

import torch
import torchvision
import PIL.Image
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torchvision import datasets
import torch.utils.data as data

from modules.networks import ANN


import numpy as np
# datasets.MNIST.resources = [
#         (
#         'https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
#         (
#         'https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'),
#         ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'),
#         ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c')
#     ]


def get_rgb_data_mean_std(dataset):
    data_r = np.dstack([dataset[i][:, :, 0] for i in range(len(dataset))])
    data_g = np.dstack([dataset[i][:, :, 1] for i in range(len(dataset))])
    data_b = np.dstack([dataset[i][:, :, 2] for i in range(len(dataset))])
    mean = np.mean(data_r) / 255., np.mean(data_g) / 255., np.mean(data_b) / 255.
    std = np.std(data_r) / 255., np.std(data_g) / 255., np.std(data_b) / 255.

    return mean, std


def get_dataset(dataset_name, working_directory, batch_size, use_validation):
    data_dir = os.path.join(working_directory, 'Data')

    if dataset_name == 'mnist':
        train_data_loader, validation_data_loader, test_data_loader = get_mnist_dataset(data_dir,
                                                                                        train_batch_size=batch_size,
                                                                                        use_validation=use_validation)
    elif dataset_name == 'cifar10':
        train_data_loader, validation_data_loader, test_data_loader = get_cifar10_dataset(data_dir,
                                                                                          train_batch_size=batch_size,
                                                                                          use_validation=use_validation)

    elif dataset_name == 'cifar100':
        train_data_loader, validation_data_loader, test_data_loader = get_cifar100_dataset(data_dir,
                                                                                           train_batch_size=batch_size,
                                                                                           use_validation=use_validation)

    elif dataset_name == 'tinyimagenet':
        train_data_loader, validation_data_loader, test_data_loader = get_tiny_imagenet_dataset(
            'Data/tiny-imagenet-200',
            train_batch_size=batch_size,
            use_validation=use_validation)

    elif dataset_name == 'imagenet':
        train_data_loader, validation_data_loader, test_data_loader = get_imagenet_dataset(
            'D:\imagenet_object_localization_patched2019\ILSVRC\Data\CLS-LOC',
            train_batch_size=batch_size,
            use_validation=use_validation)

    else:
        raise NotImplementedError()

    return train_data_loader, validation_data_loader, test_data_loader


def get_mnist_dataset(data_dir, train_batch_size, test_batch_size=1000, use_validation=False, train_subset_size=None):
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    full_train_set = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform_train)

    if use_validation:
        train_set, validation_set = torch.utils.data.random_split(full_train_set, [50000, 10000])
    else:
        train_set, validation_set = full_train_set, None

        # train_set, validation_set = torch.utils.data.random_split(full_train_set, [5000, 55000])[0], None

    test_set = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform_test)

    kwargs = {'num_workers': 0, 'pin_memory': True}

    if train_subset_size is not None:
        train_set.data = train_set.data[:train_subset_size]

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True, **kwargs)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=test_batch_size, shuffle=False, **kwargs) if validation_set is not None else None
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False, **kwargs)

    return train_loader, validation_loader, test_loader


def get_cifar10_dataset(data_dir, train_batch_size, test_batch_size=1000, use_validation=False):

    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    full_train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)

    if use_validation:
        train_set, validation_set = torch.utils.data.random_split(full_train_set, [40000, 10000])
    else:
        train_set, validation_set = full_train_set, None

    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    kwargs = {'num_workers': 0, 'pin_memory': True}

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True, **kwargs)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=test_batch_size, shuffle=False, **kwargs) if validation_set is not None else None
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False, **kwargs)

    print("CIFAR-10: " + str(get_rgb_data_mean_std(full_train_set.data)))

    return train_loader, validation_loader, test_loader


def get_cifar100_dataset(data_dir, train_batch_size, test_batch_size=1000, use_validation=False):

    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    full_train_set = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)

    if use_validation:
        train_set, validation_set = torch.utils.data.random_split(full_train_set, [40000, 10000])
    else:
        train_set, validation_set = full_train_set, None

    test_set = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)

    kwargs = {'num_workers': 0, 'pin_memory': True}

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True, **kwargs)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=test_batch_size, shuffle=False, **kwargs) if validation_set is not None else None
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False, **kwargs)

    return train_loader, validation_loader, test_loader

def get_xor_dataset():
    X_train = torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y_train = torch.Tensor([0, 1, 1, 0]).view(-1, 1)#.to(torch.int64)

    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)

    return train_loader, train_loader, train_loader