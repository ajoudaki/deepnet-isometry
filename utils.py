from scipy.integrate import quad
import math
import numpy as np
from scipy.special import expit, softmax
from scipy.special import hermitenorm
import torch
from torchvision import datasets, transforms
from scipy.special import roots_hermite
import torch


import pandas as pd
import numpy as np
import torch
import torch.nn as nn

import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt

from itertools import product
from collections import namedtuple

from scipy.special import hermitenorm
from statsmodels.distributions.empirical_distribution import ECDF

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plot_metrics(metrics):
    metrics['epoch'] = metrics['epoch'] + 1
    plt.figure(figsize=(15, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    sns.lineplot(x='epoch', y='loss', hue='stage', data=metrics)
    plt.title('Model loss through epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    # Accuracy plot
    plt.subplot(1, 2, 2)
    sns.lineplot(x='epoch', y='accuracy', hue='stage', data=metrics)
    plt.title('Model accuracy through epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')

    plt.show()


OPTIMIZERS = {
    'SGD': optim.SGD,
    'Adam': optim.Adam,
}

CRITERIONS = {
    'CrossEntropy': nn.CrossEntropyLoss()
}
    
ACTIVATIONS = {
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
    'selu': nn.SELU(),
    'leaky_relu': nn.LeakyReLU(negative_slope=.5),
    'identity': nn.Identity()
}


       
def dict_configs(d, return_dict=False):
    for k,v in d.items():
        if not hasattr(v, '__len__'):
            d[k] = [v]
    Config = namedtuple('Config', d.keys())
    for vcomb in product(*d.values()):
        c = dict(zip(d.keys(), vcomb))
        if return_dict:
            yield Config(**c), c
        else:
            yield Config(**c)
              
def compute_beta0(act):
    x = torch.randn(1000000)
    c0 = act(x).mean()
    c1 = (act(x)*x).mean()
    csum2 = (act(x)**2).mean()
    beta0 = 2 - c1**2 / (csum2 - c0**2)
    # c0, c1, csum2, beta0
    return beta0.item()

def hermit_coefs(f, n):
    coefs = np.zeros(n+1)
    for n in range(n+1):
        h = hermitenorm(n=n)
        c = quad(lambda x: h(x)*f(x)*np.exp(-x**2/2), -np.inf, np.inf)[0]/math.factorial(n)/np.sqrt(2*np.pi)
        coefs[n] = c
    return coefs


def create_input(dist, n, d, degeneracy):
    if dist=='norm':
        X = torch.randn(n, d) 
    elif dist=='unif':
        X = torch.rand(n, d) - .5 
    u, s, v = torch.svd(X)
    s[:1] *= degeneracy
    X = u @ torch.diag(s) @ v.t()
    X = X / (X**2).mean()**.5
    return X


def Hermite_act(coefs):
    coefs = np.array(coefs)
    coefs = coefs / np.sum(coefs)
    def f(x):
        r = 0
        for n,c in enumerate(coefs):
            r += c*torch.special.hermite_polynomial_he(x,n=n) / np.sqrt(math.factorial(n)/np.sqrt(2*np.pi))
        return r
    return f

def get_activation_function(name):
    """Returns the requested activation function."""
    if name == 'sigmoid':
        return torch.sigmoid
    elif name == 'relu':
        return torch.nn.functional.relu
    elif name == 'step':
        return lambda x: torch.heaviside(x,torch.tensor(0.0))
    elif name == 'leaky_relu':
        return torch.nn.functional.leaky_relu
    elif name == 'elu':
        return torch.nn.ELU()
    elif name == 'selu':
        return torch.selu
    elif name == 'tanh':
        return torch.tanh
    elif name == 'sin':
        return torch.sin
    elif name == 'exp':
        return torch.exp
    elif name == 'softmax':
        return lambda x: torch.nn.functional.softmax(x, dim=-1)
    elif name == 'prelu':
        return torch.nn.functional.prelu
    elif name == 'celu':
        return torch.nn.functional.celu
    elif name == 'gelu':
        return torch.nn.functional.gelu
    elif name == 'silu':
        return silu
    elif name == 'identity':
        return nn.Identity()
    elif name == 'rbf':
        return lambda x: torch.exp(x-2)
    elif name[:2] == 'He':
        n = int(name[2:])
        return lambda x: torch.special.hermite_polynomial_he(x,n=n)
    else:
        raise ValueError('Unsupported activation function: %s' % name)
        

def isometry_gap(matrix, epsilon = 0):
    vals = torch.linalg.eigvalsh(matrix) + epsilon
    log_vals = torch.log(vals) 
    mean_log_vals = torch.mean(log_vals)
    log_mean_vals = torch.log(torch.mean(vals))
    iso_gap = log_mean_vals - mean_log_vals
    return iso_gap


def potential_gap(matrix, epsilon = 0):
    matrix = (matrix).abs()
    d = torch.diag(matrix.diag() ** -0.5)
    matrix = d @ matrix @ d
    matrix = matrix / (1-matrix)
    matrix = matrix[torch.eye(matrix.size(0), dtype=bool) == 0]
    return torch.max(matrix)

def variance_norms(tensor,dim=0):
    tensor = tensor - tensor.mean(dim=dim,keepdim=True)
    row_norms = torch.norm(tensor, dim=dim)  # Compute the norms of each row
    mean_norm = torch.mean(row_norms)  # Compute the mean norm
    var_norm = torch.var(row_norms)  # Compute the variance of norms
    return var_norm / mean_norm**2

def calc_stats(X):
    C = X @ X.t()
    ig = isometry_gap(C).item()
    pg = potential_gap(C).item()
    ig = isometry_gap(X @ X.t()).item()
    pg = potential_gap(X @ X.t()).item()
    return dict(iso_gap=ig, gamma=pg, )


dataset_configs = {
    'MNIST': {
        'input_size': (28, 28, 1),
        'num_classes': 10,
        'transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
        'train_dataset': datasets.MNIST,
        'test_dataset': datasets.MNIST
    },
    'FashionMNIST': {
        'input_size': (28 , 28, 1),
        'num_classes': 10,
        'transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
        'train_dataset': datasets.FashionMNIST,
        'test_dataset': datasets.FashionMNIST
    },
    'CIFAR10': {
        'input_size': (32, 32, 3),
        'num_classes': 10,
        'transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        'train_dataset': datasets.CIFAR10,
        'test_dataset': datasets.CIFAR10
    },
    'CIFAR100': {
        'input_size': (32, 32, 3),
        'num_classes': 100,
        'transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        'train_dataset': datasets.CIFAR100,
        'test_dataset': datasets.CIFAR100
    },
    'SVHN': {
        'input_size': (32, 32, 3),
        'num_classes': 10,
        'transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        'train_dataset': datasets.SVHN,
        'test_dataset': datasets.SVHN
    },
    'ImageNet': {
        'input_size': (224, 224, 3),
        'num_classes': 1000,
        'transform': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
        'train_dataset': datasets.ImageNet,
        'test_dataset': datasets.ImageNet
    }
}



  

        
