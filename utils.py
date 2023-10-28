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

def generate_configs(config_grid):
    config_list = list(product(*config_grid.values()))
    config_dicts = []
    for config_tuple in config_list:
        config_dict = dict(zip(config_grid.keys(), config_tuple))
        config_dicts.append(config_dict)
    return config_dicts



def compute_beta(act):
    x = torch.randn(1000000)
    c0 = act(x).mean()
    c1 = (act(x)*x).mean()
    csum2 = (act(x)**2).mean()
    beta0 = 2 - c1**2 / (csum2 - c0**2)
    # c0, c1, csum2, beta0
    return beta0.item()



# Define the MLP model with layer normalization
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, activation, normalization, **kwargs):
        super(MLP, self).__init__()
        if normalization == 'LN':
            self.norm = nn.LayerNorm
        elif normalization == 'BN':
            self.norm = nn.BatchNorm1d
        elif normalization == 'None':
            self.norm = nn.Identity
        if not isinstance(input_size, int):
            p = 1
            for s in input_size:
                p *= s
            input_size = p

        self.layers = nn.Sequential()
        hidden_sizes = [input_size] + hidden_sizes + [num_classes]
        num_layers = len(hidden_sizes)
        # self.layers.add_module('fc-0', nn.Linear(input_size, hidden_size))
        # self.layers.add_module('norm-0', self.norm(hidden_size))

        for i, (h1,h2) in enumerate(zip(hidden_sizes[:-1],hidden_sizes[1:])):
            self.layers.add_module(f'norm-{i}', self.norm((h1)))
            self.layers.add_module(f'fc-{i}', nn.Linear(h1, h2))
            self.layers.add_module(f'act-{i}', getattr(nn, activation)())

        # self.layers.add_module(f'fc-{num_layers - 1}', nn.Linear(hidden_size, num_classes))
        self.apply(self.init_weights)

    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            std = 1 / math.sqrt(layer.weight.size(1))
            nn.init.normal_(layer.weight, mean=0, std=std)

    def forward(self, x):
        x = x.view(x.size(0), -1)        
        hidden_layers = []
        for i, (name, layer) in enumerate(self.layers.named_children()):
            hidden_layers.append((name, x, layer(x)))
            x = layer(x)
        output = x
        return output, hidden_layers


def train_model(model, train_loader, criterion, optimizer, clip_norm, device):
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs, hidden = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        if clip_norm is not None:
            for l in model.layers:
                torch.nn.utils.clip_grad_norm(l.parameters(),max_norm=clip_norm)
        loss.backward()
        optimizer.step()
    return loss


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs, hidden = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total



  

        
