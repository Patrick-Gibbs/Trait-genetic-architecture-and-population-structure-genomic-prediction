
import copy
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from skorch.callbacks import LRScheduler, EarlyStopping
from Main.HelperClasses.GetAraData import *
from sklearn.metrics import *
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from skorch import NeuralNetRegressor, NeuralNet
import skorch
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_predict, GridSearchCV


"""A set of Torch models for use with skorch."""
class MLP(nn.Module):
    """MLP model with one hidden layer only and no dropout"""
    """MLP model with one hidden layer only and no dropout"""
    def __init__(self, input_dim = 100, hidden_layer_size = 100):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.input_dim = input_dim
        self.input = nn.Linear(self.input_dim, self.hidden_layer_size)
        self.act1 = nn.ReLU()
        self.output = nn.Linear(self.hidden_layer_size, 1)

    def forward(self, x):
        x = self.act1(self.input(x))
        x = self.output(x)
        return x

class MLP_complex(nn.Module):
    """MLP model with three hidden layers and dropout"""
    def __init__(self,  input_dim = 100, hidden_layer_size = 100, num_classes=1, dropout=0.25, activation=nn.ReLU(), hidden=5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_layer_size))
        for i in range(hidden - 1):
            self.layers.append(nn.Linear(hidden_layer_size, hidden_layer_size))
        self.layers.append(nn.Linear(hidden_layer_size, num_classes))
        self.dropout = nn.Dropout(p=dropout)
        self.activation = activation

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
            x = self.dropout(x)
        return self.layers[-1](x)