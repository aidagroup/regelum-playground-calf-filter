import numpy as np
import torch


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def construct_default_layer(
    in_features, out_features, std=np.sqrt(2), bias_const=0.0, bias=True
):
    layer = torch.nn.Linear(in_features, out_features, bias=bias)
    torch.nn.init.orthogonal_(layer.weight, std)
    if bias:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer
