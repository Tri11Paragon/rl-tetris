from torch import nn
import torch.nn.functional as F
import torch
import pathlib

def make_conv2d(inc, outc, kernel_size):
    return nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=kernel_size),
            nn.GroupNorm(1, outc),
            nn.ReLU())

def make_conv1d(inc, outc, kernel_size):
    return nn.Sequential(
            nn.Conv1d(inc, outc, kernel_size=kernel_size),
            nn.GroupNorm(1, outc),
            nn.ReLU())

def make_lazy_linear(out, p):
    return nn.Sequential(
        nn.LazyLinear(out),
        nn.LayerNorm(out),
        nn.ReLU(),
        nn.Dropout(p))

def make_linear(inp, out, p):
    return nn.Sequential(
        nn.Linear(inp, out),
        nn.LayerNorm(out),
        nn.ReLU(),
        nn.Dropout(p))