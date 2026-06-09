from abc import abstractmethod

from torch import nn
import torch.nn.functional as F
import torch
import pathlib

from typing import Protocol, Any, Generic, TypeVar, get_type_hints

def make_conv2d(inc, outc, kernel_size, padding=0):
    return nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(1, outc),
            nn.GELU())

def make_conv1d(inc, outc, kernel_size):
    return nn.Sequential(
            nn.Conv1d(inc, outc, kernel_size=kernel_size),
            nn.GroupNorm(1, outc),
            nn.GELU())

def make_lazy_linear(out, p):
    return nn.Sequential(
        nn.LazyLinear(out),
        nn.LayerNorm(out),
        nn.GELU(),
        nn.Dropout(p))

def make_linear(inp, out, p):
    return nn.Sequential(
        nn.Linear(inp, out),
        nn.LayerNorm(out),
        nn.GELU(),
        nn.Dropout(p))

class ActorNetwork(Protocol):
    @abstractmethod
    def step(self) -> None:
        ...

    @abstractmethod
    def zero(self) -> None:
        ...

    @abstractmethod
    def compute(self, state):
        ...

    @abstractmethod
    def act(self):
        ...

class ActorCriticNetwork(ActorNetwork):
    @abstractmethod
    def step(self) -> None:
        ...

    @abstractmethod
    def zero(self) -> None:
        ...

    @abstractmethod
    def compute(self, state):
        ...

    @abstractmethod
    def act(self):
        ...

    @abstractmethod
    def critic(self):
        ...

    @abstractmethod
    def extra_learn(self, state):
        ...

