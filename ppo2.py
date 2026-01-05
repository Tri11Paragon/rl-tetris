import math
import pathlib
import pickle
import random
from array import array

import MaTris.matris as tetris
import pygame
import numpy as np
import cv2
import torch
from torch import nn
import torch.nn.functional as F

from MaTris.matris import MATRIX_WIDTH, MATRIX_HEIGHT
from experience import ExperienceLike
import argparse
import time

from collections import namedtuple, deque

from MaTris.matris import GameOver

import network as net

class PPONetwork(nn.Module, net.PPONetwork):
    def __init__(self, actor_output = 5, critic_output = 1, p = 0.25):
        super().__init__()

        # Layers for calculating the height
        self.conv2d_height_column = net.make_conv2d(2, 64, kernel_size=(22, 1))
        self.conv2d_height_row = net.make_conv2d(64, 32, kernel_size=(1, 10))

        # FF Layer for aggregating the height
        self.ff_height = net.make_lazy_linear(32, p)
        self.ff_height_out = nn.Linear(32, 1)

        # Layers for calculating the bumpiness
        self.conv2d_bump1 = net.make_conv2d(32, 32, kernel_size=(1, 3))
        self.conv2d_bump2 = net.make_conv2d(32, 32, kernel_size=(1, 3))

        self.ff_bump = net.make_lazy_linear(32, p)
        self.ff_bump_out = nn.Linear(32, 1)

        # Layers for calculating the holes
        self.conv2d_holes = net.make_conv2d(2, 32, kernel_size=(3, 3))

        self.ff_holes = net.make_lazy_linear(32, p)
        self.ff_holes_out = nn.Linear(32, 1)

        self.ff_critic = net.make_lazy_linear(128, p)
        self.ff_critic_value = nn.Linear(128, critic_output)

        self.ff_actor = net.make_lazy_linear(128, p)
        self.ff_actor_value = nn.Linear(128, actor_output)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, _):
        raise NotImplementedError

    def step(self) -> None:
        pass

    def act(self, state):
        pass

    def critic(self, state):
        pass