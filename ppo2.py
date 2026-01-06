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
    def __init__(self, learn_rate=1e-3, actor_output=5, critic_output=1, p=0.25):
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
        self.conv2d_holes1 = net.make_conv2d(2, 32, kernel_size=(3, 3))
        self.conv2d_holes2 = net.make_conv2d(32, 64, kernel_size=(3, 3))

        self.ff_holes = net.make_lazy_linear(32, p)
        self.ff_holes_out = nn.Linear(32, 1)

        # Layers for actor-critic network
        self.ff_actor_critic1 = net.make_lazy_linear(512, p)
        self.ff_actor_critic2 = net.make_linear(512, 128, p)

        self.ff_critic1 = net.make_linear(128, 512, p)
        self.ff_critic2 = net.make_linear(512, 128, p)
        self.ff_critic_value = nn.Linear(128, critic_output)

        self.ff_actor1 = net.make_linear(128, 512, p)
        self.ff_actor2 = net.make_linear(512, 128, p)
        self.ff_actor_value = nn.Linear(128, actor_output)

        matris = tetris.Matris()
        state = matris.state(None, True)
        state = torch.Tensor(state).unsqueeze(0).to(self.device)
        self.act(state)
        self.critic(state)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learn_rate)

    def zero(self):
        self.optimizer.zero_grad()

    def calculate_internal_state(self, x):
        column = self.conv2d_height_column(x)
        height = self.conv2d_height_row(column)

        conv2d_bump1 = self.conv2d_bump1(height)
        conv2d_bump2 = self.conv2d_bump2(conv2d_bump1)
        conv2d_holes1 = self.conv2d_holes1(x)
        conv2d_holes2 = self.conv2d_holes2(conv2d_holes1)

        height = height.flatten(start_dim=1)
        conv2d_bump2 = conv2d_bump2.flatten(start_dim=1)
        conv2d_holes2 = conv2d_holes2.flatten(start_dim=1)

        ff_height = self.ff_height(height)
        ff_height_out = self.ff_height_out(ff_height)

        ff_bump = self.ff_bump(conv2d_bump2)
        ff_bump_out = self.ff_bump_out(ff_bump)

        ff_holes = self.ff_holes(conv2d_holes2)
        ff_holes_out = self.ff_holes_out(ff_holes)

        return ff_height_out, ff_bump_out, ff_holes_out, height, conv2d_bump2, conv2d_holes2

    def detach_data(self, data_tuple):
        ff_height_out, ff_bump_out, ff_holes_out, height, conv2d_bump2, conv2d_holes2 = data_tuple
        ff_height_out = ff_height_out.detach()
        ff_bump_out = ff_bump_out.detach()
        ff_holes_out = ff_holes_out.detach()
        height = height.detach()
        conv2d_bump2 = conv2d_bump2.detach()
        conv2d_holes2 = conv2d_holes2.detach()

        return torch.cat([ff_height_out, ff_bump_out, ff_holes_out, height, conv2d_bump2, conv2d_holes2], dim=1)

    def actor_critic_shared(self, state):
        ff_actor_critic1 = self.ff_actor_critic1(self.detach_data(self.calculate_internal_state(state)))
        return self.ff_actor_critic2(ff_actor_critic1)

    def forward(self, _):
        return self.act(_)

    def step(self) -> None:
        self.optimizer.step()

    def act(self, state):
        ff_actor1 = self.ff_actor1(self.actor_critic_shared(state))
        ff_actor2 = self.ff_actor2(ff_actor1)
        return self.ff_actor_value(ff_actor2)

    def critic(self, state):
        ff_critic1 = self.ff_critic1(self.actor_critic_shared(state))
        ff_critic2 = self.ff_critic2(ff_critic1)
        return self.ff_critic_value(ff_critic2)

    def save(self, file):
        if file:
            torch.save(self.state_dict(), file)

    def load(self, file):
        if file is not None and pathlib.Path(file).exists():
            self.load_state_dict(torch.load(file, weights_only=True, map_location=self.device))


