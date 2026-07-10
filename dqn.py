import pathlib
from tqdm.auto import tqdm

import MaTris.matris as tetris
import pygame
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from config import DotDict
from experience import ERMBuffer, Experience, Trajectory
import time

import network as net

ACTOR_OUTPUT = 5

# http://vision.stanford.edu/teaching/cs231n/reports/2016/pdfs/121_Report.pdf
def AdjustedSandfordNetwork(config: DotDict, device = None):
    device = net.get_device(device)
    p = config.network.dropout
    return net.Network(
        {
            "conv": net.Lr(nn.Sequential(
                net.make_conv2d(2, 32, kernel_size=(3, 3), padding=1),
                net.make_conv2d(32, 32, kernel_size=(3, 3), padding=1),
                net.make_conv2d(32, 64, kernel_size=(3, 3), padding=1),

                net.make_conv2d(64, 64, kernel_size=(22, 1), padding=1),

                net.make_conv2d(64, 128, kernel_size=(1, 3), padding=1),
                net.make_conv2d(128, 128, kernel_size=(1, 1), padding=1),
                net.make_conv2d(128, 128, kernel_size=(1, 3), padding=1),
                nn.Flatten(),
                net.Output("conv")
            ), config.network.init_lr.convLearnRate),
            "head": net.Lr(nn.Sequential(
                net.Input("conv"),
                net.make_lazy_linear(128, p),
                net.make_linear(128, 512, p),
                nn.Linear(512, ACTOR_OUTPUT),
                net.Output("action_logits")
            ), config.network.init_lr.actorLearnRate),
        },
        default_lr=config.network.init_lr.convLearnRate, device=device
    )


def step(self: net.TrainerType , batch: tuple) -> float:
    b_state, b_action, b_reward, b_next_state, b_done = batch

    b_reward = b_reward.unsqueeze(1)
    b_done = b_done.unsqueeze(1)
    b_action = b_action.unsqueeze(1)

    assert(isinstance(self.model, dict))

    model = self.model["network"]
    target = self.model["target"]

    model(b_next_state)
    target(b_next_state)

    next_state_actions = torch.argmax(model["action_logits"].detach(), dim=1).unsqueeze(1)
    next_state_values = target["action_logits"].detach().gather(1, next_state_actions).detach()

    predicted_values = b_reward + self.config.network.gamma * next_state_values * (1.0 - b_done)

    model(b_state)

    model.zero()
    state_values = model["action_logits"].gather(1, b_action)
    criterion = nn.MSELoss()
    loss = criterion(state_values, predicted_values)
    loss.mean().backward()
    model.step()

    return loss.item()
