import json
import math
import pathlib
import pickle
import random
from array import array

from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from typing import Any

import MaTris.matris as tetris
import pygame
import numpy as np
import cv2
import torch
from torch import nn
import torch.nn.functional as F

from MaTris.matris import MATRIX_WIDTH, MATRIX_HEIGHT
from experience import ERMBuffer, PPOExperience, Trajectory
import argparse
import time
import graph

from collections import namedtuple, deque

from MaTris.matris import GameOver

import network as net

ACTOR_OUTPUT = 5
CRITIC_OUTPUT = 1


# http://vision.stanford.edu/teaching/cs231n/reports/2016/pdfs/121_Report.pdf
def AdjustedSandfordNetwork(config, device = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | HIP: {getattr(torch.version, 'hip', None) or getattr(torch.version, 'cuda', None)}")

    p = config["DROPOUT"]
    return net.Network(
        {
            "conv_filters": net.Lr(nn.Sequential(
                net.make_conv2d(2, 32, kernel_size=(3, 3), padding=1),
                net.make_conv2d(32, 32, kernel_size=(3, 3), padding=1),
                net.make_conv2d(32, 64, kernel_size=(3, 3), padding=1),

                net.make_conv2d(64, 64, kernel_size=(22, 1), padding=1),

                net.make_conv2d(64, 128, kernel_size=(1, 3), padding=1),
                net.make_conv2d(128, 128, kernel_size=(1, 1), padding=1),
                net.make_conv2d(128, 128, kernel_size=(1, 3), padding=1),
                nn.Flatten(),
                net.Output("conv")
            ), config["CONV_LEARN_RATE"]),
            "actor_head": net.Lr(nn.Sequential(
                net.Input("conv"),
                net.make_lazy_linear(128, p),
                net.make_linear(128, 512, p),
                nn.Linear(512, ACTOR_OUTPUT),
                net.Output("action_logits")
            ), config["ACTOR_LEARN_RATE"]),
            "critic_head": net.Lr(nn.Sequential(
                net.Input("conv"),
                net.make_lazy_linear(128, p),
                net.make_linear(128, 512, p),
                nn.Linear(512, CRITIC_OUTPUT),
                net.Output("state_value")
            ), config["CRITIC_LEARN_RATE"])
        },
        default_lr=config["CONV_LEARN_RATE"], device=device
    )


def step(self, batch: tuple) -> float:
    b_state, b_action, b_logprob, b_advantages, b_returns = batch

    self.model(b_state)
    logits = self.model["action_logits"]
    dist = torch.distributions.Categorical(logits=logits)
    action_logprob = dist.log_prob(b_action)
    entropy = dist.entropy()

    state_values1 = self.model["state_value"]
    state_values = state_values1.squeeze(dim=1)

    # PPO Surrogate Objective
    importance_ratio = torch.exp(action_logprob - b_logprob)
    surrogate = importance_ratio * b_advantages
    clipped_surrogate = torch.clamp(importance_ratio, 1 - self.config["CLIP_EPSILON"],
                                    1 + self.config["CLIP_EPSILON"]) * b_advantages

    actor_loss = -(torch.min(surrogate, clipped_surrogate) + self.config["ENTROPY"] * entropy)
    critic_loss = torch.functional.F.mse_loss(state_values, b_returns)
    total_loss = (actor_loss + (critic_loss * 0.5) * 0.01).mean()

    self.model.zero()
    total_loss.mean().backward()
    self.model.step()

    return total_loss.item()


class NetworkRealtimeVisualizer:
    def __init__(self, width=520, height=520):
        from pygame._sdl2.video import Window, Renderer

        self.width = width
        self.height = height
        self.window = Window("Network View", size=(width, height), position=(900, 100))
        self.renderer = Renderer(self.window)
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)

    def clear(self, color=(12, 12, 18, 255)):
        self.renderer.draw_color = color
        self.renderer.clear()

    def present(self):
        self.renderer.present()

    def fill_rect(self, rect, color):
        self.renderer.draw_color = color
        self.renderer.fill_rect(rect)

    def draw_rect(self, rect, color):
        self.renderer.draw_color = color
        self.renderer.draw_rect(rect)

    def draw_text(self, text, x, y, color=(240, 240, 240), small=False):
        from pygame._sdl2.video import Texture

        font = self.small_font if small else self.font
        text_surface = font.render(text, True, color[:3])
        texture = Texture.from_surface(self.renderer, text_surface)

        target = pygame.Rect(x, y, text_surface.get_width(), text_surface.get_height())
        texture.draw(dstrect=target)

    def draw_grid_channel(self, channel, x0, y0, title, scale=14):
        self.draw_text(title, x0, y0 - 24)

        channel = np.asarray(channel)
        for y in range(channel.shape[0]):
            for x in range(channel.shape[1]):
                value = float(channel[y, x])

                if value <= 0:
                    color = (25, 25, 30, 255)
                elif value < 2:
                    color = (90, 90, 150, 255)
                else:
                    color = (80, 210, 120, 255)

                rect = pygame.Rect(x0 + x * scale, y0 + y * scale, scale - 1, scale - 1)
                self.fill_rect(rect, color)

    def draw_action_probs(self, probs, logits, critic_value, selected_action=None):
        action_names = ["RIGHT", "LEFT", "DOWN", "ROTATE", "HARD_DROP"]

        x0 = 230
        y0 = 60
        bar_width = 220
        bar_height = 24
        gap = 12

        self.draw_text("Actor output", x0, 25)
        self.draw_text(f"Critic: {critic_value:.4f}", x0, 330)

        if selected_action is not None:
            self.draw_text(f"Selected: {action_names[selected_action]}", x0, 360)

        for i, prob in enumerate(probs):
            y = y0 + i * (bar_height + gap)

            bg_rect = pygame.Rect(x0, y, bar_width, bar_height)
            prob_rect = pygame.Rect(x0, y, int(bar_width * float(prob)), bar_height)

            self.fill_rect(bg_rect, (55, 55, 65, 255))
            self.fill_rect(prob_rect, (80, 180, 255, 255))

            if selected_action == i:
                self.draw_rect(
                    pygame.Rect(x0 - 4, y - 4, bar_width + 8, bar_height + 8),
                    (255, 220, 80, 255)
                )

            self.draw_text(
                f"{action_names[i]}: {float(prob):.3f}",
                x0,
                y + 3,
                color=(255, 255, 255),
                small=True
            )

            self.draw_text(
                f"logit {float(logits[i]):+.3f}",
                x0,
                y + bar_height + 1,
                color=(180, 180, 180),
                small=True
            )

    def update(self, state, logits, probs, critic_value, selected_action=None):
        self.clear()

        state_np = np.asarray(state)

        self.draw_grid_channel(state_np[0], 20, 55, "Board channel")
        self.draw_grid_channel(state_np[1], 20, 365, "Piece channel", scale=6)

        self.draw_action_probs(probs, logits, critic_value, selected_action)

        self.present()
