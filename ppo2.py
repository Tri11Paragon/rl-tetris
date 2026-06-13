import json
import math
import pathlib
import pickle
import random
from array import array

from matplotlib import pyplot as plt
from tqdm.auto import tqdm

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} | HIP: {getattr(torch.version, 'hip', None) or getattr(torch.version, 'cuda', None)}")

ACTOR_OUTPUT = 5
CRITIC_OUTPUT = 1

# http://vision.stanford.edu/teaching/cs231n/reports/2016/pdfs/121_Report.pdf
class AdjustedSandfordACNetwork(nn.Module, net.ActorCriticNetwork):
    def __init__(self, config, actor_output=ACTOR_OUTPUT, critic_output=CRITIC_OUTPUT):
        super().__init__()
        self.cache = {}
        self.device = device

        p = config["DROPOUT"]

        self.conv3x3_1_actor = net.make_conv2d(2, 32, kernel_size=(3, 3), padding=1)
        self.conv3x3_2_actor = net.make_conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.conv3x3_3_actor = net.make_conv2d(64, 128, kernel_size=(3, 3), padding=1)

        self.conv_collapse_actor = net.make_conv2d(128, 128, kernel_size=(22, 1), padding=1)

        self.conv3_1_actor = net.make_conv2d(128, 128, kernel_size=(1, 3), padding=1)
        self.conv1_actor = net.make_conv2d(128, 128, kernel_size=(1, 1), padding=1)
        self.conv3_2_actor = net.make_conv2d(128, 128, kernel_size=(1, 3), padding=1)

        self.conv_filters_actor = nn.Sequential(
            self.conv3x3_1_actor,
            self.conv3x3_2_actor,
            self.conv3x3_3_actor,
            self.conv_collapse_actor,
            self.conv3_1_actor,
            self.conv1_actor,
            self.conv3_2_actor
        )


        self.conv3x3_1_critic = net.make_conv2d(2, 32, kernel_size=(3, 3), padding=1)
        self.conv3x3_2_critic = net.make_conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.conv3x3_3_critic = net.make_conv2d(64, 128, kernel_size=(3, 3), padding=1)

        self.conv_collapse_critic = net.make_conv2d(128, 128, kernel_size=(22, 1), padding=1)

        self.conv3_1_critic = net.make_conv2d(128, 128, kernel_size=(1, 3), padding=1)
        self.conv1_critic = net.make_conv2d(128, 128, kernel_size=(1, 1), padding=1)
        self.conv3_2_critic = net.make_conv2d(128, 128, kernel_size=(1, 3), padding=1)

        self.conv_filters_critic = nn.Sequential(
            self.conv3x3_1_critic,
            self.conv3x3_2_critic,
            self.conv3x3_3_critic,
            self.conv_collapse_critic,
            self.conv3_1_critic,
            self.conv1_critic,
            self.conv3_2_critic
        )

        self.ff_actor_1 = net.make_lazy_linear(128, p)
        self.ff_actor_2 = net.make_linear(128, 512, p)

        self.ff_critic_1 = net.make_lazy_linear(128, p)
        self.ff_critic_2 = net.make_linear(128, 512, p)

        self.feed_forward_actor = nn.Sequential(
            self.ff_actor_1,
            self.ff_actor_2
        )

        self.feed_forward_critic = nn.Sequential(
            self.ff_critic_1,
            self.ff_critic_2
        )

        self.ff_actor = nn.Linear(512, actor_output)
        self._actor = nn.Sequential(
            self.ff_actor
        )

        self.ff_critic = nn.Linear(512, critic_output)
        self._critic = nn.Sequential(
            self.ff_critic
        )

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.conv_filters_actor.parameters(), "lr": config["CONV_LEARN_RATE"]},
                {"params": self.conv_filters_critic.parameters(), "lr": config["CONV_LEARN_RATE"]},
                {"params": self.feed_forward_actor.parameters(), "lr": config["ACTOR_LEARN_RATE"]},
                {"params": self.feed_forward_critic.parameters(), "lr": config["CRITIC_LEARN_RATE"]},
                {"params": self._actor.parameters(), "lr": config["ACTOR_LEARN_RATE"]},
                {"params": self._critic.parameters(), "lr": config["CRITIC_LEARN_RATE"]},
            ]
        )

        self.to(self.device)


    def zero(self):
        self.optimizer.zero_grad()

    def calculate_internal_state(self, x):
        conv_out_actor = self.conv_filters_actor(x)
        conv_out_critic = self.conv_filters_critic(x)

        flat_actor = conv_out_actor.flatten(start_dim=1)
        flat_critic = conv_out_critic.flatten(start_dim=1)
        feed_out_actor = self.feed_forward_actor(flat_actor)
        feed_out_critic = self.feed_forward_critic(flat_critic)

        return feed_out_actor, feed_out_critic

    def forward(self, _):
        self.compute(_)
        return self.act(), self.critic()

    def step(self) -> None:
        self.optimizer.step()

    def compute(self, state):
        self.cache.clear()
        state_actor, state_critic = self.calculate_internal_state(state)
        self.cache["act"] = self._actor(state_actor)
        self.cache["critic"] = self._critic(state_critic)

    def act(self):
        return self.cache["act"]

    def critic(self):
        return self.cache["critic"]

    def extra_learn(self, state):
        return 0

    def save(self, file):
        try:
            if file:
                torch.save(self.state_dict(), file)
        except KeyboardInterrupt:
            print("wait until save!")
            self.save(file)

    def load(self, file):
        if file is not None and pathlib.Path(file).exists():
            self.load_state_dict(torch.load(file, weights_only=True, map_location=self.device))
        return self

class PPOExperienceGenerator:
    def __init__(self, config, engine: tetris.Matris, model: net.ActorCriticNetwork):
        self.engine: tetris.Matris = engine
        self.model: net.ActorCriticNetwork = model
        self.config = config


    def generate(self):
        buffer = ERMBuffer[PPOExperience]()
        self.model.eval()
        runs_progress = tqdm(
            range(self.config["MAX_EPISODES"]),
            desc="Runs",
            dynamic_ncols=True,
            leave=False,
            position=1
        )
        state = self.engine.reset()
        trajectory = Trajectory(PPOExperience)
        for _ in runs_progress:
            episode_progress = tqdm(
                range(self.config["MAX_EPISODE_LENGTH"]),
                desc="Experiences",
                dynamic_ncols=True,
                leave=False,
                position=2
            )
            for _ in episode_progress:
                state_tensor = torch.Tensor(state).unsqueeze(0).to(self.model.device)

                with torch.no_grad():
                    self.model.compute(state_tensor)
                    state_value = self.model.critic()
                    logits = self.model.act()

                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logprob = dist.log_prob(action)

                state, reward, lines_cleared, game_over, truncated = self.engine.step(tetris.Action(action.item()))
                reward = torch.tensor([reward]).to(self.model.device)
                done = torch.tensor([int(game_over or (truncated and self.config["MATRIS_TRUNCATE_HARD_BOUNARY"]))]).to(self.model.device)

                experience = PPOExperience(state_tensor.detach(), action, reward, done, logprob, state_value)
                trajectory.append(experience)

                if game_over:
                    trajectory.set_last_value(0) # Value is masked out anyway
                    buffer.append(trajectory)
                    trajectory = Trajectory(PPOExperience)

                    state = self.engine.reset()
                    break

                if truncated:
                    state_tensor = torch.Tensor(state).unsqueeze(0).to(self.model.device)
                    self.model.compute(state_tensor)
                    trajectory.set_last_value(self.model.critic()) # Value is masked out.
                    buffer.append(trajectory)
                    trajectory = Trajectory(PPOExperience)
                    if self.config["BREAK_ON_TRUNCATE"]:
                        break


        return buffer

class PPOTrainer:
    def __init__(self, config, model: net.ActorCriticNetwork, generator: PPOExperienceGenerator):
        self.model = model
        self.generator = generator
        self.config = config

    def step(self, batch, progress):
        b_state, b_action, b_logprob, b_advantages, b_returns = batch

        b_state = b_state.to(self.model.device)
        b_action = b_action.to(self.model.device)
        b_logprob = b_logprob.to(self.model.device)
        b_advantages = b_advantages.to(self.model.device)
        b_returns = b_returns.to(self.model.device)

        self.model.compute(b_state)
        logits = self.model.act()
        dist = torch.distributions.Categorical(logits=logits)
        action_logprob = dist.log_prob(b_action)
        entropy = dist.entropy()

        state_values1 = self.model.critic()
        state_values = state_values1.squeeze(dim=1)

        # PPO Surrogate Objective
        importance_ratio = torch.exp(action_logprob - b_logprob)
        surrogate = importance_ratio * b_advantages
        clipped_surrogate = torch.clamp(importance_ratio, 1 - self.config["CLIP_EPSILON"],
                                        1 + self.config["CLIP_EPSILON"]) * b_advantages

        actor_loss = -(torch.min(surrogate, clipped_surrogate) + self.config["ENTROPY"] * entropy)
        actor_loss = actor_loss.mean()
        critic_loss = F.mse_loss(state_values, b_returns)
        critic_loss = critic_loss.mean()


        self.model.extra_learn(b_state)

        self.model.zero()
        critic_loss.backward()
        actor_loss.backward()
        progress.set_postfix(actor_loss=actor_loss.mean().item(), critic_loss=critic_loss.mean().item())
        self.model.step()

        return critic_loss.mean().item(), actor_loss.mean().item()

    def train(self):
        t0 = time.process_time()
        with torch.no_grad():
            buffer = self.generator.generate()
            t1 = time.process_time()

            tensors = buffer.to_tensors()
            advantages, returns = buffer.compute_gae(self.config["GAMMA"], self.config["LAMBDA"])

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            dataset = torch.utils.data.TensorDataset(
                tensors["state"],
                tensors["action"],
                tensors["logprob"],
                advantages,
                returns
            )

            loader = torch.utils.data.DataLoader(dataset, batch_size=self.config["BATCH_SIZE"], shuffle=self.config["SHUFFLE_EXPERIENCES"])

        epochs_progress = tqdm(
            range(self.config["EPOCHS"]),
            desc="Epochs",
            dynamic_ncols=True,
            leave=False,
            position=1
        )

        self.model.train()
        total_loss = 0
        total_critic_loss = 0
        total_actor_loss = 0
        total_batches = 0
        for _ in epochs_progress:
            for batch in loader:
                critic_loss, actor_loss= self.step(batch, epochs_progress)
                total_critic_loss += critic_loss
                total_actor_loss += actor_loss
                total_batches += 1

        t2 = time.process_time()
        average_returns, _ = buffer.compute_returns(self.config["GAMMA"])
        average_returns = average_returns.mean()
        time_taken = t2 - t0
        collection_time = t1 - t0
        return (average_returns, time_taken, collection_time,
                total_loss / total_batches, total_critic_loss / total_batches, total_actor_loss / total_batches)

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