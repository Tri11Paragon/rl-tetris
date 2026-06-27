import pathlib
from tqdm.auto import tqdm

import MaTris.matris as tetris
import pygame
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from experience import ERMBuffer, Experience, Trajectory
import time

import network as net

ACTOR_OUTPUT = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} | HIP: {getattr(torch.version, 'hip', None) or getattr(torch.version, 'cuda', None)}")

class DQNNetwork(nn.Module, net.ValueNetwork):
    def __init__(self, config, output=ACTOR_OUTPUT):
        super().__init__()
        self.cache = {}
        self.device = device

        p = config["DROPOUT"]

        self.conv3x3_1 = net.make_conv2d(2, 32, kernel_size=(3, 3), padding=1)
        self.conv3x3_2 = net.make_conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.conv3x3_3 = net.make_conv2d(64, 128, kernel_size=(3, 3), padding=1)

        self.conv_collapse = net.make_conv2d(128, 128, kernel_size=(22, 1), padding=1)

        self.conv3_1 = net.make_conv2d(128, 128, kernel_size=(1, 3), padding=1)
        self.conv1 = net.make_conv2d(128, 128, kernel_size=(1, 1), padding=1)
        self.conv3_2 = net.make_conv2d(128, 128, kernel_size=(1, 3), padding=1)

        self.conv_filters = nn.Sequential(
            self.conv3x3_1,
            self.conv3x3_2,
            self.conv3x3_3,
            self.conv_collapse,
            self.conv3_1,
            self.conv1,
            self.conv3_2
        )

        self.ff_actor_1 = net.make_lazy_linear(128, p)
        self.ff_actor_2 = net.make_linear(128, 512, p)

        self.feed_forward_actor = nn.Sequential(
            self.ff_actor_1,
            self.ff_actor_2
        )

        self.ff_actor = nn.Linear(512, output)
        self._actor = nn.Sequential(
            self.ff_actor
        )

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.conv_filters.parameters(), "lr": config["CONV_LEARN_RATE"]},
                {"params": self.feed_forward_actor.parameters(), "lr": config["ACTOR_LEARN_RATE"]},
                {"params": self._actor.parameters(), "lr": config["ACTOR_LEARN_RATE"]},
            ]
        )

        self.to(self.device)


    def zero(self):
        self.optimizer.zero_grad()

    def calculate_internal_state(self, x):
        conv_out_actor = self.conv_filters(x)

        flat_actor = conv_out_actor.flatten(start_dim=1)
        feed_out_actor = self.feed_forward_actor(flat_actor)

        return feed_out_actor

    def forward(self, _):
        self.compute(_)
        return self.act(), self.critic()

    def step(self) -> None:
        self.optimizer.step()

    def compute(self, state):
        self.cache.clear()
        state_actor = self.calculate_internal_state(state)
        self.cache["act"] = self._actor(state_actor)

    def state_value(self):
        return self.cache["act"]

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

class DQNExperienceGenerator:
    def __init__(self, config, engine: tetris.Matris, model: net.ValueNetwork):
        self.engine: tetris.Matris = engine
        self.model: net.ValueNetwork = model
        self.config = config

    def generate(self):
        buffer = ERMBuffer[Experience]()
        self.model.eval()
        runs_progress = tqdm(
            range(self.config["MAX_EPISODES"]),
            desc="Runs",
            dynamic_ncols=True,
            leave=False,
            position=1
        )
        state = self.engine.current_state()
        trajectory = Trajectory(Experience)
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
                    logits = self.model.state_value()
                    probs = torch.softmax(logits / self.config["TEMPERATURE"], dim=1)
                    dist = torch.distributions.Categorical(probs=probs)
                    action = dist.sample()

                state, reward, lines_cleared, game_over, truncated = self.engine.step(tetris.Action(action.item()))
                reward = torch.tensor([reward]).to(self.model.device)
                done = torch.tensor([int(game_over or (truncated and self.config["MATRIS_TRUNCATE_HARD_BOUNARY"]))]).to(self.model.device)
                next_state = torch.Tensor(state).unsqueeze(0).to(self.model.device)

                experience = Experience(state_tensor.detach(), action, reward, next_state.detach(), done)
                trajectory.append(experience)

                if game_over:
                    buffer.append(trajectory)
                    trajectory = Trajectory(Experience)
                    state = self.engine.reset()
                    break

                if truncated:
                    buffer.append(trajectory)
                    trajectory = Trajectory(Experience)
                    if self.config["BREAK_ON_TRUNCATE"]:
                        break

        return buffer

class DQNTrainer:
    def __init__(self, config, network: net.ValueNetwork, target: net.ValueNetwork, generator: DQNExperienceGenerator):
        self.model = network
        self.target = target
        self.generator = generator
        self.config = config

    def step(self, batch, progress):
        b_state, b_action, b_reward, b_next_state, b_done = batch

        b_state = b_state.to(self.model.device)
        b_action = b_action.to(self.model.device).unsqueeze(1)
        b_reward = b_reward.to(self.model.device).unsqueeze(1)
        b_next_state = b_next_state.to(self.model.device)
        b_done = b_done.to(self.model.device).unsqueeze(1)

        self.model.compute(b_next_state)
        next_state_actions = torch.argmax(self.model.state_value().detach(), dim=1).unsqueeze(1)
        self.target.compute(b_next_state)
        next_state_values = self.target.state_value().detach().gather(1, next_state_actions)

        predicted_values = b_reward + self.config["GAMMA"] * next_state_values * b_done
        self.model.compute(b_state)
        state_values = self.model.state_value().gather(1, b_action)

        self.model.zero()
        criterion = nn.MSELoss()
        loss = criterion(state_values, predicted_values)
        loss.backward()
        progress.set_postfix(loss=loss.item())
        self.model.step()

        return loss.item()

    def train(self):
        t0 = time.process_time()
        with torch.no_grad():
            buffer = self.generator.generate()
            t1 = time.process_time()

            tensors = buffer.to_tensors()

            dataset = torch.utils.data.TensorDataset(
                tensors["state"],
                tensors["action"],
                tensors["reward"],
                tensors["next_state"],
                tensors["done"]
            )

            loader = torch.utils.data.DataLoader(dataset, batch_size=self.config["BATCH_SIZE"], shuffle=self.config["SHUFFLE_EXPERIENCES"])

        self.model.train()

        epochs_progress = tqdm(
            range(self.config["EPOCHS"]),
            desc="Epochs",
            dynamic_ncols=True,
            leave=False,
            position=1
        )

        total_loss = 0
        total_batches = 0

        for _ in epochs_progress:
            for batch in loader:
                total_loss += self.step(batch, epochs_progress)
                total_batches += 1
            self.target.load_state_dict(self.model.state_dict())
        t2 = time.process_time()

        average_returns, _ = buffer.compute_returns(self.config["GAMMA"])
        average_returns = average_returns.mean()
        time_taken = t2 - t0
        collection_time = t1 - t0

        return average_returns, time_taken, collection_time, total_loss / total_batches
