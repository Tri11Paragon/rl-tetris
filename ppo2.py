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
from experience import ERMBuffer, PPOExperience, Trajectory
import argparse
import time

from collections import namedtuple, deque

from MaTris.matris import GameOver

import network as net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} | HIP: {getattr(torch.version, 'hip', None) or getattr(torch.version, 'cuda', None)}")

def detach_data_for_ac(data_tuple):
    ff_height_out, ff_bump_out, ff_holes_out, height, conv2d_bump2, conv2d_holes2, conv2d_ac2 = data_tuple
    # ff_height_out = ff_height_out.detach()
    # ff_bump_out = ff_bump_out.detach()
    # ff_holes_out = ff_holes_out.detach()
    height = height.detach()
    conv2d_bump2 = conv2d_bump2.detach()
    conv2d_holes2 = conv2d_holes2.detach()

    return torch.cat([height, conv2d_bump2, conv2d_holes2, conv2d_ac2], dim=1)


class PPONetwork(nn.Module, net.PPONetwork):
    def __init__(self, learn_rate=1e-4, actor_output=5, critic_output=1, p=0.25):
        super().__init__()
        self.cache = {}
        self.device = device

        # The height layer nonsense uses only the board's current state, as these values are the only ones included in the reward function.
        # This means that current piece information is lost, so we need to process that layer.
        # The actor-critic must learn from the convolutions here, conveniently saving on compute.

        # Layers for calculating the height
        self.conv2d_height_column = net.make_conv2d(1, 64, kernel_size=(22, 1))
        self.conv2d_height_row = net.make_conv2d(64, 32, kernel_size=(1, 10))

        # FF Layer for aggregating the height
        self.ff_height = net.make_lazy_linear(32, p)
        self.ff_height_out = nn.Linear(32, 1)

        # Layers for calculating the bumpiness
        self.conv2d_bump1 = net.make_conv2d(64, 32, kernel_size=(1, 3))
        self.conv2d_bump2 = net.make_conv2d(32, 32, kernel_size=(1, 3))

        self.ff_bump = net.make_lazy_linear(32, p)
        self.ff_bump_out = nn.Linear(32, 1)

        # Layers for calculating the holes
        self.conv2d_holes1 = net.make_conv2d(1, 32, kernel_size=(3, 3))
        self.conv2d_holes2 = net.make_conv2d(32, 64, kernel_size=(3, 3))

        self.ff_holes = net.make_lazy_linear(32, p)
        self.ff_holes_out = nn.Linear(32, 1)

        # Conv for actor-critic
        self.conv2d_ac1 = net.make_conv2d(2, 32, kernel_size=(3, 3))
        self.conv2d_ac2 = net.make_conv2d(32, 64, kernel_size=(3, 3))

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
        state = torch.Tensor(state).unsqueeze(0)
        self.compute(state)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learn_rate)

    def zero(self):
        self.optimizer.zero_grad()

    def calculate_internal_state(self, x):
        board, piece = torch.unbind(x, dim=1)
        board = board.unsqueeze(1)
        piece = piece.unsqueeze(1)

        column = self.conv2d_height_column(board)
        height = self.conv2d_height_row(column)

        conv2d_bump1 = self.conv2d_bump1(column)
        conv2d_bump2 = self.conv2d_bump2(conv2d_bump1)
        conv2d_holes1 = self.conv2d_holes1(board)
        conv2d_holes2 = self.conv2d_holes2(conv2d_holes1)

        conv2d_ac1 = self.conv2d_ac1(x)
        conv2d_ac2 = self.conv2d_ac2(conv2d_ac1)

        height = height.flatten(start_dim=1)
        conv2d_bump2 = conv2d_bump2.flatten(start_dim=1)
        conv2d_holes2 = conv2d_holes2.flatten(start_dim=1)
        conv2d_ac2 = conv2d_ac2.flatten(start_dim=1)

        ff_height = self.ff_height(height)
        ff_height_out = self.ff_height_out(ff_height)

        ff_bump = self.ff_bump(conv2d_bump2)
        ff_bump_out = self.ff_bump_out(ff_bump)

        ff_holes = self.ff_holes(conv2d_holes2)
        ff_holes_out = self.ff_holes_out(ff_holes)

        return ff_height_out, ff_bump_out, ff_holes_out, height, conv2d_bump2, conv2d_holes2, conv2d_ac2

    def actor_critic_shared_known(self, shared_data):
        ff_actor_critic1 = self.ff_actor_critic1(detach_data_for_ac(shared_data))
        return self.ff_actor_critic2(ff_actor_critic1)

    def actor_critic_shared(self, state):
        return self.actor_critic_shared_known(self.calculate_internal_state(state))

    def forward(self, _):
        self.compute(_)
        return self.act()

    def step(self) -> None:
        self.optimizer.step()

    def act_with_known(self, shared_data):
        ff_actor1 = self.ff_actor1(shared_data)
        ff_actor2 = self.ff_actor2(ff_actor1)
        return self.ff_actor_value(ff_actor2)

    def compute(self, state):
        internal_state_tuple = self.calculate_internal_state(state)
        shared_data = self.actor_critic_shared_known(internal_state_tuple)
        self.cache.clear()
        self.cache["act"] = self.act_with_known(shared_data)
        self.cache["critic"] = self.critic_with_known(shared_data)
        self.cache["extra"] = {
            "height": internal_state_tuple[0],
            "bump": internal_state_tuple[1],
            "holes": internal_state_tuple[2]
        }

    def act(self):
        return self.cache["act"]

    def critic_with_known(self, shared_data):
        ff_critic1 = self.ff_critic1(shared_data)
        ff_critic2 = self.ff_critic2(ff_critic1)
        return self.ff_critic_value(ff_critic2)

    def critic(self):
        return self.cache["critic"]

    def extra_learn(self, state):
        state = state.detach().cpu()

        # TODO: Direct NP array
        bumps = []
        holes = []
        heights = []

        for individual_state in torch.unbind(state, dim=0):
            grid = tetris.Grid(MATRIX_HEIGHT, MATRIX_WIDTH)
            grid.from_state(individual_state.numpy())

            b, a, h = grid.bumpy()
            hole = grid.holes()

            bumps.append([b])
            holes.append([hole])
            heights.append([a])

        bumps = torch.Tensor(np.array(bumps)).to(self.device)
        holes = torch.Tensor(np.array(holes)).to(self.device)
        heights = torch.Tensor(np.array(heights)).to(self.device)

        extra = self.cache["extra"]

        mse = nn.MSELoss()
        # print(f"{extra['height']} vs {heights} | {extra['bump']} vs {bumps} | {extra['holes']} vs {holes}")
        loss = mse(extra["height"], heights).mean()
        loss += mse(extra["bump"], bumps).mean()
        loss += mse(extra["holes"], holes).mean()
        return loss

    def save(self, file):
        if file:
            torch.save(self.state_dict(), file)

    def load(self, file):
        if file is not None and pathlib.Path(file).exists():
            self.load_state_dict(torch.load(file, weights_only=True, map_location=self.device))

class PPOExperienceGenerator:
    def __init__(self, engine: tetris.Matris, model: net.PPONetwork, runs=10, max_episode_length=1000):
        self.engine: tetris.Matris = engine
        self.model: net.PPONetwork = model
        self.runs = runs
        self.max_episode_length = max_episode_length

    def generate(self):
        buffer = ERMBuffer[PPOExperience]()
        self.model.eval()
        for _ in range(self.runs):
            state = self.engine.reset()
            trajectory = Trajectory(PPOExperience)
            for i in range(self.max_episode_length):
                state_tensor = torch.Tensor(state).unsqueeze(0).to(self.model.device)

                with torch.no_grad():
                    self.model.compute(state_tensor)
                    state_value = self.model.critic()
                    logits = self.model.act()

                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logprob = dist.log_prob(action)

                state, reward, game_over = self.engine.step(tetris.Action(action.item()))
                reward = torch.tensor([reward]).to(self.model.device)
                go = torch.tensor([int(game_over)]).to(self.model.device)

                trajectory.append(PPOExperience(state_tensor.detach(), action, reward, go, logprob, state_value))

                if game_over:
                    break
            state_tensor = torch.Tensor(state).unsqueeze(0).to(self.model.device)
            self.model.compute(state_tensor)
            state_value = self.model.critic()
            trajectory.set_last_value(state_value)
            buffer.append(trajectory)
        return buffer

class PPOTrainer:
    def __init__(self, model: net.PPONetwork, generator: PPOExperienceGenerator, gamma=0.99, gae_discount=0.95, entropy=0.01, clip_epsilon=0.2, batch_size=64, load_file=None):
        self.model = model
        self.generator = generator
        self.gamma = gamma
        self.gae_discount = gae_discount
        self.entropy = entropy
        self.clip_epsilon = clip_epsilon
        self.batch_size = batch_size
        self.load_file = load_file

        self.model.load(load_file)

    def train(self, epochs):
        t0 = time.process_time()
        with torch.no_grad():
            buffer = self.generator.generate()
        t1 = time.process_time()

        tensors = buffer.to_tensors()
        advantages, returns = buffer.compute_gae(self.gamma, self.gae_discount)

        # print(f"{tensors['state'].shape} | {tensors['action'].shape} | {tensors['logprob'].shape} | {advantages.shape} | {returns.shape}")

        dataset = torch.utils.data.TensorDataset(
            tensors["state"],
            tensors["action"],
            tensors["logprob"],
            advantages,
            returns
        )

        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for _ in range(epochs):
            for batch in loader:
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

                state_values = self.model.critic().squeeze()

                # PPO Surrogate Objective
                importance_ratio = torch.exp(action_logprob - b_logprob)
                surrogate = importance_ratio * b_advantages
                clipped_surrogate = torch.clamp(importance_ratio, 1 - self.clip_epsilon,
                                                1 + self.clip_epsilon) * b_advantages

                actor_loss = -torch.min(surrogate, clipped_surrogate) - self.entropy * entropy
                critic_loss = F.mse_loss(state_values, b_returns)

                loss = self.model.extra_learn(b_state)

                self.model.zero()
                loss.backward()
                (actor_loss + 0.5 * critic_loss).mean().backward()
                self.model.step()

        t2 = time.process_time()
        print(f"Time taken: {t2 - t0:.6f} | Collection time {t1 - t0:.6f} | Train time {t2 - t1:.6f}")

def gui_test():
    pygame.init()
    screen = pygame.display.set_mode((tetris.WIDTH, tetris.HEIGHT))
    pygame.display.set_caption("MaTris")
    matris = tetris.Matris()
    game = tetris.Game()
    game.main(screen, matris)

    engine = tetris.Matris()
    state = engine.reset()
    network = PPONetwork().to(device)
    network.load("ppo2.pt")
    network.eval()
    run = False

    while True:
        game.clock.tick(50)
        actions = game.get_user_actions()
        if game.is_key(pygame.K_r):
            run = not run

        if game.is_key(pygame.K_v) or run:
            network.compute(torch.Tensor(state).unsqueeze(0).to(network.device))
            logits = network.act()
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()
            print(f"Action taken: {action}")
            actions.append(tetris.Action(action))

        if len(actions) == 0:
            game.redraw()
            continue

        for action in actions:
            next_state, reward, game_over = matris.step(action)
            game.redraw()
            state = next_state
            grid_state = tetris.Grid(MATRIX_HEIGHT, MATRIX_WIDTH).from_state(state)
            bump, agg, heights = grid_state.bumpy()
            holes = grid_state.holes()

            network.compute(torch.Tensor(state).unsqueeze(0).to(network.device))
            extras = network.cache["extra"]
            state_value = network.critic()
            print(f"State Value: {grid_state.reward_metric(0)} | Reward {reward} | Critic value: {state_value.item()}")
            print(f"State Height: {agg} | Predicted Height: {extras['height'].item()}")
            print(f"State Bumpy: {bump} | Predicted Bumpy: {extras['bump'].item()}")
            print(f"State Holes: {holes} | Predicted Holes: {extras['holes'].item()}")
            if game_over:
                state = matris.reset()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--max_episode_length", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save_frequency", type=int, default=5)
    parser.add_argument("--load_file", type=str, default="ppo2.pt")
    parser.add_argument("--gui_test", action="store_true")
    args = parser.parse_args()

    if args.gui_test:
        gui_test()
        return

    engine = tetris.Matris()
    network = PPONetwork().to(device)
    generator = PPOExperienceGenerator(engine, network, runs=args.runs, max_episode_length=args.max_episode_length)

    trainer = PPOTrainer(network, generator, load_file=args.load_file)

    counter = 0
    try:
        while True:
            trainer.train(args.epochs)

            counter += 1
            counter %= args.save_frequency
            if counter == 0:
                trainer.model.save(args.load_file)
    except KeyboardInterrupt:
        network.save(args.load_file)

if __name__ == "__main__":
    main()