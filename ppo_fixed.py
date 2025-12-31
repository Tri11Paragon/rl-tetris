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
from experience import Experience, PPOExperience
import argparse
import time

from collections import namedtuple, deque

from MaTris.matris import GameOver

import network as net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} | HIP: {getattr(torch.version, 'hip', None) or getattr(torch.version, 'cuda', None)}")

class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, exc_type, exc, tb):
        self.dt = time.perf_counter() - self.t0

class EvaluatorNetwork(torch.nn.Module):
    def __init__(self, output_shape = 5, lr=0.001):
        super().__init__()
        p = 0.25

        self.conv2d_column = net.make_conv2d(2, 32, kernel_size=(22, 1))
        self.conv2d_row = net.make_conv2d(2, 32, kernel_size=(1, 10))
        self.conv2d = net.make_conv2d(2, 32, kernel_size=(3, 3))

        self.l1 = net.make_lazy_linear(320, p)
        self.c1 = net.make_conv1d(1, 16, kernel_size=1)
        self.fc = nn.Sequential(
            net.make_lazy_linear(512, p),
            nn.Linear(512, output_shape),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        c1 = self.conv2d(x).flatten(start_dim=1)
        c2 = self.conv2d_row(x).flatten(start_dim=1)
        c3 = self.conv2d_column(x).flatten(start_dim=1)
        x = torch.cat([c1, c2, c3], dim=1)
        x = self.l1(x).unsqueeze(1)
        x = self.c1(x).flatten(start_dim=1)
        x = self.fc(x)
        return x

    def save(self, file):
        if file:
            torch.save(self.state_dict(), file)

    def load(self, file):
        if file is not None and pathlib.Path(file).exists():
            self.load_state_dict(torch.load(file, weights_only=True, map_location=device))

def normalize_adv(x):
    return (x - x.mean()) / (x.std() + 1e-8)

class TrajectoryBuffer:
    def __init__(self):
        self.buffer = []

    def clear(self):
        self.buffer = []

    def append(self, experience):
        self.buffer.append(experience)

    def to_tensors(self):
        # Efficiently convert list of namedtuples to tensors
        experience = PPOExperience(*zip(*self.buffer))

        state = torch.tensor(np.array(experience.state), dtype=torch.float32).to(device)
        action = torch.stack(experience.action).squeeze().detach()
        logprob = torch.stack(experience.logprob).squeeze().detach()
        state_value = torch.stack(experience.state_value).squeeze().detach()

        return PPOExperience(state, action, None, None, logprob, state_value), experience

    def compute_gae(self, gamma, lamda, last_value, last_done):
        tensors, experience = self.to_tensors()
        rewards = torch.tensor(experience.reward, dtype=torch.float32).to(device)
        dones = torch.tensor(experience.done, dtype=torch.float32).to(device)
        values = tensors.state_value
        
        N = len(self.buffer)
        gae = torch.zeros(N).to(device)
        returns = torch.zeros(N).to(device)
        
        advantage = 0
        next_value = last_value
        
        # Iterate backwards to compute GAE and returns correctly
        for t in reversed(range(N)):
            # Masking terminal states: if dones[t] is True, the next state is 0-valued
            mask = 1.0 - dones[t]
            
            # TD error delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = rewards[t] + gamma * next_value * mask - values[t]
            
            # GAE: A_t = delta_t + gamma * lambda * mask * A_{t+1}
            # The mask here ensures that we reset the accumulation at episode boundaries
            advantage = delta + gamma * lamda * mask * advantage
            gae[t] = advantage
            
            # Target return for critic training (Q-estimate)
            returns[t] = gae[t] + values[t]
            
            next_value = values[t]
            
        return normalize_adv(gae), returns, tensors

    def __len__(self):
        return len(self.buffer)

class PPO:
    def __init__(self, engine, actor_lr=1e-3, critic_lr=1e-3, gamma = 0.99, gae_discount=0.95, entropy = 0.01, clip_epsilon=0.2, load_file=None):
        self.engine = engine
        self.gamma = gamma
        self.gae_discount = gae_discount
        self.entropy = entropy
        self.clip_epsilon = clip_epsilon
        self.actor = EvaluatorNetwork(lr=actor_lr).to(device)
        self.actor.load(load_file + ".actor")
        self.old_actor = EvaluatorNetwork(lr=actor_lr).to(device)
        self.old_actor.load_state_dict(self.actor.state_dict())
        self.critic = EvaluatorNetwork(output_shape=1, lr=critic_lr).to(device)
        self.critic.load(load_file + ".critic")
        self.file = load_file

    def learn(self, training_steps, epochs, runs = 1000, max_episode_length = 1000, minibatch_size = 256):
        for _ in range(training_steps):
            t0 = time.process_time()
            buffer = TrajectoryBuffer()

            print(f"------[Run {_}]------")

            self.old_actor.load_state_dict(self.actor.state_dict())
            self.old_actor.eval()
            self.critic.eval()

            state = self.engine.reset()
            game_over = False
            average_episode_length = 0
            total_lines_cleared = 0
            
            for r in range(runs):
                for i in range(max_episode_length):
                    old_lines = self.engine.lines
                    state_tensor = torch.Tensor(state).unsqueeze(0).to(device)
                    
                    # Memory optimization: no_grad during collection
                    with torch.no_grad():
                        state_value = self.critic(state_tensor)
                        logits = self.old_actor(state_tensor)
                    
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                    logprob = dist.log_prob(action)
                    
                    old_state = state
                    state, reward, game_over = self.engine.step(tetris.Action(action.item()))

                    buffer.append(PPOExperience(old_state, action, reward, game_over, logprob, state_value))
                    average_episode_length += 1
                    total_lines_cleared += self.engine.lines - old_lines

                    if game_over:
                        state = self.engine.reset()
                        break
                        
            average_episode_length /= runs
            print(f"Average episode length: {average_episode_length} | Lines cleared: {total_lines_cleared}")

            t1 = time.process_time()
            
            # Bootstrap value for the last state in the buffer (if not terminal)
            with torch.no_grad():
                last_state_tensor = torch.Tensor(state).unsqueeze(0).to(device)
                last_value = self.critic(last_state_tensor).squeeze()

            advantages, returns, batch_experiences = buffer.compute_gae(self.gamma, self.gae_discount, last_value, game_over)

            dataset = torch.utils.data.TensorDataset(
                batch_experiences.state,
                batch_experiences.action,
                batch_experiences.logprob,
                advantages,
                returns
            )

            loader = torch.utils.data.DataLoader(dataset, batch_size=minibatch_size, shuffle=True)

            self.actor.train()
            self.critic.train()
            for e in range(epochs):
                for b_state, b_action, b_logprob, b_advantages, b_returns in loader:
                    logits = self.actor(b_state)
                    dist = torch.distributions.Categorical(logits=logits)
                    action_logprob = dist.log_prob(b_action)
                    entropy = dist.entropy()
                    
                    state_values = self.critic(b_state).squeeze()

                    # PPO Surrogate Objective
                    importance_ratio = torch.exp(action_logprob - b_logprob)
                    surrogate = importance_ratio * b_advantages
                    clipped_surrogate = torch.clamp(importance_ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * b_advantages
                    
                    # Minimize negative of surrogate objective + entropy bonus
                    actor_loss = -torch.min(surrogate, clipped_surrogate).mean() - self.entropy * entropy.mean()
                    critic_loss = F.mse_loss(state_values, b_returns)

                    # Update Actor
                    self.actor.optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor.optimizer.step()

                    # Update Critic
                    self.critic.optimizer.zero_grad()
                    critic_loss.backward()
                    self.critic.optimizer.step()

            t2 = time.process_time()
            print(f"Time taken: {t2 - t0:.6f} | Collection time {t1 - t0:.6f} | Train time {t2 - t1:.6f}")
            if _ % 5 == 0:
                self.actor.save(self.file + ".actor")
                self.critic.save(self.file + ".critic")

def train_ppo():
    matris = tetris.Matris()
    ppo = PPO(matris, load_file="ppo")
    while True:
        ppo.learn(1000, 10, 10)

def gui_test():
    pygame.init()
    screen = pygame.display.set_mode((tetris.WIDTH, tetris.HEIGHT))
    pygame.display.set_caption("MaTris")
    matris = tetris.Matris()
    game = tetris.Game()
    game.main(screen, matris)

    state = matris.reset()
    ppo = PPO(matris, load_file="ppo")
    ppo.actor.eval()
    ppo.critic.eval()
    run = False

    while True:
        game.clock.tick(50)
        actions = game.get_user_actions()
        if game.is_key(pygame.K_r):
            run = not run

        if game.is_key(pygame.K_v) or run:
            logits = ppo.actor(torch.Tensor(state).unsqueeze(0).to(device))
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
            state_value = ppo.critic(torch.Tensor(state).unsqueeze(0).to(device))
            print(f"State Value: {grid_state.reward_metric(0)} | Critic value: {state_value.item()}")
            if game_over:
                state = matris.reset()

def test(runs):
    matris = tetris.Matris()
    ppo = PPO(matris, load_file="ppo")
    average_episode_length = 0
    average_total_reward = 0
    average_lines_cleared = 0
    for r in range(runs):
        state = matris.reset()
        episode_length = 0
        total_reward = 0
        lines_cleared = 0
        while True:
            old_lines = matris.lines
            action = torch.argmax(ppo.actor(torch.Tensor(state).unsqueeze(0).to(device))).item()
            next_state, reward, game_over = matris.step(tetris.Action(action))
            state = next_state
            episode_length += 1
            total_reward += reward
            lines_cleared += matris.lines - old_lines
            if game_over:
                average_episode_length += episode_length
                average_total_reward += total_reward
                average_lines_cleared += lines_cleared
                print(f"Episode {r}: {episode_length} | Total reward: {total_reward} | Lines cleared: {lines_cleared}")
                break
    print(f"Average episode length: {average_episode_length/runs} | Average reward: {average_total_reward/runs} | Average lines: {average_lines_cleared/runs}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", nargs="?", const=100, default=False, type=int)
    parser.add_argument("--gui_test", action="store_true")
    args = parser.parse_args()
    if args.gui_test:
        gui_test()
    elif args.test:
        test(args.test)
    elif args.train:
        train_ppo()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting...")
