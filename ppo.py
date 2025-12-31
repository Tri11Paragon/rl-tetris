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

def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-8)
    # return x - x.mean()

class TrajectoryBuffer:
    def __init__(self):
        self.buffer = []

    def clear(self):
        self.buffer = []

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def sample_batch(self, batch_size):
        return PPOExperience(*zip(*self.sample(batch_size)))

    def get_batch(self):
        return PPOExperience(*zip(*self.buffer))

    def to_tensors(self):
        experience = self.get_batch()

        state = torch.tensor(np.array(experience.state), dtype=torch.float32).to(device)
        action = torch.stack(experience.action).squeeze().detach()
        logprob = torch.stack(experience.logprob).squeeze().detach()
        state_value = torch.stack(experience.state_value).squeeze().detach()

        return PPOExperience(state, action, None, None, logprob, state_value), experience

    def compute_returns(self, gamma):
        # Standard reward calculation
        returns = []
        discounted_reward = 0
        for experience in reversed(self.buffer):
            if experience.done:
                discounted_reward = 0
            discounted_reward = experience.reward + (gamma * discounted_reward)
            returns.insert(0, discounted_reward)
        returns = torch.tensor(np.array(returns), dtype=torch.float32).to(device)
        print(f"Total discounted return: {returns.sum()}")
        print(f"Average return {returns.mean()}")
        print(f"Std return {returns.std()}")
        print(f"Min return {returns.min()}")
        print(f"Max return {returns.max()}")
        # Normalize the rewards, add a small value to prevent dividing by zero
        returns = normalize(returns)
        return returns

    def compute_gae(self, gamma, lamda, last_value):
        tensors, experience = self.to_tensors()
        values = tensors.state_value
        done = torch.tensor(experience.done, dtype=torch.int32)
        # Normalize reward
        reward = torch.tensor(experience.reward, dtype=torch.float32)
        reward = normalize(reward)

        gae = torch.zeros(len(self.buffer))
        returns = torch.zeros(len(self.buffer))
        next_value = last_value

        # Calculate GAE using algorithm presented on slides (updated to be O(n))
        estimate = 0
        for i in reversed(range(len(self.buffer))):
            non_terminal = 1 - int(done[i])
            next_state_value = gamma * next_value * non_terminal
            funny_s = reward[i] + next_state_value - tensors.state_value[i]
            estimate = (funny_s + (gamma * lamda) * estimate) * non_terminal
            gae[i] = estimate
            returns[i] = reward[i] + tensors.state_value[i]
            next_value = values[i]

        # Normalize advantage. Disabled because I'm not sure if it is needed or a good idea.
        gae = gae.to(device)
        gae = normalize(gae)
        return gae, returns, tensors


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

            self.old_actor.train()
            self.old_actor.load_state_dict(self.actor.state_dict())
            self.old_actor.eval()
            self.critic.eval()

            state = self.engine.reset()
            average_episode_length = 0
            total_lines_cleared = 0
            for r in range(runs):
                for i in range(max_episode_length):
                    old_lines = self.engine.lines
                    state_tensor = torch.Tensor(state).unsqueeze(0).to(device)
                    state_value = self.critic.forward(state_tensor)
                    with torch.no_grad():
                        logits = self.old_actor.forward(state_tensor)
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
            # batch_experiences = buffer.to_tensors()
            # discounted_returns = buffer.compute_returns(self.gamma)

            # print(f"{discounted_returns.shape} vs {batch_experiences.state_value.shape}")
            # Use the discounted return as the Q estimate,
            # and the previous critic value for the state as the value function estimate
            # advantages = discounted_returns - batch_experiences.state_value

            with torch.no_grad():
                last_state_tensor = torch.Tensor(state).unsqueeze(0).to(device)
                last_value = self.critic(last_state_tensor).squeeze()

            advantages, returns, batch_experiences = buffer.compute_gae(self.gamma, self.gae_discount, last_value)

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
                actor_losses = []
                critic_losses = []
                for b_state, b_action, b_logprob, b_advantages, b_returns in loader:
                    b_state = b_state.to(device)
                    b_action = b_action.to(device)
                    b_logprob = b_logprob.to(device)
                    b_advantages = b_advantages.to(device)
                    b_returns = b_returns.to(device)
                    logits = self.actor.forward(b_state)
                    dist = torch.distributions.Categorical(logits=logits)

                    # Given the current state, calculate what the current actor's probability of selecting the action
                    action_logprob = dist.log_prob(b_action)
                    entropy = dist.entropy()
                    state_values = self.critic.forward(b_state).squeeze()

                    # I've seen this used a lot on implementations online and was very confused as to why it worked.
                    # I tried using just the normal division, but that didn't work and lead to NaNs
                    # https://www.desmos.com/calculator/fsicwr9dkq
                    # The logprobs. Obviously, we are working in log space. Subtracting the logs,
                    # raising to the power of e leads to dividing the underlying probabilities. No NaNs too!
                    # This is very neat!!!! (it all comes together)
                    importance_ratio = torch.exp(action_logprob - b_logprob)
                    # importance_ratio = action_logprob / batch_experiences.logprob

                    # print(importance_ratio - importance_ratio2)

                    surrogate = importance_ratio * b_advantages
                    clipped_surrogate = torch.clamp(importance_ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * b_advantages
                    # lambda is a reserved word and I don't like using unicode characters in code.
                    actor_losses.append(torch.min(surrogate, clipped_surrogate) + self.entropy * entropy)

                    mse = nn.MSELoss()
                    critic_losses.append(mse(state_values, b_returns))

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                (-torch.cat(actor_losses).mean()).backward()
                torch.stack(critic_losses).mean().backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

            t2 = time.process_time()
            print(f"Time taken: {t2 - t0:.6f} | Step time {t1 - t0:.6f} | Train time {t2 - t1:.6f}")
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
            logits = ppo.actor.forward(torch.Tensor(state).unsqueeze(0).to(device))
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()
            # action = torch.argmax(torch.softmax(logits, 1)).item()
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
            state_value = ppo.critic.forward(torch.Tensor(state).unsqueeze(0).to(device))
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
            action = torch.argmax(ppo.actor.forward(torch.Tensor(state).unsqueeze(0).to(device))).item()
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
    average_episode_length /= runs
    average_total_reward /= runs
    average_lines_cleared /= runs
    print(f"Average episode length: {average_episode_length} | "
          f"Average total reward: {average_total_reward} | "
          f"Average lines cleared: {average_lines_cleared}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", nargs="?", const=100, default=False, type=int)
    parser.add_argument("--gui_test", action="store_true")

    args = parser.parse_args()

    if args.gui_test:
        gui_test()
        return

    if args.test:
        test(args.test)
        return

    if args.train:
        train_ppo()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting...")