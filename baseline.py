import argparse
import itertools
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

from experience import Experience

from collections import namedtuple, deque

from MaTris.matris import GameOver

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} | HIP: {getattr(torch.version, 'hip', None)}")
# device = torch.device("cpu")


class PolicyType:
    def __call__(self, logits, step):
        pass

class BoltzmannPolicy(PolicyType):
    def __init__(self, decay = 1, temperature=1.0):
        self.temperature = temperature
        self.decay = decay

    def __call__(self, logits, step):
        temp = self.temperature * math.pow(self.decay, step)
        return torch.softmax((logits - logits.max()) / temp, dim=0)

class GreedyPolicy(PolicyType):
    def __call__(self, logits, step):
        logits = logits.squeeze(0)
        probs = torch.zeros_like(logits)
        probs[torch.argmax(logits)] = 1
        return probs

class EpsilonGreedyPolicy(PolicyType):
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def __call__(self, logits, step):
        logits = logits.squeeze(0)
        action_e = self.epsilon / logits.shape[0]
        probs = torch.ones_like(logits) * action_e
        probs[torch.argmax(logits)] += action_e + (1 - self.epsilon)
        return probs

class RandomPolicy(PolicyType):
    def __call__(self, logits, step):
        logits = logits.squeeze(0)
        zeros = torch.zeros_like(logits)
        zeros[random.randrange(logits.shape[0])] = 1
        return zeros

class ERM:
    def __init__(self, maxlen = 10000):
        self.experts = {}
        self.buffer = deque(maxlen=maxlen)
        self.weights = np.ones(maxlen) / maxlen

    def full(self):
        return self.buffer.maxlen == len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), size=batch_size, replace=False, p=self.weights)
        return deque(self.buffer[i] for i in idx)

    def sample_batch(self, batch_size):
        return Experience(*zip(*self.sample(batch_size)))

    def recalculate_sweep(self, gamma, network, epsilon = 0.01, n = 1):
        with torch.no_grad():
            network.eval()
            buffer = list(self.buffer)
            for i in range(0, len(self.buffer), 128):
                end = min(i + 128, len(self.buffer))
                experiences = Experience(*zip(*buffer[i:end]))

                states = torch.tensor(np.array(experiences.state), dtype=torch.float32).to(device)
                next_states = torch.tensor(np.array(experiences.next_state), dtype=torch.float32).to(device)
                actions = torch.tensor(np.array(experiences.action), dtype=torch.int32).to(device).unsqueeze(1)
                rewards = torch.tensor(np.array(experiences.reward), dtype=torch.float32).to(device)
                dones = torch.tensor(np.array(experiences.done), dtype=torch.int32).to(device)

                next_state_values = torch.max(network(next_states), dim=1).values.detach()

                predicted_values = rewards + gamma * next_state_values * dones
                state_values = network(states).gather(1, actions).squeeze(1)

                td_error = torch.abs(predicted_values - state_values)
                self.weights[i:end] = np.pow(td_error.cpu().numpy() + epsilon, n)
        self.weights /= self.weights.sum()

        network.train()

    def load_expert_file(self, file):
        with open(file, "rb") as f:
            self.experts[file] = pickle.load(f)
            print(f"Loaded expert file {file} with {len(self.experts[file])} experiences")

    def load(self, file, amount = -1):
        if self.experts.get(file) is None:
            self.load_expert_file(file)
        if amount == -1:
            amount = len(self.experts[file])
        data = random.sample(self.experts[file], amount)
        self.buffer.extend(data)

    def __len__(self):
        return len(self.buffer)

class DQNNetwork(torch.nn.Module):
    def __init__(self, output_shape = 5, lr=0.001):
        super().__init__()
        p = 0.25

        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=(18, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.LazyLinear(128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p),

            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p),

            nn.Linear(512, output_shape),
        )

        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=lr, momentum=0.95)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

    def save(self, file):
        if file:
            torch.save(self.state_dict(), file)

    def load(self, file):
        if file is not None and pathlib.Path(file).exists():
            self.load_state_dict(torch.load(file, weights_only=True, map_location=device))

class DQN:
    def __init__(self, engine, policy: PolicyType = BoltzmannPolicy(), erm_size = 10000, lr=0.001, gamma = 0.95, uprate = 0.001, sweep_rate=100, load_file=None):
        self.gamma = gamma
        self.network = DQNNetwork(lr=lr).to(device)
        self.target_network = DQNNetwork(lr=lr).to(device)
        self.network.load(load_file)
        self.target_network.load_state_dict(self.network.state_dict())
        self.erm = ERM(erm_size)
        self.engine = engine
        self.uprate = uprate
        self.sweep_rate = sweep_rate
        self.policy = policy

    def run_network(self, state, step):
        logits = self.network(torch.tensor(state).unsqueeze(0).to(device))
        probs = self.policy(logits, step)
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample().item()
        return action

    def run_expert(self, engine, state, max_steps = 1):
        actions = engine.best_action_set()
        for action in itertools.islice(actions, max_steps):
            next_state, reward, gameover = self.engine.step(action)
            self.erm.append(Experience(state, action.value, reward, next_state, 0 if gameover else 1))

            state = next_state

            if gameover:
                state = self.engine.reset()

        return state

    def fill_erm(self, engine):
        state = self.engine.reset()
        while not self.erm.full():
            self.run_expert(engine, state, 1024)

    def train(self, max_steps, batches = 128, updates = 1, batch_size = 128, experiences = 128, save_file = None):
        state = self.engine.reset()
        average_reward = 0
        for step in range(max_steps):
            print(f"Running step {step}")
            with torch.no_grad():
                self.network.eval()
                total_reward = 0
                for k in range(experiences):
                    if random.random() < 0.01:
                        state = self.run_expert(self.engine, state, 1)
                    else:
                        action = self.run_network(state, step)
                        next_state, reward, gameover = self.engine.step(action)
                        self.erm.append(Experience(state, action, reward, next_state, 0 if gameover else 1))
                        total_reward += reward

                        state = next_state

                        if gameover:
                            state = self.engine.reset()
                self.network.train()

            print(f"Total reward of experiences: {total_reward}")
            average_reward += 1./max_steps * total_reward

            if len(self.erm) < batch_size:
                continue
            for b in range(batches):
                batch = self.erm.sample_batch(batch_size)

                states = torch.tensor(np.array(batch.state), dtype=torch.float32).to(device)
                next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(device)
                actions = torch.tensor(np.array(batch.action), dtype=torch.int32).to(device).unsqueeze(1)
                rewards = torch.tensor(np.array(batch.reward), dtype=torch.float32).to(device).unsqueeze(1)
                dones = torch.tensor(np.array(batch.done), dtype=torch.int32).to(device).unsqueeze(1)

                for u in range(updates):
                    with torch.no_grad():
                        next_state_actions = torch.argmax(self.network(next_states), dim=1).unsqueeze(1)
                        next_state_values = self.target_network(next_states).gather(1, next_state_actions).detach()
                    predicted_values = rewards + self.gamma * next_state_values * dones
                    state_values = self.network(states).gather(1, actions)

                    self.network.optimizer.zero_grad()
                    # criterion = nn.SmoothL1Loss()
                    criterion = nn.MSELoss()
                    loss = criterion(state_values, predicted_values)
                    loss.backward()
                    self.network.optimizer.step()

            if self.uprate < 1:
                # Polyak update
                target_net_state_dict = self.target_network.state_dict()
                policy_net_state_dict = self.network.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.uprate + target_net_state_dict[key]*(1-self.uprate)
                self.target_network.load_state_dict(target_net_state_dict)
            else:
                # Replacement update
                if step % int(self.uprate) == 0:
                    self.target_network.load_state_dict(self.network.state_dict())

            if step % 50 == 0:
                self.network.save(save_file)
            if step % self.sweep_rate == 0:
                self.erm.recalculate_sweep(self.gamma, self.target_network)

def gui_test(args):
    pygame.init()
    screen = pygame.display.set_mode((tetris.WIDTH, tetris.HEIGHT))
    pygame.display.set_caption("MaTris")
    matris = tetris.Matris()
    game = tetris.Game()
    game.main(screen, matris)

    engine = tetris.Matris()
    state = engine.reset()
    network = DQN(matris, policy=EpsilonGreedyPolicy(), load_file="dqn.pth", erm_size=100000, gamma=0.95, lr=2e-5, uprate=5, sweep_rate=25)
    network.network.eval()
    policy = GreedyPolicy()
    run = False

    episodes = []
    probs = []
    returns = []

    try:
        while True:
            # game.clock.tick(120)
            actions = game.get_user_actions()
            if game.is_key(pygame.K_r):
                run = not run

            if game.is_key(pygame.K_v) or run:
                logits = network.network(torch.tensor(state).unsqueeze(0).to(device))
                l_probs = policy(logits, 0)
                dist = torch.distributions.Categorical(probs=l_probs)
                action = tetris.Action(dist.sample().item())
                probs.append(dist.probs.squeeze())
                actions.append(tetris.Action(action))

            if len(actions) == 0:
                game.redraw()
                continue

            for action in actions:
                next_state, reward, game_over = matris.step(action)
                returns.append(reward)
                game.redraw()
                state = next_state
                if game_over:
                    state = matris.reset()
                    np_rewards = np.array(returns)

                    discounted_rewards = np_rewards.copy()
                    for i in reversed(range(len(returns) - 1)):
                        discounted_rewards[i] = discounted_rewards[i + 1] * 0.95 + np_rewards[i]

                    episodes.append( (torch.stack(probs), discounted_rewards, np_rewards) )
                    probs.clear()
                    returns.clear()
                    raise SystemExit("Game Over")
    except SystemExit or KeyboardInterrupt:
        for i, episode in enumerate(episodes):

            avg = episode[0].mean(dim=0)
            std = episode[0].std(dim=0)
            med = episode[0].median(dim=0).values
            print(f"Item (Samples: {episode[0].shape} |:| Average: {avg.tolist()} | Std: {std.tolist()} | Med: {med.tolist()} |:|")
            import graph
            from matplotlib import pyplot as plt
            plt.close(graph.plot_episode_action_probabilities_full(episode, i))
            plt.close(graph.plot_episode_action_probabilities(episode, i))
            plt.close(graph.plot_rewards_and_discounted_returns(episode, i, 0.95))

        pygame.image.save(screen, f"episode.png")
    except Exception as e:
        raise e

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui_test", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.gui_test:
        gui_test(args)
        return

    # if args.test:
    #     compute_measures(args)
    #     return

    pygame.init()

    matris = tetris.Matris()

    deep_q = DQN(matris, policy=EpsilonGreedyPolicy(), load_file="dqn.pth", erm_size=100000, gamma=0.95, lr=2e-5, uprate=5, sweep_rate=10)
    deep_q.fill_erm(matris)
    policy = GreedyPolicy()

    rewards = []
    average_rewards = []
    eval = 0
    try:
        while True:
            steps = 0
            deep_q.train(1000, save_file = "dqn.pth", experiences=1024, batch_size=128, batches=1000)

            state = matris.reset()
            deep_q.network.eval()
            with torch.no_grad():
                eval += 1
                total_reward = 0
                while True:
                    steps += 1
                    logits = deep_q.network(torch.tensor(state).unsqueeze(0).to(device))
                    probs = policy(logits, 0)
                    dist = torch.distributions.Categorical(probs=probs)
                    action = tetris.Action(dist.sample().item())
                    next_state, reward, gameover = matris.step(action)
                    total_reward += reward
                    state = next_state
                    if gameover:
                        break
                rewards.append(total_reward)
                average_rewards.append(total_reward / steps)
                print(total_reward)
            deep_q.network.train()
    except KeyboardInterrupt:
        import graph
        from matplotlib import pyplot as plt
        plt.close(graph.plot_rewards_and_discounted_returns((torch.empty(0), np.array(average_rewards), np.array(rewards)), -1, 0.95))
        print(rewards)

if __name__ == "__main__":
    main()