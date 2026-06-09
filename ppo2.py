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

DEFAULT_ACTOR_LEARN_RATE = 2e-5
DEFAULT_CRITIC_LEARN_RATE = 1e-4
DEFAULT_CONV_LEARN_RATE = 5e-5
DROPOUT_CHANCE = 0.2
DEFAULT_GAMMA = 0.99
DEFAULT_LAMBDA = 0.1
SHUFFLE_EXPERIENCES = False

ACTOR_OUTPUT = 5
CRITIC_OUTPUT = 1

def detach_data_for_ac(data_tuple):
    ff_height_out, ff_bump_out, ff_holes_out, height, conv2d_bump2, conv2d_holes2, conv2d_ac2 = data_tuple
    # ff_height_out = ff_height_out.detach()
    # ff_bump_out = ff_bump_out.detach()
    # ff_holes_out = ff_holes_out.detach()
    height = height.detach()
    conv2d_bump2 = conv2d_bump2.detach()
    conv2d_holes2 = conv2d_holes2.detach()

    return torch.cat([height, conv2d_bump2, conv2d_holes2, conv2d_ac2], dim=1)

# http://vision.stanford.edu/teaching/cs231n/reports/2016/pdfs/121_Report.pdf
class AdjustedSandfordACNetwork(nn.Module, net.ActorCriticNetwork):
    def __init__(self, actor_output=ACTOR_OUTPUT, critic_output=CRITIC_OUTPUT, p=DROPOUT_CHANCE):
        super().__init__()
        self.cache = {}
        self.device = device

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
                {"params": self.conv_filters.parameters(), "lr": DEFAULT_CONV_LEARN_RATE},
                {"params": self.feed_forward_actor.parameters(), "lr": DEFAULT_ACTOR_LEARN_RATE},
                {"params": self.feed_forward_critic.parameters(), "lr": DEFAULT_CRITIC_LEARN_RATE},
                {"params": self._actor.parameters(), "lr": DEFAULT_ACTOR_LEARN_RATE},
                {"params": self._critic.parameters(), "lr": DEFAULT_CRITIC_LEARN_RATE},
            ]
        )


    def zero(self):
        self.optimizer.zero_grad()

    def calculate_internal_state(self, x):
        conv_out = self.conv_filters(x)

        flat = conv_out.flatten(start_dim=1)
        feed_out_actor = self.feed_forward_actor(flat)
        feed_out_critic = self.feed_forward_critic(flat)

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

class PPOExperienceGenerator:
    def __init__(self, engine: tetris.Matris, model: net.ActorCriticNetwork, runs=10, max_episode_length=10000):
        self.engine: tetris.Matris = engine
        self.model: net.ActorCriticNetwork = model
        self.runs = runs
        self.max_episode_length = max_episode_length


    def generate(self):
        buffer = ERMBuffer[PPOExperience]()
        clearing_buffer = ERMBuffer[PPOExperience]()
        self.model.eval()
        runs_progress = tqdm(
            range(self.runs),
            desc="Runs",
            dynamic_ncols=True,
            leave=False,
            position=1
        )
        episode_progress = tqdm(
            total=self.max_episode_length,
            desc="Experiences",
            dynamic_ncols=True,
            leave=False,
            position=2
        )
        for _ in runs_progress:
            state = self.engine.reset()
            trajectory = Trajectory(PPOExperience)
            clearing_actions = Trajectory(PPOExperience)
            episode_progress.reset()
            for i in range(self.max_episode_length):
                state_tensor = torch.Tensor(state).unsqueeze(0).to(self.model.device)

                with torch.no_grad():
                    self.model.compute(state_tensor)
                    state_value = self.model.critic()
                    logits = self.model.act()

                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logprob = dist.log_prob(action)

                state, reward, game_over, lines_cleared = self.engine.step(tetris.Action(action.item()))
                reward = torch.tensor([reward]).to(self.model.device)
                go = torch.tensor([int(game_over)]).to(self.model.device)

                experience = PPOExperience(state_tensor.detach(), action, reward, go, logprob, state_value)
                if lines_cleared > 0:
                    clearing_actions.append(experience)
                trajectory.append(experience)

                if game_over:
                    break
                episode_progress.update(1)
            state_tensor = torch.Tensor(state).unsqueeze(0).to(self.model.device)
            self.model.compute(state_tensor)
            state_value = self.model.critic()
            trajectory.set_last_value(state_value)
            buffer.append(trajectory)

        return buffer, clearing_buffer

class PPOTrainer:
    def __init__(self, model: net.ActorCriticNetwork, generator: PPOExperienceGenerator, gamma=DEFAULT_GAMMA, gae_discount=DEFAULT_LAMBDA, entropy=0.01, clip_epsilon=0.2, batch_size=64, load_file=None):
        self.model = model
        self.generator = generator
        self.gamma = gamma
        self.gae_discount = gae_discount
        self.entropy = entropy
        self.clip_epsilon = clip_epsilon
        self.batch_size = batch_size
        self.load_file = load_file

        self.model.load(load_file)

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
        clipped_surrogate = torch.clamp(importance_ratio, 1 - self.clip_epsilon,
                                        1 + self.clip_epsilon) * b_advantages

        actor_loss = -(torch.min(surrogate, clipped_surrogate) + self.entropy * entropy)
        critic_loss = F.mse_loss(state_values, b_returns)

        loss = self.model.extra_learn(b_state)

        self.model.zero()
        loss += (actor_loss + 0.01 * critic_loss * 0.5).mean()
        loss.backward()
        progress.set_postfix(loss=loss.item())
        self.model.step()

        return loss.item(), critic_loss.mean().item(), actor_loss.mean().item()

    def train(self, epochs):
        t0 = time.process_time()
        with torch.no_grad():
            buffer, clearing_buffer = self.generator.generate()
            t1 = time.process_time()

            tensors = buffer.to_tensors()
            advantages, returns = buffer.compute_gae(self.gamma, self.gae_discount)

            dataset = torch.utils.data.TensorDataset(
                tensors["state"],
                tensors["action"],
                tensors["logprob"],
                advantages,
                returns
            )

            loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=SHUFFLE_EXPERIENCES)

            if len(clearing_buffer) > 0:
                clearing_tensors = clearing_buffer.to_tensors()
                clear_advantages, clear_returns = clearing_buffer.compute_gae(self.gamma, self.gae_discount)

                clearing_dataset = torch.utils.data.TensorDataset(
                    clearing_tensors["state"],
                    clearing_tensors["action"],
                    clearing_tensors["logprob"],
                    clear_advantages,
                    clear_returns
                )

                clearing_loader = torch.utils.data.DataLoader(clearing_dataset, batch_size=self.batch_size, shuffle=True)

        epochs_progress = tqdm(
            range(epochs),
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
            if len(clearing_buffer) > 0:
                batch = next(iter(clearing_loader), None)
                if batch:
                    avg_loss, critic_loss, actor_loss= self.step(batch, epochs_progress)
                    total_loss += avg_loss
                    total_critic_loss += critic_loss
                    total_actor_loss += actor_loss
                    total_batches += 1

            for batch in loader:
                avg_loss, critic_loss, actor_loss= self.step(batch, epochs_progress)
                total_loss += avg_loss
                total_critic_loss += critic_loss
                total_actor_loss += actor_loss
                total_batches += 1

        t2 = time.process_time()
        clearing_average_returns, _ = clearing_buffer.compute_returns(self.gamma)
        clearing_average_returns = clearing_average_returns.mean()
        average_returns, _ = buffer.compute_returns(self.gamma)
        average_returns = average_returns.mean()
        time_taken = t2 - t0
        collection_time = t1 - t0
        return (average_returns, clearing_average_returns, time_taken,
                collection_time, total_loss / total_batches, total_critic_loss / total_batches, total_actor_loss / total_batches)



def gui_test(args):
    pygame.init()
    screen = pygame.display.set_mode((tetris.WIDTH, tetris.HEIGHT))
    pygame.display.set_caption("MaTris")
    matris = tetris.Matris()
    state = matris.reset()
    game = tetris.Game()
    game.main(screen, matris)

    network = AdjustedSandfordACNetwork().to(device)
    network.load(args.load_file)
    network.eval()
    run = False
    best = False

    episodes = []
    probs = []
    returns = []

    try:
        while True:
            # game.clock.tick(120)
            actions = game.get_user_actions()
            if game.is_key(pygame.K_r):
                run = not run

            if game.is_key(pygame.K_b):
                best = not best
                run = False

            if game.is_key(pygame.K_v) or run:
                network.compute(torch.Tensor(state).unsqueeze(0).to(network.device))
                logits = network.act()
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample().item()
                # print(f"Action taken: {action} with dist {dist.probs.tolist()}")
                probs.append(dist.probs.squeeze())
                actions.append(tetris.Action(action))

            if best and not run:
                actions.extend(matris.best_action_set())

            if len(actions) == 0:
                game.redraw()
                continue

            for action in actions:
                network.compute(torch.Tensor(state).unsqueeze(0).to(network.device))
                print(f"Critic says state is: {network.critic().item()} | ", end='')
                next_state, reward, game_over, lines_cleared = matris.step(action)
                print(f"Reward was: {reward}")
                returns.append(reward)
                game.redraw()
                state = next_state
                # time.sleep(0.1)
                # print(f"Reward Metric: {matris.grid.brett_reward_metric()}")
                # grid_state = tetris.Grid(MATRIX_HEIGHT, MATRIX_WIDTH).from_state(state)
                # bump, agg, heights = grid_state.bumpy()
                # holes = grid_state.holes()

                # network.compute(torch.Tensor(state).unsqueeze(0).to(network.device))
                # extras = network.cache["extra"]
                # state_value = network.critic()
                # print(f"Reward {reward} | Critic value: {state_value.item()}")

                # print(f"State Height: {agg} | Predicted Height: {extras['height'].item()}")
                # print(f"State Bumpy: {bump} | Predicted Bumpy: {extras['bump'].item()}")
                # print(f"State Holes: {holes} | Predicted Holes: {extras['holes'].item()}")
                if game_over:
                    state = matris.reset()
                    np_rewards = np.array(returns)

                    discounted_rewards = np_rewards.copy()
                    for i in reversed(range(len(returns) - 1)):
                        discounted_rewards[i] = discounted_rewards[i + 1] * DEFAULT_GAMMA + np_rewards[i]

                    if len(probs) > 0:
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
            plt.close(graph.plot_episode_action_probabilities_full(episode, i))
            plt.close(graph.plot_episode_action_probabilities(episode, i))
            plt.close(graph.plot_rewards_and_discounted_returns(episode, i, DEFAULT_GAMMA))

        pygame.image.save(screen, f"episode.png")
    except Exception as e:
        raise e

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--teacher_runs", type=int, default=1)
    parser.add_argument("--max_episode_length", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save_frequency", type=int, default=5)
    parser.add_argument("--load_file", type=str, default="ppo6.pt")
    parser.add_argument("--gui_test", action="store_true")
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    args = parser.parse_args()

    if args.gui_test:
        gui_test(args)
        return

    engine = tetris.Matris()
    network = AdjustedSandfordACNetwork().to(device)
    generator = PPOExperienceGenerator(engine, network, runs=args.runs, max_episode_length=args.max_episode_length)

    trainer = PPOTrainer(network, generator, gamma=args.gamma, load_file=args.load_file)

    progress = tqdm(
        total=None,
        desc="[PPO] Training",
        unit=" round",
        dynamic_ncols=True,
        position=0
    )

    counter = 0
    average_rets = 0
    loss_over_time = []
    try:
        while True:
            rets, clear_rets, t, collects, average_loss, critic_loss, actor_loss = trainer.train(args.epochs)
            amounts = 2
            mins, maxes = 1 - 1 / amounts, 1 / amounts
            average_rets = average_rets * mins + rets * maxes


            counter += 1
            if counter % args.save_frequency == 0:
                trainer.model.save(args.load_file)

            loss_over_time.append(average_loss)

            progress.update(1)
            progress.set_postfix({
                "Return": f"{float(rets):.6f}",
                "Avg Loss": f"{float(average_loss):.6f}",
                "Actor Loss": f"{float(actor_loss):.6f}",
                "Critic Loss": f"{float(critic_loss):.6f}",
                "Avg Return": f"{average_rets:.6f}",
                "Clearing": f"{float(clear_rets):.6f}",
                "Time": f"{t:.2f}s",
                "Collection": f"{collects:.2f}s",
            })
    except KeyboardInterrupt:
        network.save(args.load_file)

    plt.plot(loss_over_time)
    plt.savefig("loss.png")
    plt.show()

if __name__ == "__main__":
    main()