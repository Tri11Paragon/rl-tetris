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


class PPONetwork(nn.Module, net.ActorCriticNetwork):
    def __init__(self, learn_rate=DEFAULT_ACTOR_LEARN_RATE, actor_output=ACTOR_OUTPUT, critic_output=CRITIC_OUTPUT, p=DROPOUT_CHANCE, eval_now = False):
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

        if eval_now:
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
        try:
            if file:
                torch.save(self.state_dict(), file)
        except KeyboardInterrupt:
            print("wait until save!")
            self.save(file)

    def load(self, file):
        if file is not None and pathlib.Path(file).exists():
            self.load_state_dict(torch.load(file, weights_only=True, map_location=self.device))

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