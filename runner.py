import time

import numpy as np
import pygame
import random
from matplotlib import pyplot as plt

import MaTris.matris
import default_config
import experience
import graph
import ppo2 as ppo
# import dqn as dqn
from collections import defaultdict
import argparse
import json
import pickle
import os
import subprocess

import MaTris.matris as tetris
import torch
from tqdm.auto import tqdm
from pathlib import Path
from config import Config

from experience import PPOExperience

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest="command")

parser.add_argument("-l", "--location", type=str, default="experiments")
parser.add_argument("-s", "--seed", type=int, default=0)
parser.add_argument("--default", action="store_true")

parser_tester = subparsers.add_parser("test")
parser_tester.add_argument("file", type=str)
parser_tester.add_argument("type", type=str, choices=["ppo", "dqn"])
parser_tester.add_argument("-r", "--refresh", "--reset", action="store_true", help="Rebuild objects. Used to reset environment variables. "
                                                                            "Doesn't lose model training progress. ")

parser_trainer = subparsers.add_parser("train")
parser_trainer.add_argument("file", type=str)
parser_trainer.add_argument("type", type=str, choices=["ppo", "dqn"])
parser_trainer.add_argument("-r", "--refresh", "--reset", action="store_true", help="Rebuild objects. Used to reset environment variables. "
                                                                            "Doesn't lose model training progress. ")

parser_initializer = subparsers.add_parser("init")
parser_initializer.add_argument("file", type=str)


class Runner:
    def __init__(self, file, location="experiments"):
        self.file = file
        self.location = location
        self.folder = Path(location) / file
        self.config_path = self.folder / "config.nix"
        self.config = Config(self.config_path).load()
        self.object_storage = {}
        self.run_data = defaultdict(list, {
            "runs": 0
        })
        self.init_run()

    def __getitem__(self, item):
        return self.object_storage[item]

    def set(self, key, obj):
        self.object_storage[key] = obj
        return self.object_storage[key]

    def object(self, key, value):
        if key not in self.object_storage:
            self.object_storage[key] = value
        return self.object_storage[key]

    def init_run(self):
        self.folder.mkdir(parents=True, exist_ok=True)

        self.config = Config(self.config_path).load()

        # object_path = self.folder / "objects.pkl"
        # if object_path.exists():
        #     self.object_storage = pickle.load(object_path.open(mode="rb"))

        loss_path = self.folder / "state.json"
        if loss_path.exists():
            self.run_data = defaultdict(list, json.load(loss_path.open(mode="r")))

        self.save()

    def save_network(self, network_name):
        if network_name in self.object_storage:
            network = self.object_storage[network_name]
            network_path = self.folder / f"{network_name}.pt"
            network.save(network_path)

    def save(self):
        self.folder.mkdir(parents=True, exist_ok=True)

        # object_path = self.folder / "objects.pkl"
        # pickle.dump(self.object_storage, object_path.open(mode="wb"))

        loss_path = self.folder / "state.json"
        json.dump(self.run_data, loss_path.open(mode="w"), indent=4)

        self.save_network("network")
        self.save_network("target")

    def runs(self):
        return self.run_data["runs"]

    def increment_runs(self):
        self.run_data["runs"] += 1

    def loss(self, returns, loss, critic, actor):
        self.run_data["returns"].append(float(returns))
        self.run_data["loss"].append(loss)
        self.run_data["critic_loss"].append(critic)
        self.run_data["actor_loss"].append(actor)

    def tetris_engine(self, name = "engine"):
        return self.object(name, tetris.Matris(self.config))

    def tetris_network(self, name = "network", ttype = "ppo"):
        if ttype == "ppo":
            return self.object(name, ppo.AdjustedSandfordNetwork(self.config).load(self.folder / f"{name}.pt"))
        else:
            raise TypeError(f"Invalid type '{ttype}'")


def test_ppo(args):
    runner = Runner(args.file, args.location)
    engine = runner.tetris_engine()
    network = runner.tetris_network()

    pygame.init()
    screen = pygame.display.set_mode((tetris.WIDTH, tetris.HEIGHT))
    pygame.display.set_caption("MaTris")
    visualizer = ppo.NetworkRealtimeVisualizer()

    game = tetris.Game()
    game.main(screen, engine)

    state = engine.reset()
    network.eval()

    run = False
    best = False

    episodes = []
    probs = []
    returns = []

    from experience import Trajectory, PPOExperience
    trajectory = Trajectory()
    try:
        while True:
            try:
                actions = game.get_user_actions()
            except:
                raise SystemExit("Game Over")
            if game.is_key(pygame.K_r):
                run = not run

            if game.is_key(pygame.K_b):
                best = not best
                run = False

            if pygame.K_q in game.extra_keys:
                state_tensor = torch.Tensor(state).unsqueeze(0).to(network.device)
                network(state_tensor)
                logits = network["action_logits"]
                dist = torch.distributions.Categorical(logits=logits)
                action = torch.argmax(logits).item()

                visualizer.update(state, logits.squeeze().cpu(), dist.probs.squeeze().cpu(), network["state_value"].item(), action)

            if pygame.K_v in game.extra_keys or run:
                state_tensor = torch.Tensor(state).unsqueeze(0).to(network.device)
                network(state_tensor)
                logits = network["action_logits"]
                dist = torch.distributions.Categorical(logits=logits)
                # action = dist.sample().item()
                action = torch.argmax(logits).item()
                probs.append(dist.probs.squeeze())
                actions.append(tetris.Action(action))

                visualizer.update(state, logits.squeeze().cpu(), dist.probs.squeeze().cpu(), network["state_value"].item(), action)

            if best and not run:
                actions.extend(engine.best_action_set())

            if len(actions) == 0:
                game.redraw()
                # update_models(tetris.Action(0))
                continue

            for action in actions:
                state_tensor = torch.Tensor(state).unsqueeze(0).to(network.device)
                network(state_tensor)
                logits = network["action_logits"]
                dist = torch.distributions.Categorical(logits=logits)

                print(f"Critic says state is: {network["state_value"].item()} | ", end='')
                next_state, reward, lines_cleared, game_over, truncated = engine.step(action)

                print(f"Reward was: {reward}")
                returns.append(reward)
                game.redraw()
                state = next_state

                reward = torch.tensor([reward]).to(network.device)
                done = torch.tensor([int(game_over or truncated)]).to(network.device)

                # trajectory.append(PPOExperience(
                #     state_tensor.detach(), action, reward, done, dist.log_prob(torch.tensor(action.value).to(network.device)), network.critic()))

                if game_over:
                    state = engine.reset()
                    np_rewards = np.array(returns)

                    discounted_rewards = np_rewards.copy()
                    for i in reversed(range(len(returns) - 1)):
                        discounted_rewards[i] = discounted_rewards[i + 1] * runner.config["GAMMA"] + np_rewards[i]

                    if len(probs) > 0:
                        episodes.append( (torch.stack(probs), discounted_rewards, np_rewards) )
                        probs.clear()
                    returns.clear()
                    trajectory.set_last_value(network["state_value"])
                    raise SystemExit("Game Over")
    except SystemExit or KeyboardInterrupt or MaTris.matris.GameOver:
        location = Path(runner.location) / runner.file / "runs"/ time.strftime("%Y-%m-%d_%H-%M-%S")
        location.mkdir(parents=True, exist_ok=True)

        for i, episode in enumerate(episodes):
            episode_action_probs = graph.plot_episode_action_probabilities_full(episode, i)
            episode_action_probs.savefig(location / f"episode_{i}_action_probs.png")
            plt.close(episode_action_probs)
            episode_action_probs = graph.plot_episode_action_probabilities(episode, i)
            episode_action_probs.savefig(location / f"episode_{i}_action_probs_seperate.png")
            plt.close(episode_action_probs)
            discounted_returns = graph.plot_rewards_and_discounted_returns(episode, i, runner.config["GAMMA"])
            discounted_returns.savefig(location / f"episode_{i}_discounted_returns.png")
            plt.close(discounted_returns)

        gae, returns = trajectory.compute_gae(runner.config["GAMMA"], runner.config["LAMBDA"])
        gae_returns = graph.plot_gae_and_returns(
            gae,
            returns,
            gamma=runner.config["GAMMA"],
            lamda=runner.config["LAMBDA"]
        )
        terminals = trajectory.get_terminals()
        for term in terminals:
            gae_returns.axes[0].axvline(term, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
            gae_returns.axes[1].axvline(term, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
        gae_returns.savefig(location / "episode_gae_returns.png")
        plt.close(gae_returns)

        pygame.image.save(screen, location / f"episode.png")

def train_ppo(args):
    runner = Runner(args.file, args.location)
    network = runner.tetris_network()
    generator = runner.object("generator", experience.ExperienceGenerator(runner.config, network, PPOExperience))
    trainer = runner.object("trainer", experience.NetworkEpochTrainer(runner.config, network, generator, ppo.step))

    try:
        progress = tqdm(
            total=None,
            desc="[Runner] Training",
            unit=" round",
            dynamic_ncols=True,
            position=0,
        )

        progress.update(runner.runs())
        progress.refresh()

        while True:
            total_loss, total_batches = trainer.train()
            runner.run_data["loss"].append(total_loss / total_batches)
            runner.increment_runs()
            progress.update(1)
            progress.set_postfix({
                "Loss": f"{float(total_loss / total_batches):.6f}"
            })

            if runner.runs() % runner.config["SAVE_INTERVAL"] == 0:
                runner.save()
    except KeyboardInterrupt:
        runner.save()

def test_dqn(args):
    runner = Runner(args.file, args.location)
    engine = runner.object("engine", tetris.Matris(runner.config))
    network = runner.object("network", dqn.DQNNetwork(runner.config).load(runner.folder / "network.pt"))

    pygame.init()
    screen = pygame.display.set_mode((tetris.WIDTH + 512, tetris.HEIGHT))
    pygame.display.set_caption("MaTris")

    game = tetris.Game()
    game.main(screen, engine)

    state = engine.reset()
    network.eval()

    run = False
    best = False

    episodes = []
    probs = []
    returns = []
    speed = 0

    try:
        while True:
            try:
                actions = game.get_user_actions()
            except:
                raise SystemExit("Game Over")
            if game.is_key(pygame.K_r):
                run = not run

            if game.is_key(pygame.K_b):
                best = not best
                run = False

            if pygame.K_EQUALS in game.extra_keys or pygame.K_PLUS in game.extra_keys:
                speed = max(0.0, speed - 0.02)
            if pygame.K_MINUS in game.extra_keys:
                speed += 0.02

            if pygame.K_v in game.extra_keys or run:
                state_tensor = torch.Tensor(state).unsqueeze(0).to(network.device)
                network.compute(state_tensor)
                logits = network.state_value()
                action = torch.argmax(logits).item()
                probs.append(logits.squeeze())
                actions.append(tetris.Action(action))
                game.draw_model_output(logits, action)
                game.draw_sleep(speed)

            if best and not run:
                actions.extend(engine.best_action_set())

            if len(actions) == 0:
                game.redraw()
                # update_models(tetris.Action(0))
                continue

            for action in actions:
                next_state, reward, lines_cleared, game_over, truncated = engine.step(action)

                print(f"Reward was: {reward}")
                returns.append(reward)
                game.draw_reward(reward)
                game.redraw()
                state = next_state

                if game_over:
                    state = engine.reset()
                    np_rewards = np.array(returns)

                    discounted_rewards = np_rewards.copy()
                    for i in reversed(range(len(returns) - 1)):
                        discounted_rewards[i] = discounted_rewards[i + 1] * runner.config["GAMMA"] + np_rewards[i]

                    if len(probs) > 0:
                        episodes.append( (torch.stack(probs), discounted_rewards, np_rewards) )
                        probs.clear()
                    returns.clear()
                    raise SystemExit("Game Over")
                time.sleep(speed)
    except SystemExit or KeyboardInterrupt or MaTris.matris.GameOver:
        location = Path(runner.location) / runner.file / "runs"/ time.strftime("%Y-%m-%d_%H-%M-%S")
        location.mkdir(parents=True, exist_ok=True)

        for i, episode in enumerate(episodes):
            episode_action_probs = graph.plot_episode_action_probabilities_full(episode, i)
            episode_action_probs.savefig(location / f"episode_{i}_action_probs.png")
            plt.close(episode_action_probs)
            episode_action_probs = graph.plot_episode_action_probabilities(episode, i)
            episode_action_probs.savefig(location / f"episode_{i}_action_probs_seperate.png")
            plt.close(episode_action_probs)
            discounted_returns = graph.plot_rewards_and_discounted_returns(episode, i, runner.config["GAMMA"])
            discounted_returns.savefig(location / f"episode_{i}_discounted_returns.png")
            plt.close(discounted_returns)

        pygame.image.save(screen, location / f"episode.png")

def train_dqn(args):
    runner = Runner(args.file, args.location)
    engine = runner.object("engine", tetris.Matris(runner.config))
    network = runner.object("network", dqn.DQNNetwork(runner.config).load(runner.folder / "network.pt"))
    target = runner.object("target", dqn.DQNNetwork(runner.config).load(runner.folder / "target.pt"))
    generator = runner.object("generator", dqn.DQNExperienceGenerator(runner.config, engine, network))
    trainer = runner.object("trainer", dqn.DQNTrainer(runner.config, network, target, generator))

    engine.reset()

    try:
        progress = tqdm(
            total=None,
            desc="[Runner] Training",
            unit=" round",
            dynamic_ncols=True,
            position=0,
        )

        progress.update(runner.runs())
        progress.refresh()

        while True:
            average_returns, time_taken, collection_time, average_loss = trainer.train()
            runner.loss(average_returns, average_loss, 0, 0)
            runner.increment_runs()
            progress.update(1)
            progress.set_postfix({
                "Return": f"{float(average_returns):.6f}",
                "Avg Loss": f"{float(average_loss):.6f}",
                "Time": f"{time_taken:.2f}s",
                "Collection": f"{collection_time:.2f}s",
            })

            if runner.runs() % runner.config["SAVE_INTERVAL"] == 0:
                runner.save()
    except KeyboardInterrupt:
        runner.save()

def main():
    args = parser.parse_args()

    if args.seed == 0:
        args.seed = int(time.time())

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.default:
        Config("./i.should.not.exist.json", Path.cwd() / "default_config.json").save_defaults()

    if args.command == "test":
        if args.type == "dqn":
            test_dqn(args)
        elif args.type == "ppo":
            test_ppo(args)
    elif args.command == "train":
        if args.type == "ppo":
            train_ppo(args)
        elif args.type == "dqn":
            train_dqn(args)
    elif args.command == "init":
        runner = Runner(args.file, args.location)
        runner.init_run()

if __name__ == "__main__":
    main()

