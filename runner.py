import time

import numpy as np
import pygame
import random
from matplotlib import pyplot as plt

import MaTris.matris
import apa
import dqn
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
from config import Config, DotDict

from experience import PPOExperience, DQNExperience

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest="command")

parser.add_argument("-l", "--location", type=str, default="experiments")
parser.add_argument("-s", "--seed", type=int, default=0)

parser_tester = subparsers.add_parser("test")
parser_tester.add_argument("file", type=str)

parser_trainer = subparsers.add_parser("train")
parser_trainer.add_argument("file", type=str)

parser_initializer = subparsers.add_parser("init")
parser_initializer.add_argument("file", type=str)


class Runner:
    def __init__(self, file, location="experiments"):
        self.file = file
        self.location = location
        self.folder = Path(location) / file
        self.config_path = self.folder / "config.nix"
        self.object_storage = {}
        self.run_data = defaultdict(list, {
            "runs": 0,
            "time": 0
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

    def log_to_list(self, list_name, value):
        self.run_data[list_name].append(value)

    def tetris_engine(self, name = "engine"):
        return self.object(name, tetris.Matris(self.config))

    def tetris_network(self, name = "network", ttype = "ppo"):
        if ttype == "ppo":
            return self.object(name, ppo.AdjustedSandfordNetwork(self.config).load(self.folder / f"{name}.pt"))
        elif ttype == "apa":
            return self.object(name, ppo.AdjustedSandfordNetwork(self.config).load(self.folder / f"{name}.pt"))
        elif ttype == "dqn":
            return self.object(name, dqn.AdjustedSandfordNetwork(self.config).load(self.folder / f"{name}.pt"))
        else:
            raise TypeError(f"Invalid type '{ttype}'")

def evaluate_network(config: DotDict, network) -> tuple[float, float, int]:
    with torch.no_grad():
        engine = tetris.Matris(config)
        state = engine.current_state()
        returns = []
        while True:
            state_tensor = torch.Tensor(state).unsqueeze(0).to(network.device)
            network(state_tensor)
            action = torch.argmax(network["action_logits"]).item()
            state, reward, lines_cleared, game_over, truncated = engine.step(action)
            returns.append(reward)

            if game_over:
                break
        discounted_returns = []
        accum = 0
        for ret in reversed(returns):
            accum = accum * config.network.gamma + ret
            discounted_returns.append(accum)
        return float(np.array(returns).mean()), float(np.array(discounted_returns).mean()), engine.lines



def test(args):
    runner = Runner(args.file, args.location)
    engine = runner.tetris_engine()
    network = runner.tetris_network(ttype=runner.config.network.type.lower())

    pygame.init()
    screen = pygame.display.set_mode((tetris.WIDTH + 512, tetris.HEIGHT))
    pygame.display.set_caption("MaTris")
    visualizer = ppo.NetworkRealtimeVisualizer(screen, pygame.Rect(tetris.WIDTH, 0, 512, tetris.HEIGHT))

    game = tetris.Game()
    game.main(screen, engine)
    game.extra_text.append(f"Training Type: {runner.config.network.type}")
    game.extra_text.append(f"Steps: {runner.run_data["runs"]}")
    game.extra_text.append(f"Timer: {0}")

    state = engine.reset()
    network.eval()

    run = False
    best = False

    episodes = []
    probs = []
    returns = []
    dones = []

    timer = 0

    from experience import Trajectory
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

                if "state_value" in network:
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

                visualizer.update(state, logits.squeeze().cpu(), dist.probs.squeeze().cpu(),
                                  network["state_value"].item() if "state_value" in network else 0, action)

            if pygame.K_PLUS in game.extra_keys or pygame.K_KP_PLUS in game.extra_keys:
                timer += 0.01

            if pygame.K_MINUS in game.extra_keys or pygame.K_KP_MINUS in game.extra_keys:
                timer -= 0.01
            game.extra_text[2] = f"Timer: {timer:.2f}"

            if best and not run:
                actions.extend(engine.best_action_set())

            if len(actions) == 0:
                game.redraw()
                # update_models(tetris.Action(0))
                continue

            for action in actions:
                time.sleep(timer)
                state_tensor = torch.Tensor(state).unsqueeze(0).to(network.device)
                network(state_tensor)
                logits = network["action_logits"]
                dist = torch.distributions.Categorical(logits=logits)

                if "state_value" in network:
                    print(f"Critic says state is: {network["state_value"].item()} | ", end='')
                next_state, reward, lines_cleared, game_over, truncated = engine.step(action)

                print(f"Reward was: {reward}")
                returns.append(reward)
                game.redraw()
                state = next_state

                # reward = torch.tensor([reward]).to(network.device)
                dones.append(int(game_over or truncated))

                # trajectory.append(PPOExperience(
                #     state_tensor.detach(), action, reward, done, dist.log_prob(torch.tensor(action.value).to(network.device)), network.critic()))

                if game_over:
                    state = engine.reset()
                    np_rewards = np.array(returns)

                    discounted_rewards = np.zeros(np_rewards.shape)
                    with_dones = np.zeros(np_rewards.shape)
                    for i in reversed(range(len(returns) - 1)):
                        discounted_rewards[i] = discounted_rewards[i + 1] * runner.config.network.gamma + np_rewards[i]
                        with_dones[i] = (1 - dones[i]) * with_dones[i + 1] * runner.config.network.gamma + np_rewards[i]

                    if len(probs) > 0:
                        episodes.append( (torch.stack(probs), discounted_rewards, with_dones, np_rewards) )
                        probs.clear()
                    returns.clear()
                    if "state_value" in network:
                        trajectory.set_last_value(network["state_value"])
                    raise SystemExit("Game Over")
    except (SystemExit, KeyboardInterrupt, MaTris.matris.GameOver):
        location = Path(runner.location) / runner.file / "runs"/ time.strftime("%Y-%m-%d_%H-%M-%S")
        location.mkdir(parents=True, exist_ok=True)
        pygame.image.save(screen, location / f"episode.png")

        for i, episode in enumerate(episodes):
            episode_action_probs = graph.plot_episode_action_probabilities_full(episode, i)
            episode_action_probs.savefig(location / f"episode_{i}_action_probs.png")
            plt.close(episode_action_probs)
            episode_action_probs = graph.plot_episode_action_probabilities(episode, i)
            episode_action_probs.savefig(location / f"episode_{i}_action_probs_seperate.png")
            plt.close(episode_action_probs)
            discounted_returns = graph.plot_rewards_and_discounted_returns(episode, i, runner.config.network.gamma)
            discounted_returns.savefig(location / f"episode_{i}_discounted_returns.png")
            plt.close(discounted_returns)

        # gae, returns = trajectory.compute_gae(runner.config.network.gamma, runner.config.network.ppo.lamda)
        # gae_returns = graph.plot_gae_and_returns(
        #     gae,
        #     returns,
        #     gamma=runner.config.network.gamma,
        #     lamda=runner.config.network.ppo.lamda
        # )
        # terminals = trajectory.get_terminals()
        # for term in terminals:
        #     gae_returns.axes[0].axvline(term, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
        #     gae_returns.axes[1].axvline(term, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
        # gae_returns.savefig(location / "episode_gae_returns.png")
        # plt.close(gae_returns)

def train(args):
    runner = Runner(args.file, args.location)
    network_type = runner.config.network.type.lower()

    network = runner.tetris_network(ttype=network_type)
    if network_type == "ppo":
        experience_type = PPOExperience
        step_function = ppo.step
        networks = network
    elif network_type == "apa":
        experience_type = PPOExperience
        step_function = apa.step
        networks = network
    elif network_type == "dqn":
        experience_type = DQNExperience
        step_function = dqn.step
        target = runner.tetris_network("target", ttype=network_type)
        networks = {"network": network, "target": target}
    else:
        raise TypeError(f"Invalid type '{network_type}'")
    generator: experience.ExperienceGenerator = runner.object("generator", experience.ExperienceGenerator(runner.config, network, experience_type))
    trainer = runner.object("trainer", experience.NetworkEpochTrainer(runner, networks, generator, step_function))

    try:
        progress = tqdm(
            total=None,
            desc="[Runner] Training",
            unit=" round",
            dynamic_ncols=True,
            position=0,
        )

        while True:
            start = time.time()
            total_loss, total_batches = trainer.train()
            runner.run_data["loss"].append(total_loss / total_batches)
            traj_returns = generator.buffer.compute_returns(runner.config.network.gamma)[0]
            runner.run_data["average_trajectory_return"].append(float(traj_returns.mean()))
            lines_cleared = np.array(generator.lines_cleared)
            runner.run_data["average_lines_cleared"].append(float(lines_cleared.mean()) if len(generator.lines_cleared) > 0 else 0)
            runner.increment_runs()
            progress.update(1)
            progress.set_postfix({
                "Loss": f"{float(total_loss / total_batches):.6f}"
            })
            end = time.time()
            runner.run_data["time"] += end - start

            # avgrets, avgdiscountedrets, lines = evaluate_network(runner.config, network)
            # runner.run_data["average_test_return"].append(avgrets)
            # runner.run_data["average_discounted_test_return"].append(avgdiscountedrets)
            # runner.run_data["test_lines_cleared"].append(lines)

            if runner.runs() % runner.config.training.saveInterval == 0:
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

    if args.command == "test":
        test(args)
    elif args.command == "train":
        train(args)
    elif args.command == "init":
        runner = Runner(args.file, args.location)
        runner.init_run()

if __name__ == "__main__":
    main()

