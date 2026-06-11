import time

import numpy as np
import pygame
from matplotlib import pyplot as plt

import graph
import ppo2 as ppo
import argparse
import json
import pickle
import os

import MaTris.matris as tetris
import torch
from tqdm.auto import tqdm
from pathlib import Path

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest="command")

parser.add_argument("-l", "--location", type=str, default="experiments")

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
        self.config = {}
        self.object_storage = {}
        self.run_data = {
            "loss": [],
            "actor_loss": [],
            "critic_loss": [],
            "runs": 0
        }
        self.init_run()

    def __getitem__(self, item):
        return self.object_storage[item]

    def object(self, key, value):
        if key not in self.object_storage:
            self.object_storage[key] = value
        return self.object_storage[key]

    def update(self, key, value):
        if key not in self.config:
            self.config[key] = value

    def init_run(self):
        folder = Path(self.location) / self.file
        folder.mkdir(parents=True, exist_ok=True)

        config_path = folder / "config.json"
        if config_path.exists():
            self.config = json.load(config_path.open(mode="r"))

        self.update("ACTOR_LEARN_RATE", 1e-5)
        self.update("CRITIC_LEARN_RATE", 1e-4)
        self.update("CONV_LEARN_RATE", 1e-5)
        self.update("DROPOUT", 0.2)
        self.update("GAMMA", 0.99)
        self.update("LAMBDA", 0.1)
        self.update("SHUFFLE_EXPERIENCES", False)
        self.update("MAX_EPISODE_LENGTH", 10000)
        self.update("MAX_EPISODES", 100)
        self.update("EPOCHS", 5)
        self.update("BATCH_SIZE", 64)
        self.update("ENTROPY", 0.01)
        self.update("CLIP_EPSILON", 0.2)

        self.update("SAVE_INTERVAL", 5)

        self.update("MATRIS_ACTIONS_UNTIL_DROP", 3)
        self.update("MATRIS_PLACEMENT_HORIZON", 50)
        self.update("MATRIS_DECAY", True)
        self.update("MATRIS_EPISODIC_TRUNCATE", True)

        self.update("MATRIS_GAMEOVER_PENALTY", 100)
        self.update("MATRIS_TRUNCATE_PENALTY", 10)
        self.update("MATRIS_DISCOURAGE_PENALTY", 0.1)
        self.update("MATRIS_ENCOURAGE_REWARD", 1)


        object_path = folder / "objects.pkl"
        if object_path.exists():
            self.object_storage = pickle.load(object_path.open(mode="rb"))

        loss_path = folder / "state.json"
        if loss_path.exists():
            self.run_data = json.load(loss_path.open(mode="r"))

        self.save()

    def save(self):
        folder = Path(self.location) / self.file
        folder.mkdir(parents=True, exist_ok=True)
        config_path = folder / "config.json"
        json.dump(self.config, config_path.open(mode="w"))

        object_path = folder / "objects.pkl"
        pickle.dump(self.object_storage, object_path.open(mode="wb"))

        loss_path = folder / "state.json"
        json.dump(self.run_data, loss_path.open(mode="w"))

    def runs(self):
        return self.run_data["runs"]

    def increment_runs(self):
        self.run_data["runs"] += 1

    def loss(self, loss, critic, actor):
        self.run_data["loss"].append(loss)
        self.run_data["critic"].append(critic)
        self.run_data["actor"].append(actor)

def test(args):
    runner = Runner(args.file, args.location)
    engine = runner.object("engine", tetris.Matris(runner.config))
    network = runner.object("network", ppo.AdjustedSandfordACNetwork(runner.config))

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

    try:
        while True:
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
                probs.append(dist.probs.squeeze())
                actions.append(tetris.Action(action))

            if best and not run:
                actions.extend(engine.best_action_set())

            if len(actions) == 0:
                game.redraw()
                # update_models(tetris.Action(0))
                continue

            for action in actions:
                print(f"Critic says state is: {network.critic().item()} | ", end='')
                next_state, reward, lines_cleared, game_over, truncated = engine.step(action)
                print(f"Reward was: {reward}")
                returns.append(reward)
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
    except SystemExit or KeyboardInterrupt:
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

def train(args):
    runner = Runner(args.file, args.location)
    engine = runner.object("engine", tetris.Matris(runner.config))
    network = runner.object("network", ppo.AdjustedSandfordACNetwork(runner.config))
    generator = runner.object("generator", ppo.PPOExperienceGenerator(runner.config, engine, network))
    trainer = runner.object("trainer", ppo.PPOTrainer(runner.config, network, generator))

    try:
        progress = tqdm(
            total=None,
            desc="[Runner] Training",
            unit=" round",
            dynamic_ncols=True,
            position=0
        )

        progress.update(runner.runs())

        while True:
            rets, t, collects, average_loss, critic_loss, actor_loss = trainer.train()
            runner.loss(average_loss, critic_loss, actor_loss)
            runner.increment_runs()
            progress.update(1)
            progress.set_postfix({
                "Return": f"{float(rets):.6f}",
                "Avg Loss": f"{float(average_loss):.6f}",
                "Actor Loss": f"{float(actor_loss):.6f}",
                "Critic Loss": f"{float(critic_loss):.6f}",
                "Time": f"{t:.2f}s",
                "Collection": f"{collects:.2f}s",
            })

            if runner.runs() % runner.config["SAVE_INTERVAL"] == 0:
                runner.save()


    except KeyboardInterrupt:
        runner.save()

def main():
    args = parser.parse_args()

    if args.command == "test":
        test(args)
    elif args.command == "train":
        train(args)

if __name__ == "__main__":
    main()

