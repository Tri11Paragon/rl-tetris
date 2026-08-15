import time

import numpy as np
import pygame
import random
from matplotlib import pyplot as plt

import apa
import bitwise
import dqn
import experience
import graph
import ppo2 as ppo
from collections import defaultdict
import argparse
import json

import tetris
import torch
from tqdm.auto import tqdm
from pathlib import Path
from config import Config, DotDict

from experience import PPOExperience, DQNExperience

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest="command")

parser.add_argument("-l", "--location", type=str, default="experiments")
parser.add_argument("-s", "--seed", type=int, default=0)

parser_evaler = subparsers.add_parser("eval")
parser_evaler.add_argument("file", type=str)
parser_evaler.add_argument("--runs", type=int, default=100)

parser_tester = subparsers.add_parser("test")
parser_tester.add_argument("file", type=str)

parser_compare = subparsers.add_parser("compare")
parser_compare.add_argument("file", type=str)

parser_trainer = subparsers.add_parser("train")
parser_trainer.add_argument("file", type=str)

parser_initializer = subparsers.add_parser("init")
parser_initializer.add_argument("file", type=str)


class Runner:
    def __init__(self, file, location="experiments"):
        self.file = file
        self._location = location
        self.folder = Path(location) / file
        self.config_path = self.folder / "config.nix"
        self.object_storage = {}
        self.run_data = defaultdict(list, {
            "runs": 0,
            "time": 0
        })
        self.init_run()

    def make_folder(self, folder_name: str):
        f = self.folder / folder_name / time.strftime("%Y-%m-%d_%H-%M-%S")
        f.mkdir(parents=True, exist_ok=True)
        return f

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

    def tetris_engine(self, name = "engine", seed: int | None = None):
        return self.object(name, tetris.PyTetrisEngine(seed or time.time_ns(), self.config.json_str))

    def tetris_network(self, name = "network", ttype = "ppo", expects = "normal"):
        if expects == "normal":
            if ttype == "ppo":
                return self.object(name, ppo.AdjustedSandfordNetwork(self.config).load(self.folder / f"{name}.pt"))
            elif ttype == "apa":
                return self.object(name, ppo.AdjustedSandfordNetwork(self.config).load(self.folder / f"{name}.pt"))
            elif ttype == "dqn":
                return self.object(name, dqn.AdjustedSandfordNetwork(self.config).load(self.folder / f"{name}.pt"))
            else:
                raise TypeError(f"Invalid type '{ttype}'")
        elif expects == "bitwise":
            if ttype == "ppo":
                return self.object(name, bitwise.network(self.config).load(self.folder / f"{name}.pt"))
            elif ttype == "apa":
                return self.object(name, bitwise.network(self.config).load(self.folder / f"{name}.pt"))
            elif ttype == "dqn":
                print("WARNING BITWISE UNDEFINED FOR DQN CURRENTLY")
                return self.object(name, dqn.AdjustedSandfordNetwork(self.config).load(self.folder / f"{name}.pt"))
            else:
                raise TypeError(f"Invalid type '{ttype}'")
        raise ValueError(f"Invalid network expects value '{expects}'")


def test(args):
    runner = Runner(args.file, args.location)
    engine: tetris.PyTetrisEngine = runner.tetris_engine()
    network = runner.tetris_network(ttype=runner.config.network.mode.lower(), expects=runner.config.network.expects.lower())

    pygame.init()
    from MaTris import matris
    screen = pygame.display.set_mode((matris.WIDTH + 512, matris.HEIGHT))
    pygame.display.set_caption("MaTris")
    visualizer = ppo.NetworkRealtimeVisualizer(screen, pygame.Rect(matris.WIDTH, 0, 512, matris.HEIGHT))

    game = matris.Game()
    game.main(screen, engine)
    game.extra_text.append(f"Training Type: {runner.config.network.mode}")
    game.extra_text.append(f"Steps: {runner.run_data['runs']}")
    game.extra_text.append(f"Timer: {0}")

    engine.reset()
    if runner.config.network.expects == "normal":
        state = engine.current_state()
    elif runner.config.network.expects == "bitwise":
        state = engine.current_state_bitwise()
    else:
        raise ValueError(f"Invalid value for network.expects: {runner.config.network.expects}")
    network.eval()

    run = False
    best = False

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
                actions.append(action)

                visualizer.update(state, logits.squeeze().cpu(), dist.probs.squeeze().cpu(),
                                  network["state_value"].item() if "state_value" in network else 0, action, runner.config.network.expects)

            if pygame.K_EQUALS in game.extra_keys or pygame.K_KP_PLUS in game.extra_keys:
                timer += 0.01

            if pygame.K_MINUS in game.extra_keys or pygame.K_KP_MINUS in game.extra_keys:
                timer -= 0.01
            timer = max(0, timer)
            game.extra_text[2] = f"Timer: {timer:.2f}"

            # if best and not run:
            #     actions.extend(engine.best_action_set())

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

                if runner.config.network.expects == "normal":
                    next_state, reward, lines_cleared, game_over, truncated = engine.step(action)
                elif runner.config.network.expects == "bitwise":
                    next_state, reward, lines_cleared, game_over, truncated = engine.step_bitwise(action)
                else:
                    raise ValueError(f"Invalid value for network.expects: {runner.config.network.expects}")
                # print(engine.current_state_bitwise())

                if "state_value" in network:
                    print(f"Critic says state is: {network['state_value'].item()} | Advantage: {reward - network['state_value'].item()}| ", end='')
                print(f"Reward was: {reward}")
                game.redraw()
                state = next_state

                is_done = int(game_over or (truncated and runner.config.tetris.truncate.rewardBoundary))

                data_dict = {
                    "state": state_tensor.detach().cpu().squeeze(),
                    "action": torch.tensor(action),
                    "reward": torch.tensor(reward),
                    "done": torch.tensor(is_done),
                    "logprob": dist.log_prob(torch.tensor(action).to(network.device)),
                    "logits": logits.detach().cpu().squeeze(),
                }

                if "state_value" in network:
                    data_dict["state_value"] = network.get("state_value").detach().cpu().squeeze()

                trajectory.append(data_dict)

                if game_over:
                    if "state_value" in network:
                        trajectory.set_last_value(network["state_value"])
                    print("Game Over")
                    engine.reset()
                    state = engine.current_state()
                    # raise SystemExit("Game Over")
    except (SystemExit, KeyboardInterrupt):
        print("Soft Exit")
        location = runner.make_folder("runs")
        pygame.image.save(screen, location / f"episode.png")

        trajectory.set_last_value(0)
        advantages, returns = trajectory.compute_gae(runner.config.network.gamma, runner.config.network.ppo.lamda)
        returns = returns.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.detach()
        dones = torch.stack(trajectory.buffer["done"])

        logits = torch.stack(trajectory["logits"])
        rewards = torch.stack(trajectory["reward"])

        episode_action_probs = graph.plot_episode_action_probabilities_full(logits)
        episode_action_probs.savefig(location / f"episode_action_probs.png")
        plt.close(episode_action_probs)
        episode_action_probs = graph.plot_episode_action_probabilities(logits, returns)
        episode_action_probs.savefig(location / f"episode_action_probs_seperate.png")
        plt.close(episode_action_probs)
        discounted_returns = graph.plot_rewards_and_discounted_returns(returns, advantages, rewards, dones, runner.config.network.gamma)
        discounted_returns.savefig(location / f"episode_discounted_returns.png")
        plt.close(discounted_returns)
    except Exception as e:
        print(e)

def compare(args):
    from MaTris import matris
    from MaTris.matris import Action
    runner = Runner(args.file, args.location)
    tetris_engine: tetris.PyTetrisEngine = tetris.PyTetrisEngine(0, runner.config.json_str)
    matris_engine = matris.Matris(runner.config)

    pygame.init()
    screen = pygame.display.set_mode((512 * 2, matris.HEIGHT))
    pygame.display.set_caption("MaTris")
    visualizer = ppo.NetworkRealtimeVisualizer(screen, pygame.Rect(256, 0, 512, matris.HEIGHT))
    visualizer2 = ppo.NetworkRealtimeVisualizer(screen, pygame.Rect(0, 0, 512, matris.HEIGHT))

    steps = 0

    while True:
        pygame.event.get(pygame.KEYDOWN)
        keyups = pygame.event.get(pygame.KEYUP)

        actions = []
        for event in keyups:
            if event.key == pygame.K_SPACE:
                actions.append(Action.HARD_DROP.value)
            elif event.key == pygame.K_UP or event.key == pygame.K_w:
                actions.append(Action.ROTATE.value)
            elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                actions.append(Action.LEFT.value)
            elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                actions.append(Action.RIGHT.value)
            elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                actions.append(Action.DOWN.value)
            elif event.key == pygame.K_ESCAPE:
                raise SystemExit("Game Over")

        for action in actions:
            steps += 1
            matris_tuple = matris_engine.step(action)
            tetris_tuple = tetris_engine.step(action)

            import engine_benchmark as bench

            if not bench.validate_tuple(action, tetris_tuple, matris_tuple, should_exit=False):
                print(f"Failed after {steps} steps")
            visualizer2.update(tetris_tuple[0], None, None, None, None, runner.config.network.expects)
            visualizer.update(matris_tuple[0], None, None, None, None, runner.config.network.expects)
            pygame.display.flip()

            if matris_tuple[3]:
                matris_engine.reset()
            if tetris_tuple[3]:
                tetris_engine.reset()


def _eval(args):
    runner = Runner(args.file, args.location)
    network = runner.tetris_network(ttype=runner.config.network.mode.lower())
    engines = [runner.tetris_engine(str(name)) for name in range(args.runs)]
    states = np.array([engine.current_state() for engine in engines])
    finished = [False] * len(engines)

    while True:
        state_tensor = torch.Tensor(states).to(network.device)
        network(state_tensor)
        actions = torch.argmax(network["action_logits"], dim=1).numpy(force=True)
        for i, (engine, action) in enumerate(zip(engines, actions)):
            if finished[i]:
                continue
            state, reward, lines_cleared, game_over, truncated = engine.step(action)
            states[i] = state
            if game_over:
                finished[i] = True
        if all(finished):
            break

    lines = np.array([engine.lines for engine in engines])
    score = np.array([engine.score for engine in engines])

    mean_lines = lines.mean()
    mean_score = score.mean()

    print(f"Average Lines: {mean_lines}")
    print(f"Average Score: {mean_score}")

    folder = runner.make_folder("evals")
    with open(f"{folder / 'results.json'}", "w+") as f:
        json.dump({"lines": mean_lines, "score": mean_score}, f)

def train(args):
    runner = Runner(args.file, args.location)
    network_type = runner.config.network.mode.lower()

    network = runner.tetris_network(ttype=network_type, expects=runner.config.network.expects.lower())
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
        target = runner.tetris_network("target", ttype=network_type, expects=runner.config.network.expects.lower())
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
            _ = trainer.train()
            traj_returns = generator.buffer.compute_returns(runner.config.network.gamma)
            runner.run_data["average_trajectory_return"].append(float(traj_returns.mean()))
            lines_cleared = np.array(generator.lines_cleared)
            game_length = np.array(generator.game_length)
            runner.run_data["average_lines_cleared"].append(float(lines_cleared.mean()) if len(generator.lines_cleared) > 0 else 0)
            runner.run_data["average_game_length"].append(float(game_length.mean()) if len(generator.game_length) > 0 else 0)
            runner.increment_runs()
            progress.update(1)
            # progress.set_postfix({
            #     "Loss": f"{float(total_loss / total_batches):.6f}"
            # })
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
    elif args.command == "compare":
        compare(args)
    elif args.command == "eval":
        _eval(args)
    elif args.command == "init":
        runner = Runner(args.file, args.location)
        runner.init_run()

if __name__ == "__main__":
    main()

