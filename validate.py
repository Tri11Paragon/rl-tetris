import torch
import ppo2 as ppo
import numpy as np
import argparse
import MaTris.matris as tetris
from tqdm.auto import tqdm
import pygame
from experience import ERMBuffer, PPOExperience, Trajectory
from runner import Runner, Config

def train(runner, engine: tetris.Matris, network: ppo.AdjustedSandfordACNetwork):
    progress = tqdm(
        total=None,
        desc="[Game] Progress",
        unit=" placements",
        dynamic_ncols=True,
        position=1,
    )
    total_lines_cleared = 0

    state = engine.reset()

    while True:
        run_states = []
        run_rewards = []
        actions = engine.best_action_set(decay=False)
        if len(actions) == 0:
            runner.save()
            raise RuntimeError(
                "engine.best_action_set(decay=False) returned no actions; "
                "training would advance progress without doing optimizer steps."
                f"are we gameover? {engine.is_game_over}"
            )

        for action in actions:
            run_states.append(state)
            state, reward, lines_cleared, game_over, truncated = engine.step(action)
            run_rewards.append(reward)
            total_lines_cleared += lines_cleared

            if game_over:
                return total_lines_cleared

        runner.run_data["runs"] += 1
        progress.update(1)
        progress.set_postfix({"Lines Cleared": total_lines_cleared})

        states = torch.Tensor(np.array(run_states)).to(network.device)
        actions = torch.tensor(np.array([action.value for action in actions]), dtype=torch.int64).to(network.device)
        rewards = torch.Tensor(np.array(run_rewards)).unsqueeze(1).to(network.device)

        network.compute(states)
        logits = network.act()
        predicted_values = network.critic()

        actor_loss = torch.nn.functional.cross_entropy(logits, actions)
        critic_loss = torch.nn.functional.mse_loss(predicted_values, rewards)

        total_loss = actor_loss + critic_loss

        network.zero()
        total_loss.mean().backward()
        network.step()

        if progress.n % (runner.config["SAVE_INTERVAL"] * 100) == 0:
            # runner.loss(rewards.sum().item(), total_loss.mean().item(), actor_loss.mean().item(), critic_loss.mean().item())
            runner.save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    args = parser.parse_args()
    runner = Runner(f"nw_validate_{args.file}")

    runner.config["MATRIS_DECAY"] = False
    runner.config["MATRIS_ACTIONS_UNTIL_DROP"] = 10000

    engine = runner.tetris_engine()
    network = runner.tetris_network()

    progress = tqdm(
        total=None,
        desc="[Trainer] Training",
        unit=" game",
        dynamic_ncols=True,
        position=0,
    )

    lines_cleared = []
    try:
        while True:
            total_lines_cleared = train(runner, engine, network)
            lines_cleared.append(total_lines_cleared)

            runner.save()
            progress.update(1)
            progress.set_postfix({
                "Average Cleared": float(np.array(lines_cleared[-16:]).mean())
            })
    except KeyboardInterrupt:
        runner.save()
        pass

if __name__ == "__main__":
    main()