# https://arxiv.org/pdf/2306.02231
import pygame

from ppo2 import PPOExperienceGenerator, AdjustedSandfordACNetwork, DEFAULT_GAMMA, DEFAULT_LEARN_RATE, device
import network as net
import torch
import numpy as np
import time
import argparse
import graph
import matplotlib.pyplot as plt
import grads

import MaTris.matris as tetris
from MaTris.matris import MATRIX_WIDTH, MATRIX_HEIGHT
from torch.utils.tensorboard import SummaryWriter

global_steps = 0

class APATrainer:
    def __init__(self, model: net.ActorCriticNetwork, generator: PPOExperienceGenerator, gamma = DEFAULT_GAMMA, lamda = DEFAULT_LEARN_RATE, batch_size = 64, load_file=None):
        self.model = model
        self.generator = generator
        self.lamda = lamda
        self.gamma = gamma
        self.batch_size = batch_size

        self.model.load(load_file)

    def train(self, epochs, local_model: net.ActorCriticNetwork, loss_callback=None):
        global global_steps

        t0 = time.process_time()
        with torch.no_grad():
            buffer = self.generator.generate()
            t1 = time.process_time()

            tensors = buffer.to_tensors()
            advantages, returns = buffer.compute_gae(self.gamma, self.lamda)

        dataset = torch.utils.data.TensorDataset(
            tensors["state"],
            tensors["action"],
            tensors["logprob"],
            advantages,
            tensors["state_value"]
        )

        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        local_model.load_state_dict(self.model.state_dict())
        local_model.train()

        for _ in range(epochs):
            for i, batch in enumerate(loader):
                b_state, b_action, b_logprob, b_advantages, b_state_value = batch

                b_state = b_state.to(self.model.device)
                b_action = b_action.to(self.model.device)
                b_logprob = b_logprob.to(self.model.device)
                b_advantages = b_advantages.to(self.model.device)
                b_state_value = b_state_value.to(self.model.device)

                local_model.compute(b_state)
                logits = local_model.act()
                dist = torch.distributions.Categorical(logits=logits)
                action_logprob = dist.log_prob(b_action)

                loss_apa = (action_logprob - b_advantages / self.lamda - b_logprob) ** 2

                loss_v = (local_model.critic() - b_advantages - b_state_value) ** 2

                loss_apa = loss_apa.mean()
                loss_v = loss_v.mean()

                loss = loss_apa + loss_v

                if loss_callback:
                    loss_callback(loss_apa, loss_v, loss)

                local_model.zero()
                loss.backward()
                local_model.step()

                global_steps += 1

        self.model.load_state_dict(local_model.state_dict())
        t2 = time.process_time()
        average_returns, returns = buffer.compute_returns(self.gamma)
        average_returns = average_returns.mean()
        print(f"Time taken: {t2 - t0:.6f} | Collection time {t1 - t0:.6f} | Average Returns {average_returns:.12f}")

def gui_test(args):
    pygame.init()
    screen = pygame.display.set_mode((tetris.WIDTH, tetris.HEIGHT))
    pygame.display.set_caption("MaTris")
    matris = tetris.Matris()
    game = tetris.Game()
    game.main(screen, matris)

    engine = tetris.Matris()
    state = engine.reset()
    network = AdjustedSandfordACNetwork().to(device)
    network.load(args.load_file)
    network.eval()
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
                network.compute(torch.Tensor(state).unsqueeze(0).to(network.device))
                logits = network.act()
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample().item()
                # print(f"Action taken: {action} with dist {dist.probs.tolist()}")
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
                        discounted_rewards[i] = discounted_rewards[i + 1] * DEFAULT_GAMMA + np_rewards[i]

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

def compute_measures(args):
    writer = SummaryWriter("runs/" +
        args.load_file + "_" + str(args.gamma) + "Y_" + str(args.epochs) + "e_" + str(args.runs) + "r_" + str(args.lamda) + "l_" + str(
            int(time.time())))

    engine = tetris.Matris()
    network = AdjustedSandfordACNetwork().to(device)
    local_model = AdjustedSandfordACNetwork().to(device)

    layers, grad = grads.get_all_layers(local_model, grads.hook_forward, grads.hook_backward)

    generator = PPOExperienceGenerator(engine, network, runs=args.runs, max_episode_length=args.max_episode_length)

    trainer = APATrainer(network, generator, gamma=args.gamma, lamda=args.lamda, load_file=args.load_file)

    loss_callback = lambda loss_apa, loss_v, loss: (
        writer.add_scalar("Loss/APA", loss_apa, global_steps),
        writer.add_scalar("Loss/V", loss_v, global_steps),
        writer.add_scalar("Loss/Total", loss, global_steps))

    trainer.train(args.epochs, local_model, loss_callback=loss_callback)

    layer_idx, avg_grads = grads.get_grads(grad)
    layer_names = list(reversed(layers.values()))


    for layer_index, grad in zip(reversed(layer_idx), reversed(avg_grads)):
        writer.add_scalar("Gradients/Average", grad, global_step=layer_index, walltime=time.time() + layer_index)


    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(layer_idx, avg_grads, label="APA", marker="o")
    ax.set_xticks(layer_idx)
    ax.set_xticklabels(layer_names, rotation=45, ha="right")
    ax.set_xlabel("Layer depth")
    ax.set_ylabel("Average gradient")
    ax.set_title("Gradient flow")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    writer.add_figure("Gradients", fig, global_steps)

    state = engine.empty()
    writer.add_graph(network, torch.tensor(state).unsqueeze(0).to(device))
    writer.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--max_episode_length", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save_frequency", type=int, default=5)
    parser.add_argument("--load_file", type=str, default="apa.pt")
    parser.add_argument("--gui_test", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--lamda", type=float, default=0.1)
    args = parser.parse_args()

    if args.gui_test:
        gui_test(args)
        return

    if args.test:
        compute_measures(args)
        return

    engine = tetris.Matris()
    network = AdjustedSandfordACNetwork().to(device)
    generator = PPOExperienceGenerator(engine, network, runs=args.runs, max_episode_length=args.max_episode_length)

    trainer = APATrainer(network, generator, gamma=args.gamma, lamda=0.1, load_file=args.load_file)

    counter = 0
    try:
        local_model = AdjustedSandfordACNetwork().to(device)
        while True:
            trainer.train(args.epochs, local_model)

            counter += 1
            counter %= args.save_frequency
            if counter == 0:
                trainer.model.save(args.load_file)
    except KeyboardInterrupt:
        network.save(args.load_file)

if __name__ == "__main__":
    main()