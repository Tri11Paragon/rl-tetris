import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_episode_action_probabilities_full(episode_tuple, episode_num):
    episode_tensor, discounted_rewards, np_rewards = episode_tuple

    episode_tensor = episode_tensor.detach().cpu().numpy()
    action_names = ['NONE', 'RIGHT', 'LEFT', 'DOWN', 'ROTATE', 'HARD_DROP']
    timesteps = np.arange(episode_tensor.shape[0])

    fig = plt.figure(figsize=(12, 6))
    for action_idx in range(episode_tensor.shape[1]):
        plt.plot(timesteps, episode_tensor[:, action_idx], label=action_names[action_idx], marker='o', markersize=3)

    plt.xlabel('Timestep in Episode')
    plt.ylabel('Probability')
    plt.title(f'Action Probabilities Over Time - Episode {episode_num}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def plot_episode_action_probabilities(episode_tuple, episode_num):
    episode_tensor, discounted_rewards, np_rewards = episode_tuple

    episode_tensor = episode_tensor.detach().cpu().numpy()
    # action_names = ['NONE', 'RIGHT', 'LEFT', 'DOWN', 'ROTATE', 'HARD_DROP']
    action_names = ['RIGHT', 'LEFT', 'DOWN', 'ROTATE', 'HARD_DROP']
    timesteps = np.arange(episode_tensor.shape[0])
    timesteps_dr = np.arange(discounted_rewards.shape[0])

    fig, axes = plt.subplots(episode_tensor.shape[1], 1, figsize=(12, 10), sharex=True)

    for action_idx in range(episode_tensor.shape[1]):
        ax = axes[action_idx]

        ax.plot(timesteps, episode_tensor[:, action_idx],
                              label=action_names[action_idx], linewidth=2, color='C' + str(action_idx))
        ax.fill_between(timesteps, episode_tensor[:, action_idx], alpha=0.3, color='C' + str(action_idx))
        ax.set_ylabel('Probability')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        ax_r = ax.twinx()

        ax_r.plot(timesteps_dr, discounted_rewards, color='black', alpha=0.5, linewidth=2, label='Discounted Return', zorder=5)

        ax_r.set_ylabel('Reward')
        ax_r.grid(False)

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax_r.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc='upper right')

    axes[-1].set_xlabel('Timestep in Episode')
    fig.suptitle(f'Action Probabilities Over Time - Episode {episode_num}', fontsize=14, y=0.995)
    plt.tight_layout()

    return fig

def plot_rewards_and_discounted_returns(episode_tuple: tuple[torch.Tensor, np.ndarray, np.ndarray], episode_num, gamma=None, rolling_window=None):
    episode_tensor, discounted_returns, rewards = episode_tuple
    if rolling_window is None:
        rolling_window = len(rewards) // 10
    rolling_window = max(1, rolling_window)

    if discounted_returns.ndim != 1 or rewards.ndim != 1:
        raise ValueError(
            f"Expected 1D inputs. Got discounted_returns shape={discounted_returns.shape}, rewards shape={rewards.shape}"
        )

    T = min(len(discounted_returns), len(rewards))
    if T == 0:
        raise ValueError("Inputs are empty after conversion.")

    discounted_returns = discounted_returns[:T]
    rewards = rewards[:T]
    timesteps = np.arange(T)

    # Calculate rolling average for rewards
    rolling_avg = np.convolve(rewards, np.ones(rolling_window) / rolling_window, mode='valid')
    rolling_timesteps = np.arange(rolling_window - 1, T)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Top plot: Discounted returns
    ax1.plot(timesteps, discounted_returns, color="black", linewidth=2, label="discounted_return")
    ax1.set_ylabel("Discounted Return")
    ax1.set_xlabel("Timestep")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    title = "Discounted Returns"
    if episode_num is not None:
        title += f" - Episode {episode_num}"
    if gamma is not None:
        title += f" (gamma={gamma})"
    ax1.set_title(title)

    # Bottom plot: Rewards with rolling average
    ax2.plot(timesteps, rewards, color="0.5", alpha=0.25, linewidth=2, label="reward")
    ax2.plot(rolling_timesteps, rolling_avg, color="blue", linewidth=2, label=f"rolling_avg (window={rolling_window})")
    ax2.set_ylabel("Reward")
    ax2.set_xlabel("Timestep")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")
    ax2.set_title("Rewards with Rolling Average")

    fig.suptitle(f"Rewards and Discounted Returns - Episode {episode_num}" if episode_num is not None else "Rewards and Discounted Returns", y=0.995)

    plt.tight_layout()

    return fig

def plot_gae_and_returns(gae: torch.Tensor, returns: torch.Tensor, episode_num=None, gamma=None, lamda=None):
    gae = gae.detach().cpu().numpy()
    returns = returns.detach().cpu().numpy()

    if gae.ndim != 1 or returns.ndim != 1:
        raise ValueError(
            f"Expected 1D inputs. Got gae shape={gae.shape}, returns shape={returns.shape}"
        )

    T = min(len(gae), len(returns))
    if T == 0:
        raise ValueError("Inputs are empty after conversion.")

    gae = gae[:T]
    returns = returns[:T]
    timesteps = np.arange(T)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    ax1.plot(timesteps, gae, color="blue", linewidth=2, label="GAE")
    ax1.fill_between(timesteps, gae, alpha=0.25, color="blue")
    ax1.set_ylabel("GAE")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")
    ax1.set_title("GAE")

    ax2.plot(timesteps, returns, color="black", alpha=0.75, linewidth=2, label="Returns")
    ax2.fill_between(timesteps, returns, alpha=0.20, color="black")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Returns")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")
    ax2.set_title("Returns")

    title = "GAE and Returns"
    if episode_num is not None:
        title += f" - Episode {episode_num}"

    details = []
    if gamma is not None:
        details.append(f"gamma={gamma}")
    if lamda is not None:
        details.append(f"lambda={lamda}")
    if details:
        title += f" ({', '.join(details)})"

    fig.suptitle(title, y=0.995)
    plt.tight_layout()

    return fig