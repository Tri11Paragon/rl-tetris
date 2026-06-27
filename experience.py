from collections import namedtuple, defaultdict
from typing import Protocol, Any, Generic, TypeVar, overload, SupportsIndex
from dataclasses import fields, dataclass
import numpy as np
import MaTris.matris as tetris
from tqdm.auto import tqdm
import network as net
import torch

# Trajectory holds a set of related experiences as part of an episode
class Trajectory:
    def __init__(self):
        # Buffer of experiences
        self.buffer: defaultdict[str, list] = defaultdict(list)

        # Optional,
        self.last_value = None

    ''' Append an experience to the trajectory. 
        If this is the last experience which will be pushed to the buffer,
        please make sure to supply the last_value argument.
        This allows any algorithm using these experiences to bootstrap using last state's value.'''
    def append(self, experience: dict[str, torch.Tensor], last_value=None):
        for key, value in experience.items():
            self.buffer[key].append(value)
        self.last_value = last_value

    def set_last_value(self, value):
        self.last_value = value

    def clear(self):
        for _, value in self.buffer.items():
            value.clear()

    def compute_returns(self, gamma: float) -> np.ndarray:
        discounted_returns = np.zeros(len(self), dtype=np.float32)
        rewards = self.buffer["reward"]

        for i in reversed(range(len(self) - 1)):
            discounted_returns[i] = self.buffer["done"][i] * discounted_returns[i + 1] * gamma + rewards[i]

        return discounted_returns

    def compute_gae(self, gamma : float, lamda: float):
        size = len(self)
        # print(size)
        advantage = 0
        next_value = self.last_value

        assert "done" in self.buffer and "reward" in self.buffer and "state_value" in self.buffer

        dones = self.buffer["done"]
        rewards = self.buffer["reward"]
        values = self.buffer["state_value"]

        device = values[0].device if len(values) > 0 else torch.device("cpu")

        gae = torch.zeros(size, device=device)
        returns = torch.zeros(size, device=device)

        # Iterate backwards to compute GAE and returns correctly
        for t in reversed(range(size)):
            # Masking terminal states: if dones[t] is True, the next state is 0-valued
            mask = 1.0 - dones[t]

            # TD error delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = rewards[t] + gamma * next_value * mask - values[t]

            # GAE: A_t = delta_t + gamma * lambda * mask * A_{t+1}
            # The mask here ensures that we reset the accumulation at episode boundaries
            advantage = delta + gamma * lamda * mask * advantage
            gae[t] = advantage

            # Target return for critic training (Q-estimate)
            returns[t] = gae[t] + values[t]

            next_value = values[t]

        return gae, returns

    def get_terminals(self):
        return [i for i, v in enumerate(self.buffer["done"]) if v]

    def to_tensors(self):
        return {field: torch.cat(arr) for field, arr in self.buffer.items()}

    def __len__(self):
        return len(next(iter(self.buffer.values())))

    def __getitem__(self, idx):
        return self.buffer[idx]

    def __iter__(self):
        return iter(self.buffer)

    def __reversed__(self):
        return reversed(self.buffer)

class ERMBuffer:
    def __init__(self):
        self.buffer: list[Trajectory] = []

    def append(self, trajectory):
        self.buffer.append(trajectory)

    def clear(self):
        for trajectory in self.buffer:
            trajectory.clear()

    def compute_returns(self, gamma: float) -> tuple[np.ndarray, list[np.ndarray]]:
        average_returns = np.zeros(len(self.buffer), dtype=np.float32)
        returns = []

        for i in range(len(self.buffer)):
            rt = self.buffer[i].compute_returns(gamma)
            average_returns[i] = rt.mean()
            returns.append(rt)

        return average_returns, returns

    def to_tensors(self):
        tensor_dict = defaultdict(list)
        for trajectory in self.buffer:
            for field, arr in trajectory.buffer.items():
                tensor_dict[field].append(torch.stack(arr))
        return {field: torch.cat(arr) for field, arr in tensor_dict.items()}

    def compute_gae(self, gamma : float, lamda: float):
        gaes = []
        returns = []
        for traj in self.buffer:
            gae, ret = traj.compute_gae(gamma, lamda)
            gaes.append(gae)
            returns.append(ret)
        return torch.cat(gaes), torch.cat(returns)

    def __len__(self):
        return sum(len(traj) for traj in self.buffer)


@overload
def sample(buffer: Trajectory, amount: int):
    first_value = len(buffer)
    size = len(buffer.buffer)
    if size < amount:
        raise ValueError("Cannot sample from empty buffer // buffer size is too small.")
    indices = np.random.choice(size, size=amount, replace=False)
    return {field: arr[indices] for field, arr in buffer.buffer.items()}

@overload
def sample(buffer: ERMBuffer, amount: int):
    combined = defaultdict(list)
    for trajectory in buffer.buffer:
        for field, arr in trajectory.buffer.items():
            combined[field].extend(arr)
    size = len(next(iter(combined.values())))
    if size < amount:
        raise ValueError("Cannot sample from empty buffer // buffer size is too small.")
    indices = np.random.choice(size, size=amount, replace=False)
    return {field: arr[indices] for field, arr in combined.items()}

@dataclass
class Experience(Protocol):
    @staticmethod
    def distribution(config: dict[str, Any], logits: torch.Tensor) -> tuple[torch.Tensor, torch.distributions.Distribution]:
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        return actions, dist

    @staticmethod
    def build_dataset(config: dict[str, Any], buffer: ERMBuffer) -> torch.utils.data.Dataset:
        ...

@dataclass
class PPOExperience(Experience):
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    logprob: torch.Tensor
    state_value: torch.Tensor

    @staticmethod
    def build_dataset(config: dict[str, Any], buffer: ERMBuffer):
        tensors = buffer.to_tensors()
        advantages, returns = buffer.compute_gae(config["GAMMA"], config["LAMBDA"])

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return torch.utils.data.TensorDataset(
            tensors["state"],
            tensors["action"],
            tensors["logprob"],
            advantages,
            returns
        )

@dataclass
class DQNExperience(Experience):
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    done: torch.Tensor

    @staticmethod
    def distribution(config, logits):
        return Experience.distribution(config, logits / config["TEMPERATURE"])

    @staticmethod
    def build_dataset(config: dict[str, Any], buffer: ERMBuffer):
        tensors = buffer.to_tensors()

        return torch.utils.data.TensorDataset(
            tensors["state"],
            tensors["action"],
            tensors["reward"],
            tensors["next_state"],
            tensors["done"]
        )

class ExperienceGenerator[T: Experience]:
    def __init__(self, config, model: net.Network, experience_type: type[T]):
        self.engines: list[tetris.Matris] = []
        for _ in range(config["PARALLEL_ENVS"]):
            self.engines.append(tetris.Matris(config))
        self.model: net.Network = model
        self.config = config
        self.experience_type = experience_type
        self.field_names = [field.name for field in fields(experience_type)]

    def state_storage(self):
        return np.ndarray((self.config["PARALLEL_ENVS"], 2, tetris.MATRIX_HEIGHT, tetris.MATRIX_WIDTH), dtype=np.uint8)

    def generate(self):
        buffer = ERMBuffer()
        self.model.eval()
        runs_progress = tqdm(
            range(self.config["MAX_EPISODES"]),
            desc="Runs",
            dynamic_ncols=True,
            leave=False,
            position=1
        )
        states = self.state_storage()
        for i, engine in enumerate(self.engines):
            states[i] = engine.current_state()
        trajectories = [Trajectory() for _ in range(self.config["PARALLEL_ENVS"])]
        for _ in runs_progress:
            episode_progress = tqdm(
                range(self.config["MAX_EPISODE_LENGTH"]),
                desc="Experiences",
                dynamic_ncols=True,
                leave=False,
                position=2
            )
            finished = [False] * self.config["PARALLEL_ENVS"]
            for _ in episode_progress:
                state_tensor = torch.Tensor(states).to(self.model.device)

                with torch.no_grad():
                    self.model(state_tensor)

                actions, dist = self.experience_type.distribution(self.config, self.model.get("action_logits"))
                requires_logprob = "logprob" in self.field_names
                if requires_logprob:
                    logprob = dist.log_prob(actions).cpu()

                for i in range(self.config["PARALLEL_ENVS"]):
                    if finished[i]:
                        continue
                    state, reward, lines_cleared, game_over, truncated = self.engines[i].step(tetris.Action(actions[i].item()))

                    is_done = game_over or (truncated and self.config["MATRIS_TRUNCATE_HARD_BOUNARY"])

                    reward = torch.tensor(reward)
                    done = torch.tensor(int(is_done))

                    data_dict = {
                        "state": state_tensor[i].detach().cpu(),
                        "action": actions[i].cpu(),
                        "reward": reward,
                        "done": done
                    }

                    if requires_logprob:
                        data_dict["logprob"] = logprob[i]

                    if "state_value" in self.field_names:
                        data_dict["state_value"] = self.model.get("state_value")[i].detach().cpu()

                    if "next_state" in self.field_names:
                        data_dict["next_state"] = torch.tensor(state)

                    trajectories[i].append(data_dict)

                    if game_over:
                        trajectories[i].set_last_value(0) # Value is masked out anyway
                        buffer.append(trajectories[i])
                        trajectories[i] = Trajectory()

                        states[i] = self.engines[i].reset()
                        finished[i] = True
                    elif truncated:
                        if not is_done:
                            self.model(torch.Tensor(state).unsqueeze(0).to(self.model.device))
                            trajectories[i].set_last_value(self.model.get("state_value"))
                        else:
                            trajectories[i].set_last_value(0)
                        buffer.append(trajectories[i])
                        trajectories[i] = Trajectory()
                        if self.config["BREAK_ON_TRUNCATE"]:
                            finished[i] = True
                        else:
                            states[i] = state
                    else:
                        states[i] = state

                if all(finished):
                    break

        return buffer

class NetworkEpochTrainer[T: Experience]:
    def __init__(self, config, model: net.Network | list[net.Network] | dict[Any, net.Network], generator: ExperienceGenerator[T], step_func):
        self.model = model
        self.generator = generator
        self.config = config
        self.step_func = step_func

        if isinstance(self.model, list):
            self.device = self.model[0].device
        elif isinstance(self.model, dict):
            self.device = next(iter(self.model.values())).device
        else:
            self.device = self.model.device

    def train(self) -> tuple[float | int, int]:
        with torch.no_grad():
            buffer = self.generator.generate()

            dataset = self.generator.experience_type.build_dataset(self.config, buffer)

            loader = torch.utils.data.DataLoader(dataset, batch_size=self.config["BATCH_SIZE"], shuffle=self.config["SHUFFLE_EXPERIENCES"])

        epochs_progress = tqdm(
            range(self.config["EPOCHS"]),
            desc="Epochs",
            dynamic_ncols=True,
            leave=False,
            position=1
        )

        epochs_progress.set_postfix({
            "Gathered Experiences": len(buffer)
        })

        total_loss = 0
        total_batches = 0

        self.model.train()
        for _ in epochs_progress:
            for batch in loader:
                total_loss += self.step_func(self, tuple(b.to(self.device) for b in batch))
                total_batches += 1

        return total_loss, total_batches