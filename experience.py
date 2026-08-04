from collections import namedtuple, defaultdict, deque
from typing import Protocol, Any, Generic, TypeVar, overload, SupportsIndex
from dataclasses import fields, dataclass
import numpy as np
import MaTris.matris as tetris
from tqdm.auto import tqdm
import network as net
import torch

from config import DotDict

class Util:
    @staticmethod
    def compute_returns(gamma: float, rewards, dones) -> np.ndarray:
        assert len(rewards) == len(dones), "rewards and dones must have the same length"
        discounted_returns = np.zeros(len(rewards), dtype=np.float32)

        discounted_returns[-1] = rewards[-1]
        for i in reversed(range(len(rewards) - 1)):
            discounted_returns[i] = (1.0 - dones[i]) * discounted_returns[i + 1] * gamma + rewards[i]

        return discounted_returns

    @staticmethod
    def generalized_advantage_estimate(gamma: float, lamda: float, rewards, state_values, dones, next_value = 0) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(rewards) == len(dones), "rewards and dones must have the same length"
        assert len(rewards) == len(state_values), "rewards and state_values must have the same length"

        size = len(rewards)
        advantage = 0

        device = state_values[0].device if len(state_values) > 0 else torch.device("cpu")

        gae = torch.zeros(size, device=device)
        returns = torch.zeros(size, device=device)

        # Iterate backwards to compute GAE and returns correctly
        for t in reversed(range(size)):
            # Masking terminal states: if dones[t] is True, the next state is 0-valued
            mask = 1.0 - dones[t]

            # TD error delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = rewards[t] + gamma * next_value * mask - state_values[t]

            # GAE: A_t = delta_t + gamma * lambda * mask * A_{t+1}
            # The mask here ensures that we reset the accumulation at episode boundaries
            gae[t] = delta + gamma * lamda * mask * advantage

            # Target return for critic training (Q-estimate)
            returns[t] = gae[t] + state_values[t]

            next_value = state_values[t]

        return gae, returns

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
        assert "done" in self.buffer and "reward" in self.buffer
        return Util.compute_returns(gamma, self.buffer["reward"], self.buffer["done"])

    def compute_gae(self, gamma : float, lamda: float):
        assert "done" in self.buffer and "reward" in self.buffer and "state_value" in self.buffer

        return Util.generalized_advantage_estimate(
            gamma, lamda,
            self.buffer["reward"],
            self.buffer["state_value"],
            self.buffer["done"],
            self.last_value)

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
    def __init__(self, config: DotDict):
        self.config = config
        self.buffer: deque[Trajectory] = deque()

    def trim(self):
        if not self.config.collection.erm.enabled:
            return
        while True:
            if len(self.buffer) <= self.config.collection.erm.minTrajectories:
                break
            if len(self) < self.config.collection.erm.length:
                break
            self.buffer.popleft()

    def append(self, trajectory):
        self.buffer.append(trajectory)
        self.trim()

    def clear(self):
        self.buffer.clear()

    def join(self, other: ERMBuffer):
        self.buffer.extend(other.buffer)
        self.trim()

    def compute_returns(self, gamma: float) -> torch.Tensor:
        rewards = np.concat([np.array(traj.buffer["reward"]) for traj in self.buffer])
        dones = np.concat([np.array(traj.buffer["done"]) for traj in self.buffer])
        return torch.Tensor(Util.compute_returns(gamma, rewards, dones))

    def to_tensors(self):
        tensor_dict = defaultdict(list)
        for trajectory in self.buffer:
            for field, arr in trajectory.buffer.items():
                tensor_dict[field].append(torch.stack(arr))
        return {field: torch.cat(arr) for field, arr in tensor_dict.items()}

    def compute_gae(self, gamma : float, lamda: float) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.buffer[0].buffer["state_value"][0].device
        rewards = torch.Tensor(np.concat([np.array(traj.buffer["reward"]) for traj in self.buffer]), device=device)
        dones = torch.Tensor(np.concat([np.array(traj.buffer["done"]) for traj in self.buffer]), device=device)
        state_values = torch.cat([torch.cat(traj.buffer["state_value"]) for traj in self.buffer])
        return Util.generalized_advantage_estimate(gamma, lamda,
                                                   rewards, state_values, dones)

    def __len__(self):
        return sum(len(traj) for traj in self.buffer)

@dataclass
class Experience(Protocol):
    @staticmethod
    def distribution(config: DotDict, logits: torch.Tensor) -> tuple[torch.Tensor, torch.distributions.Distribution]:
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        return actions, dist

    @staticmethod
    def build_dataset(config: DotDict, buffer: ERMBuffer) -> torch.utils.data.Dataset:
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
    def build_dataset(config: DotDict, buffer: ERMBuffer):
        tensors = buffer.to_tensors()
        advantages, returns = buffer.compute_gae(config.network.gamma, config.network.ppo.lamda)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.clip(-1, 1)

        return torch.utils.data.TensorDataset(
            tensors["state"],
            tensors["action"],
            tensors["logprob"],
            advantages,
            returns
        )

@dataclass
class ReinforceExperience(Experience):
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor

    @staticmethod
    def build_dataset(config: DotDict, buffer: ERMBuffer):
        tensors = buffer.to_tensors()
        returns = buffer.compute_returns(config.network.gamma)

        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return torch.utils.data.TensorDataset(
            tensors["state"],
            tensors["action"],
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
    def distribution(config: DotDict, logits):
        return Experience.distribution(config, logits / config.network.dqn.temperature)

    @staticmethod
    def build_dataset(config: DotDict, buffer: ERMBuffer):
        tensors = buffer.to_tensors()

        return torch.utils.data.TensorDataset(
            tensors["state"],
            tensors["action"],
            tensors["reward"],
            tensors["next_state"],
            tensors["done"]
        )

    @staticmethod
    def post_train(trainer):
        trainer.model["target"].load_state_dict(trainer.model["network"].state_dict())

class ExperienceGenerator[T: Experience]:
    def __init__(self, config: DotDict, model: net.Network, experience_type: type[T]):
        self.engines: list[tetris.Matris] = []
        for _ in range(config.collection.parallelEnvs):
            self.engines.append(tetris.Matris(config))
        self.model: net.Network = model
        self.config = config
        self.experience_type = experience_type
        self.experience_fields = fields(experience_type)
        self.field_names = [field.name for field in fields(experience_type)]
        self.states = self.state_storage()
        self.trajectories = [Trajectory() for _ in range(self.config.collection.parallelEnvs)]
        self.buffer = ERMBuffer(self.config)
        self.lines_cleared = []

    def state_storage(self):
        return np.ndarray((self.config.collection.parallelEnvs, 2, tetris.MATRIX_HEIGHT, tetris.MATRIX_WIDTH), dtype=np.uint8)

    def generate(self):
        self.lines_cleared = []
        buffer = ERMBuffer(self.config)
        self.model.eval()
        runs_progress = tqdm(
            range(self.config.collection.runs) if self.config.collection.type == "runs" else None,
            desc="Runs",
            dynamic_ncols=True,
            leave=False,
            position=1
        )
        for i, engine in enumerate(self.engines):
            self.states[i] = engine.current_state()
        run_iter = runs_progress
        if self.config.collection.type == "experiences":
            def nxt():
                runs_progress.update(1)
                return not (len(buffer) > self.config.collection.experiences)
            run_iter = iter(nxt, False)
        for _ in run_iter:
            episode_progress = tqdm(
                range(self.config.collection.maxExperiencesPerTrajectory),
                desc="Experiences",
                dynamic_ncols=True,
                leave=False,
                position=2
            )
            finished = [False] * self.config.collection.parallelEnvs
            for _ in episode_progress:
                state_tensor = torch.Tensor(self.states).to(self.model.device)

                with torch.no_grad():
                    self.model(state_tensor)

                actions, dist = self.experience_type.distribution(self.config, self.model.get("action_logits"))
                requires_logprob = "logprob" in self.field_names
                if requires_logprob:
                    logprob = dist.log_prob(actions).cpu()

                for i in range(self.config.collection.parallelEnvs):
                    if finished[i]:
                        continue
                    state, reward, lines_cleared, game_over, truncated = self.engines[i].step(tetris.Action(actions[i].item()))

                    is_done = game_over or (truncated and self.config.tetris.truncate.rewardBoundary)
                    should_stop_collection = truncated and self.config.tetris.truncate.stopCollection

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

                    self.trajectories[i].append(data_dict)
                    self.states[i] = state


                    if game_over or should_stop_collection:
                        self.trajectories[i].set_last_value(0) # Value is masked out anyway
                        buffer.append(self.trajectories[i])
                        self.trajectories[i] = Trajectory()

                        if game_over:
                            self.lines_cleared.append(self.engines[i].lines)
                            self.states[i] = self.engines[i].reset()

                        finished[i] = game_over or should_stop_collection

                if all(finished):
                    break

        if self.config.collection.erm.enabled:
            self.buffer.join(buffer)
        else:
            self.buffer = buffer
        return self.buffer

class NetworkEpochTrainer[T: Experience](net.TrainerType):
    def __init__(self, runner, model: net.Network | list[net.Network] | dict[Any, net.Network], generator: ExperienceGenerator[T], step_func):
        self.model = model
        self.generator = generator
        self.runner = runner
        self.config: DotDict = runner.config
        self.step_func = step_func
        self.has_lobprobs = "lobprob" in self.generator.experience_fields
        self.should_exit = False
        self.storage = defaultdict[str, Any](list)

        if isinstance(self.model, list):
            self.device = self.model[0].device
            self.train_func = lambda: [network.train() for network in self.model]
        elif isinstance(self.model, dict):
            self.device = next(iter(self.model.values())).device
            self.train_func = lambda: [network.train() for network in self.model.values()]
        else:
            self.device = self.model.device
            self.train_func = self.model.train

    def build_dataset(self, progress_bar = None):
        with torch.no_grad():
            buffer = self.generator.generate()

            dataset = self.generator.experience_type.build_dataset(self.config, buffer)

            loader = torch.utils.data.DataLoader(dataset, batch_size=self.config.training.batchSize, shuffle=self.config.training.shuffle)

        if progress_bar is not None:
            progress_bar.set_postfix({
                "Gathered Experiences": len(buffer)
            })

        self.train_func()
        return loader, buffer

    def train(self) -> float:
        epochs_progress = tqdm(
            range(self.config.training.epoch.epochs) if self.config.training.type == "epoch" else None,
            desc="Epochs",
            dynamic_ncols=True,
            leave=False,
            position=1
        )

        loop_iter = epochs_progress
        if self.config.training.type == "kl":
            def up():
                epochs_progress.update(1)
                if self.config.training.kl.useEpochLimit and epochs_progress.n > self.config.training.epoch.epochs:
                    return False

                # Could use this local KL estimate to limit the number of experiences alongside the kl estimate
                # take the number of experiences before hitting kl cutoff and min(double of value, constant) as the generation amount
                # or don't have the min and use this for fully automatic scaling.
                if len(self.storage["kl_estimate"]) > 0:
                    kl_estimate = np.array(self.storage["_kl_batch_estimate"]).mean()
                    epochs_progress.set_postfix({
                        "Gathered Experiences": len(buffer),
                        "KL Estimate": kl_estimate
                    })
                    return kl_estimate < self.config.training.kl.kl_cutoff
                return True
            loop_iter = iter(up, False)


        loader, buffer = self.build_dataset(epochs_progress)

        total_batches = 0

        self.storage.clear()
        for _ in loop_iter:
            for batch in loader:
                self.step_func(self, tuple(b.to(self.device) for b in batch))
                total_batches += 1
                if self.should_exit:
                    break
            if "_kl_batch_estimate" in self.storage:
                self.storage["kl_estimate"].append(np.array(self.storage["_kl_batch_estimate"]).mean())
            if self.should_exit:
                break

        for key, value in self.storage.items():
            if key.startswith("_"):
                continue
            if isinstance(value, list):
                self.runner.log_to_list(key, np.array(value).mean())
            else:
                self.runner.log_to_list(key, value)

        if hasattr(self.generator.experience_type, "post_train"):
            self.generator.experience_type.post_train(self)

        return float(total_batches)





