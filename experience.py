from collections import namedtuple, defaultdict
from typing import Protocol, Any, Generic, TypeVar, get_type_hints, overload
import numpy as np

T = TypeVar("T")

Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])

PPOExperience = namedtuple("PPOExperience", ["state", "action", "reward", "done", "logprob", "state_value"])

class ExperienceLike(Protocol):
    state: Any
    action: int
    reward: float
    done: int

class PPOExperienceLike(ExperienceLike):
    logprob: float
    state_value: float

class DQNExperienceLike(ExperienceLike):
    next_state: Any

# Trajectory holds a set of related experiences as part of an episode
class Trajectory(Generic[T]):
    def __init__(self):
        self.Keys = get_type_hints(T).keys()
        # Buffer of experiences
        self.buffer: dict[str, list] = {}

        for field in self.Keys:
            self.buffer[field] = []

        # Optional,
        self.last_value = None

    ''' Append an experience to the trajectory. 
        If this is the last experience which will be pushed to the buffer,
        please make sure to supply the last_value argument.
        This allows any algorithm using these experiences to bootstrap using last state's value.'''
    def append(self, experience: T, last_value=None):
        for key in self.Keys:
            self.buffer[key].append(getattr(experience, key))
        self.last_value = last_value

    def clear(self):
        for _, value in self.buffer.items():
            value.clear()

    def compute_gae(self, gamma : float, lamda: float):
        size = len(self.buffer)
        gae = np.zeros(size)
        returns = np.zeros(size)
        advantage = 0
        next_value = self.last_value

        dones = self.buffer["done"]
        rewards = self.buffer["reward"]
        values = self.buffer["state_value"]

        # Iterate backwards to compute GAE and returns correctly
        for t in reversed(range(N)):
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

    def __len__(self):
        return len(next(iter(self.buffer.values())))

    def __getitem__(self, idx):
        return self.buffer[idx]

    def __iter__(self):
        return iter(self.buffer)

    def __reversed__(self):
        return reversed(self.buffer)

class ERMBuffer(Generic[T]):
    def __init__(self):
        self.buffer: list[Trajectory[T]] = []

    def append(self, trajectory):
        self.buffer.append(trajectory)

    def clear(self):
        for trajectory in self.buffer:
            trajectory.clear()

@overload
def sample(buffer: Trajectory[T], amount: int):
    first_value = len(buffer)
    size = len(buffer.buffer)
    if size < amount:
        raise ValueError("Cannot sample from empty buffer // buffer size is too small.")
    indices = np.random.choice(size, size=amount, replace=False)
    return {field: arr[indices] for field, arr in buffer.buffer.items()}

@overload
def sample(buffer: ERMBuffer[T], amount: int):
    combined = defaultdict(list)
    for trajectory in buffer.buffer:
        for field, arr in trajectory.buffer.items():
            combined[field].extend(arr)
    size = len(next(iter(combined.values())))
    if size < amount:
        raise ValueError("Cannot sample from empty buffer // buffer size is too small.")
    indices = np.random.choice(size, size=amount, replace=False)
    return {field: arr[indices] for field, arr in combined.items()}

def sample(_, __):
    raise NotImplementedError
