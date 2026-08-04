from collections import defaultdict

import json
from abc import abstractmethod

from torch import nn
import torch.nn.functional as F
import torch
import pathlib

from typing import Any, Protocol

from config import DotDict

def make_conv2d(inc, outc, kernel_size, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(inc, outc, kernel_size=kernel_size, padding=padding, stride=stride),
        nn.GroupNorm(1, outc),
        nn.ReLU())


def make_conv1d(inc, outc, kernel_size, padding = 0, stride = 1):
    return nn.Sequential(
        nn.Conv1d(inc, outc, kernel_size=kernel_size, padding=padding, stride=stride),
        nn.GroupNorm(1, outc),
        nn.ReLU())


def make_lazy_linear(out, p):
    return nn.Sequential(
        nn.LazyLinear(out),
        nn.LayerNorm(out),
        nn.ReLU(),
        nn.Dropout(p))


def make_linear(inp, out, p):
    return nn.Sequential(
        nn.Linear(inp, out),
        nn.LayerNorm(out),
        nn.ReLU(),
        nn.Dropout(p))


class Parallel(nn.Module):
    def __init__(self, *branches, batched=True):
        super().__init__()
        self.branches = nn.ModuleList(branches)
        if batched:
            self.dim = 1
        else:
            self.dim = 0

    def forward(self, x):
        return torch.cat([torch.flatten(branch(x), start_dim=self.dim) for branch in self.branches], dim=self.dim)


class Input(nn.Module):
    def __init__(self, input_key):
        super().__init__()
        self.blt_input_key = input_key

    def forward(self, x):
        return x


class Output(nn.Module):
    def __init__(self, output_key):
        super().__init__()
        self.blt_output_key = output_key

    def forward(self, x):
        return x


class Lr(nn.Module):
    def __init__(self, module, lr_value):
        super().__init__()
        self.module = module
        self.blt_lr_value = lr_value

    def forward(self, x):
        return self.module(x)


_MISSING = object()
_NOTHING = object()


def getattr_recursive(module: nn.Module, name: str, default: Any = _MISSING) -> Any:
    for submodule in module.modules():
        value = getattr(submodule, name, _NOTHING)
        if value is not _NOTHING:
            return value

    if default is _MISSING:
        raise AttributeError(f"No submodule has attribute {name!r}")

    return default


class Network(nn.Module):
    def __init__(self, branches: dict[str, nn.Module], default_lr=1e-3, device=torch.device("cpu")):
        super().__init__()
        # self.branch_modules_list = nn.ModuleList(branches.values())
        self.branches: dict[str, dict[str, Any]] = {
            branch_name: {
                "lr": getattr_recursive(branch, "blt_lr_value", default_lr),
                "input": getattr_recursive(branch, "blt_input_key", None),
                "output": getattr_recursive(branch, "blt_output_key", "out")
            } for branch_name, branch in branches.items()
        }
        for branch_name, branch in branches.items():
            setattr(self, branch_name, branch)
        self.cache = {}
        self.optimizer = torch.optim.AdamW([
            {
                "name": branch_name,
                "params": branch.parameters(),
                "lr": self.branches[branch_name]["lr"]
            } for branch_name, branch in branches.items()
        ])
        self.device = device
        self.to(device)

    def forward(self, x):
        self.cache = {}
        unevaluated: list[str] = [branch_name for branch_name in self.branches.keys()]
        while len(unevaluated) > 0:
            uneval2 = []
            for branch_name in unevaluated:
                branch: dict[str, Any] = self.branches[branch_name]
                if branch["input"] is None:
                    self.cache[branch["output"]] = getattr(self, branch_name)(x)
                elif branch["input"] in self.cache:
                    # print(f"type {branch["input"]} for {self.cache[branch["input"]].shape}")
                    self.cache[branch["output"]] = getattr(self, branch_name)(self.cache[branch["input"]])
                else:
                    uneval2.append(branch_name)
            if len(uneval2) == len(unevaluated):
                raise ValueError(f"Loop without evaluation occurred. Unevaluated Branches: {unevaluated}")
            unevaluated = uneval2

        return next(reversed(self.cache.values()))

    def __contains__(self, item) -> bool:
        return item in self.cache

    def __getitem__(self, item) -> torch.Tensor:
        return self.cache[item]

    def get(self, key) -> torch.Tensor:
        return self.cache[key]

    def zero(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def save(self, file):
        if file:
            file = pathlib.Path(file)
            tmp_file = file.with_suffix(file.suffix + ".tmp")
            torch.save(self.state_dict(), tmp_file)
            tmp_file.replace(file)

            learn_rate_path = file.with_suffix(".lr")
            with learn_rate_path.open(mode="w") as w:
                json.dump([group["lr"] for group in self.optimizer.param_groups], w, indent=4)

    def load(self, file):
        if file is not None and pathlib.Path(file).exists():
            self.load_state_dict(torch.load(file, weights_only=True, map_location=self.device))

            learn_rate_path = file.with_suffix(".lr")
            if learn_rate_path.exists():
                learn_rate = json.load(learn_rate_path.open(mode="r"))
                for i, group in enumerate(learn_rate):
                    self.optimizer.param_groups[i]["lr"] = group
        return self

class TrainerType(Protocol):
    model: Network | list[Network] | dict[Any, Network]
    config: DotDict
    should_exit: bool
    runner: Any
    storage: defaultdict[str, Any]

    @staticmethod
    def kl_approx(old_log_probs, new_log_probs):
        log_ratio = new_log_probs - old_log_probs
        approx_kl = ((log_ratio.exp() - 1) - log_ratio).mean()
        return approx_kl

    def train(self) -> float:
        ...

class MultiVariableScheduler:
    def __init__(self, config, model: Network):
        self.config = config
        self.model = model

    def is_improvement(self, name, value_dict: dict[str, Any]):
        if "ABS" in value_dict["mode"]:
            value_transform = lambda value: abs(value)
        else:
            value_transform = lambda value: value

        new_value = value_transform(value_dict["value"])
        old_value = value_transform(self.config["CURRENT_PATIENCE"][name]["value"])

        if "MIN" in value_dict["mode"]:
            comparator = lambda nv, cv: nv < cv
        elif "MAX" in value_dict["mode"]:
            comparator = lambda nv, cv: nv > cv
        else:
            raise SystemExit(f"Invalid comparison mode {value_dict["mode"]}!")

        if name not in self.config["CURRENT_PATIENCE"] or comparator(new_value, old_value):
            self.config["CURRENT_PATIENCE"][name] = {
                "value": float(new_value),
                "timer": 0
            }
            return True
        return False

    def update_group(self, group_name=None):
        for group in self.model.optimizer.param_groups:
            if group_name is None or group.get("name") is group_name:
                group["lr"] = max(group["lr"] * self.config["DECAY_RATE"], self.config["MIN_LEARN_RATE"])

    def reset_timers(self):
        for value in self.config["CURRENT_PATIENCE"].values():
            value["timer"] = 0

    def schedular_step(self, average_returns, actor_loss, critic_loss):
        stats = {
            "actor": {
                "value": actor_loss,
                "mode": "ABS_MIN",
            },
            "critic": {
                "value": critic_loss,
                "mode": "ABS_MIN",
            },
            "returns": {
                "value": average_returns,
                "mode": "MAX",
            }
        }
        for name, values in stats.items():
            self.is_improvement(name, values)
            self.config["CURRENT_PATIENCE"][name]["timer"] += 1
        overtime = {name: self.config["CURRENT_PATIENCE"][name]["timer"] >= self.config["PATIENCE"] for name in stats.keys()}

        if all(overtime.values()):
            self.update_group()
            self.reset_timers()
            self.config["ENTROPY"] = max(self.config["ENTROPY_MIN"], self.config["ENTROPY"] - self.config["ENTROPY_DECAY_AMOUNT"])

    def get_last_lr(self) -> list[float]:
        return [group.get("lr", 0.0) for group in self.model.optimizer.param_groups]

    def get_average_lr(self):
        lr = self.get_last_lr()
        return sum(lr) / len(lr)

def get_device(div = None, echo=True):
    if div is None:
        div = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if echo:
            print(f"Using device: {div} | HIP: {getattr(torch.version, 'hip', None) or getattr(torch.version, 'cuda', None)}")
    return div