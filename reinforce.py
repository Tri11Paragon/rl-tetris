import torch

from config import DotDict

import network as net
import ppo2

def network(config: DotDict, device = None):
    _network = ppo2.AdjustedSandfordNetwork(config, device)
    _network.branches.pop("critic")
    return _network

def step(self: net.TrainerType , batch: tuple):
    b_state, b_action, b_returns = batch

    assert(isinstance(self.model, net.Network))

    self.model(b_state)
    logits = self.model["action_logits"]

    dist = torch.distributions.Categorical(logits=logits)
    action_logprob = dist.log_prob(b_action)
    entropy = dist.entropy()

    objective = action_logprob * b_returns

    actor_loss = -(objective + self.config.network.ppo.entropy * entropy).mean()

    self.storage["actor_loss"].append(actor_loss.item())

    self.model.zero()
    actor_loss.backward()
    self.model.step()