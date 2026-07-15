# https://arxiv.org/pdf/2306.02231
import pygame

import network as net
import torch

global_steps = 0

def step(self: net.TrainerType , batch: tuple) -> float:
    b_state, b_action, b_logprob, b_advantages, b_returns = batch

    assert(isinstance(self.model, net.Network))

    self.model(b_state)
    logits = self.model["action_logits"]
    state_values = self.model["state_value"].squeeze(dim=1)

    dist = torch.distributions.Categorical(logits=logits)
    entropy = dist.entropy().mean()
    action_logprob = dist.log_prob(b_action)

    log_importance_ratio = action_logprob - b_logprob

    loss_apa = torch.mean((log_importance_ratio - b_advantages / self.config.network.apa.lamda) ** 2)

    loss_v = torch.nn.functional.smooth_l1_loss(state_values, b_returns).mean()

    entropy_loss = -entropy * self.config.network.apa.entropy

    loss = loss_apa + loss_v * 0.028 + entropy_loss

    self.storage["_kl_batch_estimate"].append(self.kl_approx(b_logprob, action_logprob).mean().item())
    self.storage["actor_loss"].append(loss_apa.item())
    self.storage["critic_loss"].append(loss_v.item())
    self.storage["entropy_loss"].append(entropy_loss.item())
    self.storage["total_loss"].append(loss.item())

    self.model.zero()
    loss.backward()
    self.model.step()

    return loss.item()