import pygame
import numpy as np
import torch
from torch import nn

from config import DotDict

import network as net

ACTOR_OUTPUT = 5
CRITIC_OUTPUT = 1


# http://vision.stanford.edu/teaching/cs231n/reports/2016/pdfs/121_Report.pdf
def AdjustedSandfordNetwork(config: DotDict, device = None):
    device = net.get_device(device)
    p = config.network.dropout
    return net.Network(
        {
            "conv_filters": net.Lr(nn.Sequential(
                net.make_conv2d(2, 32, kernel_size=(3, 3), padding=1),
                net.make_conv2d(32, 32, kernel_size=(3, 3), padding=1),
                net.make_conv2d(32, 64, kernel_size=(3, 3), padding=1),

                net.make_conv2d(64, 64, kernel_size=(22, 1), padding=1),

                net.make_conv2d(64, 128, kernel_size=(1, 3), padding=1),
                net.make_conv2d(128, 128, kernel_size=(1, 1), padding=1),
                net.make_conv2d(128, 128, kernel_size=(1, 3), padding=1),
                nn.Flatten(),
                net.Output("conv")
            ), config.network.init_lr.convLearnRate),
            "actor_head": net.Lr(nn.Sequential(
                net.Input("conv"),
                net.make_lazy_linear(128, p),
                net.make_linear(128, 512, p),
                nn.Linear(512, ACTOR_OUTPUT),
                net.Output("action_logits")
            ), config.network.init_lr.actorLearnRate),
            "critic_head": net.Lr(nn.Sequential(
                net.Input("conv"),
                net.make_lazy_linear(128, p),
                net.make_linear(128, 512, p),
                nn.Linear(512, CRITIC_OUTPUT),
                net.Output("state_value")
            ), config.network.init_lr.criticLearnRate)
        },
        default_lr=config.network.init_lr.convLearnRate, device=device
    )


def step(self: net.TrainerType , batch: tuple):
    b_state, b_action, b_logprob, b_advantages, b_returns = batch

    assert(isinstance(self.model, net.Network))

    self.model(b_state)
    logits = self.model["action_logits"]

    dist = torch.distributions.Categorical(logits=logits)
    action_logprob = dist.log_prob(b_action)
    entropy = dist.entropy()

    state_values1 = self.model["state_value"]
    state_values = state_values1.squeeze(dim=1)

    # PPO Surrogate Objective
    importance_ratio = torch.exp(action_logprob - b_logprob)
    surrogate = importance_ratio * b_advantages
    clipped_surrogate = torch.clamp(importance_ratio, 1 - self.config.network.clipping.actor_epsilon,
                                    1 + self.config.network.clipping.actor_epsilon) * b_advantages

    actor_loss = -(torch.min(surrogate, clipped_surrogate) + self.config.network.ppo.entropy * entropy).mean()
    critic_loss = torch.nn.functional.huber_loss(state_values, b_returns).mean()
    total_loss = (actor_loss + (critic_loss * 0.5) * 0.01)

    self.storage["_kl_batch_estimate"].append(self.kl_approx(b_logprob, action_logprob).mean().item())
    self.storage["actor_loss"].append(actor_loss.item())
    self.storage["critic_loss"].append(critic_loss.item())
    self.storage["total_loss"].append(total_loss.item())
    self.storage["advantages"].append(b_advantages.mean().item())
    # self.runner.log_to_list("kl_estimate", kl_estimate)

    self.model.zero()
    total_loss.backward()
    self.model.step()


class NetworkRealtimeVisualizer:
    def __init__(self, surface: pygame.Surface, rect):
        self.rect = rect
        self.surface = surface
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)

    def clear(self, color=(12, 12, 18, 255)):
        self.surface.fill(color, self.rect)

    def present(self):
        pass

    def fill_rect(self, rect, color):
        self.surface.fill(color, rect)

    def draw_rect(self, rect, color):
        pygame.draw.rect(self.surface, color, rect, width=1)

    def draw_text(self, text, x, y, color=(240, 240, 240), small=False):
        font = self.small_font if small else self.font
        text_surface = font.render(text, True, color[:3])

        target = pygame.Rect(x, y, text_surface.get_width(), text_surface.get_height())
        self.surface.blit(text_surface, target)

    def draw_grid_channel(self, channel, x0, y0, title, scale=14):
        self.draw_text(title, x0, y0 - 24)

        channel = np.asarray(channel)
        for y in range(channel.shape[0]):
            for x in range(channel.shape[1]):
                value = float(channel[y, x])

                if value <= 0:
                    color = (25, 25, 30, 255)
                elif value < 2:
                    color = (90, 90, 150, 255)
                else:
                    color = (80, 210, 120, 255)

                rect = pygame.Rect(x0 + x * scale, y0 + y * scale, scale - 1, scale - 1)
                self.fill_rect(rect, color)

    def draw_action_probs(self, probs, logits, critic_value, selected_action=None):
        action_names = ["RIGHT", "LEFT", "DOWN", "ROTATE", "HARD_DROP"]

        x0 = self.rect[0] + 230
        y0 = 60
        bar_width = 220
        bar_height = 24
        gap = 24

        self.draw_text("Actor output", x0, 25)
        self.draw_text(f"Critic: {critic_value:.4f}", x0, 330)

        if selected_action is not None:
            self.draw_text(f"Selected: {action_names[selected_action]}", x0, 360)

        for i, prob in enumerate(probs):
            y = y0 + i * (bar_height + gap)

            bg_rect = pygame.Rect(x0, y, bar_width, bar_height)
            prob_rect = pygame.Rect(x0, y, int(bar_width * float(prob.detach())), bar_height)

            self.fill_rect(bg_rect, (55, 55, 65, 255))
            self.fill_rect(prob_rect, (80, 180, 255, 255))

            if selected_action == i:
                self.draw_rect(
                    pygame.Rect(x0 - 4, y - 4, bar_width + 8, bar_height + 8),
                    (255, 220, 80, 255)
                )

            self.draw_text(
                f"{action_names[i]}: {float(prob):.3f}",
                x0,
                y + 3,
                color=(255, 255, 255),
                small=True
            )

            self.draw_text(
                f"logit {float(logits[i]):+.3f}",
                x0,
                y + bar_height + 6,
                color=(180, 180, 180),
                small=True
            )

    def update(self, state, logits, probs, critic_value, selected_action=None, expects="normal"):
        self.clear()

        state_np = np.asarray(state)

        if expects == "normal":
            self.draw_grid_channel(state_np[0], self.rect[0] + 20, 55, "Board channel")
            self.draw_grid_channel(state_np[1], self.rect[0] + 20, 365, "Piece channel", scale=6)

        if probs is not None and logits is not None and critic_value is not None:
            self.draw_action_probs(probs, logits, critic_value, selected_action)

        self.present()
