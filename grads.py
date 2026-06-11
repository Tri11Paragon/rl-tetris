# https://docs.pytorch.org/tutorials/intermediate/visualizing_gradients_tutorial.html

from collections import defaultdict

import torch
from torch import nn


def hook_forward(module_name, grads, hook_backward):
    def hook(module, args, output):
        """Forward pass hook which attaches backward pass hooks to intermediate tensors"""
        output.register_hook(hook_backward(module_name, grads))

    return hook


def hook_backward(module_name, grads):
    def hook(grad):
        """Backward pass hook which appends gradients"""
        grads[module_name].append(grad.abs().mean())

    return hook


def get_all_layers(model, hook_forward, hook_backward):
    """Register forward pass hook (which registers a backward hook) to model outputs

    Returns:
        - layers: a dict with keys as layer/module and values as layer/module names
                  e.g. layers[nn.Conv2d] = layer1.0.conv1
        - grads: a list of tuples with module name and tensor output gradient
                 e.g. grads[0] == (layer1.0.conv1, tensor.Torch(...))
    """
    layers = dict()
    grads = defaultdict(list)

    for name, layer in model.named_modules():
        # skip Sequential and/or wrapper modules
        if not any(layer.children()):
            layers[layer] = name
            layer.register_forward_hook(hook_forward(name, grads, hook_backward))
    return layers, grads


def get_grads(grads):
    layer_idx = []
    avg_grads = []
    for idx, layer_name in enumerate(grads):
        grad_list = grads[layer_name]
        if len(grad_list) > 0:
            # Mean absolute gradients per step
            new_list = torch.stack(grad_list)
            avg_grads.append(new_list.mean().cpu())
            # idx is backwards since we appended in backward pass
            layer_idx.append(len(grads) - 1 - idx)
    return layer_idx, avg_grads

class ActivationRecorder:
    def __init__(self, model, include_layer_types=(nn.Conv2d, nn.Linear, nn.LazyLinear, nn.GroupNorm, nn.LayerNorm, nn.GELU)):
        self.model = model
        self.include_layer_types = include_layer_types
        self.activations = {}
        self.handles = []
        self.enabled = True

        self.register()

    def register(self):
        for name, module in self.model.named_modules():
            if not name:
                continue

            if not isinstance(module, self.include_layer_types):
                continue

            handle = module.register_forward_hook(self.make_hook(name))
            self.handles.append(handle)

    def make_hook(self, name):
        def hook(module, args, output):
            if not self.enabled:
                return

            if isinstance(output, tuple):
                output = output[0]

            if not torch.is_tensor(output):
                return

            detached = output.detach()

            self.activations[name] = {
                "module": module.__class__.__name__,
                "shape": tuple(detached.shape),
                "mean": detached.mean().item(),
                "std": detached.std().item() if detached.numel() > 1 else 0.0,
                "min": detached.min().item(),
                "max": detached.max().item(),
                "abs_mean": detached.abs().mean().item(),
                "data": detached.float().cpu(),
            }

        return hook

    def clear(self):
        self.activations.clear()

    def snapshot(self):
        return dict(self.activations)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
