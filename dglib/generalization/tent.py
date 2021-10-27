"""
Modified from https://github.com/KaiyangZhou/mixstyle-release
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dalib.modules.entropy import entropy


class Tent(nn.Module):
    """A wrapper from `Tent: Fully Test-Time Adaptation by Entropy Minimization (ICLR 2021)
    <https://openreview.net/pdf?id=uXl3bZLkr3c>`_. During test phase, this module automatically update classifier by
    entropy minimization.

    Args:
        model (nn.Module): classifier.
        optimizer: optimizer for test-time adaptation.
        steps (int, optional): steps to update for a single batch. Default: 1
    """

    def __init__(self, model, optimizer, steps=1):
        super().__init__()
        self.model = model
        self.optimizer = optimizer

        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.steps = steps

    def forward(self, x):
        for i in range(self.steps):
            outputs = forward_and_adapt(x, self.model, self.optimizer)

        return outputs


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    outputs, _ = model(x)
    outputs = F.softmax(outputs, dim=1)
    # adapt
    loss = entropy(outputs, reduction='mean')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return outputs


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names
