"""Smish activation; Wang et al., Electronics 11.4 (2022): 540."""

import torch
import torch.nn.functional as F
from torch import nn

import utils.AF.Fsmish as Func

class Smish(nn.Module):
    """Smish activation."""

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return Func.smish(input)
