"""Tool MLP head, causal multi-stage TCN, and the LoRA layer."""
import math

import torch
import torch.nn.functional as F
from torch import nn

from data import PHASES, TOOLS


def tool_head(in_dim=1536):
    """MLP 1536 -> 1024 -> 7 (GELU, dropout 0.5); logits for the 7 tools."""
    return nn.Sequential(nn.Linear(in_dim, 1024), nn.GELU(), nn.Dropout(0.5), nn.Linear(1024, len(TOOLS)))


class DilatedResidual(nn.Module):
    """x + dropout(conv1x1(relu(dilated conv3(x)))); causal: padded on the left only, so frame t sees frames <= t."""

    def __init__(self, dilation, channels, dropout):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, dilation=dilation)
        self.out = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)
        self.pad = (2 * dilation, 0)

    def forward(self, x):
        return x + self.dropout(self.out(F.relu(self.conv(F.pad(x, self.pad)))))


class TCNStage(nn.Module):
    def __init__(self, in_dim, layers, channels, dropout):
        super().__init__()
        self.inp = nn.Conv1d(in_dim, channels, 1)
        self.layers = nn.Sequential(*[DilatedResidual(2 ** i, channels, dropout) for i in range(layers)])
        self.out = nn.Conv1d(channels, len(PHASES), 1)

    def forward(self, x):
        return self.out(self.layers(self.inp(x)))


class CausalTCN(nn.Module):
    """Causal multi-stage TCN (TeCNO-style). Stage 1 reads the frame features, every later stage refines the
    previous stage's phase probabilities. Dilations 1, 2, 4, ... per stage. (T, d) -> list of (T, 7) logits per stage."""

    def __init__(self, in_dim, layers, stages=2, channels=64, dropout=0.5):
        super().__init__()
        self.stages = nn.ModuleList([TCNStage(in_dim if s == 0 else len(PHASES), layers, channels, dropout)
                                     for s in range(stages)])

    def forward(self, x):
        h, outs = x.T.unsqueeze(0), []
        for s, stage in enumerate(self.stages):
            h = stage(h if s == 0 else F.softmax(h, dim=1))
            outs.append(h.squeeze(0).T)
        return outs


class LoRALinear(nn.Module):
    """y = W x + b + (alpha / r) B A x with W, b frozen; A random, B zero (starts as the original layer)."""

    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.empty(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale
