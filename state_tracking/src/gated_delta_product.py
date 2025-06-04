"""Input-dependent S4 (IDS4)."""

from itertools import accumulate
import math
import torch
from einops import einsum
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812
from fla import GatedDeltaProductForCausalLM


class GatedDeltaProduct(GatedDeltaProductForCausalLM):
    @property
    def num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: Tensor) -> Tensor:
        output = super().forward(input_ids=x)
        return output.logits

    @torch.no_grad
    def get_useful_stats(self):
        print("Empty useful stats for this model, to be implemented")
        return dict()
