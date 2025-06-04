import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812

def cumulative_sequence_accuracies(
    predictions: Tensor, targets: Tensor, ignore_index: int | None = None
) -> dict[str, Tensor]:
    """
    Computes the sequence_accuracy for each sequence length from 1 to the length of the sequences.
    The `predictions` can be passed in as logits or as tokens; if passed as logits, 
    `argmax(dim=-1)` will be applied.
    
    Args:
        predictions (Tensor): The predictions.
        targets (Tensor): The targets.
        ignore_index (int, optional): An index to ignore in the accuracy calculation. 
                                      Defaults to None.
    Returns:
        dict[str, Tensor]: A dictionary with 'accuracies' and 'n_samples'.
    """
    # Adjust predictions if they are logits
    if predictions.dim() != targets.dim():
        predictions = predictions.argmax(dim=-1)

    assert predictions.size() == targets.size(), f"{predictions.size()} != {targets.size()}"

    # Create a mask for valid positions if ignore_index is specified
    if ignore_index is not None:
        valid_mask = targets != ignore_index
    else:
        valid_mask = torch.ones_like(targets, dtype=torch.bool)
    
    correct = (predictions == targets) & valid_mask
    cumulative_correct = correct.cumsum(dim=1)
    cumulative_valid = valid_mask.cumsum(dim=1)

    accuracies = (cumulative_correct / cumulative_valid).float().floor().mean(dim=0).cpu().numpy()

    return {
        "value": accuracies,
        "n_samples": targets.size(0),
    }