from typing import Literal

import numpy as np
import torch
from tqdm import tqdm


def tensor_to_plottable(img: torch.Tensor):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()

    return np.transpose(npimg, (1, 2, 0))


def update_pbar(
    pbar: tqdm,
    n_epochs: int,
    epoch: int,
    loss: float,
    acc: float,
    phase: Literal['train', 'val'],
):
    pbar.set_description(
        f'Epoch {epoch}/{n_epochs} | {phase} loss: {loss:.4f} | {phase} acc: {acc:.4f}'
    )
    pbar.update()
