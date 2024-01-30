from typing import List

from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_pij_matrix(
    ax: plt.Axes,
    pij: np.ndarray,
    qubit_labels: List[str] = None,
    num_rounds: int = None,
    max_prob: float = 0.05,
) -> plt.Axes:
    """
    Replicates the pij matrix plot in Figure 2c from:
    Google Quantum AI. Exponential suppression of bit or phase errors with cyclic
    error correction. Nature 595, 383–387 (2021). https://doi.org/10.1038/s41586-021-03588-y
    """
    # rotate matrix 90 regrees anticlock-wise
    upper_mask = np.rot90(np.triu(np.ones_like(pij)).astype(bool))
    lower_mask = np.rot90(np.tril(np.ones_like(pij)).astype(bool))
    pij = np.rot90(deepcopy(pij))

    extent = [0, len(pij), 0, len(pij)]

    # Plot the upper triangle with the 'Blues' colormap
    colorbar_full = ax.imshow(
        np.ma.array(pij, mask=lower_mask),  # mask invalidates the given elements
        cmap="Blues",
        interpolation="nearest",  # "none" does not work
        vmin=0,
        extent=extent,
    )

    # Plot the lower triangle with the 'Reds' colormap
    colorbar_zoom = ax.imshow(
        np.ma.array(pij, mask=upper_mask),  # mask invalidates the given elements
        cmap="Reds",
        interpolation="nearest",  # "none" does not work
        vmin=0,
        vmax=max_prob,
        extent=extent,
    )

    # Add a colorbars to the right of the plot
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.3)
    colorbar = plt.colorbar(colorbar_zoom, cax=cax)
    colorbar.ax.yaxis.set_ticks_position("left")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.8)
    colorbar = plt.colorbar(colorbar_full, cax=cax)
    colorbar.ax.yaxis.set_label_position("right")

    # Set labels at the center of their regions
    if (num_rounds is not None) and (qubit_labels is not None):
        assert pij.shape == (
            num_rounds * len(qubit_labels),
            num_rounds * len(qubit_labels),
        )

        tick_positions_small = np.arange(0, len(pij) + 1, num_rounds)
        tick_positions_big = np.arange(num_rounds / 2, len(pij) + 1, num_rounds)
        ax.set_xticks(tick_positions_big)
        ax.set_xticks(tick_positions_small, minor=True)
        ax.set_yticks(tick_positions_big)
        ax.set_yticks(tick_positions_small, minor=True)

        # Set tick labels (optional)
        ax.set_xticklabels(qubit_labels)
        ax.set_yticklabels(qubit_labels)

        ax.tick_params(axis="both", which="major", size=0)
        ax.tick_params(axis="both", which="minor", size=5)

    return ax
