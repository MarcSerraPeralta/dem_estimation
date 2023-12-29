from typing import List, Dict
from itertools import combinations

import numpy as np


def get_edge_probabilities(
    defects: np.ndarray,
    edges: List = None,
    boundary_edges: List = None,
    neighbors: Dict = None,
    avoid_nans: bool = True,
) -> Dict:
    """
    Infers the probability of each edges between the given pairs of defects
    (including the boundary node) based on the observed defects provided.

    For the theory behind this formulat see Eqns. 11 and 16 from the article
    "Exponential suppression of bit or phase errors with cyclic error correction" by Google Quantum AI,
    found in the Supplementary information, accessible from https://doi.org/10.1038/s41586-021-03588-y.

    Parameters
    ----------
    defects
        Defect observations with shape (n_shots, n_defects)
    edges
        List containing the edges to calculate, corresponding to pairs of defects.
        If not specified, calculates all possible edges.
    boundary_edges
        List containing the edges to calculate, corresponding to defects.
        If not specified, calculates all possible boundary edges.
    avoid_nans
        If True, ensures that the values inside square roots are positive.

    Returns
    -------
    edge_probs
        Dictionary containing the edges and their estimated probabilities.
    """
    # checks
    if (not isinstance(edges, list)) and (edges is not None):
        raise ValueError(f"'edges' must be a list or None, but {type(edges)} was given")
    if (not isinstance(boundary_edges, list)) and (boundary_edges is not None):
        raise ValueError(
            f"'boundary_edges' must be a list or None, but {type(boundary_edges)} was given"
        )

    # setup inputs and ouputs
    _, n_defects = defects.shape
    edge_probs = {}

    if edges is None:
        edges = list(combinations(range(n_defects), 2))
        # avoid expensive calculation
        if neighbors is None:
            all_d = np.arange(n_defects)
            neighbors = {d: all_d[all_d != d] for d in range(n_defects)}

    if boundary_edges is None:
        boundary_edges = np.arange(n_defects)

    if neighbors is None:
        neighbors = {d: [] for d in range(n_defects)}
        for e1, e2 in edges:
            neighbors[e1].append(e2)
            neighbors[e2].append(e1)
        neighbors = {d: np.array(n) for d, n in neighbors.items()}

    # obtain <didj> and <di>
    cross_corr_mat = np.einsum("ni, nj -> nij", defects, defects)
    didj = np.average(cross_corr_mat, axis=0)
    di = np.average(defects, axis=0)
    di_matrix = np.repeat(di[np.newaxis, :], len(di), axis=0)

    # get edges using Eq. 11 from the reference above
    numerator = 4 * (didj - di_matrix * di_matrix.T)
    denominator = 1 - 2 * di_matrix - 2 * di_matrix.T + 4 * didj
    fraction = numerator / denominator

    if avoid_nans:
        fraction[fraction < 0] = 0

    pij = 0.5 - 0.5 * np.sqrt(1 - fraction)

    for edge in edges:
        edge_probs[tuple(edge)] = pij[edge[0], edge[1]]

    # get boundary edges using Eq. 16 from the reference above
    for d in boundary_edges:
        probs = pij[neighbors[d], d]
        p_sigma = add_probs(probs)
        edge_probs[(d,)] = (di[d] - p_sigma) / (1 - 2 * p_sigma)

    return edge_probs


def add_probs(probs: list) -> float:
    prob = probs[0]
    for p in probs[1:]:
        prob = g(prob, p)
    return prob


def g(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return p * (1 - q) + (1 - p) * q
