from typing import List, Dict, Tuple
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
    n_shots, n_defects = defects.shape
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

    # get edges using Eq. 11 from the reference above
    di = np.average(defects, axis=0)
    for e1, e2 in edges:
        xi, xj = defects[:, e1], defects[:, e2]
        xixj = np.einsum("i,i -> ", xi, xj, dtype=np.int32) / n_shots
        numerator = 4 * (xixj - di[e1] * di[e2])
        denominator = 1 - 2 * di[e1] - 2 * di[e2] + 4 * xixj
        tmp = 1 - numerator / denominator

        if avoid_nans:
            tmp = 0 if tmp < 0 else tmp

        edge_probs[e_pair(e1, e2)] = 0.5 - 0.5 * np.sqrt(tmp)

    # get boundary edges using Eq. 16 from the reference above
    for d in boundary_edges:
        probs = [edge_probs[e_pair(e1, d)] for e1 in neighbors[d]]
        p_sigma = add_probs(probs)
        edge_probs[(d,)] = (di[d] - p_sigma) / (1 - 2 * p_sigma)

    return edge_probs


def e_pair(e1: int, e2: int) -> Tuple:
    """
    Returns a standarized edge pair (e1, e2)
    """
    return tuple(sorted((e1, e2)))


def add_probs(probs: list) -> float:
    """
    Returns the joined probability that an odd number
    of independent mechanisms are triggered.

    Parameters
    ----------
    probs
        Probabilities of the independent mechanisms

    Returns
    -------
    prob
        Joined probability that an odd number
        of independent mechanisms are triggered
    """
    prob = probs[0]
    for p in probs[1:]:
        prob = g(prob, p)

    return prob


def g(p: float, q: float) -> float:
    """
    Returns the joined probability that only one
    indepedent mechanism is triggered from a set of two.

    Parameters
    ----------
    p
        Probability for the first mechanism
    q
        Probability for the second mechanism

    Returns
    -------
    Joined probability that only one indepedent mechanism is
    triggered from a set of two
    """
    return p * (1 - q) + (1 - p) * q
