from typing import Dict, Tuple, List

import numpy as np
import stim


def stim_to_edges(dem: stim.DetectorErrorModel) -> Tuple[List, List, Dict]:
    """
    Returns the edges, boundary edges and the logical effect of them from a
    stim detector error model.

    Parameters
    ----------
    dem
        Detector error model in stim format.

    Returns
    -------
    edges
        List of edges present in dem
    boundary_edges
        List of boundary edges present in dem
    edge_logicals
        Dictionary with the edges (keys) and logical flips (values)
    """
    edges = []
    boundary_edges = []
    edge_logicals = {}

    for instr in dem.flattened():
        if instr.type != "error":
            continue

        defects = [d.val for d in instr.targets_copy() if d.is_relative_detector_id()]
        logicals = [d.val for d in instr.targets_copy() if d.is_logical_observable_id()]

        if len(defects) == 1:
            boundary_edges.append(defects[0])
        else:
            edges.append(defects)

        edge_logicals[tuple(defects)] = logicals

    return edges, boundary_edges, edge_logicals


def edges_to_stim(edge_probs: Dict, edge_logicals: Dict) -> stim.DetectorErrorModel:
    """
    Returns a decoding graph from the given edge probabilities and logical effects

    Parameters
    ----------
    edge_probs
        Dictionary containing the edges (key) and probabilities (values)
        for the given error model
    edge_logicals
        Dictionary containing the edges (key) and logical flips (values)
        for the given error model

    Returns
    -------
    dem
        Decoding graph (stim format) corresponding to the inputs
    """
    dem = stim.DetectorErrorModel()

    for edge, prob in edge_probs.items():
        logicals = [
            stim.DemTarget.logical_observable_id(l) for l in edge_logicals[edge]
        ]
        defects = [stim.DemTarget.relative_detector_id(d) for d in edge]
        dem.append("error", prob, defects + logicals)

    return dem


def get_pij_matrix(defects: np.ndarray, avoid_nans: bool = True) -> np.ndarray:
    """
    Calculates the Pij matrix.

    For the theory behind this formulat see Eqns. 11 from the article
    "Exponential suppression of bit or phase errors with cyclic error correction" by Google Quantum AI,
    found in the Supplementary information, accessible from https://doi.org/10.1038/s41586-021-03588-y.

    Parameters
    ----------
    defects
        Defect observations with shape (n_shots, n_defects)
    avoid_nans
        If True, ensures that the values inside square roots are positive.

    Returns
    -------
    pij
        Dictionary containing the edges and their estimated probabilities.
    """
    n_shots, n_defects = defects.shape

    # obtain <didj> and <di>
    didj = np.einsum("ni, nj -> ij", defects, defects, dtype=np.int32) / n_shots
    di = np.average(defects, axis=0)
    di_matrix = np.repeat(di[np.newaxis, :], len(di), axis=0)

    # get edges using Eq. 11 from the reference above
    numerator = 4 * (didj - di_matrix * di_matrix.T)
    denominator = 1 - 2 * di_matrix - 2 * di_matrix.T + 4 * didj
    tmp = 1 - numerator / denominator

    if avoid_nans:
        tmp[tmp < 0] = 0

    pij = 0.5 - 0.5 * np.sqrt(tmp)

    return pij
