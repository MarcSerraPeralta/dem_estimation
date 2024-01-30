from typing import Dict, Tuple, List

from itertools import chain

import numpy as np
import stim
import networkx as nx

from .dem_estimation import g as g_funct


def stim_to_edges(
    dem: stim.DetectorErrorModel, return_coords: bool = False
) -> Tuple[Dict, Dict, Dict]:
    """
    Returns the edges, boundary edges and the logical effect of them from a
    stim detector error model.

    Parameters
    ----------
    dem
        Detector error model in stim format.
    return_coords
        If True, returns the a 'detector_coords'

    Returns
    -------
    edges
        List of edges present in dem
    boundary_edges
        List of boundary edges present in dem
    edge_logicals
        Dictionary with the edges (keys) and logical flips (values)
    detector_coords
        Dictionary containing the detectors (key) and their coordinates (values)
    """
    edges = {}
    boundary_edges = {}
    edge_logicals = {}
    detector_coords = {}
    shift_coords = 0

    for instr in dem.flattened():
        if instr.type == "error":
            prob = instr.args_copy()[0]
            defects = tuple(
                [d.val for d in instr.targets_copy() if d.is_relative_detector_id()]
            )
            logicals = [
                d.val for d in instr.targets_copy() if d.is_logical_observable_id()
            ]

            if len(defects) == 1:
                # stim allows for multiple defects having the same effect
                # why ?
                if defects not in boundary_edges:
                    boundary_edges[defects] = prob
                else:
                    boundary_edges[defects] = g_funct(boundary_edges[defects], prob)
            else:
                # stim allows for multiple defects having the same effect
                # why ?
                if defects not in edges:
                    edges[defects] = prob
                else:
                    edges[defects] = g_funct(edges[defects], prob)

            edge_logicals[defects] = logicals
        elif instr.type == "detector":
            detector = instr.targets_copy()[0].val
            coords = np.array(instr.args_copy())
            detector_coords[detector] = (coords + shift_coords).tolist()
        elif instr.type == "shift_detectors":
            shift_coords += np.array(instr.args_copy())

    if return_coords:
        return edges, boundary_edges, edge_logicals, detector_coords
    else:
        return edges, boundary_edges, edge_logicals


def edges_to_stim(
    edge_probs: Dict, edge_logicals: Dict, detector_coords: Dict = None
) -> stim.DetectorErrorModel:
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
    detector_coords
        Dictionary containing the detectors (key) and their coordinates (values)

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

    # add coordinates
    if detector_coords is not None:
        for detector, coords in detector_coords.items():
            dem.append(
                "detector",
                targets=[stim.DemTarget.relative_detector_id(detector)],
                parens_arguments=coords,
            )

    return dem


def stim_to_nx(dem: stim.DetectorErrorModel) -> nx.Graph:
    edges = {}  # detectors : {"prob": prob, "log": logicals_flipped}
    nodes = {}  # detector : coords

    bulk_edges, boundary_edges, edge_logicals, detector_coords = stim_to_edges(
        dem, return_coords=True
    )
    nodes = detector_coords
    for e, prob in bulk_edges.items():
        edges[e] = {"prob": prob, "log": edge_logicals[e]}
    for e, prob in boundary_edges.items():
        edges[e] = {"prob": prob, "log": edge_logicals[e]}

    # check for undefined nodes and
    # if all of them have coordinates
    nodes_in_edges = set(chain.from_iterable(edges.keys()))
    for node in nodes_in_edges:
        if (node not in nodes) or (len(nodes[node]) == 0):
            raise ValueError(
                f"All nodes must have coordinates, but {node} does not have them"
            )

    # create graph
    graph = nx.Graph()
    for node, coords in nodes.items():
        graph.add_node(node, coords=coords)
    graph.add_node("boundary")
    for nodes, attrs in edges.items():
        if len(nodes) == 1:
            graph.add_edge(nodes[0], "boundary", **attrs)
        else:
            graph.add_edge(*nodes, **attrs)

    return graph


def floor_boundary_edges(
    dem: Dict[Tuple, float], boundary_edges: Dict[Tuple, float]
) -> Dict[Tuple, float]:
    floored_dem = {}
    for edge, prob in dem.items():
        if len(edge) != 1:
            floored_dem[edge] = prob
            continue

        # ensure that all the boundary edges in 'dem'
        # specified in 'boundary_edges'
        if prob < boundary_edges[edge]:
            floored_dem[edge] = boundary_edges[edge]
        else:
            floored_dem[edge] = prob

    return floored_dem


def clip_negative_edges(dem: Dict[Tuple, float]) -> Dict[Tuple, float]:
    return {e: p if p > 0 else 0 for e, p in dem.items()}


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
