from typing import Dict

import stim


def stim_to_edges(dem: stim.DetectorErrorModel) -> Dict:
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
    dem = stim.DetectorErrorModel()

    for edge, prob in edge_probs.items():
        logicals = [
            stim.DemTarget.logical_observable_id(l) for l in edge_logicals[edge]
        ]
        defects = [stim.DemTarget.relative_detector_id(d) for d in edge]
        dem.append("error", prob, defects + logicals)

    return dem
