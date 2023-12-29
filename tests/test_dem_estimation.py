import pytest
import stim

from dem_estimation import get_edge_probabilities, stim_to_edges


def test_get_edge_probabilities():
    circuit = stim.Circuit.generated(
        code_task="repetition_code:memory",
        rounds=3,
        distance=3,
        after_clifford_depolarization=0.1,
        before_measure_flip_probability=0.1,
        after_reset_flip_probability=0.1,
    )
    sampler = circuit.compile_detector_sampler()
    defects = sampler.sample(shots=500_000)

    dem = circuit.detector_error_model()
    edges, boundary_edges, _ = stim_to_edges(dem)

    edge_probs = get_edge_probabilities(
        defects, edges=edges, boundary_edges=boundary_edges
    )

    for instr in dem.flattened():
        if instr.type != "error":
            continue
        defects = tuple(
            [d.val for d in instr.targets_copy() if d.is_relative_detector_id()]
        )
        prob_stim = instr.args_copy()[0]
        prob_estimated = edge_probs[defects]

        assert pytest.approx(prob_stim, abs=2e-2) == prob_estimated

    return
