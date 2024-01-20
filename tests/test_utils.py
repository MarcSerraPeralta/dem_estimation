import stim

from dem_estimation.utils import stim_to_nx


def test_dem_to_nx():
    circuit = stim.Circuit.generated(
        code_task="repetition_code:memory",
        rounds=5,
        distance=3,
        after_clifford_depolarization=0.01,
        before_measure_flip_probability=0.05,
        after_reset_flip_probability=0.01,
    )
    dem = circuit.detector_error_model()
    graph = stim_to_nx(dem)
    print(graph)
    return
