import stim
import numpy as np
import matplotlib.pyplot as plt

from dem_estimation import stim_to_edges, get_edge_probabilities


circuit = stim.Circuit.generated(
    code_task="repetition_code:memory",
    rounds=10,
    distance=5,
    after_clifford_depolarization=0.1,
    before_measure_flip_probability=0.1,
    after_reset_flip_probability=0.1,
)
sampler = circuit.compile_detector_sampler()
defects = sampler.sample(shots=1_000_000)

dem = circuit.detector_error_model()
edges, boundary_edges, _ = stim_to_edges(dem)

edge_probs = get_edge_probabilities(defects, edges=edges, boundary_edges=boundary_edges)

difference = {"all": [], "boundary": [], "edges": []}
for instr in dem.flattened():
    if instr.type != "error":
        continue
    defects = tuple(
        [d.val for d in instr.targets_copy() if d.is_relative_detector_id()]
    )
    prob_stim = instr.args_copy()[0]
    prob_estimated = edge_probs[defects]

    error = prob_stim - prob_estimated

    difference["all"].append(error)
    if len(defects) == 1:
        difference["boundary"].append(error)
    else:
        difference["edges"].append(error)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))

bins = np.linspace(min(difference["all"]), max(difference["all"]), 50)

axes[0].hist(difference["all"], bins=bins)
axes[1].hist(difference["boundary"], bins=bins)
axes[2].hist(difference["edges"], bins=bins)

axes[0].set_xlabel("p - p_estimated (all)")
axes[1].set_xlabel("p - p_estimated (boundary)")
axes[2].set_xlabel("p - p_estimated (non-boundary)")
for i in range(3):
    axes[i].set_ylabel("# edges")

fig.tight_layout()
fig.savefig("accuracy.pdf", format="pdf")
plt.show()
