from time import perf_counter
import os

import numpy as np
import matplotlib.pyplot as plt

from imitation_learning_deep_dive.rollouts import policy_rollout


def get_success_rate(task, policy, num_instances=10, seed_offset=0, filename_prefix="oracle"):
    """Roll out `policy` on `num_instances` seeded instances of `task` and return the success rate."""
    t0 = perf_counter()
    successes = []
    for i in range(num_instances):
        print(f"Running instance {i} of {num_instances} for calculating success rate...")
        successes.append(policy_rollout(task, seed=seed_offset + i, policy=policy)[0])
        print(f'{"SUCCESS!" if successes[-1] else "FAILURE..."}')
    t1 = perf_counter()
    success_rate_percentage = np.mean(successes)*100.0

    os.makedirs("out", exist_ok=True)
    fig, ax = plt.subplots()
    ax.bar([task], [success_rate_percentage])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Success rate")
    ax.set_title(f"{filename_prefix} success rate over ({num_instances} episodes)")
    fig.tight_layout()
    fig.savefig(f"out/{filename_prefix}_success_rate.png")
    plt.close(fig)

    return success_rate_percentage, num_instances, t1 - t0
