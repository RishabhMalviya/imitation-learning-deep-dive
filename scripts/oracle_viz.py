from time import perf_counter
import os

import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from imitation_learning_deep_dive.envs import TASK
from imitation_learning_deep_dive.rollouts import oracle_rollout
from imitation_learning_deep_dive.visualizations import visualize_behaviour



def render_video(task, seed=0):
    print(f"\nRolling out oracle on 1 instance of {task} for rendering...")

    t0 = perf_counter()
    solved, steps, frames, _ = oracle_rollout(task, seed=seed, render=True)
    t1 = perf_counter()
    
    print(f"  solved={solved} steps={steps} frames={len(frames)}")
    print(f"  Render speed: {len(frames) / (t1 - t0):.0f} fps (total time {t1 - t0:.1f}s)")

    out = f"out/oracle_{task.split('-v3')[0].replace('-', '_')}.mp4"
    print(f"\n  Saving rendered video to {out}")
    imageio.mimsave(out, frames, fps=15, macro_block_size=1)
    print(f"  Saved rendered video to {out}")


def get_success_rate(task, num_instances=10):
    t0 = perf_counter()
    successes = [False for _ in range(num_instances)]
    for i in range(num_instances):
        print(f"Running instance {i} of {num_instances} for calculating success rate...")
        oracle_rollout(task, seed=i, return_trajectory=False, render=False)
    successes = [oracle_rollout(task, seed=s)[0] for s in range(num_instances)]
    t1 = perf_counter()
    success_rate_percentage = np.mean(successes)*100.0

    os.makedirs("out", exist_ok=True)
    fig, ax = plt.subplots()
    ax.bar([task], [success_rate_percentage])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Success rate")
    ax.set_title(f"Oracle success rate over ({num_instances} episodes)")
    fig.tight_layout()
    fig.savefig("out/oracle_success_rate.png")
    plt.close(fig)

    return success_rate_percentage, num_instances, t1 - t0


if __name__ == "__main__":
    # # Oracle Success Rate Test
    # success_rate, num_instances, elapsed_time = get_success_rate(TASK)
    # print(f"  Success rate: {success_rate:.0%} over {num_instances} eps in {elapsed_time:.1f}s")

    # # Oracle Rollout for Video
    # render_video(TASK, seed=0)

    # Visualize Oracle Behaviour
    visualize_behaviour(oracle_rollout(TASK, seed=0, return_trajectory=True)[-1], TASK, "oracle")
