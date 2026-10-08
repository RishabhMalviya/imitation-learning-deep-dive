from time import perf_counter

import imageio.v2 as imageio

from imitation_learning_deep_dive.envs import TASK, oracle_for
from imitation_learning_deep_dive.evaluation import get_success_rate
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


if __name__ == "__main__":
    # Oracle Success Rate Test
    success_rate, num_instances, elapsed_time = get_success_rate(TASK, oracle_for(TASK).get_action)
    print(f"  Success rate: {success_rate:.0%} over {num_instances} eps in {elapsed_time:.1f}s")

    # Oracle Rollout for Video
    render_video(TASK, seed=0)

    # Visualize Oracle Behaviour
    visualize_behaviour(oracle_rollout(TASK, seed=0, return_trajectory=True)[-1], TASK, "oracle")
