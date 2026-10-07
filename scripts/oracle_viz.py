from time import perf_counter
import os

import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from imitation_learning_deep_dive.envs import TASK, GRIPPER_POS, PUCK_POS, GRIPPER_CLOSE_AMOUNT
from imitation_learning_deep_dive.rollouts import oracle_rollout



def render_video(task, seed=0):
    print(f"\nRolling out oracle on 1 instance of {task} for rendering...")

    t0 = perf_counter()
    solved, steps, frames, _ = oracle_rollout(task, seed=0, render=True)
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


def visualize_oracle_behaviour(task, seed=0):
    _, _, _, trajectory = oracle_rollout(task, seed=seed, return_trajectory=True)

    observations = np.asarray(trajectory.get("obs", trajectory.get("obs")))
    actions = np.asarray(trajectory.get("act", trajectory.get("actions")))
    timesteps = np.asarray(range(observations.shape[0]))

    # Gripper-Puck Distance
    gripper_pos = observations[:, GRIPPER_POS]
    puck_pos = observations[:, PUCK_POS]
    gripper_puck_distance = np.linalg.norm(gripper_pos - puck_pos, axis=1)

    # Gripper z-position
    gripper_z = observations[:, GRIPPER_POS.stop]

    # Gripper Close Amount and Signal
    gripper_close_amount = observations[:, GRIPPER_CLOSE_AMOUNT]
    gripper_close_signal = actions[:, -1]

    # Plot and Save Plots against Timesteps   
    fig, ax = plt.subplots()

    ax.plot(timesteps, gripper_puck_distance, label="Gripper-Puck Distance")
    ax.plot(timesteps, gripper_z, label="Gripper Z-Position")
    ax.plot(timesteps, gripper_close_amount, label="Gripper Close Amount")
    ax.plot(timesteps, gripper_close_signal, label="Gripper Close Signal")

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()

    os.makedirs("out", exist_ok=True)
    task_name = task.split("-v3")[0].replace("-", "_")
    fig.savefig(f"out/oracle_{task_name}_behaviour_vs_timesteps.png")

    plt.close(fig)

    # Plot and Save Plots against Gripper-Puck Distance
    fig, ax = plt.subplots()

    ax.plot(gripper_puck_distance, gripper_close_signal, label="Gripper Close Signal versus Gripper-Puck Distance")

    ax.set_xlabel("Gripper-Puck Distance")
    ax.set_ylabel("Gripper Close Signal (Action)")
    ax.legend()
    fig.tight_layout()

    os.makedirs("out", exist_ok=True)
    task_name = task.split("-v3")[0].replace("-", "_")
    fig.savefig(f"out/oracle_{task_name}_behaviour__action_versus_obs.png")

    plt.close(fig)



if __name__ == "__main__":
    # Oracle Success Rate Test
    success_rate, num_instances, elapsed_time = get_success_rate(TASK)
    print(f"  Success rate: {success_rate:.0%} over {num_instances} eps in {elapsed_time:.1f}s")

    # Oracle Rollout for Video
    render_video(TASK, seed=0)

    # Visualize Oracle Behaviour
    visualize_oracle_behaviour(TASK, seed=0)
