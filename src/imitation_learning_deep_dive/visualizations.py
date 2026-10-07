import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

from imitation_learning_deep_dive.envs import GRIPPER_POS, PUCK_POS, GRIPPER_CLOSE_AMOUNT


def _visualizations_over_time(timesteps, gripper_puck_distance, gripper_z, gripper_close_amount, gripper_close_signal, 
                              task: str, filename_prefix: str):
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
    fig.savefig(f"out/{filename_prefix}_{task_name}_behaviour_vs_timesteps.png")

    plt.close(fig)


def _actions_versus_obs(gripper_puck_distance, gripper_close_signal, task: str, filename_prefix: str):
    fig, ax = plt.subplots()

    ax.plot(gripper_puck_distance, gripper_close_signal, label="Gripper Close Signal versus Gripper-Puck Distance")

    ax.set_xlabel("Gripper-Puck Distance")
    ax.set_ylabel("Gripper Close Signal (Action)")
    ax.legend()
    fig.tight_layout()

    os.makedirs("out", exist_ok=True)
    task_name = task.split("-v3")[0].replace("-", "_")
    fig.savefig(f"out/{filename_prefix}_{task_name}_behaviour__action_versus_obs.png")

    plt.close(fig)


def _position_gif(gripper_pos, puck_pos, values, value_name: str, task: str, filename_prefix: str):
    """Save an animation of gripper positions relative to the puck, grouped by value."""
    relative_positions = np.asarray(gripper_pos)[:, :2] - np.asarray(puck_pos)[:, :2]
    values = np.asarray(values).reshape(-1)
    value_levels = [i/10.0 for i in range(0,11)]
    bin_width = 0.1

    finite_positions = relative_positions[np.all(np.isfinite(relative_positions), axis=1)]
    if finite_positions.size:
        limits = max(float(np.max(np.abs(finite_positions))), 0.05) * 1.05
    else:
        limits = 1.0

    fig, ax = plt.subplots()
    points = ax.scatter([], [], c=[], cmap="coolwarm", vmin=0.0, vmax=1.0)
    fig.colorbar(points, ax=ax, label=value_name)
    ax.set_xlim(-limits, limits)
    ax.set_ylim(-limits, limits)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Gripper X relative to puck")
    ax.set_ylabel("Gripper Y relative to puck")

    def update(g):        
        mask = (values >= g) & (values <= g + bin_width if g == value_levels[-1] else values < g + bin_width)
        frame_positions = relative_positions[mask]
        frame_positions = frame_positions[np.all(np.isfinite(frame_positions), axis=1)]
        points.set_offsets(frame_positions.reshape(-1, 2))
        frame_values = values[mask]
        frame_values = frame_values[np.all(np.isfinite(relative_positions[mask]), axis=1)]
        points.set_array(frame_values)
        ax.set_title(f"{value_name} = {g:.2f}")
        return (points,)

    animation = FuncAnimation(fig, update, frames=value_levels, interval=1500, blit=False, repeat=True)
    os.makedirs("out", exist_ok=True)
    task_name = task.split("-v3")[0].replace("-", "_")
    value_slug = value_name.lower().replace(" ", "_")
    animation.save(
        f"out/{filename_prefix}_{task_name}_{value_slug}_positions.gif",
        writer=PillowWriter(fps=2),
    )
    plt.close(fig)


def visualize_behaviour(trajectory, task: str, filename_prefix: str):
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

    _visualizations_over_time(timesteps, gripper_puck_distance, gripper_z, gripper_close_amount, gripper_close_signal, task, filename_prefix)
    _actions_versus_obs(gripper_puck_distance, gripper_close_signal, task, filename_prefix)
    _position_gif(gripper_pos, puck_pos, gripper_close_amount, "Gripper Close Amount", task, filename_prefix)
    _position_gif(gripper_pos, puck_pos, gripper_close_signal, "Gripper Close Signal", task, filename_prefix)
