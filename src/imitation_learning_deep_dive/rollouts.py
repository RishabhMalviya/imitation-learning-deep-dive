import numpy as np

from imitation_learning_deep_dive.envs import make_env, oracle_for


def oracle_rollout(task, seed, return_trajectory=False, render=False, noisy_states=False):
    return policy_rollout(task, seed, oracle_for(task).get_action, return_trajectory, render, noisy_states)


def policy_rollout(task, seed, policy, return_trajectory=False, render=False, noisy_states=False):
    """Roll out `policy` (a callable mapping an observation to an action) for one episode."""
    env = make_env(task, seed=seed, render=render)
    rng = np.random.default_rng(seed)

    obs, _ = env.reset(seed=seed)
    obs = obs.copy()
    if noisy_states:
        obs[:7] += rng.normal(0, 0.01, size=7)
    
    trajectory = {}
    if return_trajectory:
        trajectory = {
            "obs": [],
            "actions": [],
            "next_obs": [],
            "rewards": [],
            "terminated": [],
            "truncated": [],
            "infos": [],
        }

    t = 0
    terminated, truncated = False, False
    frames, solved = [], False
    while not (terminated or truncated):
        t += 1

        action = policy(obs)
        action[:3] = np.clip(action[:3], -1.0, 1.0)  # Target Gripper Position
        action[-1] = np.clip(action[-1], 0.0, 1.0)  # Gripper Close Signal
        next_obs, reward, terminated, truncated, info = env.step(action)

        if noisy_states:
            next_obs = next_obs.copy()
            next_obs[:7] += rng.normal(0, 0.01, size=7)

        if return_trajectory:
            trajectory["obs"].append(obs.copy()) 
            trajectory["actions"].append(np.array(action, copy=True))
            trajectory["rewards"].append(reward)
            trajectory["next_obs"].append(next_obs.copy())
            trajectory["terminated"].append(terminated)
            trajectory["truncated"].append(truncated)
            trajectory["infos"].append(info.copy())

        obs = next_obs

        if render:
            frames.append(np.rot90(np.rot90(env.render())))

        if int(info.get("success", 0)) == 1:
            solved = True
            break

    env.close()

    return solved, t, frames, trajectory
