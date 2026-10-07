import time

import imageio.v2 as imageio
import numpy as np

from imitation_learning_deep_dive.envs import TASK, make_env, oracle_for



def run(task, seed, gripper_action_override: float, render=False, every=2, noisy_states=False):
    env, oracle = make_env(task, seed=seed, render=render), oracle_for(task)
    rng = np.random.default_rng(seed)

    obs, _ = env.reset(seed=seed)
    obs = obs.copy()
    obs[:7] += rng.normal(0, 0.01, size=7)

    t = 0
    truncate, terminate = False, False
    frames, solved = [], False
    while not (truncate or terminate):
        t += 1
        
        action = np.clip(oracle.get_action(obs), -1, 1)
        # Override to see effect of -1.0 on gripper
        action[-1] = gripper_action_override
        obs, _, truncate, terminate, info = env.step(action)

        # Inject state noise
        if noisy_states:
            obs = obs.copy()
            obs[:7] += rng.normal(0, 0.01, size=7)

        # Frame render
        if render and t % every == 0:
            frames.append(np.rot90(np.rot90(env.render())))

        # Check for success
        if int(info.get("success", 0)) == 1:
            solved = True
            break

        if t > 1000:
            break

    env.close()

    return solved, t, frames


if __name__ == "__main__":
    # Oracle Rollout for Video
    for gripper_action_override in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        print(f"\nRolling out oracle with gripper action override set to {gripper_action_override}...")

        t0 = time.time()
        solved, steps, frames = run(TASK, seed=0, gripper_action_override=gripper_action_override, render=True)
        t1 = time.time()

        out = f"out/oracle_{TASK.split('-v3')[0].replace('-', '_')}_gripper_{gripper_action_override}.mp4"
        print(f"\n  Saving rendered video to {out}")
        imageio.mimsave(out, frames, fps=15, macro_block_size=1)
        print(f"  Saved rendered video to {out}")

