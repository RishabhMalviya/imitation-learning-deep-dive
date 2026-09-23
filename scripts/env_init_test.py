import time

import imageio.v2 as imageio
import numpy as np

from imitation_learning_deep_dive.envs import TASK, make_env, oracle_for



def run(task, seed, render=False, every=2, noisy_states=False):
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

    env.close()

    return solved, t, frames



if __name__ == "__main__":
    # Oracle Success Rate Test
    num_instances = 10

    print(f"\nRolling out oracle on {num_instances} instances of {TASK} without rendering ...")

    t0 = time.time()
    successes = [run(TASK, seed=s)[0] for s in range(num_instances)]
    t1 = time.time()

    print(f"  Success rate: {np.mean(successes):.0%} over {num_instances} eps in {t1 - t0:.1f}s")


    # Oracle Rollout for Video
    print(f"\nRolling out oracle on 1 instance of {TASK} for rendering...")

    t0 = time.time()
    solved, steps, frames = run(TASK, seed=0, render=True)
    t1 = time.time()
    print(f"  solved={solved} steps={steps} frames={len(frames)}")
    print(f"  Render speed: {len(frames) / (t1 - t0):.0f} fps (total time {t1 - t0:.1f}s)")

    out = f"out/oracle_{TASK.split('-v3')[0].replace('-', '_')}.mp4"
    print(f"\n  Saving rendered video to {out}")
    imageio.mimsave(out, frames, fps=15, macro_block_size=1)
    print(f"  Saved rendered video to {out}")

