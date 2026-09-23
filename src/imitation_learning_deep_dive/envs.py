"""
MetaWorld V3 environment + scripted-oracle plumbing.

Here are some tasks that were considered. The oracles for these tasks run with 100% success rate:
============================= ================= =============================================
Task                          Bottleneck        Description
============================= ================= =============================================
``reach-v3``                  None              Pure servoing to a point; BC control group.
``button-press-topdown-v3``   Mild              Align over the button, then press down.
``drawer-open-v3``            Contact           Engage the handle, then pull hard (p=50).
``door-open-v3``              Contact           Hook the handle, then swing.
``pick-place-v3``             Grasp             Missing the grasp shifts later states off-distribution.
============================= ================= =============================================
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")  # must precede the mujoco import (change to "egl" if you have a working EGL backend)

import metaworld  # noqa: E402
import metaworld.policies as mw_policies  # noqa: E402

import gymnasium as gym


TASK = 'pick-place-v3'
assert TASK in metaworld.ALL_V3_ENVIRONMENTS, f"{TASK!r} is not a MetaWorld V3 task" # pyright: ignore[reportPrivateImportUsage]

"""
Observation layout in Metaworld environments (https://metaworld.farama.org/benchmark/state_space/):
  [ 0: 3] the XYZ coordinates of the end-effector
  [ 3: 4] a scalar value that represents how open/closed the gripper is
  [ 4:11] object 1 pos (3) + quat (4)
  [11:18] object 2 pos (3) + quat (4)   (zeros for single-object tasks)
  [18:36] the previous frame's [0:18]
  [36:39] goal xyz
"""
OBS_DIM = 39
HAND_POS = slice(0, 3)
GRIPPER = slice(3, 4)
OBJ1_POS = slice(4, 7)
PREV_OBS = slice(18, 36)
GOAL_POS = slice(36, 39)

ACT_DIM = 4  # [dx, dy, dz, grab_effort], each clipped to [-1, 1]


def oracle_for(task: str):
    """
    Return the oracle class for a task
    
    For example:
        - 'pick-place-v3', would produce `metaworld.policies.SawyerPickPlaceV3Policy` (https://github.com/Farama-Foundation/Metaworld/blob/main/metaworld/policies/sawyer_pick_place_v3_policy.py)
        - 'drawer-open-v3' would produce `metaworld.policies.SawyerDrawerOpenV3Policy` (https://github.com/Farama-Foundation/Metaworld/blob/main/metaworld/policies/sawyer_drawer_open_v3_policy.py).
    """
    stem = task.replace("-v3", "")
    cls_name = "Sawyer" + "".join(p.capitalize() for p in stem.split("-")) + "V3Policy"

    try:
        return getattr(mw_policies, cls_name)()
    except AttributeError as e:
        raise KeyError(f"no scripted oracle {cls_name!r} for task {task!r}") from e


def make_env(task: str, seed: int = 0, render: bool = False, camera: str = 'corner'):
    """
    Build a single goal-observable MetaWorld env with diverse, seeded resets.
    """
    if task not in metaworld.ALL_V3_ENVIRONMENTS: # pyright: ignore[reportPrivateImportUsage]
        raise KeyError(f"{task!r} is not a MetaWorld V3 task")

    env = gym.make(
        'Meta-World/MT1', env_name=task,
        seed=seed,
        render_mode='rgb_array' if render else None,
        camera_name=camera if render else None,
    )

    return env

