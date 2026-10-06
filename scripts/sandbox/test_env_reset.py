import tempfile
from pathlib import Path

import mujoco
import numpy as np
import pyvista as pv
from PIL import Image

from typing import List, Dict, Optional, Tuple

from imitation_learning_deep_dive.envs import TASK, make_env


CAMERA = 'corner'
OUT_PATH = Path(__file__).resolve().parents[2] / 'out' / 'starting_points_viz.png'


def get_camera(env, camera_name: str) -> Dict:
    """
    World-frame pose and vertical FOV of a MuJoCo camera (camera looks along its -z, with +y up).
    """
    model, data = env.unwrapped.model, env.unwrapped.data
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)  # pyright: ignore[reportAttributeAccessIssue]
    return {
        "pos": data.cam_xpos[cam_id].copy(),
        "rot": data.cam_xmat[cam_id].reshape(3, 3).copy(),
        "fovy": float(model.cam_fovy[cam_id]),
    }


def visualize_3d(
    positions: Dict[str, List[Tuple[int, int, int]]],
    background: Optional[np.ndarray] = None,
    camera: Optional[Dict] = None,
    save_path: Optional[Path] = None,
):
    """
    If `background` (an RGB render) and the `camera` it was rendered from are given, the pyvista camera is
    matched to the MuJoCo one so the scatter lines up with the render. The alignment only holds for the
    initial view — the background stays fixed if you rotate the scene.

    If `save_path` is given, a screenshot of the initial view is saved there.
    """
    if background is not None:
        height, width = background.shape[:2]
        plotter = pv.Plotter(window_size=(width, height))
    else:
        plotter = pv.Plotter()
    colors = {
        "puck_poses": "tomato",
        "goal_poses": "dodgerblue",
        "gripper_poses": "mediumseagreen",
    }

    for name, poses in positions.items():
        if len(poses) == 0:
            continue

        points = np.asarray(poses, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"{name} must contain 3D positions")

        plotter.add_points(
            points,
            color=colors.get(name, "white"),
            point_size=10,
            render_points_as_spheres=True,
            label=name,
        )

    plotter.add_axes()
    if any(len(poses) for poses in positions.values()):
        plotter.add_legend(bcolor="white", border=True, background_opacity=0.85)

    if background is not None and camera is not None:
        # The 'corner' camera is rolled upside down, so flip both the image and the view-up to keep the
        # scene upright (rotating the image 180 deg == negating the camera's x and y axes)
        rot = camera["rot"]
        plotter.camera.position = camera["pos"]
        plotter.camera.focal_point = camera["pos"] - rot[:, 2]
        plotter.camera.up = -rot[:, 1]
        plotter.camera.view_angle = camera["fovy"]
        plotter.camera.clipping_range = (0.01, 10.0)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            Image.fromarray(np.ascontiguousarray(background[::-1, ::-1])).save(f.name)
        plotter.add_background_image(f.name)
    else:
        plotter.show_grid(xlabel="X", ylabel="Y", zlabel="Z")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.show(screenshot=save_path or False)


if __name__ == "__main__":
    N = 1_000
    goal_poses    = list()
    puck_poses    = list()
    gripper_poses = list()

    env = make_env(TASK, seed=42, render=True, camera=CAMERA)
    for n in range(N):
        obs, _ = env.reset()

        gripper_pos = obs[0:3]
        puck_pos = obs[4:7]
        goal_pos = obs[-3:]

        gripper_poses.append(tuple(gripper_pos))
        puck_poses.append(tuple(puck_pos))
        goal_poses.append(tuple(goal_pos))

    frame = env.render()  # scene from the last reset
    camera = get_camera(env, CAMERA)
    env.close()

    print(f"\nUnique gripper poses: {len(list(set(gripper_poses)))}")

    print(f'SANITY CHECK: Num puck_poses before list-set: {len(puck_poses)}')
    print(f"Unique puck poses: {len(list(set(puck_poses)))}")
    print(f'SANITY CHECK: Num puck_poses after list-set: {len(puck_poses)}')

    print(f"Unique goal poses: {len(list(set(goal_poses)))}")

    visualize_3d({
        'puck_poses': puck_poses,
        'goal_poses': goal_poses,
        'gripper_poses': gripper_poses
    }, background=frame, camera=camera, save_path=OUT_PATH)
    print(f"Saved visualization to {OUT_PATH}")
