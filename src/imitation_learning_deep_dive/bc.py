import numpy as np
import torch
import torch.nn as nn

from imitation_learning_deep_dive.envs import OBS_DIM, ACT_DIM, filter_observation
from imitation_learning_deep_dive.rollouts import oracle_rollout


class BCPolicy(nn.Module):
    """MLP mapping (normalized) observations to actions."""
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden_dim=256, num_layers=3):
        super().__init__()
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))

        layers, in_dim = [], obs_dim
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, act_dim))
        self.net = nn.Sequential(*layers)

    def set_normalization(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32)
        self.obs_mean.copy_(obs.mean(dim=0))
        self.obs_std.copy_(obs.std(dim=0).clamp_min(1e-6))

    def forward(self, obs):
        return self.net((obs - self.obs_mean) / self.obs_std)

    @torch.no_grad()
    def act(self, obs: np.ndarray, task: str = "pick-place-v3") -> np.ndarray:
        """Numpy-in, numpy-out action for use as a rollout policy."""
        obs = filter_observation(obs, task=task)
        device = self.obs_mean.device
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        return self(obs_tensor).squeeze(0).cpu().numpy().astype(np.float64)


def collect_demos(task, num_demos, seed_offset=0):
    """Return stacked (obs, actions) from `num_demos` successful oracle rollouts."""
    obs, actions = [], []
    for i in range(num_demos):
        solved, _, _, trajectory = oracle_rollout(task, seed=seed_offset + i, return_trajectory=True)
        if not solved:
            print(f"  Oracle failed on seed {seed_offset + i}; skipping demo")
            continue
        obs.extend(trajectory["obs"])
        actions.extend(trajectory["actions"])

    obs = filter_observation(np.asarray(obs, dtype=np.float32), task)

    return np.asarray(obs, dtype=np.float32), np.asarray(actions, dtype=np.float32)


def train_bc(policy, train_data, val_data, epochs=100, batch_size=256, lr=1e-3, device="cpu"):
    """Train `policy` with MSE on (obs, action) pairs; return per-epoch train and val losses."""
    policy.to(device)
    train_obs, train_act = (torch.as_tensor(x, device=device) for x in train_data)
    val_obs, val_act = (torch.as_tensor(x, device=device) for x in val_data)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_losses, val_losses = [], []
    for epoch in range(epochs):
        policy.train()
        perm = torch.randperm(len(train_obs), device=device)
        epoch_loss = 0.0
        for start in range(0, len(train_obs), batch_size):
            idx = perm[start:start + batch_size]
            loss = loss_fn(policy(train_obs[idx]), train_act[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        train_losses.append(epoch_loss / len(train_obs))

        policy.eval()
        with torch.no_grad():
            val_losses.append(loss_fn(policy(val_obs), val_act).item())

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:4d}  train {train_losses[-1]:.5f}  val {val_losses[-1]:.5f}")

    return train_losses, val_losses
