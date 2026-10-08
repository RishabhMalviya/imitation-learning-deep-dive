import os

import torch
import matplotlib.pyplot as plt

from imitation_learning_deep_dive.envs import TASK
from imitation_learning_deep_dive.bc import BCPolicy, collect_demos, train_bc
from imitation_learning_deep_dive.evaluation import get_success_rate


NUM_TRAIN_DEMOS = 50
NUM_VAL_DEMOS = 10
NUM_EVAL_EPISODES = 10
EVAL_SEED_OFFSET = 10_000  # keep eval instances disjoint from demo seeds
EPOCHS = 100
BATCH_SIZE = 256
LR = 1e-3
HIDDEN_DIM = 8
NUM_LAYERS = 2


def save_loss_curves(train_losses, val_losses, path):
    fig, ax = plt.subplots()
    ax.plot(train_losses, label="Train")
    ax.plot(val_losses, label="Val")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("BC loss curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def load_or_collect_demos(task, num_demos, seed_offset, path):
    if os.path.exists(path):
        print(f"Loading demos from {path}")
        return torch.load(path, map_location="cpu", weights_only=False)

    demos = collect_demos(task, num_demos, seed_offset=seed_offset)
    torch.save(demos, path)
    print(f"Saved demos to {path}")
    return demos


if __name__ == "__main__":
    os.makedirs("out", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    task_name = TASK.split("-v3")[0].replace("-", "_")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Collecting {NUM_TRAIN_DEMOS} train + {NUM_VAL_DEMOS} val oracle demos on {TASK}...")
    train_data = load_or_collect_demos(
        TASK, NUM_TRAIN_DEMOS, seed_offset=0, path=f"data/bc_{task_name}_train_demos.pt"
    )
    val_data = load_or_collect_demos(
        TASK, NUM_VAL_DEMOS, seed_offset=NUM_TRAIN_DEMOS, path=f"data/bc_{task_name}_val_demos.pt"
    )
    print(f"  {len(train_data[0])} train / {len(val_data[0])} val transitions")

    policy = BCPolicy(
        obs_dim=train_data[0].shape[1],  # if we only keep the relevant parts of the observation ([:11] + [36:])
        hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS
    )
    policy.set_normalization(train_data[0])

    print(f"Training BC on {device}...")
    train_losses, val_losses = train_bc(policy, train_data, val_data, epochs=EPOCHS, batch_size=BATCH_SIZE,
                                        lr=LR, device=device)
    save_loss_curves(train_losses, val_losses, f"out/bc_{task_name}_loss_curves.png")

    policy.to("cpu").eval()
    checkpoint_path = f"out/bc_{task_name}.pt"
    torch.save({
        "state_dict": policy.state_dict(),
        "config": {"task": TASK, "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS},
    }, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

    print(f"Evaluating BC policy online over {NUM_EVAL_EPISODES} episodes...")
    success_rate, num_instances, elapsed_time = get_success_rate(
        TASK, policy.act, num_instances=NUM_EVAL_EPISODES, seed_offset=EVAL_SEED_OFFSET, filename_prefix=f"bc_{task_name}")
    print(f"  Success rate: {success_rate:.1f}% over {num_instances} eps in {elapsed_time:.1f}s")
