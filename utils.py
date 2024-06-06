import os

import numpy as np

from copy import deepcopy
from tqdm import tqdm


def constant_lr(timestep: float) -> float:
    return 1e-4


def decreasing_lr(timestep: float) -> float:
    start = 3e-4
    end = 5e-5
    print((start - end / 307000) * timestep + start)

    return (start - end / 307000) * timestep + start

def stack_obs_frames(frames, history_length=4):
    trajectory = []
    frame_buffer = [np.zeros(shape=frames[0].shape) for i in range(history_length)]
    for i in range(frames.shape[0]):
        frame_buffer.pop(0)
        frame_buffer.insert(history_length - 1, frames[i])
        trajectory.insert(i, deepcopy(frame_buffer))

    return np.array(trajectory)


def create_traj_metadata(trajectories_path):
    counts = np.zeros(shape=(3,))
    for i in tqdm(os.listdir(trajectories_path), desc="Creating trajectories metadata file"):
        with open(f"{trajectories_path}/trajectories_lengths.csv", "a+") as f:
            data = np.load(f"{trajectories_path}/{i}", allow_pickle=True)
            for x in data["actions"]:
                counts[x] += 1
            f.write(f"{i.split(sep='.')[0]},{int(data['actions'].shape[0])}\n")
            f.close()

    print(f"actions: {counts}")
