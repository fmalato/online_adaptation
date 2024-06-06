import os
import argparse
import pyglet
#pyglet.options["headless"] = True

import numpy as np
import gymnasium as gym
import miniworld
from tqdm import tqdm

import torch

from torch.utils.data import Dataset, DataLoader
from stable_baselines3.common.policies import NatureCNN, ActorCriticPolicy
from pathlib import Path

from imitation.algorithms import bc
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import save_stable_model, load_policy
from imitation.util.util import make_vec_env

from utils import stack_obs_frames, create_traj_metadata
from resnet_encoder import CausalIDMEncoder
from wrappers import TorchObsGymWrapper, decreasing_lr, constant_lr

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class MiniWorldDataset(Dataset):
    def __init__(self, data_path, history_length=4, use_all=False, max_steps=1000):
        super().__init__()
        self.data_path = data_path
        self.history_length = history_length
        # Create metadata file if it doesn't exist
        if "trajectories_lengths.csv" not in os.listdir(data_path):
            create_traj_metadata(data_path)
        self.traj_metadata = np.genfromtxt(os.path.join(data_path, "trajectories_lengths.csv"), delimiter=",")
        # Remove fails if requested (valid only for examples where success/failure can be determined)
        if not use_all:
            failures = np.where(self.traj_metadata[:, 1] == max_steps)
            self.traj_metadata = np.delete(self.traj_metadata, failures, axis=0)
        # Compute offsets
        self._compute_offsets()
        # Avoids repetitive computation later on
        self._num_samples = int(np.sum(self.traj_metadata[:, 1]))

    def __len__(self):
        return self._num_samples

    def num_traj(self):
        return int(self.traj_metadata.shape[0])

    def __getitem__(self, item):
        # Find belonging bin
        start, end = self.traj_metadata[:, 2], self.traj_metadata[:, 3]
        mask = (item >= start) & (item <= end)
        traj_idx = np.argmax(mask)
        # Load data
        traj = np.load(os.path.join(self.data_path, f"{int(self.traj_metadata[traj_idx, 0])}.npz"))
        # Build observation
        offset = item - int(start[traj_idx])
        if offset < self.history_length - 1:
            zeros_stack = np.zeros(shape=(self.history_length - 1 - offset, *traj["observations"][0, :].shape))
            obs = traj["observations"][0: offset + 1, :]
            obs = np.concatenate([zeros_stack, obs], axis=0)
        else:
            obs = traj["observations"][offset - self.history_length + 1: offset + 1, :]
        act = traj["actions"][offset]

        return {"obs": torch.FloatTensor(obs.astype(np.float32) / 255).permute((0, 3, 1, 2)), "act": act}

    def _compute_offsets(self):
        total_count = 0
        offsets = np.zeros(shape=self.traj_metadata.shape)
        for i, v in enumerate(list(self.traj_metadata[:, 1])):
            offsets[i, 0] = total_count
            offsets[i, 1] = total_count + v - 1
            total_count += v

        self.traj_metadata = np.append(self.traj_metadata, offsets, axis=1)


def evaluate_agent(agent_type, bc_agent, env, num_games, history_length, trajectories_path):
    os.makedirs(f"training/{agent_type}/", exist_ok=True)
    agent = bc_agent.policy
    suff = ""

    num_timesteps = []
    success = []
    ep_durations = []
    ep_rewards = []
    progress_bar = tqdm(range(num_games))
    for i in progress_bar:
        observation, info = env.reset()
        if history_length <= 1:
            observation = observation[0]
        terminated = False
        truncated = False
        timestep = 0
        ep_reward = 0.0
        while not (terminated or truncated):
            action, _ = agent.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            if args.history_length <= 1:
                observation = observation[0]
            ep_reward += reward
            timestep += 1

            if terminated or truncated:
                num_timesteps.append(timestep)
                success.append(1 if (terminated and ep_reward > 0.0) else 0)
        ep_rewards.append(ep_reward)

        progress_bar.set_description(
            f"[Testing {agent_type}] Avg. number of steps: {np.mean(num_timesteps):.3f} +/- {np.std(num_timesteps):.3f} | Success percentage: {np.mean(success) * 100:.3f}% | Avg. reward: {np.mean(ep_rewards):.3f} +/- {np.std(ep_rewards):.3f}")

    avg_steps = np.mean(num_timesteps)
    std_steps = np.std(num_timesteps)
    success_percentage = np.mean(success) * 100
    ep_duration = np.mean(ep_durations)
    ep_duration_std = np.std(ep_durations)
    avg_reward = np.mean(ep_rewards)
    std_reward = np.std(ep_rewards)
    with open(f"training/{agent_type}/eval_per_epoch_{trajectories_path.split(sep='/')[-1]}{suff}.txt", "a+") as f:
        f.write(
            f"Avg. number of steps: {avg_steps:.3f} +/- {std_steps:.3f} | Success percentage: {success_percentage:.3f}% | Avg. episode duration: {ep_duration} +/- {ep_duration_std} | Avg. reward: {avg_reward:.3f} +/- {std_reward:.3f}\n")
    f.close()

    return success_percentage


def train_bc(agent, env, data_loader, epochs, args, device="cuda", num_eval_games=30):
    assert device in ["cuda", "cpu"], "Unknown device."
    max_success = 0.0
    for e in range(epochs):
        progress_bar = tqdm(enumerate(data_loader), desc=f"Epoch {e}")
        avg_loss = 0.0
        for i, batch in progress_bar:
            agent.optimizer.zero_grad()
            # Forward pass + metrics computation
            training_metrics = agent.loss_calculator(agent.policy, batch["obs"].to(device), batch["act"].to(device))
            # Compute loss (rescale if minibatch_size != batch_size, see original BC code)
            loss = training_metrics.loss * batch["act"].shape[0] / agent.batch_size
            loss.backward()
            agent.optimizer.step()
            avg_loss += loss.item()
            progress_bar.set_description(f"Epoch {e} - Avg. Loss: {avg_loss / (i + 1)}")

        # Test agent on env at the end of each epoch
        success = evaluate_agent(
            agent_type="bc",
            bc_agent=agent,
            env=env,
            num_games=num_eval_games,
            history_length=args.history_length,
            trajectories_path=f"{args.trajectories_base_path}/{args.env_name}"
        )
        if success > max_success:
            print("Saving new best")
            save_stable_model(Path(f"models/bc/{args.env_name}/checkpoints/"), agent.policy, "best_model.zip")
            torch.save(agent.policy.state_dict(), f"models/bc/{args.env_name}/checkpoints/best_model.pth")
            max_success = success

    return agent


def collate_fn(batch):
    return {
        'observations': batch["obs"],
        'actions': torch.tensor([x['labels'] for x in batch])
    }


def main(args):
    if args.headless:
        pyglet.options["headless"] = True
    ds = MiniWorldDataset(f"{args.trajectories_base_path}/{args.env_name}", history_length=args.history_length)
    data_loader = DataLoader(ds, batch_size=args.batch_size, num_workers=8, shuffle=True)

    # Env for testing
    test_env = gym.make(args.env_name, render_mode="human")
    test_env = TorchObsGymWrapper(test_env, history_length=args.history_length)

    net_arch = dict(pi=[256, 256], vf=[256, 256])
    feats_dim = 1024
    lr_schedule = "constant"

    input_size = (1, args.history_length, 3, 60, 80)
    post_wrappers = [lambda env, _: TorchObsGymWrapper(RolloutInfoWrapper(env), history_length=args.history_length)]

    rng = np.random.default_rng(0)
    env = make_vec_env(
        env_name=args.env_name,
        rng=rng,
        n_envs=1,
        post_wrappers=post_wrappers
    )

    idm_encoder_kwargs = dict(
        feats_dim=feats_dim,
        conv3d_in_channels=args.history_length,
        conv3d_out_channels=128,
        resnet_in_channels=[128, 64, 128],
        resnet_out_channels=[64, 128, 128],
        input_size=input_size,
        use_conv3d=True,
        device="cuda"
    )
    policy = ActorCriticPolicy(
        observation_space=gym.spaces.Box(low=0.0, high=1.0, shape=input_size[1:]),
        action_space=env.action_space,
        features_extractor_class=CausalIDMEncoder,
        features_extractor_kwargs=idm_encoder_kwargs,
        activation_fn=torch.nn.ReLU,
        net_arch=net_arch,
        lr_schedule=constant_lr if lr_schedule == "constant" else decreasing_lr
    )

    bc_trainer = bc.BC(
        policy=policy,
        observation_space=gym.spaces.Box(low=0.0, high=1.0, shape=input_size[1:]),
        action_space=env.action_space,
        demonstrations=None,
        rng=rng,
        optimizer_kwargs=dict(lr=1e-4)
    )

    env.close()

    os.makedirs(f"models/bc/{args.env_name}/checkpoints", exist_ok=True)
    os.makedirs(f"logs/{args.env_name}", exist_ok=True)

    agent = train_bc(bc_trainer, test_env, data_loader, epochs=args.epochs, args=args)
    save_stable_model(Path(f"models/bc/{args.env_name}"), agent.policy, f"model_{args.epochs}_ep_{len(os.listdir(f'trajectories/{args.env_name}')) - 1}_traj_{args.history_length}_hist_len.zip")
    torch.save(agent.policy.state_dict(), f=f"models/bc/{args.env_name}/model_{args.epochs}_ep_{len(os.listdir(f'{args.trajectories_base_path}/{args.env_name}')) - 1}_traj_{args.history_length}_hist_len.pth")


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('-env', '--env-name', required=True, type=str)
    arg_parser.add_argument('-d', '--trajectories-base-path', required=True, type=str)
    arg_parser.add_argument('-vd', '--validation-traj', required=True, type=str)
    arg_parser.add_argument('-e', '--epochs', required=True, type=int)
    arg_parser.add_argument('-b', '--batch-size', default=32, type=int)
    arg_parser.add_argument('-hist', '--history-length', default=16, type=int)
    arg_parser.add_argument('-head', '--headless', action='store_true')
    arg_parser.add_argument('-p', '--precompute', action="store_true")

    args = arg_parser.parse_args()

    main(args)
