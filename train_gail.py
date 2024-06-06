import os
import argparse
import random

import pyglet
# pyglet.options["headless"] = True

import numpy as np
import gymnasium as gym
import torch as th
from tqdm import tqdm
from pathlib import Path

import torch

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy, CnnPolicy

from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
from imitation.policies.serialize import save_stable_model
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data.serialize import Trajectory
from imitation.util.util import make_vec_env
from imitation.rewards.reward_nets import BasicRewardNet

from resnet_encoder import CausalIDMEncoder
from wrappers import TorchObsGymWrapper
from train_bc import MiniWorldDataset
from utils import stack_obs_frames


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SEED = 100000


class MiniWorldAdversarialDataset(MiniWorldDataset):
    def __init__(self, data_path, history_length=16):
        super().__init__(data_path=data_path, history_length=history_length)

    def __len__(self):
        return self.num_traj()

    def __getitem__(self, item):
        traj = np.load(os.path.join(self.data_path, f"{int(self.traj_metadata[item, 0])}.npz"), allow_pickle=True)
        if traj["actions"].shape[0] > 1:
            obs = traj["observations"].astype(np.float32) / 255
            obs = obs.transpose((0, 3, 1, 2))
            obs = stack_obs_frames(frames=obs, history_length=self.history_length)
            trajectory = Trajectory(obs=obs,
                                    acts=traj["actions"][:-1],
                                    infos=None,
                                    terminal=True)
        else:
            trajectory = None

        return trajectory

class TemporallyExtendedCnnRewardNet(BasicRewardNet):
    def __init__(self, action_space, observation_space, feats_dim,
                 conv3d_in_channels,
                 conv3d_out_channels,
                 resnet_in_channels,
                 resnet_out_channels,
                 input_size,
                 use_conv3d,
                 device,
                 use_next_state=False,
                 use_done=False):
        super().__init__(observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(feats_dim,)),
                         action_space=action_space)
        self.network = CausalIDMEncoder(
            observation_space=observation_space,
            feats_dim=feats_dim,
            conv3d_in_channels=conv3d_in_channels,
            conv3d_out_channels=conv3d_out_channels,
            resnet_in_channels=resnet_in_channels,
            resnet_out_channels=resnet_out_channels,
            input_size=input_size,
            use_conv3d=use_conv3d,
            device=device
        )
        self.use_next_state = use_next_state
        self.use_done = use_done
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(feats_dim + action_space.n,))

    def forward(self, state: th.Tensor, action: th.Tensor, next_state: th.Tensor, done: th.Tensor) -> th.Tensor:
        image_feats = self.network(state)
        next_image_feats = None
        if self.use_next_state:
            next_image_feats = self.network(next_state)

        return super().forward(image_feats, action, next_image_feats, done)


def adversarial_collate_fn(trajectories):
    return trajectories


def evaluate_agent(agent_type, gail_trainer, test_env, num_test_games, trajectories_path):
    os.makedirs(f"training/{agent_type}/", exist_ok=True)

    agent = gail_trainer.policy
    suff = ""

    num_timesteps = []
    success = []
    ep_durations = []
    ep_rewards = []
    progress_bar = tqdm(range(num_test_games))
    for j in progress_bar:
        observation, _ = test_env.reset()
        terminated = False
        truncated = False
        timestep = 0
        ep_reward = 0.0
        while not (terminated or truncated):
            action, _ = agent.predict(observation, deterministic=True)

            observation, reward, terminated, truncated, info = test_env.step(action)
            ep_reward += reward

            timestep += 1

            if terminated or truncated:
                num_timesteps.append(timestep)
                success.append(1 if (terminated and ep_reward > 0.0) else 0)

        ep_durations.append(timestep)
        ep_rewards.append(ep_reward)
        progress_bar.set_description(
            f"[Testing {agent_type}] Avg. number of steps: {np.mean(num_timesteps):.3f} +/- {np.std(num_timesteps):.3f} | Success percentage: {np.mean(success) * 100:.3f}% | Avg. reward: {np.mean(ep_rewards):.3f} +/- {np.std(ep_rewards):.3f}")

    avg_steps = np.mean(num_timesteps)
    std_steps = np.std(num_timesteps)
    success_percentage = np.mean(success) * 100
    avg_reward = np.mean(ep_rewards)
    std_reward = np.std(ep_rewards)
    with open(f"training/{agent_type}/eval_per_epoch_{trajectories_path.split(sep='/')[-1]}{suff}.txt", "a+") as f:
        f.write(
            f"Avg. number of steps: {avg_steps:.3f} +/- {std_steps:.3f} | Success percentage: {success_percentage:.3f}% | Avg. reward: {avg_reward:.3f} +/- {std_reward:.3f}\n")
    f.close()

    return success_percentage


def train_gail(gail_trainer, test_env, epochs, trajectories_path, num_test_games=30, device="cuda"):
    assert device in ["cuda", "cpu"], "Unknown device."
    steps_per_batch = gail_trainer.gen_train_timesteps * 10
    max_success = 0.0
    for e in range(epochs):
        gail_trainer.train(total_timesteps=steps_per_batch)
        # Test
        success = evaluate_agent(
            agent_type="gail",
            gail_trainer=gail_trainer,
            test_env=test_env,
            num_test_games=num_test_games,
            trajectories_path=trajectories_path
        )
        if success > max_success:
            save_stable_model(Path(f"models/gail/{args.env_name}/checkpoints"), gail_trainer.gen_algo,
                              f"best_model_run_{args.run}.zip")
            max_success = success

    return gail_trainer


def collate_fn(batch):
    return {
        'observations': batch["obs"],
        'actions': torch.tensor([x['labels'] for x in batch])
    }


def main(args):
    if args.headless:
        pyglet.options["headless"] = True

    net_arch = dict(pi=[256, 256], vf=[256, 256])
    feats_dim = 1024
    reward_feats_dim = 256
    policy_type = "CnnPolicy"
    batch_size = 64
    gamma = 0.99
    lr = 1e-4
    demo_batch_size = 128
    replay_buffer_capacity = 5120
    n_disc_updates = 4
    gen_train_timesteps = 2048
    print(f"Total training steps: {gen_train_timesteps * args.epochs}")
    # Env specific options
    input_size = (1, args.history_length, 3, 60, 80)
    post_wrappers = [lambda env, _: TorchObsGymWrapper(RolloutInfoWrapper(env), history_length=args.history_length)]

    rng = np.random.default_rng(SEED)
    env = make_vec_env(
        env_name=args.env_name,
        rng=rng,
        n_envs=1,
        post_wrappers=post_wrappers,
        max_episode_steps=args.max_steps
        # for computing rollouts
    )
    test_env = gym.make(args.env_name, render_mode="human", max_episode_steps=args.max_steps)
    test_env = TorchObsGymWrapper(test_env, history_length=args.history_length)

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
    policy_kwargs = dict(
        features_extractor_class=CausalIDMEncoder,
        features_extractor_kwargs=idm_encoder_kwargs,
        net_arch=net_arch,
        activation_fn=torch.nn.ReLU
    )
    learner = PPO(
        env=env,
        policy=CnnPolicy if policy_type == "CnnPolicy" else MlpPolicy,
        batch_size=batch_size,
        ent_coef=0.02,
        learning_rate=lr,
        gamma=gamma,
        n_epochs=10,
        seed=random.randint(a=0, b=SEED),
        policy_kwargs=policy_kwargs,
        n_steps=gen_train_timesteps
    )
    reward_net = TemporallyExtendedCnnRewardNet(
        action_space=env.action_space,
        observation_space=env.observation_space,
        feats_dim=reward_feats_dim,
        conv3d_in_channels=idm_encoder_kwargs["conv3d_in_channels"],
        conv3d_out_channels=idm_encoder_kwargs["conv3d_out_channels"],
        resnet_in_channels=idm_encoder_kwargs["resnet_in_channels"],
        resnet_out_channels=idm_encoder_kwargs["resnet_out_channels"],
        input_size=idm_encoder_kwargs["input_size"],
        use_conv3d=idm_encoder_kwargs["use_conv3d"],
        device=idm_encoder_kwargs["device"]
    )

    trajects = []
    traj_path = os.listdir(f'{args.trajectories_base_path}/{args.env_name}')
    traj_path.remove("trajectories_lengths.csv")
    for traj in traj_path:
        trajectory_path = os.path.join(f'{args.trajectories_base_path}/{args.env_name}', traj)
        data = np.load(trajectory_path, allow_pickle=True)
        if data["actions"].shape[0] > 1:
            obs = data["observations"].astype(np.float32) / 255
            obs = obs.transpose((0, 3, 1, 2))
            obs = stack_obs_frames(obs, history_length=args.history_length)
            trajectory = Trajectory(obs=obs,
                                    acts=data["actions"][:-1],
                                    infos=None,
                                    terminal=True)
            trajects.append(trajectory)
        else:
            print(f"Trajectory {traj} excluded for lack of actions.")
    print(f"Total number of trajectories in the dataset: {len(trajects)}")

    transitions = rollout.flatten_trajectories(trajects)

    gail_trainer = GAIL(
        demonstrations=transitions,
        demo_batch_size=demo_batch_size,
        gen_replay_buffer_capacity=replay_buffer_capacity,
        n_disc_updates_per_round=n_disc_updates,
        venv=env,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True,
        log_dir="tensorboard_logs",
        init_tensorboard=True,
        init_tensorboard_graph=True,
    )

    os.makedirs(f"models/gail/{args.env_name}/checkpoints", exist_ok=True)
    os.makedirs(f"logs/{args.env_name}", exist_ok=True)

    agent = train_gail(gail_trainer, test_env, epochs=args.epochs, trajectories_path=f'{args.trajectories_base_path}/{args.env_name}')
    save_stable_model(Path(f"models/gail/{args.env_name}"), agent.gen_algo,
                      f"model_{args.epochs}_ep_{len(os.listdir(f'{args.trajectories_base_path}/{args.env_name}')) - 1}_traj_{args.history_length}_hist_len_run_{args.run}.zip")


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('-env', '--env-name', required=True, type=str)
    arg_parser.add_argument('-e', '--epochs', required=True, type=int)
    arg_parser.add_argument('-t', '--trajectories-base-path', required=True, type=str)
    arg_parser.add_argument('-b', '--batch-size', default=4, type=int)
    arg_parser.add_argument('-hist', '--history-length', default=16, type=int)
    arg_parser.add_argument('-s', '--max-steps', default=1000, type=int)
    arg_parser.add_argument('-head', '--headless', action='store_true')
    arg_parser.add_argument('-p', '--precompute', action="store_true")
    arg_parser.add_argument('-r', '--run', default=1, type=int)
    args = arg_parser.parse_args()

    main(args)
