import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import time

import numpy as np
import gymnasium as gym
import miniworld
import torch

from tqdm import tqdm
from stable_baselines3.common.policies import NatureCNN, ActorCriticPolicy
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper

from resnet_encoder import CausalIDMEncoder
from wrappers import TorchObsGymWrapper
from utils import constant_lr


SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main(args):
    env = gym.make(args.env_name, render_mode="human", max_episode_steps=args.max_steps)
    input_size = (1, args.history_length, 3, 60, 80)

    env = TorchObsGymWrapper(env, history_length=args.history_length)

    net_arch = dict(pi=[256, 256], vf=[256, 256]) if args.env_name != "rlgym" else dict(pi=[64, 64], vf=[64, 64])
    feats_dim = 512 if args.history_length <= 1 else 1024
    lr_schedule = "constant"

    os.makedirs(f"results/{args.env_name}", exist_ok=True)
    with open(f"results/{args.env_name}/bc_test_results.txt", "a+") as f:
        if args.deterministic:
            f.write(f"NEW RUN - DETERMINISTIC - MODEL: {args.model_name}\n")
        else:
            f.write(f"NEW RUN - STOCHASTIC - MODEL: {args.model_name}\n")
    f.close()

    if args.history_length <= 1:
        agent = ActorCriticPolicy(
            observation_space=gym.spaces.Box(low=0.0, high=255.0, shape=(3, 60, 80), dtype=np.uint8),
            action_space=env.action_space,
            features_extractor_class=NatureCNN,
            features_extractor_kwargs=dict(features_dim=feats_dim),
            activation_fn=torch.nn.ReLU,
            net_arch=net_arch,
            lr_schedule=constant_lr if lr_schedule == "constant" else None
        )
    else:
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
        agent = ActorCriticPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            features_extractor_class=CausalIDMEncoder,
            features_extractor_kwargs=idm_encoder_kwargs,
            activation_fn=torch.nn.ReLU,
            net_arch=net_arch,
            lr_schedule=constant_lr if lr_schedule == "constant" else None
        )
        agent.action_net.to(DEVICE)

    agent.load_state_dict(torch.load(f"models/bc/{args.env_name}/{args.model_name}"))

    for r in range(args.num_test_runs):
        num_timesteps = []
        success = []
        ep_durations = []
        ep_rewards = []
        progress_bar = tqdm(range(args.eval_episodes))
        progress_bar.set_description(f"[RUN {r + 1}/{args.num_test_runs}] Testing {args.model_name}")

        for i in progress_bar:
            start = time.time()
            observation, info = env.reset(seed=SEED)
            if args.history_length <= 1:
                observation = observation[0]
            terminated = False
            truncated = False
            obs = []
            acts = []
            timestep = 0
            ep_reward = 0.0
            while not (terminated or truncated):
                action, _ = agent.predict(observation, deterministic=args.deterministic)
                if args.record:
                    obs.append(observation[args.history_length - 1])
                    acts.append(action)
                observation, reward, terminated, truncated, info = env.step(action)
                if args.history_length <= 1:
                    observation = observation[0]
                ep_reward += reward
                if args.show:
                    env.render()
                timestep += 1

                if terminated or truncated:
                    num_timesteps.append(timestep)
                    success.append(1 if (terminated and ep_reward > 0.0) else 0)

            end = time.time()
            ep_durations.append(float(end - start))
            ep_rewards.append(ep_reward)

            progress_bar.set_description(f"Avg. number of steps: {np.mean(num_timesteps):.3f} +/- {np.std(num_timesteps):.3f} | Success percentage: {np.mean(success) * 100:.3f}% | Avg. reward: {np.mean(ep_rewards):.3f} +/- {np.std(ep_rewards):.3f}")

            if args.record:
                os.makedirs(f"recorded/{args.env_name}/BC/", exist_ok=True)
                np.savez_compressed(f"recorded/{args.env_name}/BC/game_{i}.npz", observations=np.array(obs), actions=np.array(acts))

        avg_steps = np.mean(num_timesteps)
        std_steps = np.std(num_timesteps)
        success_percentage = np.mean(success) * 100
        ep_duration = np.mean(ep_durations)
        ep_duration_std = np.std(ep_durations)
        avg_reward = np.mean(ep_rewards)
        std_reward = np.std(ep_rewards)
        with open(f"results/{args.env_name}/bc_test_results.txt", "a+") as f:
            f.write(f"Avg. number of steps: {avg_steps:.3f} +/- {std_steps:.3f} | Success percentage: {success_percentage:.3f}% | Avg. episode duration: {ep_duration} +/- {ep_duration_std} | Avg. reward: {avg_reward:.3f} +/- {std_reward:.3f}\n")
        f.close()

    env.close()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('-env', "--env-name", required=True, type=str)
    arg_parser.add_argument('-d', '--model-name', required=True, type=str)
    arg_parser.add_argument('-t', '--eval-episodes', required=True, type=int)
    arg_parser.add_argument('-hist', '--history-length', default=16, type=int)
    arg_parser.add_argument('-st', '--max-steps', default=1000, type=int)
    arg_parser.add_argument('-r', '--num-test-runs', default=3, type=int)
    arg_parser.add_argument('-rec', '--record', action="store_true")
    arg_parser.add_argument('-s', '--show', action='store_true')
    arg_parser.add_argument('-det', '--deterministic', action="store_true")

    args = arg_parser.parse_args()

    main(args)
