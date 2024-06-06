import os
import argparse

import numpy as np
import gymnasium as gym
import miniworld
import torch

from stable_baselines3.common.policies import ActorCriticPolicy
from tqdm import tqdm

from zip_agent import ZIPAgent
from resnet_encoder import CausalIDMEncoder
from utils import constant_lr
from wrappers import TorchObsGymWrapper


def test(args):
    env = gym.make(args.env_name, render_mode="human", max_episode_steps=args.max_steps)
    env = TorchObsGymWrapper(env, history_length=args.history_length)
    if args.encoder:
        net_arch = dict(pi=[256, 256], vf=[256, 256])
        feats_dim = 1024
        lr_schedule = "constant"
        idm_encoder_kwargs = dict(
            feats_dim=feats_dim,
            conv3d_in_channels=args.history_length,
            conv3d_out_channels=128,
            resnet_in_channels=[128, 64, 128],
            resnet_out_channels=[64, 128, 128],
            input_size=(1, args.history_length, 3, 60, 80),
            use_conv3d=True,
            device="cuda"
        )
        encoder = ActorCriticPolicy(
            observation_space=gym.spaces.Box(low=0.0, high=1.0, shape=(args.history_length, 3, 60, 80)),
            action_space=env.action_space,
            features_extractor_class=CausalIDMEncoder,
            features_extractor_kwargs=idm_encoder_kwargs,
            activation_fn=torch.nn.ReLU,
            net_arch=net_arch,
            lr_schedule=constant_lr if lr_schedule == "constant" else None
        )
        encoder.load_state_dict(torch.load(args.encoder_path))
    else:
        encoder = None
    agent = ZIPAgent(embeddings_path=f"trajectories/{args.env_name}",
                     max_followed=args.max_followed,
                     divergence_scaling_factor=args.divergence_factor,
                     debug=args.debug,
                     grayscale=args.grayscale,
                     encoder=encoder,
                     history_length=args.history_length
                     )

    for r in range(args.num_test_runs):
        num_timesteps = []
        success = []
        ep_rewards = []
        progress_bar = tqdm(range(args.num_games))

        for i in progress_bar:
            observation, info = env.reset()
            agent.reset()
            terminated = False
            truncated = False
            obs = []
            acts = []
            timestep = 0
            ep_reward = 0.0
            while not (terminated or truncated):
                if not args.encoder:
                    observation = observation.astype(np.float32) / 255
                action = agent.get_action(observation)
                if args.record:
                    obs.append(observation[args.history_length - 1])
                    acts.append(action)
                observation, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                if args.show:
                    env.render()
                timestep += 1

                if terminated or truncated:
                    num_timesteps.append(timestep)
                    if truncated:
                        success.append(0)
                    else:
                        success.append(1)
            progress_bar.set_description(f"Current success rate: {(float(np.sum(success)) / (i + 1)) * 100:.3f}%")
            ep_rewards.append(ep_reward)
            if args.record:
                os.makedirs(f"recorded/{args.env_name}/ZIP/", exist_ok=True)
                np.savez_compressed(f"recorded/{args.env_name}/ZIP/game_{i}.npz", observations=np.array(obs), actions=np.array(acts))

        env.close()
        avg_steps = np.mean(num_timesteps)
        std_steps = np.std(num_timesteps)
        avg_reward = np.mean(ep_rewards)
        std_reward = np.std(ep_rewards)
        success_percentage = (float(np.sum(success)) / args.num_games) * 100
        print(f"Avg. number of steps: {avg_steps:.3f} +/- {std_steps:.3f} | Success percentage: {success_percentage:.3f}% | Avg. reward: {avg_reward:.3f} +/- {std_reward:.3f}\n")


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('-g', '--env-name', required=True, type=str)
    arg_parser.add_argument('-n', '--num-games', required=True, type=int)
    arg_parser.add_argument('-tr', '--num-test-runs', required=True, type=int)
    arg_parser.add_argument('-f', '--max-followed', default=64, type=int)
    arg_parser.add_argument('-d', '--divergence-factor', default=2.0, type=float)
    arg_parser.add_argument('-s', '--max-steps', default=1000, type=int)
    arg_parser.add_argument('-hist', '--history-length', default=16, type=int)
    arg_parser.add_argument('-b', '--debug', action="store_true")
    arg_parser.add_argument('-gs', '--grayscale', action="store_true")
    arg_parser.add_argument('-e', '--encoder', action='store_true')
    arg_parser.add_argument('-p', '--encoder-path', default="", type=str)
    arg_parser.add_argument('-sh', '--show', action="store_true")
    arg_parser.add_argument('-rec', '--record', action="store_true")

    args = arg_parser.parse_args()

    test(args)
